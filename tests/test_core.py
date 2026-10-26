import unittest

import numpy as np

from care_ecg.agents import CausalRAGAgent, DiagnosisAgent, ResponseAgent
from care_ecg.causal import BayesianCausalModel, DiscreteBayesianNetwork, K2StructureLearner
from care_ecg.counterfactual import CounterfactualEngine
from care_ecg.data import preprocess_ecg
from care_ecg.discretization import QuantileDiscretizer
from care_ecg.generation import TemplateGenerator
from care_ecg.retrieval import CausalRetriever, KnowledgeDocument
from care_ecg.types import DiagnosisState, HistorySummary, RetrievedFact


class CoreTests(unittest.TestCase):
    def test_preprocess_orientation_resample_and_nonfinite(self):
        signal = np.zeros((100, 12), dtype=np.float32)
        signal[10, 0] = np.nan
        result = preprocess_ecg(signal, source_rate=100, target_rate=50, seconds=1)
        self.assertEqual(result.shape, (12, 50))
        self.assertTrue(np.isfinite(result).all())

    def test_quantile_roundtrip(self):
        values = np.arange(60, dtype=float).reshape(20, 3)
        discretizer = QuantileDiscretizer(3).fit(values)
        states = discretizer.transform(values)
        self.assertEqual(states.shape, values.shape)
        restored = QuantileDiscretizer.from_dict(discretizer.to_dict())
        np.testing.assert_array_equal(states, restored.transform(values))

    def _causal_model(self):
        network = DiscreteBayesianNetwork(
            ["z0", "diagnosis"],
            {"z0": 3, "diagnosis": 2},
            {"z0": [], "diagnosis": ["z0"]},
        )
        z = np.repeat(np.arange(3), 20)
        y = (z >= 1).astype(int)
        network.fit(np.column_stack([z, y]), alpha=0.1)
        model = BayesianCausalModel(["negative", "positive"], ["z0"], 3)
        model.network = network
        return model

    def test_variable_elimination(self):
        model = self._causal_model()
        self.assertGreater(model.posterior([2])["positive"], 0.98)
        self.assertGreater(model.posterior([0])["negative"], 0.98)

    def test_k2_learns_predictive_parent(self):
        z0 = np.tile(np.arange(3), 50)
        z1 = np.random.default_rng(3).integers(0, 3, len(z0))
        diagnosis = (z0 > 0).astype(int)
        data = np.column_stack([z0, z1, diagnosis])
        parents = K2StructureLearner(max_parents=2).learn(
            data,
            ["z0", "z1", "diagnosis"],
            {"z0": 3, "z1": 3, "diagnosis": 2},
        )
        self.assertIn("z0", parents["diagnosis"])

    def test_counterfactual_single_edit(self):
        model = self._causal_model()
        result = CounterfactualEngine(model).search([0], "positive")
        self.assertTrue(result.found)
        self.assertEqual(result.feature, "z0")
        self.assertEqual(result.old_state, 0)
        self.assertEqual(result.new_state, 1)

    def test_counterfactual_needs_no_edit_when_target_already_wins(self):
        model = self._causal_model()
        result = CounterfactualEngine(model).search([2], "positive")
        self.assertTrue(result.found)
        self.assertEqual(result.edit_distance, 0)
        self.assertIsNone(result.feature)

    def test_neural_only_fusion_does_not_claim_causal_influence(self):
        model = self._causal_model()
        model.neural_weight = 1.0
        model.class_biases = np.asarray([0.0, 0.7])
        neural = np.asarray([0.1, 0.9])

        drivers = model.rank_drivers(
            [0], target_label="positive", neural_probabilities=neural
        )
        counterfactual = CounterfactualEngine(model).search(
            [0], "negative", neural_probabilities=neural
        )

        self.assertEqual(drivers, [])
        self.assertFalse(counterfactual.found)
        self.assertIsNone(counterfactual.feature)
        self.assertIn("reported fused model", counterfactual.message)

    def test_neural_only_diagnosis_exposes_separate_bayesian_component_sensitivity(self):
        model = self._causal_model()
        model.neural_weight = 1.0

        diagnosis = DiagnosisAgent(model).diagnose(
            np.asarray([0]), neural_probabilities=np.asarray([0.1, 0.9])
        )

        self.assertEqual(diagnosis.predicted_label, "positive")
        self.assertEqual(diagnosis.attribution_scope, "bayesian_component")
        self.assertTrue(diagnosis.drivers)
        self.assertTrue(all(driver.scope == "bayesian_component" for driver in diagnosis.drivers))

    def test_counterfactual_can_explicitly_analyze_bayesian_component(self):
        model = self._causal_model()
        model.neural_weight = 1.0

        result = CounterfactualEngine(model).search(
            [0],
            "positive",
            neural_probabilities=np.asarray([0.9, 0.1]),
            posterior_scope="bayesian_component",
        )

        self.assertTrue(result.found)
        self.assertEqual(result.scope, "bayesian_component")
        self.assertEqual(result.new_state, 1)

    def test_class_biases_round_trip_and_legacy_default(self):
        model = self._causal_model()
        model.class_biases = np.asarray([0.0, 0.7])

        serialized = model.to_dict()
        restored = BayesianCausalModel.from_dict(serialized)
        np.testing.assert_allclose(restored.class_biases, [0.0, 0.7])
        np.testing.assert_allclose(restored.fused_posterior_array([0]), model.fused_posterior_array([0]))

        serialized.pop("class_biases")
        legacy = BayesianCausalModel.from_dict(serialized)
        np.testing.assert_array_equal(legacy.class_biases, np.zeros(2))

    def test_zero_class_biases_preserve_uncalibrated_posterior(self):
        model = self._causal_model()

        raw = model.posterior_array([2])
        fused = model.fused_posterior_array([2])

        np.testing.assert_array_equal(fused, raw)

    def test_causal_retrieval(self):
        documents = [
            KnowledgeDocument("a", "atrial rhythm evidence", "source-a"),
            KnowledgeDocument("b", "ventricular conduction evidence", "source-b"),
        ]
        retriever = CausalRetriever().fit(documents)
        result = retriever.retrieve("atrial rhythm", top_k=1)
        self.assertEqual(result[0].fact_id, "a")

    def test_retrieval_reranks_with_diagnosis_metadata_and_records_scores(self):
        documents = [
            KnowledgeDocument("norm", "PTB-XL diagnostic category", "source", {"diagnostic_class": "NORM"}),
            KnowledgeDocument("mi", "PTB-XL diagnostic category", "source", {"diagnostic_class": "MI"}),
        ]
        retriever = CausalRetriever(metadata_weight=0.5).fit(documents)

        result = retriever.retrieve(
            "What is the diagnosis?", probabilities={"NORM": 0.1, "MI": 0.9}, top_k=1
        )

        self.assertEqual(result[0].fact_id, "mi")
        self.assertAlmostEqual(result[0].metadata["retrieval_structured_score"], 0.9)
        self.assertEqual(CausalRetriever.from_dict(retriever.to_dict()).metadata_weight, 0.5)

    def test_retrieval_excludes_competing_superclasses_but_keeps_generic_evidence(self):
        documents = [
            KnowledgeDocument(
                "generic",
                "A clinical differential can discuss MI and absence of NORM findings.",
                "review",
                {"scope": "method"},
            ),
            KnowledgeDocument(
                "norm-taxonomy",
                "PTB-XL defines NORM as a diagnostic superclass label.",
                "metadata",
                {"target_label": "NORM", "relation": "contains_label"},
            ),
            KnowledgeDocument(
                "norm-scp",
                "The NORM SCP statement belongs to the NORM superclass.",
                "metadata",
                {"diagnostic_class": "NORM", "scp_code": "NORM"},
            ),
            KnowledgeDocument(
                "mi-taxonomy",
                "PTB-XL defines MI as a diagnostic superclass label.",
                "metadata",
                {"target_label": "MI", "relation": "contains_label"},
            ),
            KnowledgeDocument(
                "mi-scp",
                "The IMI SCP statement belongs to the MI superclass.",
                "metadata",
                {"diagnostic_class": "MI", "scp_code": "IMI"},
            ),
        ]
        retriever = CausalRetriever(metadata_weight=0.5).fit(documents)
        probabilities = {"NORM": 0.49, "MI": 0.51}

        mi_result = retriever.retrieve(
            "Explain the diagnosis and differential.",
            probabilities=probabilities,
            top_k=5,
            diagnosed_label="MI",
        )
        repeated = retriever.retrieve(
            "Explain the diagnosis and differential.",
            probabilities=probabilities,
            top_k=5,
            diagnosed_label="MI",
        )
        norm_result = retriever.retrieve(
            "Explain the diagnosis and differential.",
            probabilities=probabilities,
            top_k=5,
            diagnosed_label="NORM",
        )

        mi_ids = [fact.fact_id for fact in mi_result]
        self.assertEqual(mi_ids, [fact.fact_id for fact in repeated])
        self.assertIn("generic", mi_ids)
        self.assertIn("mi-taxonomy", mi_ids)
        self.assertIn("mi-scp", mi_ids)
        self.assertNotIn("norm-taxonomy", mi_ids)
        self.assertNotIn("norm-scp", mi_ids)
        norm_ids = {fact.fact_id for fact in norm_result}
        self.assertIn("generic", norm_ids)
        self.assertIn("norm-taxonomy", norm_ids)
        self.assertIn("norm-scp", norm_ids)
        self.assertNotIn("mi-taxonomy", norm_ids)
        self.assertNotIn("mi-scp", norm_ids)
        self.assertTrue(mi_result)
        self.assertTrue(norm_result)

        agent_result = CausalRAGAgent(retriever, top_k=5).retrieve(
            "Explain the diagnosis and differential.",
            DiagnosisState(probabilities, "MI", [], []),
        )
        agent_ids = {fact.fact_id for fact in agent_result}
        self.assertIn("mi-scp", agent_ids)
        self.assertNotIn("norm-scp", agent_ids)

    def test_template_states_only_predicted_class_confidence(self):
        prompt = (
            "Patient Query: What is the diagnosis?\n"
            "Predicted Diagnosis: MI\n"
            "Diagnosis Posterior: NORM=0.200, MI=0.800\n"
            "Key Causal Factors (from VAE/Graph): z0=state-1 (influence 0.300)\n"
            "Attribution Scope: bayesian component\n"
            "Counterfactual Check: Not requested."
        )
        fact = RetrievedFact("f", "MI is a PTB-XL diagnostic class.", "source", 1.0)

        report = TemplateGenerator().generate(prompt, [fact])

        self.assertIn("stated MI class: 0.800", report)
        self.assertNotIn("NORM=", report)
        self.assertIn("Bayesian-component latent sensitivity", report)

    def test_response_fallback(self):
        fact = RetrievedFact("a", "retrieved evidence supports cautious review", "source", 1.0)

        class Generator:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt, facts):
                self.calls += 1
                if self.calls == 1:
                    return "An unrelated unsupported assertion."
                return "The retrieved evidence supports cautious review. [Fact 1]"

        generator = Generator()
        agent = ResponseAgent(generator, hallucination_threshold=0.5)
        diagnosis = DiagnosisState({"normal": 1.0}, "normal", [], [])
        history = HistorySummary("none", [], {}, "No history.")
        response = agent.respond("query", diagnosis, [fact], history)
        self.assertTrue(response.used_fallback)
        self.assertEqual(generator.calls, 2)
        self.assertEqual(response.hallucination_score, 0.0)


if __name__ == "__main__":
    unittest.main()
