import unittest

import numpy as np

from care_ecg.agents import ResponseAgent
from care_ecg.causal import BayesianCausalModel
from care_ecg.cli import _prepare_signals
from care_ecg.config import CAREConfig
from care_ecg.counterfactual import CounterfactualEngine
from care_ecg.data import apply_global_standardizer, fit_global_standardizer
from care_ecg.metrics import (
    SCPTextMapper,
    grounding_metrics,
    scp_text_metrics,
    verifier_diagnostics,
)
from care_ecg.pipeline import CAREECGPipeline
from care_ecg.retrieval import CausalRetriever, KnowledgeDocument
from care_ecg.training import latent_diagnostics, stratified_holdout_indices
from care_ecg.types import CausalDriver, DiagnosisState, HistorySummary, RetrievedFact
from care_ecg.verification import hallucination_risk, unsupported_claim_risk


class ReproducibilityTests(unittest.TestCase):
    def test_crc_does_not_reward_missing_causal_attribution(self):
        missing = grounding_metrics("query", "explanation", [], [])
        available = grounding_metrics(
            "query",
            "z0 supports the result",
            [],
            [CausalDriver("z0", 1, 0.5, 0.5)],
        )

        self.assertEqual(missing["crc"], 0.0)
        self.assertEqual(missing["causal_attribution_availability"], 0.0)
        self.assertEqual(available["crc"], 1.0)
        self.assertEqual(available["causal_attribution_availability"], 1.0)

    def test_component_sensitivity_is_not_presented_as_fused_attribution(self):
        metrics = grounding_metrics(
            "query",
            "z0 supports the result",
            [],
            [CausalDriver("z0", 1, 0.5, 0.5, scope="bayesian_component")],
        )

        self.assertEqual(metrics["causal_attribution_availability"], 1.0)
        self.assertEqual(metrics["fused_causal_attribution_availability"], 0.0)

    def test_default_beta_avoids_known_collapse_prone_setting(self):
        self.assertEqual(CAREConfig().training.beta, 0.001)

    def test_model_initialization_uses_configured_seed(self):
        config = CAREConfig()
        config.model.hidden_dim = 16
        config.model.latent_dim = 4
        config.model.lstm_layers = 1

        first = CAREECGPipeline.build_model(config)
        second = CAREECGPipeline.build_model(config)

        for left, right in zip(first.parameters(), second.parameters()):
            np.testing.assert_array_equal(left.detach().numpy(), right.detach().numpy())

    def test_stratified_holdout_is_disjoint_and_keeps_every_class_in_train(self):
        labels = np.asarray([0] * 5 + [1] * 5 + [2])
        train, validation = stratified_holdout_indices(labels, 0.2, seed=7)

        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(range(len(labels))))
        self.assertEqual(set(labels[train]), {0, 1, 2})
        self.assertEqual(set(labels[validation]), {0, 1})

    def test_native_rate_training_data_is_preprocessed_unless_explicitly_marked(self):
        config = CAREConfig()
        config.data.sample_rate = 10
        config.data.seconds = 1
        base = np.arange(10, dtype=np.float32)
        signals = np.stack([np.stack([base + 100 * lead for lead in range(12)])])

        prepared = _prepare_signals(signals, config, source_rate=None)
        untouched = _prepare_signals(signals, config, source_rate=None, preprocessed=True)

        np.testing.assert_allclose(np.median(prepared, axis=2), 0.0, atol=1e-6)
        np.testing.assert_array_equal(untouched, signals)

    def test_global_standardizer_is_fitted_once_and_reused(self):
        training = np.asarray([[[1.0, 3.0]], [[5.0, 7.0]]], dtype=np.float32)
        validation = np.asarray([[[9.0, 11.0]]], dtype=np.float32)

        mean, scale = fit_global_standardizer(training)
        transformed_train = apply_global_standardizer(training, mean, scale)
        transformed_validation = apply_global_standardizer(validation, mean, scale)

        self.assertAlmostEqual(float(transformed_train.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(transformed_train.std()), 1.0, places=6)
        self.assertGreater(float(transformed_validation.mean()), 1.0)

    def test_latent_diagnostics_reports_collapsed_units(self):
        values = np.column_stack([np.arange(5), np.ones(5)])
        diagnostics = latent_diagnostics(values)
        self.assertEqual(diagnostics["active_units"], 1)
        self.assertEqual(diagnostics["latent_dimensions"], 2)

    def test_scp_mapper_uses_boundaries_negation_and_mention_order(self):
        mapper = SCPTextMapper(
            {
                "MI": ["mi", "myocardial infarction"],
                "NORM": ["normal ecg", "normal"],
            }
        )

        self.assertNotIn("MI", mapper.map("The family history is minimal."))
        self.assertNotIn("MI", mapper.map("No myocardial infarction is present."))
        self.assertEqual(
            mapper.map("Normal ECG initially; myocardial infarction is later discussed."),
            ["NORM", "MI"],
        )

    def test_scp_mapper_prefers_exact_alias_over_fuzzy_prose(self):
        mapper = SCPTextMapper(
            {"NORM": ["normal electrocardiogram"]},
            fuzzy_threshold=0.8,
        )

        self.assertEqual(
            mapper.map("Normal electrocardiographs are discussed before NORM."),
            ["NORM"],
        )

    def test_scp_mapper_retains_clinically_constrained_typo_matching(self):
        mapper = SCPTextMapper({"MI": ["myocardial infarction"]}, fuzzy_threshold=0.84)

        self.assertEqual(mapper.map("Possible myocardial infraction."), ["MI"])

    def test_scp_metrics_expose_first_vs_any_mention(self):
        mapper = SCPTextMapper({"MI": ["myocardial infarction"], "NORM": ["normal ecg"]})
        metrics = scp_text_metrics(
            ["MI"], ["Normal ECG. The final diagnosis is myocardial infarction."], mapper
        )

        self.assertEqual(metrics["scp_first_accuracy"], 0.0)
        self.assertEqual(metrics["scp_any_accuracy"], 1.0)

    def test_verifier_diagnostics_exposes_wrong_diagnosis_passes(self):
        metrics = verifier_diagnostics(["MI", "NORM"], ["NORM", "NORM"], [False, False])
        self.assertEqual(metrics["wrong_diagnosis_count"], 1.0)
        self.assertEqual(metrics["wrong_diagnosis_pass_rate"], 1.0)

    def test_claim_risk_catches_false_extra_claim_that_paper_hr_misses(self):
        facts = [RetrievedFact("f1", "Sinus rhythm is present.", "source", 1.0)]
        explanation = (
            "Sinus rhythm is present. [Fact 1]\n"
            "Acute STEMI and ventricular fibrillation are present. [Fact 1]"
        )

        self.assertEqual(hallucination_risk(explanation, facts), 0.0)
        self.assertGreater(unsupported_claim_risk(explanation, facts), 0.0)

    def test_unsafe_second_generation_uses_deterministic_evidence_fallback(self):
        fact = RetrievedFact("f1", "Sinus rhythm is present.", "source", 1.0)

        class UnsafeGenerator:
            def generate(self, prompt, facts):
                return "Acute STEMI is certain. [Fact 1]"

        diagnosis = DiagnosisState({"NORM": 1.0}, "NORM", [], [])
        history = HistorySummary("none", [], {}, "No history.")
        response = ResponseAgent(UnsafeGenerator()).respond(
            "What is the diagnosis?", diagnosis, [fact], history
        )

        self.assertTrue(response.used_fallback)
        self.assertTrue(response.abstained)
        self.assertIn("Evidence-only fallback", response.explanation)
        self.assertNotIn("Acute STEMI", response.explanation)

    def test_zero_score_retrieval_returns_no_arbitrary_document(self):
        retriever = CausalRetriever().fit(
            [KnowledgeDocument("a", "atrial rhythm evidence", "source")]
        )
        self.assertEqual(retriever.retrieve("banana", top_k=1), [])

    def test_counterfactual_uses_only_observed_states_and_honors_empty_candidates(self):
        latents = np.asarray([[2], [2], [1], [1]])
        labels = np.asarray([1, 1, 0, 0])
        model = BayesianCausalModel(["negative", "positive"], ["z0"], 3).fit(latents, labels)

        self.assertEqual(model.supported_states("z0"), [1, 2])
        result = CounterfactualEngine(model).search([1], "positive", candidate_features=[])
        self.assertFalse(result.found)
        self.assertIsNone(result.feature)


if __name__ == "__main__":
    unittest.main()
