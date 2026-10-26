import unittest
from types import SimpleNamespace

import numpy as np
import torch

from care_ecg.security.claims import ClaimOracle
from care_ecg.security.contracts import DefenseKind, ReleaseDecision, VictimSnapshot
from care_ecg.security.victim import (
    BudgetedVictim,
    InternalCAREECGVictim,
    QueryBudgetExceeded,
)
from care_ecg.types import CounterfactualResult, PipelineOutput, RetrievedFact


class TinyModel:
    def __init__(self):
        self.last_input = None

    def encode_and_classify(self, signal, sample=False):
        assert not sample
        self.last_input = signal
        latent = signal.mean(dim=(-2, -1), keepdim=False).unsqueeze(1)
        logits = torch.cat((-latent, latent), dim=1)
        return latent, latent, latent, logits


class TinyCausalModel:
    label_names = ("clean", "target")
    feature_names = ("z0",)
    neural_weight = 0.5
    class_biases = np.zeros(2, dtype=float)

    @staticmethod
    def posterior_array(states):
        del states
        return np.asarray([0.75, 0.25], dtype=float)


class TinyDiscretizer:
    thresholds_ = np.asarray([[0.0]], dtype=np.float32)

    @staticmethod
    def transform(latent):
        return (np.asarray(latent) >= 0.0).astype(int)


class TinyCounterfactualEngine:
    def __init__(self):
        self.calls = []
        self.no_edit = False
        self.fail = False

    def search(
        self,
        states,
        target_label,
        *,
        neural_probabilities,
        posterior_scope,
    ):
        self.calls.append(
            {
                "states": tuple(states),
                "target_label": target_label,
                "neural_probabilities": np.asarray(neural_probabilities).copy(),
                "posterior_scope": posterior_scope,
            }
        )
        if self.fail:
            raise RuntimeError("counterfactual backend unavailable")
        if self.no_edit:
            return CounterfactualResult(
                found=True,
                target_label=target_label,
                feature=None,
                old_state=None,
                new_state=None,
                posterior_before={"clean": 0.1, "target": 0.9},
                posterior_after={"clean": 0.1, "target": 0.9},
                edit_distance=0,
                message="target already satisfied",
                scope=posterior_scope,
            )
        return CounterfactualResult(
            found=True,
            target_label=target_label,
            feature="z0",
            old_state=0,
            new_state=1,
            posterior_before={"clean": 0.9, "target": 0.1},
            posterior_after={"clean": 0.1, "target": 0.9},
            edit_distance=1,
            message="change z0",
            scope=posterior_scope,
        )


class TinyPipeline:
    def __init__(self):
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            data=SimpleNamespace(
                leads=1,
                samples=4,
                sample_rate=100,
                normalization="global",
            )
        )
        self.normalization_stats = (1.0, 2.0)
        self.model = TinyModel()
        self.causal_model = TinyCausalModel()
        self.discretizer = TinyDiscretizer()
        self.counterfactual_engine = TinyCounterfactualEngine()
        self.diagnosis_agent = SimpleNamespace(
            counterfactual_engine=self.counterfactual_engine
        )
        self.retriever = SimpleNamespace(embeddings=None)
        self.calls = []
        self.report_override = None

    def run(
        self,
        signal,
        query,
        *,
        source_rate,
        history,
        patient_id,
        counterfactual_target,
    ):
        self.calls.append(
            {
                "signal": np.asarray(signal).copy(),
                "query": query,
                "source_rate": source_rate,
                "history": history,
                "patient_id": patient_id,
                "counterfactual_target": counterfactual_target,
            }
        )
        target = float(np.mean(signal)) >= 0.5
        probabilities = {"clean": 0.1 if target else 0.9, "target": 0.9 if target else 0.1}
        diagnosis = "target" if target else "clean"
        rationale = self.report_override or f"The supported diagnosis is {diagnosis}."
        return PipelineOutput(
            query=query,
            probabilities=probabilities,
            diagnosis=diagnosis,
            rationale=rationale,
            continuous_biomarkers={"z0": float(np.mean(signal))},
            discrete_biomarkers={"z0": int(target)},
            causal_drivers=[],
            causal_paths=[],
            retrieved_facts=[RetrievedFact("fact-1", "target evidence", "test", 0.8)],
            hallucination_score=0.1,
            used_fallback=False,
            raw_explanation_with_causal=rationale,
            unsupported_claim_score=0.0,
            abstained=False,
            counterfactual=None,
            metadata={
                "fake": True,
                "diagnosis_fusion": {
                    "neural_probabilities": probabilities,
                },
            },
        )


class RejectingVerifier:
    def __init__(self):
        self.snapshot = None
        self.kwargs = None

    def decide(self, snapshot, **kwargs):
        self.snapshot = snapshot
        self.kwargs = kwargs
        return ReleaseDecision(
            defense=DefenseKind.V2,
            released=False,
            score=2.0,
            threshold=1.0,
            reasons=("measurement conflict",),
        )


class SecurityVictimTests(unittest.TestCase):
    def test_surrogate_preserves_fused_forward_values_and_raw_mv_gradient(self):
        pipeline = TinyPipeline()
        victim = InternalCAREECGVictim(pipeline)
        signal = torch.full((1, 1, 4), 3.0, requires_grad=True)

        snapshot = victim.surrogate(signal, "target")
        neural = torch.softmax(torch.tensor([[-1.0, 1.0]]), dim=1)
        expected = 0.5 * torch.tensor([[0.75, 0.25]]) + 0.5 * neural

        torch.testing.assert_close(pipeline.model.last_input, torch.ones_like(signal))
        torch.testing.assert_close(snapshot.probabilities, expected)
        snapshot.probabilities[0, 1].backward()
        self.assertIsNotNone(signal.grad)
        self.assertGreater(float(signal.grad.abs().sum()), 0.0)
        self.assertEqual(snapshot.metadata["posterior_scope"], "internal_care_fused")

    def test_evaluate_runs_real_pipeline_and_uses_claim_oracle(self):
        pipeline = TinyPipeline()
        victim = InternalCAREECGVictim(
            pipeline,
            query="Explain the tracing",
            patient_id="patient-1",
            claim_extractor=ClaimOracle("target"),
        )

        snapshot = victim.evaluate(np.full((1, 4), 2.0, dtype=np.float32), "target")

        self.assertEqual(len(pipeline.calls), 1)
        self.assertIsNone(pipeline.calls[0]["counterfactual_target"])
        self.assertEqual(snapshot.diagnosis, "target")
        self.assertTrue(snapshot.released)
        self.assertTrue(snapshot.target_claim_present)
        self.assertEqual(snapshot.retrieved_fact_ids, ("fact-1",))
        self.assertEqual(victim.counterfactual_guidance(snapshot), (0, 1))
        self.assertTrue(snapshot.metadata["counterfactual_guidance"]["available"])
        self.assertIsNone(snapshot.metadata["counterfactual"])
        np.testing.assert_array_equal(victim.discretizer_thresholds, [[0.0]])

    def test_attack_target_does_not_change_released_victim_output(self):
        pipeline = TinyPipeline()
        victim = InternalCAREECGVictim(pipeline)
        signal = np.full((1, 4), 2.0, dtype=np.float32)

        target_snapshot = victim.evaluate(signal, "target")
        clean_snapshot = victim.evaluate(signal, "clean")

        self.assertEqual(target_snapshot.report, clean_snapshot.report)
        self.assertEqual(target_snapshot.diagnosis, clean_snapshot.diagnosis)
        self.assertEqual(target_snapshot.probabilities, clean_snapshot.probabilities)
        self.assertTrue(
            all(call["counterfactual_target"] is None for call in pipeline.calls)
        )

    def test_unavailable_counterfactual_guidance_is_explicit_and_nonfatal(self):
        pipeline = TinyPipeline()
        pipeline.counterfactual_engine.no_edit = True
        victim = InternalCAREECGVictim(pipeline)

        snapshot = victim.evaluate(np.full((1, 4), 2.0, dtype=np.float32), "target")

        self.assertIsNone(victim.counterfactual_guidance(snapshot))
        self.assertFalse(snapshot.metadata["counterfactual_guidance"]["available"])
        self.assertEqual(
            snapshot.metadata["counterfactual_guidance"]["reason"],
            "target_already_satisfied_no_edit",
        )

        pipeline.counterfactual_engine.fail = True
        failed_guidance = victim.evaluate(
            np.full((1, 4), 2.0, dtype=np.float32), "target"
        )
        self.assertIsNone(victim.counterfactual_guidance(failed_guidance))
        self.assertIn(
            "counterfactual_search_error",
            failed_guidance.metadata["counterfactual_guidance"]["reason"],
        )

    def test_claim_oracle_does_not_count_negated_target(self):
        pipeline = TinyPipeline()
        pipeline.report_override = "There is no target diagnosis on this tracing."
        victim = InternalCAREECGVictim(pipeline)

        snapshot = victim.evaluate(np.full((1, 4), 2.0, dtype=np.float32), "target")

        self.assertEqual(snapshot.diagnosis, "target")
        self.assertFalse(snapshot.target_claim_present)

    def test_nondefault_defense_requires_an_explicit_verifier(self):
        pipeline = TinyPipeline()
        victim = InternalCAREECGVictim(pipeline)

        with self.assertRaisesRegex(ValueError, "requires a configured release verifier"):
            victim.evaluate(
                np.full((1, 4), 2.0, dtype=np.float32),
                "target",
                defense=DefenseKind.V3,
            )
        self.assertEqual(pipeline.calls, [])

    def test_release_verifier_receives_snapshot_waveform_and_sample_rate(self):
        pipeline = TinyPipeline()
        verifier = RejectingVerifier()
        victim = InternalCAREECGVictim(
            pipeline,
            patient_id="patient-1",
            defense_evaluators={DefenseKind.V2: verifier},
        )
        signal = np.full((1, 4), 2.0, dtype=np.float32)

        snapshot = victim.evaluate(signal, "target", defense=DefenseKind.V2)

        self.assertIsInstance(verifier.snapshot, VictimSnapshot)
        np.testing.assert_array_equal(verifier.kwargs["waveform_mv"], signal)
        self.assertEqual(verifier.kwargs["sample_rate_hz"], 100.0)
        self.assertEqual(verifier.kwargs["patient_id"], "patient-1")
        self.assertFalse(snapshot.released)
        self.assertEqual(snapshot.defense_score, 2.0)
        self.assertEqual(snapshot.metadata["defense_decision"]["defense"], "v2")

    def test_budgeted_victim_counts_complete_pipeline_queries(self):
        victim = InternalCAREECGVictim(TinyPipeline())
        budgeted = BudgetedVictim(victim, budget=2)
        signal = np.zeros((1, 4), dtype=np.float32)

        budgeted.submit(signal, "target")
        budgeted.submit(signal, "target")

        self.assertEqual(budgeted.queries_used, 2)
        self.assertEqual(budgeted.remaining, 0)
        with self.assertRaisesRegex(QueryBudgetExceeded, "exhausted"):
            budgeted.submit(signal, "target")

    def test_budgeted_victim_charges_pipeline_errors_at_call_boundary(self):
        class RejectingVictim:
            @staticmethod
            def evaluate(*args, **kwargs):
                del args, kwargs
                raise ValueError("invalid submitted waveform")

        budgeted = BudgetedVictim(RejectingVictim(), budget=1)

        with self.assertRaisesRegex(ValueError, "invalid submitted waveform"):
            budgeted.submit(np.zeros((1, 4), dtype=np.float32), "target")

        self.assertEqual(budgeted.queries_used, 1)
        self.assertEqual(budgeted.queries_completed, 0)
        self.assertEqual(budgeted.queries_errored, 1)
        self.assertEqual(budgeted.remaining, 0)
        with self.assertRaisesRegex(QueryBudgetExceeded, "exhausted"):
            budgeted.submit(np.zeros((1, 4), dtype=np.float32), "target")

    def test_victim_rejects_wrong_shape_nonfinite_and_unresampled_input(self):
        victim = InternalCAREECGVictim(TinyPipeline())
        with self.assertRaisesRegex(ValueError, "expects preprocessed raw waveforms"):
            victim.surrogate(np.zeros((2, 4), dtype=np.float32), "target")
        with self.assertRaisesRegex(ValueError, "finite"):
            victim.surrogate(np.asarray([[0.0, np.nan, 0.0, 0.0]]), "target")

        wrong_rate = InternalCAREECGVictim(TinyPipeline(), source_rate=200)
        with self.assertRaisesRegex(ValueError, "resample"):
            wrong_rate.surrogate(np.zeros((1, 4), dtype=np.float32), "target")


if __name__ == "__main__":
    unittest.main()
