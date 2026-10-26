import unittest

import numpy as np

from care_ecg.security.contracts import (
    AttackAttempt,
    AttackFamily,
    AttackResult,
    VictimSnapshot,
)
from care_ecg.security.robustness_metrics import (
    automated_attack_metrics,
    summarize_automated_robustness,
    waveform_distortion,
)


def _snapshot(
    diagnosis: str,
    *,
    target_probability: float,
    target_claim_present: bool = False,
) -> VictimSnapshot:
    return VictimSnapshot(
        diagnosis=diagnosis,
        probabilities={
            "clean": 1.0 - target_probability,
            "target": target_probability,
        },
        latent=np.asarray([target_probability]),
        discrete_latent=np.asarray([int(target_claim_present)]),
        retrieved_fact_ids=("fact",),
        report=f"Assessment supports {diagnosis}.",
        hallucination_score=0.0,
        unsupported_claim_score=0.0,
        used_fallback=False,
        abstained=False,
        released=True,
        target_claim_present=target_claim_present,
    )


def _attack_result(*, successful: bool) -> tuple[AttackResult, np.ndarray]:
    clean = np.ones((1, 4), dtype=np.float32)
    clean_snapshot = _snapshot("clean", target_probability=0.2)
    first = AttackAttempt(
        query_index=1,
        waveform=clean + 0.05,
        objective=1.0,
        target_probability=0.3,
        diagnosis="clean",
        released=True,
        target_claim_present=False,
        admissible=True,
        success=False,
        metadata={"constraint_satisfied": True},
    )
    if successful:
        selected_waveform = clean + 0.1
        attempts = [
            first,
            AttackAttempt(
                query_index=2,
                waveform=selected_waveform,
                objective=0.1,
                target_probability=0.9,
                diagnosis="target",
                released=True,
                target_claim_present=True,
                admissible=True,
                success=True,
                metadata={"constraint_satisfied": True},
            ),
        ]
        selected_snapshot = _snapshot(
            "target",
            target_probability=0.9,
            target_claim_present=True,
        )
        selected_query = 2
    else:
        selected_waveform = first.waveform
        attempts = [first]
        selected_snapshot = _snapshot("clean", target_probability=0.3)
        selected_query = 1
    result = AttackResult(
        family=AttackFamily.F1,
        target_label="target",
        clean_snapshot=clean_snapshot,
        adversarial_snapshot=selected_snapshot,
        adversarial_signal=selected_waveform,
        attempts=attempts,
        success=successful,
        queries_used=len(attempts),
        stopped_reason="incorrect_release" if successful else "query_budget_exhausted",
        metadata={"selected_attempt_query_index": selected_query},
    )
    return result, clean


class WaveformDistortionTests(unittest.TestCase):
    def test_literature_standard_distortions(self):
        clean = np.ones((1, 4), dtype=np.float32)
        distortion = waveform_distortion(clean, clean + 0.1)

        self.assertAlmostEqual(distortion.linf_mv, 0.1, places=6)
        self.assertAlmostEqual(distortion.l2_mv, 0.2, places=6)
        self.assertAlmostEqual(distortion.mean_absolute_error_mv, 0.1, places=6)
        self.assertAlmostEqual(distortion.rmse_mv, 0.1, places=6)
        self.assertAlmostEqual(distortion.max_per_lead_l2_mv, 0.2, places=6)
        self.assertAlmostEqual(distortion.relative_l2, 0.1, places=6)
        self.assertAlmostEqual(distortion.prd_percent, 10.0, places=5)
        self.assertAlmostEqual(distortion.snr_db, 20.0, places=5)


class AutomatedRobustnessTests(unittest.TestCase):
    def test_automated_success_does_not_require_clinical_preservation(self):
        clean = np.ones((1, 4), dtype=np.float32)
        adversarial = clean + 0.1
        attempt = AttackAttempt(
            query_index=1,
            waveform=adversarial,
            objective=0.1,
            target_probability=0.9,
            diagnosis="target",
            released=True,
            target_claim_present=True,
            admissible=False,
            success=False,
            metadata={"constraint_satisfied": True},
        )
        result = AttackResult(
            family=AttackFamily.F1,
            target_label="target",
            clean_snapshot=_snapshot("clean", target_probability=0.2),
            adversarial_snapshot=_snapshot(
                "target",
                target_probability=0.9,
                target_claim_present=True,
            ),
            adversarial_signal=adversarial,
            attempts=[attempt],
            success=False,
            queries_used=1,
            stopped_reason="query_budget_exhausted",
            metadata={"selected_attempt_query_index": 1},
        )

        metrics = automated_attack_metrics(
            result,
            clean,
            expected_clean_label="clean",
        )

        self.assertTrue(metrics.targeted_diagnosis_success)
        self.assertTrue(metrics.released_target_claim_success)
        self.assertFalse(metrics.robust_correct)

    def test_attack_metrics_capture_first_success_and_probability_gain(self):
        result, clean = _attack_result(successful=True)
        metrics = automated_attack_metrics(
            result,
            clean,
            expected_clean_label="clean",
        )

        self.assertTrue(metrics.clean_correct)
        self.assertTrue(metrics.targeted_diagnosis_success)
        self.assertTrue(metrics.untargeted_diagnosis_success)
        self.assertTrue(metrics.released_target_claim_success)
        self.assertFalse(metrics.robust_correct)
        self.assertEqual(metrics.first_targeted_diagnosis_query, 2)
        self.assertEqual(metrics.first_untargeted_diagnosis_query, 2)
        self.assertEqual(metrics.first_released_target_claim_query, 2)
        self.assertAlmostEqual(metrics.target_probability_gain, 0.7)
        self.assertAlmostEqual(metrics.target_log_probability_gain, np.log(4.5))
        self.assertAlmostEqual(metrics.first_targeted_distortion.linf_mv, 0.1)

    def test_summary_conditions_asr_on_clean_correct_cases(self):
        successful, clean = _attack_result(successful=True)
        unsuccessful, _ = _attack_result(successful=False)
        rows = [
            automated_attack_metrics(
                successful,
                clean,
                expected_clean_label="clean",
            ),
            automated_attack_metrics(
                unsuccessful,
                clean,
                expected_clean_label="clean",
            ),
        ]

        summary = summarize_automated_robustness(
            rows,
            ["patient-1", "patient-2"],
            query_budgets=(1, 2),
            n_resamples=100,
            seed=7,
        )

        self.assertEqual(summary["patients"], 2)
        self.assertAlmostEqual(summary["clean_accuracy"]["estimate"], 1.0)
        self.assertAlmostEqual(summary["robust_accuracy"]["estimate"], 0.5)
        self.assertAlmostEqual(
            summary["targeted_diagnosis_attack_success_rate"]["estimate"],
            0.5,
        )
        self.assertAlmostEqual(
            summary["untargeted_diagnosis_attack_success_rate"]["estimate"],
            0.5,
        )
        self.assertAlmostEqual(
            summary["targeted_diagnosis_asr_at_budget"][1]["estimate"],
            0.0,
        )
        self.assertAlmostEqual(
            summary["targeted_diagnosis_asr_at_budget"][2]["estimate"],
            0.5,
        )
        self.assertAlmostEqual(
            summary["queries_to_first_targeted_diagnosis"]["median"],
            2.0,
        )
        self.assertTrue(summary["notes"]["clinical_primary_endpoint_unchanged"])

    def test_summary_defaults_to_publication_bootstrap_count(self):
        result, clean = _attack_result(successful=False)
        metrics = automated_attack_metrics(
            result,
            clean,
            expected_clean_label="clean",
        )

        summary = summarize_automated_robustness(
            [metrics],
            ["patient-1"],
            query_budgets=(1,),
        )

        self.assertEqual(summary["clean_accuracy"]["n_resamples"], 10_000)


if __name__ == "__main__":
    unittest.main()
