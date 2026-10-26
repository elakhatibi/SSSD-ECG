import unittest

import numpy as np

from care_ecg.security.contracts import (
    AttackAttempt,
    AttackFamily,
    AttackResult,
    VictimSnapshot,
)
from care_ecg.security.metrics import (
    bootstrap_incorrect_release_rate,
    bootstrap_paired_irr_difference,
    counterfactual_alignment,
    retrieval_overlap_at_k,
    rounds_to_pass,
    selective_metrics,
)


def attack_result(success_query=None, total=3):
    attempts = []
    for query in range(1, total + 1):
        success = query == success_query
        attempts.append(
            AttackAttempt(
                query,
                np.zeros((1, 4), dtype=np.float32),
                1.0 / query,
                0.8 if success else 0.2,
                "target" if success else "clean",
                True,
                success,
                True,
                success,
            )
        )
        if success:
            break
    selected = attempts[-1]
    adversarial = VictimSnapshot(
        diagnosis=selected.diagnosis,
        probabilities={"clean": 0.2 if selected.success else 0.8, "target": selected.target_probability},
        latent=np.asarray([float(selected.success)]),
        discrete_latent=np.asarray([int(selected.success)]),
        retrieved_fact_ids=(),
        report="Assessment supports target." if selected.success else "Assessment supports clean.",
        hallucination_score=0.0,
        unsupported_claim_score=0.0,
        used_fallback=False,
        abstained=False,
        released=True,
        target_claim_present=selected.success,
    )
    return AttackResult(
        AttackFamily.F3,
        "target",
        None,
        adversarial,
        attempts[-1].waveform,
        attempts,
        bool(success_query),
        len(attempts),
        "incorrect_release" if success_query else "query_budget_exhausted",
    )


class SecurityMetricTests(unittest.TestCase):
    def test_attack_result_rejects_incoherent_query_traces(self):
        result = attack_result(None, 2)
        duplicate = list(result.attempts)
        duplicate[1].query_index = 1
        with self.assertRaisesRegex(ValueError, "unique, contiguous"):
            AttackResult(
                AttackFamily.F3,
                "target",
                None,
                result.adversarial_snapshot,
                result.adversarial_signal,
                duplicate,
                False,
                2,
                "query_budget_exhausted",
            )

        successful = attack_result(1, 2)
        with self.assertRaisesRegex(ValueError, "presence of a successful attempt"):
            AttackResult(
                AttackFamily.F3,
                "target",
                None,
                successful.adversarial_snapshot,
                successful.adversarial_signal,
                successful.attempts,
                False,
                1,
                "query_budget_exhausted",
            )
        with self.assertRaisesRegex(ValueError, "require adversarial snapshot"):
            AttackResult(
                AttackFamily.F3,
                "target",
                None,
                None,
                None,
                successful.attempts,
                True,
                1,
                "incorrect_release",
            )
        trailing = AttackAttempt(
            2,
            np.ones((1, 4), dtype=np.float32),
            0.5,
            0.2,
            "clean",
            True,
            False,
            True,
            False,
        )
        with self.assertRaisesRegex(ValueError, "stop immediately"):
            AttackResult(
                AttackFamily.F3,
                "target",
                None,
                successful.adversarial_snapshot,
                successful.adversarial_signal,
                [*successful.attempts, trailing],
                True,
                2,
                "incorrect_release",
            )

    def test_irr_bootstrap_clusters_repeated_rows_by_patient(self):
        estimate = bootstrap_incorrect_release_rate(
            [True, True, False],
            ["p1", "p1", "p2"],
            n_resamples=100,
            seed=3,
        )
        self.assertEqual(estimate.estimate, 0.5)
        self.assertEqual(estimate.n_patients, 2)

    def test_paired_delta_uses_identical_patient_set(self):
        estimate = bootstrap_paired_irr_difference(
            {"p1": True, "p2": True},
            {"p1": False, "p2": True},
            n_resamples=100,
        )
        self.assertEqual(estimate.estimate, 0.5)
        with self.assertRaises(ValueError):
            bootstrap_paired_irr_difference({"p1": True}, {"p2": False})

    def test_rounds_and_budget_curve_use_first_success(self):
        successful = attack_result(2, 4)
        unsuccessful = attack_result(None, 4)
        summary = rounds_to_pass([successful, unsuccessful], budgets=(1, 2, 4))
        budget_two = bootstrap_incorrect_release_rate(
            [successful, unsuccessful],
            ["p1", "p2"],
            budget=2,
            n_resamples=50,
        )
        self.assertEqual(summary.median_rounds_to_pass, 2.0)
        self.assertEqual(summary.success_by_budget, {1: 0.0, 2: 0.5, 4: 0.5})
        self.assertEqual(budget_two.estimate, 0.5)

    def test_coverage_selective_risk_and_false_rejection(self):
        result = selective_metrics([True, False, True], [True, True, False])
        self.assertAlmostEqual(result.coverage, 2 / 3)
        self.assertEqual(result.selective_risk, 0.5)
        self.assertEqual(result.false_rejection_rate, 0.5)

    def test_retrieval_overlap_and_counterfactual_alignment(self):
        self.assertEqual(retrieval_overlap_at_k(["a", "b"], ["b", "c"], 2), 1 / 3)
        alignment = counterfactual_alignment([1, 0, 1], [2, 0, 0], k=1)
        self.assertAlmostEqual(alignment.cosine_similarity, 1 / np.sqrt(2))
        self.assertTrue(alignment.support_match)
        self.assertEqual(alignment.factor_overlap_at_k, 1.0)


if __name__ == "__main__":
    unittest.main()
