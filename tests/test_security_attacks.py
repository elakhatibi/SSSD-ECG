import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from care_ecg.security.attacks import (
    F0RandomAttack,
    F1CounterfactualAttack,
    F1PGDAttack,
    F2NESAttack,
    F2SimBAAttack,
    F2SquareAttack,
    F3AdaptiveAttack,
)
from care_ecg.security.constraints import (
    S2Constraint,
    SInfConstraint,
    SPhysConstraint,
    SSpecConstraint,
)
from care_ecg.security.contracts import (
    AdmissibilityResult,
    AttackFamily,
    ClinicalCaseContext,
    ClinicalPreservationEvidence,
    ClinicalTarget,
    DefenseKind,
    KnowledgeLevel,
    MeasurementStatus,
    VictimSnapshot,
)
from care_ecg.security.objectives import (
    adaptive_objective,
    latent_bin_margin_loss,
    soft_bin_memberships,
    target_evidence_loss,
    targeted_margin_loss,
)
from care_ecg.security.victim import SurrogateSnapshot

TARGET = ClinicalTarget("T", "clean", "target", "target finding", "fabrication")
CASE = SimpleNamespace(patient_id="p1", record_id="r1", target=TARGET)


class ToyVictim:
    label_names = ("clean", "target")
    device = torch.device("cpu")
    sample_rate_hz = 100.0
    discretizer_thresholds = np.asarray([[-0.05, 0.05]], dtype=np.float32)

    def __init__(self, claim_threshold: float = 0.02):
        self.claim_threshold = claim_threshold
        self.evaluations = 0

    def target_index(self, target_label):
        if isinstance(target_label, int):
            return target_label
        return self.label_names.index(target_label)

    @staticmethod
    def _probabilities(signal):
        mean = signal.mean(dim=(-2, -1))
        score = 8.0 * mean
        return torch.softmax(torch.stack((0.2 - score, score), dim=1), dim=1), mean

    def surrogate(self, signal, target_label, *, defense=DefenseKind.V1):
        del target_label, defense
        batch = torch.as_tensor(signal, dtype=torch.float32)
        if batch.ndim == 2:
            batch = batch.unsqueeze(0)
        probabilities, mean = self._probabilities(batch)
        latent = mean.unsqueeze(1)
        guidance_loss = torch.relu(0.12 - latent).mean()
        verifier_loss = torch.relu(0.10 - latent).mean()
        defense_loss = torch.relu(0.08 - latent).mean()
        return SurrogateSnapshot(
            logits=torch.log(probabilities),
            probabilities=probabilities,
            latent=latent,
            retrieval_scores=torch.stack((-mean, mean), dim=1),
            retrieval_loss=guidance_loss,
            verifier_loss=verifier_loss,
            defense_loss=defense_loss,
            metadata={"oracle": "toy", "verifier_surrogate": "toy_distinct_proxy"},
        )

    def evaluate(
        self,
        signal,
        target_label,
        *,
        defense=DefenseKind.V1,
        clean_snapshot=None,
    ):
        del clean_snapshot
        self.evaluations += 1
        waveform = torch.as_tensor(signal, dtype=torch.float32)
        probabilities, mean = self._probabilities(waveform.unsqueeze(0))
        target_name = self.label_names[self.target_index(target_label)]
        target_claim = float(mean[0]) >= self.claim_threshold
        released = True
        diagnosis = target_name if target_claim else "clean"
        report = (
            f"Assessment supports {target_name}." if target_claim else "Assessment supports clean."
        )
        return VictimSnapshot(
            diagnosis=diagnosis,
            probabilities={
                name: float(probabilities[0, index]) for index, name in enumerate(self.label_names)
            },
            latent=np.asarray([float(mean[0])]),
            discrete_latent=np.asarray([int(target_claim)]),
            retrieved_fact_ids=("target-fact",) if target_claim else ("clean-fact",),
            report=report,
            hallucination_score=0.0,
            unsupported_claim_score=0.0,
            used_fallback=False,
            abstained=False,
            released=released,
            target_claim_present=target_claim,
            defense_score=max(0.0, 0.12 - float(mean[0])),
            metadata={
                "defense": DefenseKind(defense).value,
                "counterfactual": {
                    "found": True,
                    "feature": "z0",
                    "old_state": 1,
                    "new_state": 2,
                },
            },
        )

    def counterfactual_guidance(self, clean_snapshot):
        del clean_snapshot
        return 0, 2


class RecordingAdmissibility:
    def __init__(self, result=True):
        self.result = result
        self.sample_rates = []
        self.contexts = []

    def evaluate(self, clean, candidate, sample_rate_hz, *, context=None, **kwargs):
        del clean, candidate, kwargs
        self.sample_rates.append(sample_rate_hz)
        self.contexts.append(context)
        if context is None:
            return self.result
        evidence = ClinicalPreservationEvidence(
            context=context,
            preserved_findings=("independently reviewed baseline finding",),
            evaluator="test-independent-reviewer",
            independent=True,
        )
        return AdmissibilityResult(
            admissible=self.result,
            status=MeasurementStatus.AVAILABLE,
            reasons=() if self.result else ("baseline finding not preserved",),
            preservation_evidence=evidence,
            primary_endpoint_eligible=True,
        )


class FirstQueryIneligibleAdmissibility(RecordingAdmissibility):
    def evaluate(self, clean, candidate, sample_rate_hz, *, context=None, **kwargs):
        if not self.sample_rates:
            self.sample_rates.append(sample_rate_hz)
            self.contexts.append(context)
            return False
        return super().evaluate(
            clean,
            candidate,
            sample_rate_hz,
            context=context,
            **kwargs,
        )


class ProbabilityOnlyTiebreakVictim(ToyVictim):
    def evaluate(self, *args, **kwargs):
        snapshot = super().evaluate(*args, **kwargs)
        if self.evaluations == 2:
            snapshot.probabilities = {"clean": 0.9, "target": 0.1}
        elif self.evaluations >= 3:
            snapshot.probabilities = {"clean": 0.1, "target": 0.9}
        return snapshot


class NoGuidanceVictim(ToyVictim):
    def counterfactual_guidance(self, clean_snapshot):
        del clean_snapshot


class KnowledgeSeparatedVictim(ToyVictim):
    def surrogate(self, signal, target_label, *, defense=DefenseKind.V1):
        result = super().surrogate(signal, target_label, defense=defense)
        batch = torch.as_tensor(signal, dtype=torch.float32)
        if batch.ndim == 2:
            batch = batch.unsqueeze(0)
        mean = batch.mean(dim=(-2, -1))
        result.retrieval_loss = torch.relu(0.12 - mean).mean()
        result.verifier_loss = torch.relu(mean + 0.12).mean()
        result.defense_loss = None
        result.metadata["verifier_surrogate"] = "opposing_test_proxy"
        return result


class NoVerifierVictim(ToyVictim):
    def surrogate(self, signal, target_label, *, defense=DefenseKind.V1):
        result = super().surrogate(signal, target_label, defense=defense)
        result.verifier_loss = None
        result.defense_loss = None
        result.metadata["verifier_surrogate"] = None
        return result


class SecurityObjectiveTests(unittest.TestCase):
    def test_targeted_and_adaptive_objectives_are_differentiable(self):
        logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
        diagnosis = targeted_margin_loss(logits, 1, confidence=0.5)
        terms = adaptive_objective(
            logits,
            1,
            retrieval_loss=logits[0, 0].square(),
            retrieval_weight=0.25,
            confidence=0.5,
        )
        terms.total.backward()

        self.assertAlmostEqual(float(diagnosis.detach()), 3.5)
        self.assertGreater(float(terms.total.detach()), float(diagnosis.detach()))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_soft_bins_and_counterfactual_interior_loss(self):
        latent = torch.tensor([[0.0], [0.8]], requires_grad=True)
        thresholds = torch.tensor([[-0.5, 0.5]])
        memberships = soft_bin_memberships(latent, thresholds, temperature=0.1)
        loss = latent_bin_margin_loss(latent, thresholds, 0, 2, margin=0.05)
        loss.backward()

        self.assertEqual(tuple(memberships.shape), (2, 1, 3))
        torch.testing.assert_close(memberships.sum(dim=-1), torch.ones(2, 1))
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertLess(float(latent.grad[0]), 0.0)

    def test_target_evidence_loss_prefers_target_documents(self):
        weak = target_evidence_loss(torch.tensor([[3.0, 0.0]]), torch.tensor([False, True]))
        strong = target_evidence_loss(torch.tensor([[0.0, 3.0]]), torch.tensor([False, True]))
        self.assertGreater(float(weak), float(strong))


class SecurityAttackTests(unittest.TestCase):
    def setUp(self):
        self.clean = np.zeros((1, 8), dtype=np.float32)

    def test_f0_uses_exact_budget_and_logs_auditable_attempts(self):
        victim = ToyVictim(claim_threshold=math.inf)
        result = F0RandomAttack(epsilon_mv=0.1, seed=7).run(
            victim,
            self.clean,
            "target",
            query_budget=4,
            constraint=SInfConstraint(0.1),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.queries_used, 4)
        self.assertEqual(victim.evaluations, 5)  # clean eligibility plus four attack queries
        self.assertEqual([attempt.query_index for attempt in result.attempts], [1, 2, 3, 4])
        self.assertTrue(
            all(len(attempt.metadata["waveform_sha256"]) == 64 for attempt in result.attempts)
        )
        self.assertTrue(all(attempt.metadata["linf_mv"] <= 0.100001 for attempt in result.attempts))
        self.assertTrue(
            all(
                {
                    "probabilities",
                    "latent",
                    "discrete_latent",
                    "report",
                    "verifier_score",
                    "verifier_decision",
                    "interface_compliant",
                }
                <= attempt.metadata.keys()
                for attempt in result.attempts
            )
        )

    def test_f1_pgd_linf_stops_at_first_real_pipeline_success(self):
        victim = ToyVictim()
        result = F1PGDAttack(
            epsilon_mv=0.2,
            step_size_mv=0.05,
            steps=5,
            random_start=False,
        ).run(
            victim,
            self.clean,
            "target",
            query_budget=5,
            constraint=SInfConstraint(0.2),
            admissibility_checker=RecordingAdmissibility(True),
            case=CASE,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.family, AttackFamily.F1)
        self.assertEqual(result.queries_used, 1)
        self.assertLessEqual(float(np.max(np.abs(result.adversarial_signal))), 0.200001)
        self.assertTrue(result.attempts[-1].released)
        self.assertTrue(result.attempts[-1].target_claim_present)

    def test_f1_pgd_completes_surrogate_trajectory_before_budgeted_query(self):
        victim = ToyVictim(claim_threshold=0.15)
        result = F1PGDAttack(
            epsilon_mv=0.2,
            step_size_mv=0.05,
            steps=4,
            random_start=False,
            confidence=3.0,
        ).run(
            victim,
            self.clean,
            "target",
            query_budget=1,
            constraint=SInfConstraint(0.2),
            admissibility_checker=RecordingAdmissibility(True),
            case=CASE,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.queries_used, 1)
        self.assertEqual(result.metadata["surrogate_iterations"], 4)
        self.assertEqual(
            result.metadata["complete_system_query_schedule"],
            "one_best_candidate_per_restart",
        )
        self.assertEqual(result.attempts[0].metadata["gradient_step"], 4)
        self.assertEqual(result.attempts[0].metadata["surrogate_steps_in_restart"], 4)
        self.assertEqual(victim.evaluations, 2)

    def test_f1_pgd_l2_respects_global_norm(self):
        result = F1PGDAttack(
            norm="l2",
            epsilon_mv=0.15,
            step_size_mv=0.1,
            steps=4,
            random_start=False,
        ).run(
            ToyVictim(),
            self.clean,
            "target",
            query_budget=4,
            admissibility_checker=RecordingAdmissibility(True),
            case=CASE,
        )

        self.assertTrue(result.success)
        self.assertLessEqual(
            float(np.linalg.norm(result.adversarial_signal - self.clean)), 0.150001
        )

    def test_f1_counterfactual_uses_continuous_bin_guidance(self):
        victim = ToyVictim(claim_threshold=0.11)
        victim.discretizer_thresholds = np.asarray([[-0.05, 0.15]], dtype=np.float32)
        result = F1CounterfactualAttack(
            epsilon_mv=0.2,
            step_size_mv=0.05,
            steps=5,
            random_start=False,
            latent_weight=2.0,
        ).run(
            victim,
            self.clean,
            "target",
            query_budget=5,
            admissibility_checker=RecordingAdmissibility(True),
            case=CASE,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.family, AttackFamily.F1_CF)
        self.assertEqual(result.metadata["counterfactual_feature_index"], 0)
        self.assertEqual(result.metadata["counterfactual_target_state"], 2)
        self.assertIn("latent", result.attempts[-1].metadata["objective_terms"])

    def test_all_f2_methods_obey_query_budget(self):
        attacks = (
            F2NESAttack(epsilon_mv=0.1, sigma=0.02, samples_per_step=2),
            F2SimBAAttack(epsilon_mv=0.1, step_size_mv=0.02),
            F2SquareAttack(epsilon_mv=0.1, initial_fraction=0.5),
        )
        for attack in attacks:
            with self.subTest(attack=type(attack).__name__):
                victim = ToyVictim(claim_threshold=math.inf)
                result = attack.run(victim, self.clean, "target", query_budget=5)
                self.assertEqual(result.queries_used, 5)
                self.assertEqual(victim.evaluations, 6)
                self.assertEqual(result.stopped_reason, "query_budget_exhausted")

    def test_k1_artifact_selection_never_uses_hidden_target_probability(self):
        result = F2SquareAttack(epsilon_mv=0.1, seed=3).run(
            ProbabilityOnlyTiebreakVictim(claim_threshold=math.inf),
            self.clean,
            "target",
            query_budget=2,
        )

        self.assertEqual([attempt.target_probability for attempt in result.attempts], [0.1, 0.9])
        np.testing.assert_array_equal(result.adversarial_signal, result.attempts[0].waveform)
        self.assertEqual(result.metadata["selected_attempt_query_index"], 1)
        self.assertEqual(
            result.metadata["artifact_selection_policy"],
            "observable_decision_loss_first_query_tiebreak",
        )
        self.assertEqual(
            result.metadata["attacker_feedback_contract"],
            "released_report_and_release_decision_only",
        )
        self.assertIn("probabilities", result.attempts[0].metadata["audit_only_fields"])

    def test_f2_methods_support_physiological_parameter_space(self):
        clean = np.zeros((1, 100), dtype=np.float32)
        physiological = SPhysConstraint(sample_rate_hz=100.0)
        attacks = (
            F2NESAttack(
                sigma=0.1,
                learning_rate=0.1,
                samples_per_step=1,
                parameterization=physiological,
            ),
            F2SimBAAttack(step_size_mv=0.2, parameterization=physiological),
            F2SquareAttack(parameterization=physiological),
        )
        for attack in attacks:
            with self.subTest(attack=type(attack).__name__):
                result = attack.run(
                    ToyVictim(claim_threshold=math.inf),
                    clean,
                    "target",
                    query_budget=2,
                    constraint=physiological,
                )
                self.assertEqual(result.queries_used, 2)
                self.assertTrue(result.metadata["parameterized"])
                self.assertTrue(result.metadata["normalized_parameter_coordinates"])
                self.assertTrue(
                    all(attempt.metadata["constraint_satisfied"] for attempt in result.attempts)
                )

    def test_every_attack_submits_float32_waveforms_inside_declared_constraint(self):
        sample_rate = 100.0
        time = np.arange(128) / sample_rate
        clean = np.stack(
            (
                2.0 * np.sin(2.0 * np.pi * 1.25 * time),
                1.5 * np.cos(2.0 * np.pi * 1.25 * time),
            )
        ).astype(np.float32)
        constraints = (
            SInfConstraint(0.02),
            S2Constraint(0.2, per_lead=True),
            SSpecConstraint(
                0.02,
                sample_rate,
                cutoff_hz=20.0,
                max_high_frequency_ratio=0.05,
                max_slew_mv=0.01,
            ),
            SPhysConstraint(sample_rate),
        )
        attacks = (
            F0RandomAttack(epsilon_mv=0.02, matched_magnitude=0.01),
            F1PGDAttack(
                epsilon_mv=0.02,
                step_size_mv=0.003,
                steps=1,
                random_start=False,
            ),
            F1CounterfactualAttack(
                epsilon_mv=0.02,
                step_size_mv=0.003,
                steps=1,
                random_start=False,
            ),
            F2NESAttack(
                epsilon_mv=0.02,
                sigma=0.002,
                samples_per_step=1,
            ),
            F2SimBAAttack(epsilon_mv=0.02, step_size_mv=0.002),
            F2SquareAttack(epsilon_mv=0.02, initial_fraction=0.25),
            F3AdaptiveAttack(
                knowledge_level=KnowledgeLevel.K1,
                black_box_method="square",
                epsilon_mv=0.02,
                steps=1,
            ),
            F3AdaptiveAttack(
                knowledge_level=KnowledgeLevel.K2,
                epsilon_mv=0.02,
                step_size_mv=0.003,
                steps=1,
                random_start=False,
            ),
        )

        for constraint in constraints:
            for attack in attacks:
                with self.subTest(
                    constraint=constraint.kind.value,
                    attack=type(attack).__name__,
                ):
                    result = attack.run(
                        ToyVictim(claim_threshold=math.inf),
                        clean,
                        "target",
                        query_budget=1,
                        constraint=constraint,
                        admissibility_checker=RecordingAdmissibility(True),
                        case=CASE,
                    )
                    self.assertEqual(result.queries_used, 1)
                    self.assertTrue(constraint.contains(clean, result.attempts[0].waveform))
                    self.assertTrue(result.attempts[0].metadata["constraint_satisfied"])
                    if isinstance(attack, F0RandomAttack):
                        self.assertTrue(result.attempts[0].metadata["native_constraint_sampling"])
                        self.assertAlmostEqual(
                            result.attempts[0].metadata["achieved_constraint_distance"],
                            0.01,
                            delta=2e-5,
                        )

    def test_f3_supports_k1_k2_k3_and_defense_readaptation(self):
        k1 = F3AdaptiveAttack(
            knowledge_level=KnowledgeLevel.K1,
            black_box_method="square",
            epsilon_mv=0.1,
            steps=2,
        ).run(
            ToyVictim(claim_threshold=math.inf),
            self.clean,
            "target",
            query_budget=2,
            defense=DefenseKind.V2,
        )
        self.assertEqual(k1.family, AttackFamily.F3_DEFENSE)
        self.assertEqual(k1.metadata["knowledge_level"], KnowledgeLevel.K1.value)
        self.assertTrue(k1.metadata["attack_reoptimized_for_defense"])
        self.assertEqual(k1.metadata["re_adapted_to_defense"], "v2")
        self.assertFalse(k1.metadata["reoptimization_evidence"]["hidden_target_probability_used"])

        for level in (KnowledgeLevel.K2, KnowledgeLevel.K3):
            with self.subTest(level=level):
                result = F3AdaptiveAttack(
                    knowledge_level=level,
                    epsilon_mv=0.2,
                    step_size_mv=0.05,
                    steps=4,
                    random_start=False,
                    defense_weight=1.0,
                ).run(
                    ToyVictim(),
                    self.clean,
                    "target",
                    query_budget=4,
                    admissibility_checker=RecordingAdmissibility(True),
                    case=CASE,
                )
                self.assertTrue(result.success)
                self.assertEqual(result.family, AttackFamily.F3)
                self.assertEqual(result.metadata["knowledge_level"], level.value)
                self.assertIn("defense", result.attempts[-1].metadata["objective_terms"])

    def test_f1_counterfactual_falls_back_when_guidance_is_unavailable(self):
        result = F1CounterfactualAttack(
            epsilon_mv=0.1,
            step_size_mv=0.02,
            steps=2,
            random_start=False,
        ).run(
            NoGuidanceVictim(claim_threshold=math.inf),
            self.clean,
            "target",
            query_budget=2,
            admissibility_checker=RecordingAdmissibility(True),
            case=CASE,
        )

        self.assertEqual(result.queries_used, 1)
        self.assertEqual(result.metadata["surrogate_iterations"], 2)
        self.assertFalse(result.metadata["counterfactual_guidance_available"])
        self.assertEqual(
            result.metadata["objective"],
            "targeted_diagnosis_fallback_counterfactual_unavailable",
        )
        self.assertEqual(result.metadata["effective_latent_weight"], 0.0)

    def test_missing_clinical_evidence_is_not_primary_success(self):
        result = F0RandomAttack(epsilon_mv=0.1).run(
            ToyVictim(claim_threshold=-1.0),
            self.clean,
            "target",
            query_budget=2,
        )

        self.assertFalse(result.success)
        self.assertTrue(result.metadata["primary_endpoint_eligible"])
        self.assertIsNone(result.metadata["successful_attempt_primary_eligible"])
        self.assertTrue(all(not attempt.admissible for attempt in result.attempts))
        self.assertTrue(
            all(
                attempt.metadata["clinical_preservation"] is False
                and attempt.metadata["admissibility"]["available"] is False
                for attempt in result.attempts
            )
        )

    def test_boolean_clinical_checker_is_exploratory_only(self):
        result = F0RandomAttack(epsilon_mv=0.1).run(
            ToyVictim(claim_threshold=-1.0),
            self.clean,
            "target",
            query_budget=1,
            admissibility_checker=lambda clean, candidate: True,
            case=CASE,
        )

        self.assertFalse(result.success)
        self.assertTrue(result.metadata["primary_endpoint_eligible"])
        self.assertIsNone(result.metadata["successful_attempt_primary_eligible"])
        self.assertTrue(result.metadata["exploratory_success"])
        self.assertTrue(result.attempts[0].metadata["exploratory_admissible"])
        self.assertFalse(result.attempts[0].admissible)

    def test_success_eligibility_uses_selected_attempt_and_charges_prior_query(self):
        checker = FirstQueryIneligibleAdmissibility(True)
        result = F0RandomAttack(epsilon_mv=0.1, seed=5).run(
            ToyVictim(claim_threshold=-1.0),
            self.clean,
            "target",
            query_budget=3,
            admissibility_checker=checker,
            case=CASE,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.queries_used, 2)
        self.assertFalse(result.attempts[0].metadata["primary_endpoint_eligible"])
        self.assertTrue(result.attempts[1].metadata["primary_endpoint_eligible"])
        self.assertTrue(result.metadata["successful_attempt_primary_eligible"])
        self.assertTrue(result.metadata["primary_endpoint_eligible"])
        self.assertEqual(result.metadata["selected_attempt_query_index"], 2)
        self.assertEqual(result.metadata["complete_system_queries_consumed"], 2)
        self.assertEqual(result.metadata["inadmissible_queries"], 1)

    def test_declared_per_lead_s2_is_not_reduced_to_global_l2(self):
        clean = np.zeros((2, 8), dtype=np.float32)
        result = F1PGDAttack(
            norm="l2",
            epsilon_mv=0.1,
            step_size_mv=1.0,
            steps=1,
            random_start=False,
        ).run(
            ToyVictim(claim_threshold=math.inf),
            clean,
            "target",
            query_budget=1,
            constraint=S2Constraint(0.1, per_lead=True),
            admissibility_checker=RecordingAdmissibility(True),
            case=CASE,
        )

        delta = result.adversarial_signal - clean
        self.assertGreater(float(np.linalg.norm(delta)), 0.13)
        self.assertTrue(np.all(np.linalg.norm(delta, axis=1) <= 0.100001))
        self.assertFalse(result.metadata["builtin_lp_projection"])

    def test_declared_sphys_gradient_is_not_linf_limited(self):
        clean = np.zeros((1, 100), dtype=np.float32)
        constraint = SPhysConstraint(sample_rate_hz=100.0)
        result = F1PGDAttack(
            norm="linf",
            epsilon_mv=0.0001,
            step_size_mv=0.2,
            steps=1,
            random_start=False,
        ).run(
            ToyVictim(claim_threshold=math.inf),
            clean,
            "target",
            query_budget=1,
            constraint=constraint,
            admissibility_checker=RecordingAdmissibility(True),
            case=CASE,
        )

        self.assertGreater(float(np.max(np.abs(result.adversarial_signal - clean))), 0.0005)
        self.assertTrue(constraint.contains(clean, result.adversarial_signal))
        self.assertFalse(result.metadata["builtin_lp_projection"])

    def test_k2_and_k3_use_distinct_objectives_and_candidates(self):
        common = {
            "epsilon_mv": 0.2,
            "step_size_mv": 0.05,
            "steps": 1,
            "random_start": False,
            "diagnosis_weight": 0.01,
            "retrieval_weight": 1.0,
            "verifier_weight": 2.0,
            "defense_weight": 0.0,
        }
        checker = RecordingAdmissibility(True)
        k2 = F3AdaptiveAttack(knowledge_level=KnowledgeLevel.K2, **common).run(
            KnowledgeSeparatedVictim(claim_threshold=math.inf),
            self.clean,
            "target",
            query_budget=1,
            admissibility_checker=checker,
            case=CASE,
        )
        k3 = F3AdaptiveAttack(knowledge_level=KnowledgeLevel.K3, **common).run(
            KnowledgeSeparatedVictim(claim_threshold=math.inf),
            self.clean,
            "target",
            query_budget=1,
            admissibility_checker=checker,
            case=CASE,
        )

        self.assertEqual(k2.metadata["objective"], "diagnosis_plus_retrieval")
        self.assertEqual(k2.metadata["verifier_weight"], 0.0)
        self.assertFalse(k2.metadata["verifier_objective_permitted"])
        self.assertFalse(k2.metadata["verifier_objective_applied"])
        self.assertEqual(k3.metadata["objective"], "diagnosis_plus_retrieval_plus_verifier")
        self.assertEqual(k3.metadata["verifier_weight"], 2.0)
        self.assertTrue(k3.metadata["verifier_objective_permitted"])
        self.assertTrue(k3.metadata["verifier_objective_applied"])
        self.assertGreater(float(k2.adversarial_signal.mean()), 0.0)
        self.assertLess(float(k3.adversarial_signal.mean()), 0.0)

    def test_k3_requires_a_distinct_available_verifier_surrogate(self):
        with self.assertRaisesRegex(ValueError, "no verifier surrogate"):
            F3AdaptiveAttack(
                knowledge_level=KnowledgeLevel.K3,
                epsilon_mv=0.1,
                steps=1,
                random_start=False,
            ).run(
                NoVerifierVictim(claim_threshold=math.inf),
                self.clean,
                "target",
                query_budget=1,
                admissibility_checker=RecordingAdmissibility(True),
                case=CASE,
            )

    def test_f3_only_claims_defense_readaptation_when_surrogate_is_applied(self):
        unavailable = F3AdaptiveAttack(
            knowledge_level=KnowledgeLevel.K3,
            epsilon_mv=0.1,
            step_size_mv=0.02,
            steps=1,
            random_start=False,
            defense_weight=0.0,
        ).run(
            KnowledgeSeparatedVictim(claim_threshold=math.inf),
            self.clean,
            "target",
            query_budget=1,
            defense=DefenseKind.V2,
            admissibility_checker=RecordingAdmissibility(True),
            case=CASE,
        )
        applied = F3AdaptiveAttack(
            knowledge_level=KnowledgeLevel.K3,
            epsilon_mv=0.1,
            step_size_mv=0.02,
            steps=1,
            random_start=False,
            defense_weight=1.0,
        ).run(
            ToyVictim(claim_threshold=math.inf),
            self.clean,
            "target",
            query_budget=1,
            defense=DefenseKind.V2,
            admissibility_checker=RecordingAdmissibility(True),
            case=CASE,
        )

        self.assertEqual(unavailable.family, AttackFamily.F3_DEFENSE)
        self.assertEqual(unavailable.metadata["defense_adaptation_status"], "surrogate_unavailable")
        self.assertIsNone(unavailable.metadata["re_adapted_to_defense"])
        self.assertEqual(applied.metadata["defense_adaptation_status"], "applied")
        self.assertEqual(applied.metadata["re_adapted_to_defense"], "v2")
        self.assertGreater(applied.metadata["defense_weight"], 0.0)

    def test_f3_records_fresh_v1_verifier_reoptimization(self):
        result = F3AdaptiveAttack(
            knowledge_level=KnowledgeLevel.K3,
            epsilon_mv=0.1,
            step_size_mv=0.02,
            steps=1,
            random_start=False,
            defense_weight=0.0,
        ).run(
            ToyVictim(claim_threshold=math.inf),
            self.clean,
            "target",
            query_budget=1,
            defense=DefenseKind.V1,
            admissibility_checker=RecordingAdmissibility(True),
            case=CASE,
        )

        self.assertTrue(result.metadata["attack_reoptimized_for_defense"])
        self.assertEqual(result.metadata["reoptimized_defense"], "v1")
        self.assertEqual(result.metadata["re_adapted_to_defense"], "v1")
        self.assertEqual(
            result.metadata["defense_adaptation_status"],
            "shipped_verifier_surrogate_applied",
        )
        self.assertTrue(
            result.metadata["reoptimization_evidence"]["verifier_surrogate_defense_specific"]
        )

    def test_admissibility_is_required_for_success_and_gets_sample_rate(self):
        checker = RecordingAdmissibility(result=False)
        result = F0RandomAttack(epsilon_mv=0.1).run(
            ToyVictim(claim_threshold=-1.0),
            self.clean,
            "target",
            query_budget=2,
            admissibility_checker=checker,
            case=CASE,
        )

        self.assertFalse(result.success)
        self.assertEqual(checker.sample_rates, [100.0, 100.0])
        self.assertTrue(
            all(isinstance(context, ClinicalCaseContext) for context in checker.contexts)
        )
        self.assertTrue(all(context.record_id == "r1" for context in checker.contexts))
        self.assertTrue(all(attempt.target_claim_present for attempt in result.attempts))
        self.assertTrue(all(not attempt.admissible for attempt in result.attempts))


if __name__ == "__main__":
    unittest.main()
