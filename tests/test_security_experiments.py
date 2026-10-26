import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from care_ecg.security.attacks import F3AdaptiveAttack
from care_ecg.security.constraints import SInfConstraint, SPhysConstraint
from care_ecg.security.contracts import (
    AdmissibilityResult,
    AttackAttempt,
    AttackFamily,
    AttackResult,
    ClinicalCaseContext,
    ClinicalPreservationEvidence,
    ClinicalTarget,
    DefenseKind,
    KnowledgeLevel,
    Measurements,
    MeasurementStatus,
    ReleaseDecision,
    ReportCorrectnessResult,
    VictimSnapshot,
)
from care_ecg.security.defenses import (
    HistoryAuthenticator,
    LongitudinalVerifier,
    MeasurementAwareVerifier,
    ShippedVerifier,
)
from care_ecg.security.experiments import (
    CohortCase,
    E5CalibrationCase,
    run_e0,
    run_e1,
    run_e1b,
    run_e2,
    run_e3,
    run_e4,
    run_e5,
    run_e6,
    run_e7,
)
from care_ecg.security.logging import AppendOnlySecurityLogger

TARGET = ClinicalTarget("T", "clean", "target", "target finding", "fabrication")


def independent_report_evaluator(*, context, **kwargs):
    del kwargs
    return ReportCorrectnessResult(
        correct=True,
        context=context,
        evaluator="test-independent-report-review",
        independent=True,
        all_claims_evaluated=True,
        evaluated_claims=("Assessment: clean.",),
        reference_standard="test adjudicated patient record",
    )


def make_snapshot(signal, target_claim=None, released=True, latent=None, facts=("clean",)):
    mean = float(np.mean(signal))
    claim = mean > 0.1 if target_claim is None else target_claim
    diagnosis = "target" if claim else "clean"
    latent_value = np.asarray([mean, mean / 2]) if latent is None else np.asarray(latent)
    return VictimSnapshot(
        diagnosis,
        {"clean": 0.2 if claim else 0.8, "target": 0.8 if claim else 0.2},
        latent_value,
        (latent_value > 0.1).astype(int),
        tuple(facts),
        f"Assessment: {diagnosis}.",
        0.0,
        0.0,
        False,
        not released,
        released,
        claim,
    )


class ToyVictim:
    def __init__(self, defense_evaluators=None):
        self.defense_evaluators = dict(defense_evaluators or {})

    def evaluate(self, signal, target_label, *, defense=DefenseKind.V1, clean_snapshot=None):
        del target_label, clean_snapshot
        snapshot = make_snapshot(signal)
        defense = DefenseKind(defense)
        verifier = self.defense_evaluators.get(defense)
        if verifier is not None:
            decision = verifier.decide(
                snapshot,
                waveform_mv=np.asarray(signal, dtype=np.float32),
                sample_rate_hz=100.0,
                patient_id="p1",
                current_timestamp=2.0,
            )
            snapshot.released = decision.released
            snapshot.abstained = not decision.released
            snapshot.defense_score = decision.score
            snapshot.metadata["defense_decision"] = decision.to_dict()
        return snapshot


class ToyAttack:
    def __init__(self, seed=42, family=AttackFamily.F3, verifier=None):
        self.seed = seed
        self.family = AttackFamily(family)
        self.verifier = verifier

    def run(
        self,
        victim,
        clean_signal,
        target_label,
        *,
        query_budget,
        constraint,
        admissibility_checker=None,
        case=None,
        defense=DefenseKind.V1,
        clean_snapshot=None,
    ):
        del admissibility_checker
        clean = clean_snapshot or victim.evaluate(clean_signal, target_label, defense=defense)
        attack_decision = (
            None if self.verifier is None else self.verifier.decide(clean)
        )
        direction = -0.2 if attack_decision is not None and not attack_decision.released else 0.2
        waveform = constraint.project(clean_signal, clean_signal + direction)
        attacked = victim.evaluate(waveform, target_label, defense=defense, clean_snapshot=clean)
        success = attacked.released and attacked.target_claim_present
        context = ClinicalCaseContext(case.patient_id, case.record_id, case.target)
        evidence = ClinicalPreservationEvidence(
            context=context,
            preserved_findings=("test target-specific clean finding",),
            evaluator="test-independent-preservation-review",
            independent=True,
        )
        admissibility = AdmissibilityResult(
            admissible=True,
            status=MeasurementStatus.AVAILABLE,
            reasons=(),
            preservation_evidence=evidence,
            primary_endpoint_eligible=True,
        )
        attempt = AttackAttempt(
            1,
            waveform,
            0.1,
            attacked.probabilities["target"],
            attacked.diagnosis,
            attacked.released,
            attacked.target_claim_present,
            True,
            success,
            {
                "seed": self.seed,
                "constraint_distance": float(constraint.distance(clean_signal, waveform)),
                "constraint_satisfied": True,
                "primary_endpoint_eligible": True,
                "admissibility": admissibility.to_dict(),
            },
        )
        return AttackResult(
            self.family,
            "target",
            clean,
            attacked,
            waveform,
            [attempt],
            success,
            1,
            "incorrect_release" if success else "query_budget_exhausted",
            {
                "budget": query_budget,
                "primary_endpoint_eligible": True,
                "exploratory_success": success,
                "knowledge_level": "k3",
                "attack_reoptimized_for_defense": True,
                "reoptimized_defense": DefenseKind(defense).value,
                "reoptimization_method": "test_fresh_defense_conditioned_search",
                "reoptimization_evidence": {
                    "defense": DefenseKind(defense).value,
                    "fresh_search_from_clean": True,
                    "complete_system_feedback": True,
                },
            },
        )


class CheckerMutatingAttack(ToyAttack):
    def run(self, *args, **kwargs):
        checker = kwargs.get("admissibility_checker")
        if checker is not None and hasattr(checker, "admissible"):
            checker.admissible = True
        return super().run(*args, **kwargs)


class ThresholdVerifier:
    def __init__(self, kind):
        self.kind = DefenseKind(kind)
        self.threshold = 1.0

    def decide(self, snapshot, **kwargs):
        del kwargs
        score = float(snapshot.unsupported_claim_score)
        released = bool(snapshot.released and score <= self.threshold)
        reasons = () if released else ("test threshold rejected release",)
        return ReleaseDecision(
            defense=self.kind,
            released=released,
            score=score,
            threshold=self.threshold,
            reasons=reasons,
        )


def e5_calibration_cases(scores=(0.1, 0.2, 0.8, 0.9)):
    cases = []
    for index, (score, correct) in enumerate(
        zip(scores, (True, True, False, False), strict=True)
    ):
        case = CohortCase(
            f"cal-p{index}",
            f"cal-r{index}",
            np.zeros((1, 8), dtype=np.float32),
            TARGET,
            metadata={"age": 40.0, "sample_rate_hz": 100.0, "timestamp": 2.0},
        )
        snapshot = make_snapshot(case.waveform_mv)
        snapshot.unsupported_claim_score = float(score)
        assessment = ReportCorrectnessResult(
            correct=correct,
            context=ClinicalCaseContext(case.patient_id, case.record_id, TARGET),
            evaluator="test-independent-calibration-review",
            independent=True,
            all_claims_evaluated=True,
            evaluated_claims=("Assessment: clean.",),
            reference_standard="test adjudicated patient record",
            errors=() if correct else ("incorrect claim",),
        )
        cases.append(
            E5CalibrationCase(
                case,
                snapshot,
                {kind: {} for kind in DefenseKind},
                assessment,
            )
        )
    return tuple(cases)


class IndependentPreservationChecker:
    def __init__(
        self,
        *,
        admissible=True,
        status=MeasurementStatus.AVAILABLE,
        record_id=None,
    ):
        self.admissible = admissible
        self.status = MeasurementStatus(status)
        self.record_id = record_id
        self.waveforms = []

    def check(self, clean, candidate, *, context=None, **kwargs):
        del clean, kwargs
        self.waveforms.append(np.asarray(candidate, dtype=np.float32).copy())
        if context is None:
            raise AssertionError("independent checker requires case context")
        evidence_context = (
            context
            if self.record_id is None
            else replace(context, record_id=self.record_id)
        )
        evidence = ClinicalPreservationEvidence(
            context=evidence_context,
            preserved_findings=("independently reviewed baseline finding",),
            evaluator="test-independent-preservation-boundary",
            independent=True,
        )
        return AdmissibilityResult(
            admissible=self.admissible,
            status=self.status,
            reasons=()
            if self.admissible
            else ("baseline finding not preserved",),
            preservation_evidence=evidence,
            primary_endpoint_eligible=(
                self.status is MeasurementStatus.AVAILABLE
            ),
        )


def measurements():
    return Measurements(
        "independent",
        MeasurementStatus.AVAILABLE,
        {
            "heart_rate_bpm": 72.0,
            "rr_ms": 833.0,
            "rhythm_irregularity": 0.03,
            "pr_ms": 160.0,
            "qrs_ms": 92.0,
            "qt_ms": 390.0,
            "qtc_ms": 427.0,
            "st_deviation_mv": 0.02,
        },
    )


class SecurityExperimentTests(unittest.TestCase):
    def setUp(self):
        self.case = CohortCase(
            "p1",
            "r1",
            np.zeros((1, 8), dtype=np.float32),
            TARGET,
            metadata={"age": 40.0, "timestamp": 2.0},
        )
        self.baseline = run_e0(
            [self.case],
            ToyVictim(),
            report_correctness_evaluator=independent_report_evaluator,
        )
        self.eligible = self.baseline.eligible_cases[0]
        self.self_attested_attack = ToyAttack().run(
            ToyVictim(),
            self.case.waveform_mv,
            "target",
            query_budget=2,
            constraint=SInfConstraint(0.2),
            case=self.case,
            clean_snapshot=self.eligible.clean_snapshot,
        )
        e1 = run_e1(
            [self.eligible],
            ToyVictim(),
            {AttackFamily.F3: lambda seed: ToyAttack(seed)},
            {"s_inf": SInfConstraint(0.2)},
            budgets=(1,),
            seeds=(42,),
            admissibility_checker=IndependentPreservationChecker(),
        )
        self.attack = e1.observations[0].attack_result
        if self.attack is None:
            raise AssertionError("E1 fixture did not retain its verified attack result")

    def test_e0_eligibility_and_e1_budget_matrix(self):
        self.assertEqual(self.baseline.run.summary["eligible"], 1)
        self.assertEqual(self.baseline.run.summary["endpoint_classification"], "primary")
        run = run_e1(
            [self.eligible],
            ToyVictim(),
            {AttackFamily.F3: lambda seed: ToyAttack(seed)},
            {"s_inf": SInfConstraint(0.2)},
            budgets=(1, 2),
            seeds=(1, 2),
            admissibility_checker=IndependentPreservationChecker(),
        )
        self.assertEqual(len(run.observations), 4)
        self.assertTrue(all(row.success for row in run.observations))
        self.assertTrue(all(row.queries_used == 1 for row in run.observations))
        self.assertTrue(
            all(key.startswith("target=T|") for key in run.summary if key != "paired_f3_minus_f1")
        )
        serialized = run.observations[0].to_dict()["attack_result"]
        self.assertEqual(len(serialized["constraint_distances"]), 1)
        self.assertTrue(np.isfinite(serialized["constraint_distances"][0]))
        self.assertEqual(serialized["constraint_satisfied"], [True])
        self.assertNotIn("verification_receipt", serialized)
        public_receipt = run.observations[0].metadata[
            "preservation_verification_receipt"
        ]
        self.assertNotIn("authenticator", public_receipt)

    def test_e0_heuristic_fallback_is_explicitly_exploratory(self):
        baseline = run_e0([self.case], ToyVictim())

        self.assertEqual(len(baseline.eligible_cases), 1)
        self.assertEqual(baseline.run.summary["eligible"], 0)
        self.assertEqual(baseline.run.summary["analysis_eligible"], 1)
        self.assertEqual(baseline.run.summary["exploratory_eligible"], 1)
        self.assertEqual(baseline.run.summary["endpoint_classification"], "exploratory")
        self.assertFalse(baseline.eligible_cases[0].primary_endpoint_eligible)
        self.assertFalse(baseline.run.observations[0].success)

        run = run_e1(
            baseline.eligible_cases,
            ToyVictim(),
            {
                AttackFamily.F1: lambda seed: ToyAttack(seed, AttackFamily.F1),
                AttackFamily.F3: lambda seed: ToyAttack(seed, AttackFamily.F3),
            },
            {"s_inf": SInfConstraint(0.2)},
            budgets=(1,),
            seeds=(1,),
        )
        arms = [value for key, value in run.summary.items() if key != "paired_f3_minus_f1"]
        self.assertTrue(all(value["estimate"] is None for value in arms))
        self.assertTrue(
            all(value["endpoint_classification"] == "not_evaluable" for value in arms)
        )
        paired = next(iter(run.summary["paired_f3_minus_f1"].values()))
        self.assertIsNone(paired["estimate"])
        self.assertEqual(paired["endpoint_classification"], "not_evaluable")
        self.assertEqual(paired["pairs"], 0)

    def test_e1_pairs_f3_and_f1_by_target_case_and_seed(self):
        run = run_e1(
            [self.eligible],
            ToyVictim(),
            {
                AttackFamily.F1: lambda seed: ToyAttack(seed, AttackFamily.F1),
                AttackFamily.F3: lambda seed: ToyAttack(seed, AttackFamily.F3),
            },
            {"s_inf": SInfConstraint(0.2)},
            budgets=(1,),
            seeds=(1, 2),
            admissibility_checker=IndependentPreservationChecker(),
        )
        paired = run.summary["paired_f3_minus_f1"]
        key = "target=T|f3-f1:s_inf|B=1|v1"
        self.assertEqual(paired[key]["pairs"], 2)
        self.assertEqual(paired[key]["n_patients"], 1)
        self.assertEqual(paired[key]["seeds"], [1, 2])

    def test_e1b_e2_e3_and_e4_mechanism_helpers(self):
        pairs = [(self.eligible, self.attack)]
        e1b = run_e1b(
            [self.eligible],
            {self.case.key: self.attack},
            {"full": ToyVictim()},
        )

        def replay(context_snapshot, retrieval_snapshot, draft_report=None, **kwargs):
            del kwargs
            claim = context_snapshot.target_claim_present or draft_report is not None
            released = retrieval_snapshot.target_claim_present
            return {"released": released, "target_claim_present": claim}

        e2 = run_e2(pairs, replay)

        def latent_replay(latent, **kwargs):
            del kwargs
            return make_snapshot(np.zeros((1, 8)), target_claim=bool(np.max(latent) > 0.1), latent=latent)

        e3 = run_e3(pairs, latent_replay, top_k=(1,), random_repeats=2)
        e4 = run_e4(pairs, lambda **kwargs: np.asarray([0.2, 0.0]), k=1)

        self.assertEqual(len(e1b.observations), 1)
        self.assertEqual(len(e2.observations), 6)
        self.assertEqual(e2.summary["source_pair_eligibility"]["eligible"], 1)
        self.assertTrue(
            all(
                row.metrics["source_selected_clinically_admissible"]
                for row in e2.observations
            )
        )
        self.assertEqual(len(e3.observations), 3)
        self.assertEqual(len(e4.observations), 1)
        self.assertIn("cosine_similarity", e4.observations[0].metrics)
        self.assertIn("rate_ci", e1b.summary["full"])
        self.assertIn("T", e1b.summary["full"]["by_target"])

    def test_e5_reoptimizes_each_defense_and_e6_transfers(self):
        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        with tempfile.TemporaryDirectory() as directory:
            logger = AppendOnlySecurityLogger(Path(directory) / "e5.jsonl")
            e5 = run_e5(
                [self.eligible],
                lambda case, defense, verifier: ToyVictim(
                    {defense: verifier}
                ),
                lambda seed, defense, verifier, case: ToyAttack(
                    seed, verifier=verifier
                ),
                verifiers,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
                seeds=(1,),
                query_budget=2,
                calibration_cases=e5_calibration_cases(),
                admissibility_checker=IndependentPreservationChecker(),
                report_correctness_evaluator=independent_report_evaluator,
                reoptimization_validator=lambda **kwargs: {
                    "valid": True,
                    "configured_defense": kwargs["defense"].value,
                },
                logger=logger,
            )
            records = logger.records()
            self.assertEqual(len(records), 6)
            self.assertEqual(
                {
                    row["payload"]["experiment_context"]["defense"]
                    for row in records
                },
                {"v1", "v2", "v3"},
            )
            self.assertTrue(
                all(
                    row["payload"]["experiment_context"]["constraint"] == "s_phys"
                    for row in records
                )
            )
        e6 = run_e6(
            [(self.eligible, self.attack)],
            ToyVictim(),
            report_correctness_evaluator=independent_report_evaluator,
        )
        self.assertEqual(len(e5.observations), 3)
        self.assertTrue(all(row.metadata["reoptimized"] for row in e5.observations))
        self.assertTrue(
            all(
                row.metadata["behaviorally_bound_verifier_queries"] == 1
                for row in e5.observations
            )
        )
        self.assertTrue(all(row.metadata["audit_event_ids"] for row in e5.observations))
        self.assertTrue(
            all(
                point["correctness_available"]
                for point in e5.summary["clean_operating_points"].values()
            )
        )
        self.assertEqual(
            e5.summary["clean_operating_points"][DefenseKind.V1.value][
                "selective_risk"
            ],
            0.0,
        )
        self.assertEqual(
            e5.summary["calibration_source"],
            "orchestrator_recomputed_typed_shared_cases",
        )
        self.assertEqual(len(e6.observations), 1)
        self.assertTrue(e6.observations[0].success)
        self.assertEqual(e6.summary["clean_transfer_eligibility"]["eligible"], 1)
        self.assertEqual(
            e6.summary["transfer_replay"]["endpoint_classification"],
            "primary",
        )

    def test_e2_excludes_released_claim_without_primary_source_preservation(self):
        exploratory = replace(self.eligible, primary_endpoint_eligible=False)
        self.attack.metadata["primary_endpoint_eligible"] = False

        run = run_e2(
            [(exploratory, self.attack)],
            lambda **kwargs: {
                "released": True,
                "target_claim_present": True,
            },
        )

        self.assertEqual(run.observations, ())
        eligibility = run.summary["source_pair_eligibility"]
        self.assertEqual(eligibility["submitted"], 1)
        self.assertEqual(eligibility["eligible"], 0)
        self.assertEqual(
            eligibility["exclusion_counts"][
                "source_attack_not_primary_endpoint_eligible"
            ],
            1,
        )

    def test_e6_heuristic_and_legacy_correctness_are_exploratory_only(self):
        heuristic = run_e6([(self.eligible, self.attack)], ToyVictim())
        legacy = run_e6(
            [(self.eligible, self.attack)],
            ToyVictim(),
            clean_correctness=lambda **kwargs: True,
        )
        untyped_evaluator = run_e6(
            [(self.eligible, self.attack)],
            ToyVictim(),
            report_correctness_evaluator=lambda **kwargs: True,
        )

        for run in (heuristic, legacy, untyped_evaluator):
            self.assertEqual(len(run.observations), 1)
            self.assertFalse(run.observations[0].success)
            self.assertTrue(
                run.observations[0].metrics["exploratory_incorrect_release"]
            )
            arm = run.summary["transfer_replay"]
            self.assertIsNone(arm["rate"])
            self.assertEqual(arm["endpoint_classification"], "not_evaluable")
            self.assertEqual(run.summary["endpoint_classification"], "not_evaluable")
            self.assertEqual(arm["exploratory_rate"], 1.0)
            eligibility = run.summary["clean_transfer_eligibility"]
            self.assertEqual(eligibility["eligible"], 0)
            self.assertEqual(eligibility["analysis_eligible"], 1)
            self.assertEqual(eligibility["endpoint_classification"], "not_evaluable")
            self.assertIsNone(run.summary["by_target"]["T"]["transfer_irr"])

    def test_e5_rejects_inert_instantiated_providers(self):
        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        scores = {kind: [0.1, 0.2] for kind in DefenseKind}
        with self.assertRaisesRegex(TypeError, "must be a factory"):
            run_e5(
                [self.eligible],
                ToyVictim(),
                lambda seed, defense, verifier, case: ToyAttack(seed),
                verifiers,
                scores,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
            )

        with self.assertRaisesRegex(ValueError, "invoke the bound verifier"):
            run_e5(
                [self.eligible],
                lambda case, defense, verifier: ToyVictim(),
                lambda seed, defense, verifier, case: ToyAttack(seed),
                verifiers,
                scores,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
                seeds=(1,),
                query_budget=2,
            )

        class InertVerifier:
            def __init__(self, kind):
                self.kind = kind
                self.threshold = 1.0

        inert = {kind: InertVerifier(kind) for kind in DefenseKind}
        with self.assertRaisesRegex(TypeError, "callable decide"):
            run_e5(
                [],
                lambda case, defense, verifier: ToyVictim({defense: verifier}),
                lambda seed, defense, verifier, case: ToyAttack(seed),
                inert,
                scores,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
            )

        no_labels = run_e5(
            [],
            lambda case, defense, verifier: ToyVictim(),
            lambda seed, defense, verifier, case: ToyAttack(seed),
            verifiers,
            scores,
            target_coverage=0.5,
            constraint=SPhysConstraint(sample_rate_hz=100.0),
        )
        self.assertTrue(
            all(
                point["selective_risk"] is None
                and not point["correctness_available"]
                for point in no_labels.summary["clean_operating_points"].values()
            )
        )

    def test_e5_primary_contract_rejects_missing_arm_non_sphys_and_boolean_review(self):
        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        scores = {kind: [0.1, 0.2] for kind in DefenseKind}
        factory = lambda seed, defense, verifier, case: ToyAttack(
            seed, verifier=verifier
        )
        victim = lambda case, defense, verifier: ToyVictim({defense: verifier})
        validator = lambda **kwargs: {"valid": True}

        with self.assertRaisesRegex(ValueError, "exactly V1, V2, and V3"):
            run_e5(
                [self.eligible],
                victim,
                factory,
                {DefenseKind.V1: ThresholdVerifier(DefenseKind.V1)},
                scores,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
            )
        with self.assertRaisesRegex(ValueError, "SPhys"):
            run_e5(
                [self.eligible],
                victim,
                factory,
                verifiers,
                scores,
                target_coverage=0.5,
                constraint=SInfConstraint(0.2),
            )
        with self.assertRaisesRegex(TypeError, "ReportCorrectnessResult"):
            run_e5(
                [self.eligible],
                victim,
                factory,
                verifiers,
                scores,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
                report_correctness_evaluator=lambda **kwargs: True,
                reoptimization_validator=validator,
            )

    def test_e6_excludes_transfer_victim_that_is_wrong_on_clean_input(self):
        class IneligibleTransferVictim:
            def evaluate(self, signal, target_label, **kwargs):
                del signal, target_label, kwargs
                return make_snapshot(
                    np.zeros((1, 8)), target_claim=True, released=True
                )

        run = run_e6([(self.eligible, self.attack)], IneligibleTransferVictim())
        self.assertEqual(run.observations, ())
        self.assertEqual(run.summary["clean_transfer_eligibility"]["excluded"], 1)

    def test_e7_rejects_tampered_authenticated_history(self):
        authenticator = HistoryAuthenticator(b"0123456789abcdef0123456789abcdef")
        prior = np.zeros((1, 1000), dtype=np.float32)
        history = authenticator.seal(
            patient_id="p1",
            record_id="prior",
            timestamp=1.0,
            waveform_mv=prior,
            measurements=measurements(),
        )
        verifier = LongitudinalVerifier(
            authenticator,
            measurement_verifier=MeasurementAwareVerifier(),
        )
        replayed = []

        def pre_defense_replay(source_snapshot, **kwargs):
            replayed.append(kwargs["waveform_mv"])
            return source_snapshot

        run = run_e7(
            [(self.eligible, self.attack)],
            verifier,
            lambda case: {
                "history": history,
                "prior_waveform_mv": prior,
                "measurements": measurements(),
                "current_timestamp": 2.0,
            },
            pre_defense_replay=pre_defense_replay,
        )

        by_arm = {row.arm: row for row in run.observations}
        self.assertTrue(by_arm["no_history_v2"].released)
        self.assertTrue(by_arm["trusted_history"].released)
        self.assertFalse(by_arm["tampered_history"].released)
        self.assertIn("trusted_history_gain", run.summary)
        self.assertEqual(len(replayed), 1)
        self.assertTrue(
            all(row.metrics["common_pre_defense_replay"] for row in run.observations)
        )

    def test_e0_requires_explicit_adult_metadata_for_primary_endpoint(self):
        no_age = replace(self.case, metadata={})
        baseline = run_e0(
            [no_age],
            ToyVictim(),
            report_correctness_evaluator=independent_report_evaluator,
        )

        self.assertEqual(baseline.run.summary["analysis_eligible"], 1)
        self.assertEqual(baseline.run.summary["eligible"], 0)
        self.assertEqual(
            baseline.run.observations[0].metadata["adult_status"], "not_evaluable"
        )

    def test_e1_recomputes_constraint_membership_and_query_count(self):
        class PermissiveConstraint:
            kind = "s_inf"

            @staticmethod
            def project(clean, candidate):
                del candidate
                return np.asarray(clean, dtype=np.float32) + 1.0

            @staticmethod
            def distance(clean, candidate):
                return float(np.max(np.abs(candidate - clean)))

        class OutOfSetAttack(ToyAttack):
            def run(self, victim, clean_signal, target_label, **kwargs):
                return super().run(
                    victim,
                    clean_signal,
                    target_label,
                    **{**kwargs, "constraint": PermissiveConstraint()},
                )

        with self.assertRaisesRegex(ValueError, "outside the exact constraint"):
            run_e1(
                [self.eligible],
                ToyVictim(),
                {AttackFamily.F3: lambda seed: OutOfSetAttack(seed)},
                {"s_inf": SInfConstraint(0.2)},
                budgets=(1,),
                seeds=(1,),
            )

        class MisreportingAttack(ToyAttack):
            def run(self, *args, **kwargs):
                result = super().run(*args, **kwargs)
                result.queries_used = 0
                return result

        with self.assertRaisesRegex(ValueError, "counted 1 complete-system calls"):
            run_e1(
                [self.eligible],
                ToyVictim(),
                {AttackFamily.F3: lambda seed: MisreportingAttack(seed)},
                {"s_inf": SInfConstraint(0.2)},
                budgets=(1,),
                seeds=(1,),
            )

    def test_e1_ignores_attack_authored_preservation_mapping(self):
        class WrongContextAttack(ToyAttack):
            def run(self, *args, **kwargs):
                result = super().run(*args, **kwargs)
                result.attempts[-1].metadata["admissibility"]["preservation_evidence"][
                    "context"
                ]["record_id"] = "another-record"
                return result

        run = run_e1(
            [self.eligible],
            ToyVictim(),
            {AttackFamily.F3: lambda seed: WrongContextAttack(seed)},
            {"s_inf": SInfConstraint(0.2)},
            budgets=(1,),
            seeds=(1,),
            admissibility_checker=IndependentPreservationChecker(),
        )

        row = run.observations[0]
        self.assertTrue(row.success)
        self.assertTrue(row.metrics["primary_endpoint_eligible"])
        self.assertTrue(row.metrics["exploratory_incorrect_release"])
        assessment = row.metadata["independent_preservation_assessment"]
        self.assertEqual(
            assessment["evaluations"][0]["result"]["preservation_evidence"][
                "context"
            ]["record_id"],
            self.case.record_id,
        )

    def test_e1_requires_typed_bound_checker_evidence_for_primary_endpoint(self):
        for checker in (
            None,
            lambda clean, candidate: True,
            IndependentPreservationChecker(record_id="another-record"),
            IndependentPreservationChecker(
                admissible=False,
                status=MeasurementStatus.UNAVAILABLE,
            ),
        ):
            with self.subTest(checker=type(checker).__name__):
                run = run_e1(
                    [self.eligible],
                    ToyVictim(),
                    {AttackFamily.F3: lambda seed: ToyAttack(seed)},
                    {"s_inf": SInfConstraint(0.2)},
                    budgets=(1,),
                    seeds=(1,),
                    admissibility_checker=checker,
                )
                row = run.observations[0]
                self.assertFalse(row.success)
                self.assertFalse(row.metrics["primary_endpoint_eligible"])
                arm = next(
                    value
                    for key, value in run.summary.items()
                    if key != "paired_f3_minus_f1"
                )
                self.assertIsNone(arm["estimate"])
                self.assertEqual(arm["endpoint_classification"], "not_evaluable")

    def test_e1_typed_nonpreservation_is_a_genuine_primary_zero(self):
        checker = IndependentPreservationChecker(admissible=False)
        run = run_e1(
            [self.eligible],
            ToyVictim(),
            {AttackFamily.F3: lambda seed: CheckerMutatingAttack(seed)},
            {"s_inf": SInfConstraint(0.2)},
            budgets=(1,),
            seeds=(1,),
            admissibility_checker=checker,
        )

        row = run.observations[0]
        self.assertTrue(row.metrics["primary_endpoint_eligible"])
        self.assertFalse(row.success)
        self.assertFalse(checker.admissible)
        arm = next(
            value for key, value in run.summary.items() if key != "paired_f3_minus_f1"
        )
        self.assertEqual(arm["estimate"], 0.0)
        self.assertEqual(len(checker.waveforms), 1)
        np.testing.assert_array_equal(
            checker.waveforms[0], row.attack_result.attempts[0].waveform
        )

    def test_e1_stateful_function_checker_is_invoked_once_per_query(self):
        calls = []

        def checker(clean, candidate, *, context):
            del clean, candidate
            calls.append(context)
            admissible = len(calls) > 1
            return AdmissibilityResult(
                admissible=admissible,
                status=MeasurementStatus.AVAILABLE,
                reasons=() if admissible else ("not preserved",),
                preservation_evidence=ClinicalPreservationEvidence(
                    context=context,
                    preserved_findings=("independently reviewed finding",),
                    evaluator="stateful-independent-review",
                    independent=True,
                ),
                primary_endpoint_eligible=True,
            )

        class CheckerCallingAttack(ToyAttack):
            def run(self, *args, **kwargs):
                attack_checker = kwargs.get("admissibility_checker")
                result = super().run(*args, **kwargs)
                attack_checker.check(args[1], result.attempts[0].waveform)
                return result

        run = run_e1(
            [self.eligible],
            ToyVictim(),
            {AttackFamily.F3: lambda seed: CheckerCallingAttack(seed)},
            {"s_inf": SInfConstraint(0.2)},
            budgets=(1,),
            seeds=(1,),
            admissibility_checker=checker,
        )

        self.assertEqual(len(calls), 1)
        self.assertTrue(run.observations[0].metrics["primary_endpoint_eligible"])
        self.assertFalse(run.observations[0].success)

    def test_e1_rejects_attack_supplied_preservation_receipt(self):
        receipt = self.attack.verification_receipt

        class ReceiptReplayingAttack(ToyAttack):
            def run(self, *args, **kwargs):
                result = super().run(*args, **kwargs)
                result.verification_receipt = receipt
                return result

        with self.assertRaisesRegex(ValueError, "unauthorized preservation receipt"):
            run_e1(
                [self.eligible],
                ToyVictim(),
                {AttackFamily.F3: lambda seed: ReceiptReplayingAttack(seed)},
                {"s_inf": SInfConstraint(0.2)},
                budgets=(1,),
                seeds=(1,),
                admissibility_checker=IndependentPreservationChecker(),
            )

    def test_unsuccessful_attacks_remain_primary_denominator_zeros(self):
        class NoEventAttack:
            def __init__(self, eligible, verifier=None):
                self.eligible = eligible
                self.verifier = verifier

            def run(
                self,
                victim,
                clean_signal,
                target_label,
                *,
                query_budget,
                constraint,
                defense=DefenseKind.V1,
                **kwargs,
            ):
                del kwargs
                attack_decision = (
                    None
                    if self.verifier is None
                    else self.verifier.decide(self.eligible.clean_snapshot)
                )
                direction = (
                    -0.05
                    if attack_decision is not None and not attack_decision.released
                    else 0.05
                )
                waveform = constraint.project(clean_signal, clean_signal + direction)
                snapshot = victim.evaluate(
                    waveform,
                    target_label,
                    defense=defense,
                    clean_snapshot=self.eligible.clean_snapshot,
                )
                attempt = AttackAttempt(
                    query_index=1,
                    waveform=waveform,
                    objective=1.0,
                    target_probability=snapshot.probabilities["target"],
                    diagnosis=snapshot.diagnosis,
                    released=snapshot.released,
                    target_claim_present=False,
                    admissible=False,
                    success=False,
                    metadata={
                        "constraint_distance": constraint.distance(
                            clean_signal, waveform
                        ),
                        "constraint_satisfied": True,
                    },
                )
                defense_name = DefenseKind(defense).value
                return AttackResult(
                    family=AttackFamily.F3,
                    target_label=target_label,
                    clean_snapshot=self.eligible.clean_snapshot,
                    adversarial_snapshot=snapshot,
                    adversarial_signal=waveform,
                    attempts=[attempt],
                    success=False,
                    queries_used=1,
                    stopped_reason="query_budget_exhausted",
                    metadata={
                        "query_budget": query_budget,
                        "exploratory_success": False,
                        "attack_reoptimized_for_defense": True,
                        "reoptimized_defense": defense_name,
                        "reoptimization_method": "test_fresh_defense_search",
                        "reoptimization_evidence": {
                            "defense": defense_name,
                            "fresh_search_from_clean": True,
                            "complete_system_feedback": True,
                        },
                    },
                )

        attack = NoEventAttack(self.eligible)
        e1 = run_e1(
            [self.eligible],
            ToyVictim(),
            {AttackFamily.F3: lambda seed: attack},
            {"s_inf": SInfConstraint(0.2)},
            budgets=(1,),
            seeds=(1,),
            admissibility_checker=IndependentPreservationChecker(),
        )
        self.assertTrue(e1.observations[0].metrics["primary_endpoint_eligible"])
        self.assertFalse(e1.observations[0].success)

        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        e5 = run_e5(
            [self.eligible],
            lambda case, defense, verifier: ToyVictim({defense: verifier}),
            lambda seed, defense, verifier, case: NoEventAttack(
                self.eligible, verifier
            ),
            verifiers,
            target_coverage=0.5,
            constraint=SPhysConstraint(sample_rate_hz=100.0),
            seeds=(1,),
            query_budget=1,
            calibration_cases=e5_calibration_cases(),
            admissibility_checker=IndependentPreservationChecker(),
        )
        self.assertEqual(e5.summary["endpoint_classification"], "not_evaluable")
        self.assertTrue(
            all(
                not row.metrics["primary_endpoint_eligible"]
                for row in e5.observations
            )
        )
        self.assertTrue(all(not row.success for row in e5.observations))

    def test_e2_rejects_outputs_without_explicit_claim_state(self):
        decision = ReleaseDecision(
            defense=DefenseKind.V1,
            released=True,
            score=0.0,
            threshold=1.0,
        )
        for outcome, message in (
            (decision, "does not establish target-claim presence"),
            (True, "cannot distinguish release"),
        ):
            with self.subTest(outcome=type(outcome).__name__), self.assertRaisesRegex(
                TypeError, message
            ):
                run_e2(
                    [(self.eligible, self.attack)],
                    lambda outcome=outcome, **kwargs: outcome,
                )

    def test_e5_requires_same_keyed_calibration_cohort_for_primary(self):
        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        scores = {kind: [0.1, 0.2] for kind in DefenseKind}
        unkeyed = run_e5(
            [],
            lambda case, defense, verifier: ToyVictim({defense: verifier}),
            lambda seed, defense, verifier, case: ToyAttack(seed),
            verifiers,
            scores,
            target_coverage=0.5,
            constraint=SPhysConstraint(sample_rate_hz=100.0),
        )
        self.assertEqual(
            unkeyed.summary["calibration_endpoint_classification"], "exploratory"
        )
        aligned = {
            kind: ("p1:r1:T", "p2:r2:T") for kind in DefenseKind
        }
        keyed_legacy = run_e5(
            [],
            lambda case, defense, verifier: ToyVictim({defense: verifier}),
            lambda seed, defense, verifier, case: ToyAttack(seed),
            verifiers,
            scores,
            target_coverage=0.5,
            constraint=SPhysConstraint(sample_rate_hz=100.0),
            calibration_case_keys=aligned,
        )
        self.assertTrue(keyed_legacy.summary["legacy_calibration_keys_aligned"])
        self.assertFalse(
            keyed_legacy.summary["calibration_verifier_behavior_recomputed"]
        )
        self.assertEqual(
            keyed_legacy.summary["calibration_endpoint_classification"],
            "exploratory",
        )

        mismatched = {
            DefenseKind.V1: ("p1:r1:T", "p2:r2:T"),
            DefenseKind.V2: ("p1:r1:T", "p2:r2:T"),
            DefenseKind.V3: ("p2:r2:T", "p1:r1:T"),
        }
        with self.assertRaisesRegex(ValueError, "same ordered keyed calibration cohort"):
            run_e5(
                [],
                lambda case, defense, verifier: ToyVictim({defense: verifier}),
                lambda seed, defense, verifier, case: ToyAttack(seed),
                verifiers,
                scores,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
                calibration_case_keys=mismatched,
            )

    def test_e5_external_valid_flag_cannot_replace_native_verifier_binding(self):
        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        scores = {kind: [0.1, 0.2] for kind in DefenseKind}
        with self.assertRaisesRegex(ValueError, "invoke the bound verifier"):
            run_e5(
                [self.eligible],
                lambda case, defense, verifier: ToyVictim(),
                lambda seed, defense, verifier, case: ToyAttack(seed),
                verifiers,
                scores,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
                query_budget=1,
                reoptimization_validator=lambda **kwargs: {"valid": True},
            )

    def test_e5_fresh_provenance_is_observed_not_attack_self_attested(self):
        class MetadataFreeAttack(ToyAttack):
            def run(self, *args, **kwargs):
                result = super().run(*args, **kwargs)
                for key in (
                    "attack_reoptimized_for_defense",
                    "reoptimized_defense",
                    "reoptimization_method",
                    "reoptimization_evidence",
                ):
                    result.metadata.pop(key, None)
                return result

        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        result = run_e5(
            [self.eligible],
            lambda case, defense, verifier: ToyVictim({defense: verifier}),
            lambda seed, defense, verifier, case: MetadataFreeAttack(
                seed, verifier=verifier
            ),
            verifiers,
            target_coverage=0.5,
            constraint=SPhysConstraint(sample_rate_hz=100.0),
            calibration_cases=e5_calibration_cases(),
            seeds=(1,),
            query_budget=1,
            admissibility_checker=IndependentPreservationChecker(),
        )

        self.assertEqual(result.summary["endpoint_classification"], "not_evaluable")
        for row in result.observations:
            evidence = row.metadata["reoptimization_evidence"]
            native = evidence["native"]
            self.assertFalse(row.metrics["primary_endpoint_eligible"])
            self.assertEqual(evidence["endpoint_classification"], "exploratory")
            self.assertFalse(evidence["primary_eligible"])
            self.assertTrue(native["verifier_behaviorally_bound"])
            self.assertTrue(native["attempts_match_observed_queries"])
            self.assertTrue(native["defense_conditioning_observed"])
            self.assertFalse(native["primary_defense_conditioning_attested"])
            self.assertEqual(native["observed_attack_verifier_calls"], 1)
            self.assertTrue(native["causal_attack_verifier_probe"]["valid"])
            self.assertTrue(
                native["causal_attack_verifier_probe"][
                    "intervention_changed_submitted_queries"
                ]
            )
            self.assertEqual(
                native["independently_replayed_verifier_decisions"],
                native["observed_complete_system_queries"],
            )
            self.assertFalse(native["attack_authored_metadata_used_for_validity"])

    def test_e5_exact_audited_builtin_retains_primary_eligibility(self):
        class BuiltinCompatibleVictim(ToyVictim):
            label_names = ("clean", "target")
            sample_rate_hz = 100.0

            def target_index(self, label):
                return self.label_names.index(str(label))

        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        result = run_e5(
            [self.eligible],
            lambda case, defense, verifier: BuiltinCompatibleVictim(
                {defense: verifier}
            ),
            lambda seed, defense, verifier, case: F3AdaptiveAttack(
                knowledge_level=KnowledgeLevel.K1,
                black_box_method="square",
                epsilon_mv=0.1,
                steps=1,
                seed=seed,
            ),
            verifiers,
            target_coverage=0.5,
            constraint=SPhysConstraint(sample_rate_hz=100.0),
            calibration_cases=e5_calibration_cases(),
            seeds=(1,),
            query_budget=1,
            admissibility_checker=IndependentPreservationChecker(),
        )

        self.assertEqual(result.summary["endpoint_classification"], "primary")
        for row in result.observations:
            evidence = row.metadata["reoptimization_evidence"]
            self.assertTrue(evidence["primary_eligible"])
            self.assertEqual(evidence["source"], "audited_builtin_f3")
            self.assertTrue(
                evidence["native"]["builtin_implementation_attestation"][
                    "valid"
                ]
            )
            self.assertTrue(row.metrics["primary_endpoint_eligible"])

    def test_e5_rejects_one_shot_f3_label_without_observed_conditioning(self):
        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        with self.assertRaisesRegex(ValueError, "could not establish"):
            run_e5(
                [self.eligible],
                lambda case, defense, verifier: ToyVictim({defense: verifier}),
                lambda seed, defense, verifier, case: ToyAttack(seed),
                verifiers,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
                calibration_cases=e5_calibration_cases(),
                seeds=(1,),
                query_budget=1,
                admissibility_checker=IndependentPreservationChecker(),
            )

    def test_e5_marks_ignored_attack_verifier_call_exploratory(self):
        class IgnoredVerifierAttack(ToyAttack):
            def run(self, victim, clean_signal, target_label, **kwargs):
                self.verifier.decide(kwargs["clean_snapshot"])
                self.verifier = None
                return super().run(victim, clean_signal, target_label, **kwargs)

        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        result = run_e5(
            [self.eligible],
            lambda case, defense, verifier: ToyVictim({defense: verifier}),
            lambda seed, defense, verifier, case: IgnoredVerifierAttack(
                seed, verifier=verifier
            ),
            verifiers,
            target_coverage=0.5,
            constraint=SPhysConstraint(sample_rate_hz=100.0),
            calibration_cases=e5_calibration_cases(),
            seeds=(1,),
            query_budget=1,
            admissibility_checker=IndependentPreservationChecker(),
        )
        self.assertEqual(result.summary["endpoint_classification"], "not_evaluable")
        self.assertTrue(
            all(
                not row.metadata["reoptimization_evidence"]["primary_eligible"]
                and not row.metrics["primary_endpoint_eligible"]
                for row in result.observations
            )
        )

    def test_e5_marks_structural_probe_side_channel_exploratory(self):
        class StructuralSideChannelAttack(ToyAttack):
            def run(
                self,
                victim,
                clean_signal,
                target_label,
                **kwargs,
            ):
                verifier = self.verifier
                verifier.decide(kwargs["clean_snapshot"])
                structural_branch = hasattr(verifier, "_verifier")

                class BranchDecision:
                    released = structural_branch

                class BranchVerifier:
                    @staticmethod
                    def decide(snapshot):
                        del snapshot
                        return BranchDecision()

                self.verifier = BranchVerifier()
                return super().run(victim, clean_signal, target_label, **kwargs)

        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        result = run_e5(
            [self.eligible],
            lambda case, defense, verifier: ToyVictim({defense: verifier}),
            lambda seed, defense, verifier, case: StructuralSideChannelAttack(
                seed, verifier=verifier
            ),
            verifiers,
            target_coverage=0.5,
            constraint=SPhysConstraint(sample_rate_hz=100.0),
            calibration_cases=e5_calibration_cases(),
            seeds=(1,),
            query_budget=1,
            admissibility_checker=IndependentPreservationChecker(),
        )
        self.assertEqual(result.summary["endpoint_classification"], "not_evaluable")
        self.assertTrue(
            all(
                row.metadata["reoptimization_endpoint_classification"]
                == "exploratory"
                and not row.metrics["primary_endpoint_eligible"]
                for row in result.observations
            )
        )

    def test_e5_reflective_probe_state_cannot_promote_custom_factory(self):
        class ReflectiveSideChannelAttack(ToyAttack):
            def run(self, victim, clean_signal, target_label, **kwargs):
                from care_ecg.security import experiments as experiment_module

                verifier = self.verifier
                delegate = object.__getattribute__(
                    verifier,
                    "_ObservedVerifier__verifier",
                )
                registry = getattr(
                    experiment_module,
                    "_E5_DECISION_PROBE_INTERVENTIONS",
                    {},
                )
                reflective_branch = bool(registry.get(delegate, False))
                verifier.decide(kwargs["clean_snapshot"])

                class BranchDecision:
                    released = reflective_branch

                class BranchVerifier:
                    @staticmethod
                    def decide(snapshot):
                        del snapshot
                        return BranchDecision()

                self.verifier = BranchVerifier()
                return super().run(victim, clean_signal, target_label, **kwargs)

        from care_ecg.security import experiments as experiment_module

        self.assertFalse(
            hasattr(experiment_module, "_E5_DECISION_PROBE_INTERVENTIONS")
        )
        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        result = run_e5(
            [self.eligible],
            lambda case, defense, verifier: ToyVictim({defense: verifier}),
            lambda seed, defense, verifier, case: ReflectiveSideChannelAttack(
                seed, verifier=verifier
            ),
            verifiers,
            target_coverage=0.5,
            constraint=SPhysConstraint(sample_rate_hz=100.0),
            calibration_cases=e5_calibration_cases(),
            seeds=(1,),
            query_budget=1,
            admissibility_checker=IndependentPreservationChecker(),
        )

        self.assertEqual(result.summary["endpoint_classification"], "not_evaluable")
        self.assertTrue(
            all(
                not row.metadata["reoptimization_evidence"]["primary_eligible"]
                and not row.metrics["primary_endpoint_eligible"]
                for row in result.observations
            )
        )

    def test_e5_global_builtin_rebinding_cannot_promote_custom_factory(self):
        from care_ecg.security import experiments as experiment_module

        class GlobalRebindingAttack(ToyAttack):
            def run(self, *args, **kwargs):
                experiment_module.F3AdaptiveAttack = type(self)
                experiment_module._AUDITED_F3_ADAPTIVE_RUN = type(self).run
                return super().run(*args, **kwargs)

        original_class = experiment_module.F3AdaptiveAttack
        had_run_alias = hasattr(
            experiment_module,
            "_AUDITED_F3_ADAPTIVE_RUN",
        )
        original_run_alias = getattr(
            experiment_module,
            "_AUDITED_F3_ADAPTIVE_RUN",
            None,
        )
        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        try:
            result = run_e5(
                [self.eligible],
                lambda case, defense, verifier: ToyVictim({defense: verifier}),
                lambda seed, defense, verifier, case: GlobalRebindingAttack(
                    seed, verifier=verifier
                ),
                verifiers,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
                calibration_cases=e5_calibration_cases(),
                seeds=(1,),
                query_budget=1,
                admissibility_checker=IndependentPreservationChecker(),
            )
        finally:
            experiment_module.F3AdaptiveAttack = original_class
            if had_run_alias:
                experiment_module._AUDITED_F3_ADAPTIVE_RUN = original_run_alias
            elif hasattr(experiment_module, "_AUDITED_F3_ADAPTIVE_RUN"):
                del experiment_module._AUDITED_F3_ADAPTIVE_RUN

        self.assertEqual(result.summary["endpoint_classification"], "not_evaluable")
        for row in result.observations:
            evidence = row.metadata["reoptimization_evidence"]
            attestation = evidence["native"][
                "builtin_implementation_attestation"
            ]
            self.assertFalse(attestation["valid"])
            self.assertFalse(evidence["primary_eligible"])
            self.assertFalse(row.metrics["primary_endpoint_eligible"])

    def test_e5_rejects_fixed_multiquery_sequence_as_reoptimization(self):
        class FixedTwoQueryAttack:
            def run(
                self,
                victim,
                clean_signal,
                target_label,
                *,
                query_budget,
                constraint,
                defense,
                clean_snapshot,
                **kwargs,
            ):
                del query_budget, kwargs
                first_waveform = np.asarray(clean_signal, dtype=np.float32).copy()
                second_waveform = constraint.project(
                    clean_signal,
                    clean_signal + 0.2,
                )
                first = victim.evaluate(
                    first_waveform,
                    target_label,
                    defense=defense,
                    clean_snapshot=clean_snapshot,
                )
                second = victim.evaluate(
                    second_waveform,
                    target_label,
                    defense=defense,
                    clean_snapshot=clean_snapshot,
                )
                attempts = [
                    AttackAttempt(
                        1,
                        first_waveform,
                        1.0,
                        first.probabilities["target"],
                        first.diagnosis,
                        first.released,
                        first.target_claim_present,
                        False,
                        False,
                    ),
                    AttackAttempt(
                        2,
                        second_waveform,
                        0.1,
                        second.probabilities["target"],
                        second.diagnosis,
                        second.released,
                        second.target_claim_present,
                        True,
                        bool(second.released and second.target_claim_present),
                    ),
                ]
                return AttackResult(
                    AttackFamily.F3,
                    target_label,
                    clean_snapshot,
                    second,
                    second_waveform,
                    attempts,
                    attempts[-1].success,
                    2,
                    "incorrect_release",
                )

        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        with self.assertRaisesRegex(ValueError, "could not establish"):
            run_e5(
                [self.eligible],
                lambda case, defense, verifier: ToyVictim({defense: verifier}),
                lambda seed, defense, verifier, case: FixedTwoQueryAttack(),
                verifiers,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
                calibration_cases=e5_calibration_cases(),
                seeds=(1,),
                query_budget=2,
                admissibility_checker=IndependentPreservationChecker(),
            )

    def test_e5_rejects_forged_adversarial_snapshot_payload(self):
        class SnapshotForgingAttack(ToyAttack):
            def run(self, *args, **kwargs):
                result = super().run(*args, **kwargs)
                forged = copy.deepcopy(result.adversarial_snapshot)
                forged.report = "Assessment: attacker-authored report."
                result.adversarial_snapshot = forged
                return result

        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        with self.assertRaisesRegex(ValueError, "full recorded victim output"):
            run_e5(
                [self.eligible],
                lambda case, defense, verifier: ToyVictim({defense: verifier}),
                lambda seed, defense, verifier, case: SnapshotForgingAttack(
                    seed, verifier=verifier
                ),
                verifiers,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
                calibration_cases=e5_calibration_cases(),
                seeds=(1,),
                query_budget=1,
                admissibility_checker=IndependentPreservationChecker(),
            )

    def test_e5_rejects_forged_clean_snapshot_payload(self):
        class CleanSnapshotForgingAttack(ToyAttack):
            def run(self, *args, **kwargs):
                result = super().run(*args, **kwargs)
                forged = copy.deepcopy(result.clean_snapshot)
                forged.report = "Assessment: attacker-authored clean report."
                result.clean_snapshot = forged
                return result

        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        with self.assertRaisesRegex(ValueError, "does not match its E0 input"):
            run_e5(
                [self.eligible],
                lambda case, defense, verifier: ToyVictim({defense: verifier}),
                lambda seed, defense, verifier, case: CleanSnapshotForgingAttack(
                    seed, verifier=verifier
                ),
                verifiers,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
                calibration_cases=e5_calibration_cases(),
                seeds=(1,),
                query_budget=1,
                admissibility_checker=IndependentPreservationChecker(),
            )

    def test_e5_attack_facing_victim_hides_bound_verifier(self):
        class BoundVerifierTamperAttack(ToyAttack):
            def run(self, victim, *args, **kwargs):
                bound_verifier = victim.verifier_observer
                bound_verifier.threshold = 999.0
                return super().run(victim, *args, **kwargs)

        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        with self.assertRaisesRegex(AttributeError, "does not expose"):
            run_e5(
                [self.eligible],
                lambda case, defense, verifier: ToyVictim({defense: verifier}),
                lambda seed, defense, verifier, case: BoundVerifierTamperAttack(
                    seed, verifier=verifier
                ),
                verifiers,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
                calibration_cases=e5_calibration_cases(),
                seeds=(1,),
                query_budget=1,
                admissibility_checker=IndependentPreservationChecker(),
            )

    def test_e5_calibrates_real_v1_v2_v3_decision_interfaces(self):
        authenticator = HistoryAuthenticator(
            b"0123456789abcdef0123456789abcdef"
        )
        calibration = []
        for item in e5_calibration_cases():
            panel = measurements()
            prior = item.case.waveform_mv.copy()
            history = authenticator.seal(
                patient_id=item.case.patient_id,
                record_id=f"prior-{item.case.record_id}",
                timestamp=1.0,
                waveform_mv=prior,
                measurements=panel,
            )
            calibration.append(
                replace(
                    item,
                    verifier_contexts={
                        DefenseKind.V1: {},
                        DefenseKind.V2: {"measurements": panel},
                        DefenseKind.V3: {
                            "measurements": panel,
                            "history": history,
                            "prior_waveform_mv": prior,
                        },
                    },
                )
            )
        verifiers = {
            DefenseKind.V1: ShippedVerifier(),
            DefenseKind.V2: MeasurementAwareVerifier(),
            DefenseKind.V3: LongitudinalVerifier(authenticator),
        }
        result = run_e5(
            [],
            lambda case, defense, verifier: ToyVictim({defense: verifier}),
            lambda seed, defense, verifier, case: ToyAttack(seed),
            verifiers,
            target_coverage=0.5,
            constraint=SPhysConstraint(sample_rate_hz=100.0),
            calibration_cases=calibration,
        )

        self.assertEqual(
            result.summary["calibration_endpoint_classification"], "primary"
        )
        for defense, provenance in result.summary[
            "calibration_behavioral_provenance"
        ].items():
            self.assertEqual(defense, provenance["verifier_kind"])
            self.assertEqual(provenance["typed_release_decisions"], 8)
            self.assertTrue(provenance["score_stable_across_threshold_application"])

    def test_e5_attack_cannot_mutate_the_calibrated_verifier(self):
        class ThresholdMutatingAttack(ToyAttack):
            def __init__(self, seed, verifier):
                super().__init__(seed)
                self.verifier = verifier

            def run(self, *args, **kwargs):
                self.verifier.threshold = 999.0
                return super().run(*args, **kwargs)

        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        with self.assertRaisesRegex(AttributeError, "threshold is locked"):
            run_e5(
                [self.eligible],
                lambda case, defense, verifier: ToyVictim({defense: verifier}),
                lambda seed, defense, verifier, case: ThresholdMutatingAttack(
                    seed, verifier
                ),
                verifiers,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
                calibration_cases=e5_calibration_cases(),
                seeds=(1,),
                query_budget=1,
                admissibility_checker=IndependentPreservationChecker(),
            )

    def test_e5_rejects_verifier_decision_on_a_different_snapshot(self):
        class SnapshotSwappingVictim:
            def __init__(self, defense, verifier):
                self.defense = defense
                self.verifier = verifier

            def evaluate(self, signal, target_label, **kwargs):
                del target_label, kwargs
                benign = make_snapshot(signal, target_claim=False)
                decision = self.verifier.decide(benign)
                attacked = make_snapshot(signal, target_claim=True)
                attacked.released = decision.released
                attacked.defense_score = decision.score
                return attacked

        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        with self.assertRaisesRegex(
            ValueError,
            "snapshot other than the one verified",
        ):
            run_e5(
                [self.eligible],
                (
                    lambda case, defense, verifier: SnapshotSwappingVictim(
                        defense, verifier
                    )
                ),
                lambda seed, defense, verifier, case: ToyAttack(
                    seed, verifier=verifier
                ),
                verifiers,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
                calibration_cases=e5_calibration_cases(),
                seeds=(1,),
                query_budget=1,
                admissibility_checker=IndependentPreservationChecker(),
            )

    def test_e5_rejects_victim_authored_verifier_decision_metadata(self):
        class DecisionMetadataForgingVictim(ToyVictim):
            def evaluate(self, *args, **kwargs):
                snapshot = super().evaluate(*args, **kwargs)
                snapshot.metadata["defense_decision"] = {
                    "defense": DefenseKind.V1.value,
                    "released": True,
                    "score": -999.0,
                }
                return snapshot

        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        with self.assertRaisesRegex(ValueError, "typed verifier decision"):
            run_e5(
                [self.eligible],
                (
                    lambda case, defense, verifier: DecisionMetadataForgingVictim(
                        {defense: verifier}
                    )
                ),
                lambda seed, defense, verifier, case: ToyAttack(
                    seed, verifier=verifier
                ),
                verifiers,
                target_coverage=0.5,
                constraint=SPhysConstraint(sample_rate_hz=100.0),
                calibration_cases=e5_calibration_cases(),
                seeds=(1,),
                query_budget=1,
                admissibility_checker=IndependentPreservationChecker(),
            )

    def test_e5_calibration_context_cannot_override_shared_case_fields(self):
        item = e5_calibration_cases()[0]
        contexts = dict(item.verifier_contexts)
        contexts[DefenseKind.V2] = {
            "waveform_mv": np.ones_like(item.case.waveform_mv)
        }
        with self.assertRaisesRegex(ValueError, "override shared case fields"):
            replace(item, verifier_contexts=contexts)

    def test_e5_mismatched_recomputed_coverage_is_not_primary(self):
        class ExtraRejectVerifier(ThresholdVerifier):
            def decide(self, snapshot, **kwargs):
                decision = super().decide(snapshot, **kwargs)
                if self.kind is not DefenseKind.V3:
                    return decision
                return ReleaseDecision(
                    defense=self.kind,
                    released=False,
                    score=decision.score,
                    threshold=decision.threshold,
                    reasons=("test V3 clean rejection",),
                )

        verifiers = {
            kind: ExtraRejectVerifier(kind) for kind in DefenseKind
        }
        result = run_e5(
            [],
            lambda case, defense, verifier: ToyVictim({defense: verifier}),
            lambda seed, defense, verifier, case: ToyAttack(seed),
            verifiers,
            target_coverage=0.5,
            constraint=SPhysConstraint(sample_rate_hz=100.0),
            calibration_cases=e5_calibration_cases(),
        )

        validation = result.summary["calibration_coverage_validation"]
        self.assertFalse(validation["valid"])
        self.assertFalse(validation["counts_match_across_defenses"])
        self.assertEqual(
            result.summary["calibration_endpoint_classification"],
            "exploratory",
        )

    def test_e5_independent_preservation_boundary_controls_primary_endpoint(self):
        verifiers = {kind: ThresholdVerifier(kind) for kind in DefenseKind}
        common = {
            "eligible_cases": [self.eligible],
            "victim_provider": (
                lambda case, defense, verifier: ToyVictim({defense: verifier})
            ),
            "attack_factory": (
                lambda seed, defense, verifier, case: ToyAttack(
                    seed, verifier=verifier
                )
            ),
            "verifiers": verifiers,
            "target_coverage": 0.5,
            "constraint": SPhysConstraint(sample_rate_hz=100.0),
            "calibration_cases": e5_calibration_cases(),
            "seeds": (1,),
            "query_budget": 1,
        }
        missing = run_e5(**common)
        untyped = run_e5(
            **common,
            admissibility_checker=lambda *args, **kwargs: True,
        )
        nonpreserving_checker = IndependentPreservationChecker(admissible=False)
        nonpreserving = run_e5(
            **{
                **common,
                "attack_factory": (
                    lambda seed, defense, verifier, case: CheckerMutatingAttack(
                        seed, verifier=verifier
                    )
                ),
            },
            admissibility_checker=nonpreserving_checker,
        )

        for run in (missing, untyped):
            self.assertEqual(run.summary["endpoint_classification"], "not_evaluable")
            self.assertTrue(
                all(
                    not row.metrics["primary_endpoint_eligible"]
                    for row in run.observations
                )
            )
        self.assertEqual(
            nonpreserving.summary["endpoint_classification"], "not_evaluable"
        )
        self.assertFalse(nonpreserving_checker.admissible)
        self.assertTrue(
            all(
                not row.metrics["primary_endpoint_eligible"]
                and not row.success
                for row in nonpreserving.observations
            )
        )

    def test_mechanism_replays_do_not_promote_nonprimary_sources(self):
        exploratory = replace(self.eligible, primary_endpoint_eligible=False)
        e1b = run_e1b(
            [exploratory],
            {exploratory.case.key: self.attack},
            {"full": ToyVictim()},
        )
        e3 = run_e3(
            [(exploratory, self.attack)],
            lambda latent, **kwargs: make_snapshot(
                np.zeros((1, 8)), target_claim=False, latent=latent
            ),
            top_k=(1,),
            random_repeats=1,
        )
        e4 = run_e4(
            [(exploratory, self.attack)],
            lambda **kwargs: np.asarray([0.2, 0.0]),
            k=1,
        )

        self.assertIsNone(e1b.summary["full"]["rate"])
        self.assertTrue(all(not row.success for row in e3.observations))
        self.assertEqual(e3.summary["top_1"]["endpoint_classification"], "not_evaluable")
        self.assertIsNone(e4.summary["mean_cosine_similarity"])
        self.assertEqual(e4.summary["endpoint_classification"], "not_evaluable")

    def test_downstream_experiments_do_not_promote_self_attesting_results(self):
        direct = self.self_attested_attack
        e1b = run_e1b(
            [self.eligible],
            {self.case.key: direct},
            {"full": ToyVictim()},
        )
        e2 = run_e2(
            [(self.eligible, direct)],
            lambda **kwargs: {"released": True, "target_claim_present": True},
        )
        e3 = run_e3(
            [(self.eligible, direct)],
            lambda latent, **kwargs: make_snapshot(
                np.zeros((1, 8)), target_claim=False, latent=latent
            ),
            top_k=(1,),
            random_repeats=1,
        )
        e4 = run_e4(
            [(self.eligible, direct)],
            lambda **kwargs: np.asarray([0.2, 0.0]),
            k=1,
        )
        e6 = run_e6(
            [(self.eligible, direct)],
            ToyVictim(),
            successful_only=False,
            report_correctness_evaluator=independent_report_evaluator,
        )

        authenticator = HistoryAuthenticator(
            b"0123456789abcdef0123456789abcdef"
        )
        prior = np.zeros((1, 1000), dtype=np.float32)
        history = authenticator.seal(
            patient_id="p1",
            record_id="prior",
            timestamp=1.0,
            waveform_mv=prior,
            measurements=measurements(),
        )
        e7 = run_e7(
            [(self.eligible, direct)],
            LongitudinalVerifier(
                authenticator,
                measurement_verifier=MeasurementAwareVerifier(),
            ),
            lambda case: {
                "history": history,
                "measurements": measurements(),
                "current_timestamp": 2.0,
            },
            pre_defense_replay=lambda source_snapshot, **kwargs: source_snapshot,
        )

        self.assertFalse(e1b.observations[0].metrics["primary_endpoint_eligible"])
        self.assertEqual(e2.observations, ())
        self.assertTrue(
            all(not row.metrics["primary_endpoint_eligible"] for row in e3.observations)
        )
        self.assertEqual(e4.summary["endpoint_classification"], "not_evaluable")
        self.assertTrue(
            all(not row.metrics["primary_endpoint_eligible"] for row in e6.observations)
        )
        self.assertTrue(
            all(not row.metrics["primary_endpoint_eligible"] for row in e7.observations)
        )

    def test_downstream_receipt_detects_snapshot_and_waveform_tampering(self):
        def report_tamper(result):
            result.adversarial_snapshot.report = "forged report"

        def probability_tamper(result):
            result.adversarial_snapshot.probabilities["target"] = 0.01

        def score_tamper(result):
            result.adversarial_snapshot.defense_score = 999.0

        def clean_tamper(result):
            result.clean_snapshot.report = "forged clean report"

        def waveform_tamper(result):
            result.adversarial_signal[0, 0] += np.float32(0.01)

        for tamper in (
            report_tamper,
            probability_tamper,
            score_tamper,
            clean_tamper,
            waveform_tamper,
        ):
            with self.subTest(tamper=tamper.__name__):
                result = copy.deepcopy(self.attack)
                tamper(result)
                replay = run_e1b(
                    [self.eligible],
                    {self.case.key: result},
                    {"full": ToyVictim()},
                )
                row = replay.observations[0]
                self.assertFalse(row.metrics["primary_endpoint_eligible"])
                self.assertFalse(row.success)
                self.assertFalse(row.metrics["source_preservation_evidence_valid"])

    def test_append_only_log_hashes_json_chain_and_npz_waveforms(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = AppendOnlySecurityLogger(Path(directory) / "security.jsonl")
            receipts = logger.append_attack_result(
                self.attack,
                patient_id="p1",
                record_id="r1",
                target_id="T",
                experiment_id="E1",
            )
            logger.append("summary", {"complete": True})
            verification = logger.verify()
            waveform = logger.load_waveform(receipts[0].waveform_path)

            self.assertTrue(verification.valid)
            self.assertEqual(verification.records, 3)
            np.testing.assert_array_equal(waveform, self.attack.attempts[0].waveform)
            artifact = Path(directory) / receipts[0].waveform_path
            with artifact.open("ab") as handle:
                handle.write(b"tamper")
            self.assertFalse(logger.verify().valid)

    def test_log_manifest_detects_suffix_truncation_and_missing_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "security.jsonl"
            logger = AppendOnlySecurityLogger(path)
            self.assertFalse(logger.verify().valid)

            logger.append("first", {"value": 1})
            logger.append("second", {"value": 2})
            manifest = logger.write_terminal_manifest()
            verification = logger.verify(require_manifest=True)
            self.assertTrue(verification.valid)
            self.assertTrue(verification.complete)
            self.assertEqual(verification.expected_records, 2)
            self.assertEqual(
                verification.terminal_record_sha256,
                manifest["terminal_record_sha256"],
            )

            path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n")
            truncated = logger.verify(require_manifest=True)
            self.assertFalse(truncated.valid)
            self.assertFalse(truncated.complete)

    def test_log_verifier_rejects_uncommitted_attack_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = AppendOnlySecurityLogger(Path(directory) / "security.jsonl")
            logger.append(
                "attack_attempt",
                {
                    "experiment_id": "E1",
                    "patient_id": "p1",
                    "record_id": "r1",
                    "target_id": "T",
                    "target_label": "target",
                    "family": "f1",
                    "query_index": 1,
                    "success": False,
                },
                waveform_mv=np.zeros((1, 8), dtype=np.float32),
            )
            verification = logger.verify()
            self.assertFalse(verification.valid)
            self.assertTrue(
                any("uncommitted attack attempts" in error for error in verification.errors)
            )


if __name__ == "__main__":
    unittest.main()
