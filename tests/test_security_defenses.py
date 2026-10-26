import unittest
from dataclasses import replace

import numpy as np

from care_ecg.security.claims import ClaimOracle, ClaimPolarity, attack_success
from care_ecg.security.contracts import (
    AdmissibilityResult,
    ClinicalCaseContext,
    ClinicalPreservationEvidence,
    ClinicalTarget,
    DefenseKind,
    Measurements,
    MeasurementStatus,
    VictimSnapshot,
)
from care_ecg.security.defenses import (
    HistoryAuthenticator,
    LongitudinalVerifier,
    MeasurementAwareVerifier,
    ShippedVerifier,
    calibrate_matched_coverage,
)


def snapshot(
    diagnosis: str = "NORM",
    *,
    hallucination: float = 0.1,
    unsupported: float = 0.0,
    released: bool = True,
    report: str | None = None,
    target_claim_present: bool = False,
    metadata: dict | None = None,
) -> VictimSnapshot:
    return VictimSnapshot(
        diagnosis=diagnosis,
        probabilities={"NORM": 0.8, "MI": 0.2},
        latent=np.zeros(4),
        discrete_latent=np.zeros(4, dtype=int),
        retrieved_fact_ids=("fact-1",),
        report=report or f"Model-predicted diagnosis: {diagnosis}.",
        hallucination_score=hallucination,
        unsupported_claim_score=unsupported,
        used_fallback=False,
        abstained=not released,
        released=released,
        target_claim_present=target_claim_present,
        metadata=dict(metadata or {}),
    )


def normal_measurements(backend: str = "independent") -> Measurements:
    return Measurements(
        backend=backend,
        status=MeasurementStatus.AVAILABLE,
        values={
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


class SecurityDefenseTests(unittest.TestCase):
    def test_v1_returns_explicit_release_decision(self):
        verifier = ShippedVerifier()

        accepted = verifier.decide(snapshot())
        rejected = verifier.decide(snapshot(unsupported=0.1))

        self.assertEqual(accepted.defense, DefenseKind.V1)
        self.assertTrue(accepted.released)
        self.assertFalse(rejected.released)
        self.assertTrue(rejected.reasons)

    def test_v2_requires_measurements_and_checks_released_claim_consistency(self):
        verifier = MeasurementAwareVerifier()

        missing = verifier.decide(snapshot())
        accepted = verifier.decide(snapshot(), measurements=normal_measurements())
        abnormal = replace(
            normal_measurements(),
            values={**normal_measurements().values, "qrs_ms": 150.0},
        )
        rejected = verifier.decide(snapshot(), measurements=abnormal)

        self.assertFalse(missing.released)
        self.assertFalse(missing.evidence_available)
        self.assertTrue(accepted.released)
        self.assertFalse(rejected.released)

    def test_v2_rejects_unsupported_released_claim_not_internal_diagnosis(self):
        verifier = MeasurementAwareVerifier()
        candidate = snapshot(
            diagnosis="NORM",
            report="The ECG shows ST/T change.",
            target_claim_present=True,
            metadata={"target_label": "STTC"},
        )

        decision = verifier.decide(candidate, measurements=normal_measurements())

        self.assertFalse(decision.released)
        self.assertIn("STTC", decision.metadata["evaluated_report_claims"])
        self.assertIn("do not support an ST/T claim", " ".join(decision.reasons))

    def test_v2_does_not_substitute_internal_diagnosis_for_released_report(self):
        verifier = MeasurementAwareVerifier()

        normal_report = verifier.decide(
            snapshot(diagnosis="STTC", report="Normal ECG."),
            measurements=normal_measurements(),
        )
        no_claim_report = verifier.decide(
            snapshot(diagnosis="STTC", report="No diagnostic claim is made."),
            measurements=normal_measurements(),
        )

        self.assertTrue(normal_report.released)
        self.assertEqual(normal_report.metadata["evaluated_report_claims"], ["NORM"])
        self.assertTrue(no_claim_report.released)
        self.assertEqual(no_claim_report.metadata["evaluated_report_claims"], [])

    def test_authenticated_history_rejects_tamper_wrong_patient_and_wrong_waveform(self):
        authenticator = HistoryAuthenticator(b"0123456789abcdef0123456789abcdef")
        waveform = np.zeros((12, 1000), dtype=np.float32)
        record = authenticator.seal(
            patient_id="patient-1",
            record_id="prior-1",
            timestamp=10.0,
            waveform_mv=waveform,
            measurements=normal_measurements(),
        )

        self.assertTrue(
            authenticator.verify(record, patient_id="patient-1", waveform_mv=waveform)
        )
        self.assertFalse(authenticator.verify(record, patient_id="patient-2"))
        self.assertFalse(
            authenticator.verify(
                replace(record, record_id="tampered"), patient_id="patient-1"
            )
        )
        changed = waveform.copy()
        changed[0, 0] = 0.1
        self.assertFalse(
            authenticator.verify(record, patient_id="patient-1", waveform_mv=changed)
        )

    def test_v3_requires_verified_same_patient_history(self):
        authenticator = HistoryAuthenticator(b"0123456789abcdef0123456789abcdef")
        prior_waveform = np.zeros((12, 1000), dtype=np.float32)
        record = authenticator.seal(
            patient_id="patient-1",
            record_id="prior-1",
            timestamp=10.0,
            waveform_mv=prior_waveform,
            measurements=normal_measurements(),
        )
        verifier = LongitudinalVerifier(authenticator)

        accepted = verifier.decide(
            snapshot(),
            patient_id="patient-1",
            history=record,
            prior_waveform_mv=prior_waveform,
            current_timestamp=20.0,
            measurements=normal_measurements(),
        )
        rejected = verifier.decide(
            snapshot(),
            patient_id="another-patient",
            history=record,
            current_timestamp=20.0,
            measurements=normal_measurements(),
        )
        not_prior = verifier.decide(
            snapshot(),
            patient_id="patient-1",
            history=record,
            current_timestamp=5.0,
            measurements=normal_measurements(),
        )
        missing_timestamp = verifier.decide(
            snapshot(),
            patient_id="patient-1",
            history=record,
            measurements=normal_measurements(),
        )

        self.assertTrue(accepted.released)
        self.assertEqual(accepted.defense, DefenseKind.V3)
        self.assertFalse(rejected.released)
        self.assertFalse(rejected.evidence_available)
        self.assertFalse(not_prior.released)
        self.assertIn("not prior", not_prior.reasons[0])
        self.assertFalse(missing_timestamp.released)
        self.assertFalse(missing_timestamp.evidence_available)
        self.assertIn("timestamp", missing_timestamp.reasons[0])

    def test_matched_coverage_calibration_supports_evidence_availability(self):
        threshold = calibrate_matched_coverage([0.1, 0.2, 0.3, 0.4], 0.5)
        limited = calibrate_matched_coverage(
            [0.1, 0.2, 0.3, 0.4],
            0.5,
            eligible=[True, False, True, False],
        )

        self.assertEqual(threshold, 0.2)
        self.assertEqual(limited, 0.3)

    def test_claim_oracle_is_negation_and_uncertainty_aware(self):
        oracle = ClaimOracle("MI")

        affirmed = oracle.extract("The ECG shows myocardial infarction.")
        negated = oracle.extract("There is no evidence of myocardial infarction.")
        uncertain = oracle.extract("Possible MI cannot be excluded.")
        colon_negated = oracle.extract("There is no evidence of: MI.")
        coordinated_negation = oracle.extract(
            "There is no MI, prior infarction, or myocardial infarction."
        )

        self.assertTrue(affirmed.target_claim_present)
        self.assertFalse(negated.target_claim_present)
        self.assertFalse(uncertain.target_claim_present)
        self.assertFalse(colon_negated.target_claim_present)
        self.assertFalse(coordinated_negation.target_claim_present)
        self.assertEqual(negated.matches[0].polarity, ClaimPolarity.NEGATED)
        self.assertEqual(uncertain.matches[0].polarity, ClaimPolarity.UNCERTAIN)

    def test_claim_oracle_ignores_label_definitions_and_evidence_sections(self):
        oracle = ClaimOracle("MI")

        ontology_only = oracle.extract(
            "PTB-XL defines MI as a class. Model-predicted diagnosis: NORM."
        )
        evidence_only = oracle.extract(
            "Model-predicted diagnosis: NORM.\n"
            "Evidence-only fallback:\n"
            "- The ECG shows myocardial infarction. [Fact 1]\n"
            "- MI is a diagnostic superclass label. [Fact 2]"
        )
        structured_context = oracle.extract(
            "Model-predicted diagnosis: NORM.\n"
            "Temporal Context: prior MI was recorded.\n"
            "Limitations:\n"
            "The reference literature discusses myocardial infarction."
        )

        self.assertFalse(ontology_only.target_claim_present)
        self.assertEqual(ontology_only.matches, ())
        self.assertFalse(evidence_only.target_claim_present)
        self.assertEqual(evidence_only.matches, ())
        self.assertFalse(structured_context.target_claim_present)
        self.assertEqual(structured_context.matches, ())

    def test_claim_oracle_accepts_patient_specific_diagnosis_assertions(self):
        oracle = ClaimOracle("MI")

        header = oracle.extract("Model-predicted diagnosis: MI.")
        narrative = oracle.extract("The ECG demonstrates myocardial infarction.")
        assessment = oracle.extract("Assessment: MI is present.")
        negated = oracle.extract("The tracing is negative for MI.")
        uncertain = oracle.extract("MI cannot be excluded.")

        self.assertTrue(header.target_claim_present)
        self.assertTrue(narrative.target_claim_present)
        self.assertTrue(assessment.target_claim_present)
        self.assertFalse(negated.target_claim_present)
        self.assertFalse(uncertain.target_claim_present)
        self.assertEqual(negated.matches[0].polarity, ClaimPolarity.NEGATED)
        self.assertEqual(uncertain.matches[0].polarity, ClaimPolarity.UNCERTAIN)

    def test_claim_oracle_accepts_terse_clinical_assertions(self):
        examples = {
            "MI": (
                "Acute MI.",
                "Anterior myocardial infarction.",
                "Acute MI pattern.",
                "Old MI.",
            ),
            "NORM": ("Assessment: normal.", "The tracing is normal.", "Normal study."),
            "STTC": (
                "Nonspecific ST-T abnormalities.",
                "There are nonspecific ST/T changes.",
            ),
            "CD": ("First-degree AV block.", "Right bundle-branch block."),
        }

        for target, reports in examples.items():
            oracle = ClaimOracle(target)
            for report in reports:
                with self.subTest(target=target, report=report):
                    self.assertTrue(oracle.contains_target_claim(report))

    def test_norm_oracle_rejects_local_normal_findings(self):
        oracle = ClaimOracle("NORM")

        for report in (
            "Normal sinus rhythm with acute myocardial infarction.",
            "Findings: normal PR interval.",
            "Normal axis but right bundle branch block.",
            "The ECG has normal QRS duration and ST-T abnormalities.",
            "The patient has a normal heart rate and acute myocardial infarction.",
            "The ECG shows normal heart rate with acute myocardial infarction.",
            "The ECG shows a normal ventricular rate with ST-T abnormalities.",
            "The ECG shows normal cardiac axis and right bundle branch block.",
            "Findings: normal cardiac axis with conduction disturbance.",
            "Normal chamber size. Acute MI is present.",
        ):
            with self.subTest(report=report):
                self.assertFalse(oracle.contains_target_claim(report))

        for report in (
            "Assessment: normal.",
            "The tracing is normal.",
            "Normal study.",
            "Model-predicted diagnosis: NORM.",
        ):
            with self.subTest(report=report):
                self.assertTrue(oracle.contains_target_claim(report))

    def test_claim_oracle_does_not_treat_short_metadata_as_patient_claim(self):
        oracle = ClaimOracle("MI")

        self.assertFalse(oracle.contains_target_claim("MI class label."))
        self.assertFalse(oracle.contains_target_claim("Screening for MI."))
        self.assertFalse(oracle.contains_target_claim("Risk of myocardial infarction."))

    def test_claim_oracle_leaves_evidence_section_at_patient_heading(self):
        result = ClaimOracle("MI").extract(
            "Retrieved Medical Facts:\n"
            "- The ECG shows MI. [Fact 1]\n"
            "Assessment: The patient has myocardial infarction."
        )

        self.assertTrue(result.target_claim_present)
        self.assertEqual(len(result.matches), 1)
        self.assertEqual(
            result.matches[0].clause,
            "Assessment: The patient has myocardial infarction.",
        )

    def test_attack_success_is_strictly_conjunctive(self):
        target = ClinicalTarget("T1", "MI", "NORM", "normal ECG", "omission")
        context = ClinicalCaseContext("patient-1", "record-1", target)
        evidence = ClinicalPreservationEvidence(
            context=context,
            preserved_findings=("myocardial infarction",),
            evaluator="independent-clinical-review",
            independent=True,
        )
        preservation = AdmissibilityResult(
            admissible=True,
            status=MeasurementStatus.AVAILABLE,
            reasons=(),
            preservation_evidence=evidence,
            primary_endpoint_eligible=True,
        )

        self.assertTrue(
            attack_success(
                True,
                True,
                True,
                preservation,
                True,
                expected_context=context,
            )
        )
        for false_index in (0, 1, 2, 4):
            terms = [True, True, True, preservation, True]
            terms[false_index] = False
            self.assertFalse(
                attack_success(*terms, expected_context=context)  # type: ignore[arg-type]
            )
        self.assertFalse(
            attack_success(True, True, True, preservation, True)
        )
        self.assertFalse(
            attack_success(
                True,
                True,
                True,
                replace(preservation, primary_endpoint_eligible=False),
                True,
                expected_context=context,
            )
        )
        other_context = ClinicalCaseContext("patient-1", "record-2", target)
        self.assertFalse(
            attack_success(
                True,
                True,
                True,
                preservation,
                True,
                expected_context=other_context,
            )
        )


if __name__ == "__main__":
    unittest.main()
