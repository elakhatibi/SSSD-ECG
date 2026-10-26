from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any

import numpy as np


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class AttackFamily(_StringEnum):
    F0 = "f0"
    RANDOM = "f0"
    F1 = "f1"
    DIAGNOSIS_WHITE_BOX = "f1"
    F1_CF = "f1_cf"
    COUNTERFACTUAL_GUIDED = "f1_cf"
    F2 = "f2"
    DIAGNOSIS_BLACK_BOX = "f2"
    F3 = "f3"
    VERIFIER_AWARE = "f3"
    F3_DEFENSE = "f3_defense"
    DEFENSE_ADAPTIVE = "f3_defense"


class KnowledgeLevel(_StringEnum):
    K1 = "k1"
    BLACK_BOX = "k1"
    K2 = "k2"
    GREY_BOX = "k2"
    K3 = "k3"
    VERIFIER_AWARE_WHITE_BOX = "k3"


class ConstraintKind(_StringEnum):
    S_INF = "s_inf"
    SINF = "s_inf"
    L_INF = "s_inf"
    S2 = "s2"
    L2 = "s2"
    S_SPEC = "s_spec"
    SSPEC = "s_spec"
    SPECTRAL = "s_spec"
    S_PHYS = "s_phys"
    SPHYS = "s_phys"
    PHYSIOLOGICAL = "s_phys"


class DefenseKind(_StringEnum):
    V1 = "v1"
    SHIPPED = "v1"
    V2 = "v2"
    MEASUREMENT_AWARE = "v2"
    V3 = "v3"
    LONGITUDINAL = "v3"


class MeasurementStatus(_StringEnum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
            if item.metadata.get("serialize", True)
        }
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


class SerializableContract:
    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True, slots=True)
class ClinicalTarget(SerializableContract):
    target_id: str
    clean_label: str
    target_label: str
    false_claim: str
    error_type: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.target_id,
                self.clean_label,
                self.target_label,
                self.false_claim,
                self.error_type,
            )
        ):
            raise ValueError("Clinical target fields must not be empty")

    @property
    def claim(self) -> str:
        return self.false_claim


@dataclass(frozen=True, slots=True)
class ClinicalCaseContext(SerializableContract):
    """Patient-record-target identity supplied to independent endpoint evaluators."""

    patient_id: str
    record_id: str
    target: ClinicalTarget

    def __post_init__(self) -> None:
        if not self.patient_id.strip() or not self.record_id.strip():
            raise ValueError("Clinical context requires patient and record identifiers")
        if not isinstance(self.target, ClinicalTarget):
            raise TypeError("Clinical context target must be a ClinicalTarget")


@dataclass(frozen=True, slots=True)
class ClinicalPreservationEvidence(SerializableContract):
    """Target-specific independent evidence that baseline findings were preserved."""

    context: ClinicalCaseContext
    preserved_findings: tuple[str, ...]
    evaluator: str
    independent: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        findings = tuple(str(value).strip() for value in self.preserved_findings)
        if not findings or any(not value for value in findings):
            raise ValueError("Preservation evidence requires explicit preserved findings")
        if not self.evaluator.strip():
            raise ValueError("Preservation evidence requires an evaluator identifier")
        object.__setattr__(self, "preserved_findings", findings)


@dataclass(frozen=True, slots=True)
class PreservationVerificationReceipt(SerializableContract):
    """Authenticated in-process receipt for orchestrator-owned preservation review."""

    schema_version: str
    experiment_id: str
    context: ClinicalCaseContext
    attack_family: AttackFamily
    target_label: str
    query_count: int
    clean_waveform_sha256: str
    clean_snapshot_sha256: str
    query_waveform_sha256s: tuple[str, ...]
    query_snapshot_sha256s: tuple[str, ...]
    primary_endpoint_eligible: bool
    primary_success: bool
    primary_success_query_index: int | None
    selected_waveform_sha256: str | None
    selected_snapshot_sha256: str | None
    review_payload_sha256: str
    payload_sha256: str
    authenticator: str = field(repr=False, metadata={"serialize": False})

    def __post_init__(self) -> None:
        if self.schema_version != "care-ecg-preservation-receipt.v1":
            raise ValueError("Unsupported preservation receipt schema")
        if self.experiment_id not in {"E1", "E5"}:
            raise ValueError("Preservation receipts may only be issued by E1 or E5")
        if not isinstance(self.context, ClinicalCaseContext):
            raise TypeError("Preservation receipt context must be typed")
        object.__setattr__(self, "attack_family", AttackFamily(self.attack_family))
        if not self.target_label.strip() or self.query_count < 1:
            raise ValueError("Preservation receipt target and query count are required")
        if len(self.query_waveform_sha256s) != self.query_count or len(
            self.query_snapshot_sha256s
        ) != self.query_count:
            raise ValueError("Preservation receipt query bindings must be complete")
        digests = (
            self.clean_waveform_sha256,
            self.clean_snapshot_sha256,
            *self.query_waveform_sha256s,
            *self.query_snapshot_sha256s,
            self.review_payload_sha256,
            self.payload_sha256,
            *(
                ()
                if self.selected_waveform_sha256 is None
                else (self.selected_waveform_sha256,)
            ),
            *(
                ()
                if self.selected_snapshot_sha256 is None
                else (self.selected_snapshot_sha256,)
            ),
        )
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        ):
            raise ValueError("Preservation receipt digests must be lowercase SHA-256")
        if self.primary_success:
            if not self.primary_endpoint_eligible:
                raise ValueError("A primary success requires an eligible endpoint")
            if self.primary_success_query_index is None:
                raise ValueError("A primary success requires its selected query")
            if self.selected_waveform_sha256 is None or self.selected_snapshot_sha256 is None:
                raise ValueError("A primary success requires selected artifact bindings")
        elif any(
            value is not None
            for value in (
                self.primary_success_query_index,
                self.selected_waveform_sha256,
                self.selected_snapshot_sha256,
            )
        ):
            raise ValueError("A non-success receipt cannot select success artifacts")
        if self.primary_success_query_index is not None and not (
            1 <= self.primary_success_query_index <= self.query_count
        ):
            raise ValueError("Preservation receipt selected query is out of range")
        if self.primary_success_query_index is not None:
            selected_index = self.primary_success_query_index - 1
            if (
                self.selected_waveform_sha256
                != self.query_waveform_sha256s[selected_index]
                or self.selected_snapshot_sha256
                != self.query_snapshot_sha256s[selected_index]
            ):
                raise ValueError(
                    "Preservation receipt selected artifacts must bind its query"
                )
        if not self.authenticator:
            raise ValueError("Preservation receipt requires an authenticator")


@dataclass(frozen=True, slots=True)
class ReportCorrectnessResult(SerializableContract):
    """Independent, all-claims clean-report correctness assessment for E0/E6."""

    correct: bool
    context: ClinicalCaseContext
    evaluator: str
    independent: bool
    all_claims_evaluated: bool
    evaluated_claims: tuple[str, ...]
    reference_standard: str
    errors: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.context, ClinicalCaseContext):
            raise TypeError("Report correctness context must be a ClinicalCaseContext")
        claims = tuple(str(value).strip() for value in self.evaluated_claims)
        errors = tuple(str(value).strip() for value in self.errors)
        if any(not value for value in (*claims, *errors)):
            raise ValueError("Report correctness claims and errors must not be blank")
        if not self.evaluator.strip():
            raise ValueError("Report correctness requires an evaluator identifier")
        if not self.reference_standard.strip():
            raise ValueError("Report correctness requires a reference standard")
        if self.all_claims_evaluated and not claims:
            raise ValueError("All-claims evaluation requires at least one evaluated claim")
        if self.correct and errors:
            raise ValueError("A correct report cannot carry report errors")
        object.__setattr__(self, "evaluated_claims", claims)
        object.__setattr__(self, "errors", errors)

    @property
    def primary_endpoint_eligible(self) -> bool:
        return bool(
            self.independent
            and self.all_claims_evaluated
            and self.evaluated_claims
            and self.reference_standard.strip()
        )


CLINICAL_TARGETS: dict[str, ClinicalTarget] = {
    "T1": ClinicalTarget("T1", "MI", "NORM", "normal ECG", "omission"),
    "T2": ClinicalTarget("T2", "CD", "NORM", "normal ECG", "omission"),
    "T3": ClinicalTarget(
        "T3", "NORM", "MI", "myocardial infarction", "fabrication"
    ),
    "T4": ClinicalTarget("T4", "NORM", "STTC", "ST/T change", "fabrication"),
}


@dataclass(frozen=True, slots=True)
class AttackConfig(SerializableContract):
    family: AttackFamily
    knowledge_level: KnowledgeLevel
    constraint_kind: ConstraintKind
    query_budget: int
    target: ClinicalTarget | None = None
    seed: int = 42
    steps: int = 40
    restarts: int = 1
    step_size_mv: float = 0.001
    diagnosis_weight: float = 1.0
    retrieval_weight: float = 0.0
    verifier_weight: float = 0.0
    defense_weight: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", AttackFamily(self.family))
        object.__setattr__(self, "knowledge_level", KnowledgeLevel(self.knowledge_level))
        object.__setattr__(self, "constraint_kind", ConstraintKind(self.constraint_kind))
        if self.query_budget < 1 or self.steps < 1 or self.restarts < 1:
            raise ValueError("Attack budget, steps, and restarts must be positive")
        numeric = (
            self.step_size_mv,
            self.diagnosis_weight,
            self.retrieval_weight,
            self.verifier_weight,
            self.defense_weight,
        )
        if any(not np.isfinite(value) or value < 0 for value in numeric):
            raise ValueError("Attack scales and objective weights must be finite and non-negative")

    @property
    def budget(self) -> int:
        return self.query_budget


@dataclass(slots=True)
class VictimSnapshot(SerializableContract):
    diagnosis: str
    probabilities: dict[str, float]
    latent: np.ndarray
    discrete_latent: np.ndarray
    retrieved_fact_ids: tuple[str, ...]
    report: str
    hallucination_score: float
    unsupported_claim_score: float
    used_fallback: bool
    abstained: bool
    released: bool
    target_claim_present: bool = False
    defense_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.latent = np.asarray(self.latent, dtype=float).reshape(-1)
        self.discrete_latent = np.asarray(self.discrete_latent, dtype=int).reshape(-1)
        self.retrieved_fact_ids = tuple(str(value) for value in self.retrieved_fact_ids)
        if not self.diagnosis.strip() or not self.report.strip():
            raise ValueError("Victim diagnosis and report must not be empty")
        if not self.probabilities or any(
            not np.isfinite(value) or value < 0 for value in self.probabilities.values()
        ):
            raise ValueError("Victim probabilities must be a non-empty finite mapping")
        if not np.isfinite(self.hallucination_score) or not np.isfinite(
            self.unsupported_claim_score
        ):
            raise ValueError("Verifier scores must be finite")


@dataclass(slots=True)
class AttackAttempt(SerializableContract):
    query_index: int
    waveform: np.ndarray
    objective: float
    target_probability: float
    diagnosis: str
    released: bool
    target_claim_present: bool
    admissible: bool
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.waveform = np.asarray(self.waveform, dtype=np.float32)
        if self.query_index < 1:
            raise ValueError("query_index is one-based and must be positive")
        if self.waveform.ndim != 2 or not np.isfinite(self.waveform).all():
            raise ValueError("Attack waveform must be a finite [leads, samples] array")
        if not np.isfinite(self.objective) or not 0 <= self.target_probability <= 1:
            raise ValueError("Attack objective and target probability are invalid")
        expected = bool(
            self.released and self.target_claim_present and self.admissible
        )
        if self.success != expected:
            raise ValueError("Attempt success must satisfy release, claim, and admissibility")


# Each complete-system submission is one auditable query trace.
QueryTrace = AttackAttempt


@dataclass(slots=True)
class AttackResult(SerializableContract):
    family: AttackFamily | str
    target_label: str
    clean_snapshot: Any
    adversarial_snapshot: Any | None
    adversarial_signal: np.ndarray | None
    attempts: list[AttackAttempt]
    success: bool
    queries_used: int
    stopped_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    verification_receipt: PreservationVerificationReceipt | None = field(
        default=None,
        repr=False,
        metadata={"serialize": False},
    )

    def __post_init__(self) -> None:
        self.family = AttackFamily(self.family)
        self.attempts = list(self.attempts)
        if self.verification_receipt is not None and not isinstance(
            self.verification_receipt, PreservationVerificationReceipt
        ):
            raise TypeError("Attack verification receipt must use the typed contract")
        if self.adversarial_signal is not None:
            self.adversarial_signal = np.asarray(self.adversarial_signal, dtype=np.float32)
            if self.adversarial_signal.ndim != 2 or not np.isfinite(
                self.adversarial_signal
            ).all():
                raise ValueError(
                    "Adversarial signal must be a finite [leads, samples] array"
                )
        if self.queries_used < 0 or self.queries_used != len(self.attempts):
            raise ValueError("queries_used must equal the number of complete-system attempts")
        indices = [attempt.query_index for attempt in self.attempts]
        if indices != list(range(1, len(self.attempts) + 1)):
            raise ValueError("Attack query indices must be unique, contiguous, and one-based")
        successful = [index for index, attempt in enumerate(self.attempts) if attempt.success]
        expected_success = bool(successful)
        if self.success != expected_success:
            raise ValueError("Result success must equal the presence of a successful attempt")
        if successful and successful != [len(self.attempts) - 1]:
            raise ValueError("An attack must stop immediately after its first successful attempt")
        artifacts_present = (
            self.adversarial_snapshot is not None,
            self.adversarial_signal is not None,
        )
        if artifacts_present[0] != artifacts_present[1]:
            raise ValueError("Adversarial snapshot and signal must be provided together")
        if artifacts_present[0] and not self.attempts:
            raise ValueError("Adversarial artifacts require at least one submitted query")
        if self.adversarial_signal is not None and not any(
            np.array_equal(self.adversarial_signal, attempt.waveform)
            for attempt in self.attempts
        ):
            raise ValueError("Adversarial signal must correspond to a logged query waveform")
        if self.success:
            if self.adversarial_snapshot is None or self.adversarial_signal is None:
                raise ValueError("Successful results require adversarial snapshot and signal artifacts")
            final = self.attempts[-1]
            if not np.array_equal(self.adversarial_signal, final.waveform):
                raise ValueError("Successful artifacts must correspond to the first-success query")
            if not isinstance(self.adversarial_snapshot, VictimSnapshot):
                raise TypeError("Successful results require a VictimSnapshot artifact")
            if not (
                self.adversarial_snapshot.released
                and self.adversarial_snapshot.target_claim_present
            ):
                raise ValueError("Successful snapshot must contain a released target claim")
            if self.adversarial_snapshot.diagnosis != final.diagnosis:
                raise ValueError("Successful snapshot and attempt diagnosis must agree")
        selected_query = self.metadata.get("selected_attempt_query_index")
        selected_attempt = None
        if selected_query is not None:
            if isinstance(selected_query, (bool, np.bool_)) or not isinstance(
                selected_query, (int, np.integer)
            ):
                raise TypeError("Selected attack query index must be an integer or None")
            matches = [
                attempt
                for attempt in self.attempts
                if attempt.query_index == int(selected_query)
            ]
            if len(matches) != 1 or self.adversarial_signal is None:
                raise ValueError("Selected attack query must identify the result artifacts")
            selected_attempt = matches[0]
            if not np.array_equal(selected_attempt.waveform, self.adversarial_signal):
                raise ValueError("Selected attack query and adversarial signal must agree")
        if "selected_attempt_primary_endpoint_eligible" in self.metadata:
            selected_eligible = self.metadata[
                "selected_attempt_primary_endpoint_eligible"
            ]
            if selected_attempt is None:
                if selected_eligible is not None:
                    raise ValueError(
                        "Results without selected artifacts cannot carry selected evidence"
                    )
            else:
                if not isinstance(selected_eligible, (bool, np.bool_)):
                    raise TypeError("Selected-attempt eligibility must be a boolean")
                if bool(selected_eligible) != bool(
                    selected_attempt.metadata.get(
                        "primary_endpoint_eligible", False
                    )
                ):
                    raise ValueError(
                        "Selected-attempt eligibility must match its evidence trace"
                    )
        if "complete_system_queries_consumed" in self.metadata:
            consumed = self.metadata["complete_system_queries_consumed"]
            if isinstance(consumed, (bool, np.bool_)) or not isinstance(
                consumed, (int, np.integer)
            ):
                raise TypeError("Complete-system query accounting must be an integer")
            if int(consumed) != self.queries_used:
                raise ValueError("Complete-system query accounting must equal queries_used")
        if "successful_attempt_primary_eligible" in self.metadata:
            successful_eligible = self.metadata[
                "successful_attempt_primary_eligible"
            ]
            if not self.success and successful_eligible is not None:
                raise ValueError(
                    "Unsuccessful results cannot identify successful-attempt evidence"
                )
            if not self.success:
                result_eligible = self.metadata.get(
                    "primary_endpoint_eligible", False
                )
                if not isinstance(result_eligible, (bool, np.bool_)):
                    raise TypeError("Primary endpoint eligibility must be a boolean")
                if not bool(result_eligible):
                    raise ValueError(
                        "Unsuccessful attacks remain eligible denominator observations"
                    )
            if self.success:
                if not isinstance(successful_eligible, (bool, np.bool_)):
                    raise TypeError(
                        "Successful-attempt eligibility must be a boolean or None"
                    )
                attempt_eligible = self.attempts[-1].metadata.get(
                    "primary_endpoint_eligible", False
                )
                if not isinstance(attempt_eligible, (bool, np.bool_)):
                    raise TypeError("Attempt endpoint eligibility must be a boolean")
                expected_eligible = bool(attempt_eligible)
                if bool(successful_eligible) != expected_eligible:
                    raise ValueError(
                        "Result eligibility must follow the selected successful attempt"
                    )
                result_eligible = self.metadata.get(
                    "primary_endpoint_eligible", False
                )
                if not isinstance(result_eligible, (bool, np.bool_)):
                    raise TypeError("Primary endpoint eligibility must be a boolean")
                if bool(result_eligible) != expected_eligible:
                    raise ValueError(
                        "Primary endpoint eligibility must follow successful-attempt evidence"
                    )
        if not self.target_label.strip() or not self.stopped_reason.strip():
            raise ValueError("Attack result labels and stopping reason must not be empty")


@dataclass(frozen=True, slots=True)
class Measurements(SerializableContract):
    backend: str
    status: MeasurementStatus
    values: dict[str, float] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", MeasurementStatus(self.status))
        clean_values = {str(key): float(value) for key, value in self.values.items()}
        if any(not np.isfinite(value) for value in clean_values.values()):
            raise ValueError("Measurement values must be finite")
        object.__setattr__(self, "values", clean_values)
        if not self.backend.strip():
            raise ValueError("Measurement backend must not be empty")
        if self.status is MeasurementStatus.AVAILABLE and not clean_values:
            raise ValueError("Available measurements must contain at least one value")
        if self.status in {MeasurementStatus.UNAVAILABLE, MeasurementStatus.INVALID} and clean_values:
            raise ValueError("Unavailable or invalid measurements cannot carry usable values")
        if self.status is not MeasurementStatus.AVAILABLE and not self.reason:
            raise ValueError("Non-available measurements require an explicit reason")

    @property
    def available(self) -> bool:
        return self.status is MeasurementStatus.AVAILABLE

    def value(self, name: str) -> float | None:
        return self.values.get(name)

    @property
    def heart_rate_bpm(self) -> float | None:
        return self.value("heart_rate_bpm")

    @property
    def pr_ms(self) -> float | None:
        return self.value("pr_ms")

    @property
    def qrs_ms(self) -> float | None:
        return self.value("qrs_ms")

    @property
    def qt_ms(self) -> float | None:
        return self.value("qt_ms")

    @property
    def qtc_ms(self) -> float | None:
        return self.value("qtc_ms")

    @property
    def st_deviation_mv(self) -> float | None:
        return self.value("st_deviation_mv")


@dataclass(frozen=True, slots=True)
class MeasurementSet(SerializableContract):
    by_backend: dict[str, Measurements]

    def __post_init__(self) -> None:
        if not self.by_backend:
            raise ValueError("MeasurementSet requires at least one backend result")
        if any(name != result.backend for name, result in self.by_backend.items()):
            raise ValueError("MeasurementSet keys must equal their backend names")

    def get(self, backend: str) -> Measurements | None:
        return self.by_backend.get(backend)

    @property
    def available(self) -> tuple[Measurements, ...]:
        return tuple(result for result in self.by_backend.values() if result.available)


@dataclass(frozen=True, slots=True)
class AdmissibilityResult(SerializableContract):
    admissible: bool
    status: MeasurementStatus
    reasons: tuple[str, ...]
    clean: Measurements | None = None
    attacked: Measurements | None = None
    deltas: dict[str, float] = field(default_factory=dict)
    tolerances: dict[str, float] = field(default_factory=dict)
    preservation_evidence: ClinicalPreservationEvidence | None = None
    primary_endpoint_eligible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", MeasurementStatus(self.status))
        if self.status is not MeasurementStatus.AVAILABLE and self.admissible:
            raise ValueError("Unavailable clinical evidence cannot establish admissibility")
        if not self.admissible and not self.reasons:
            raise ValueError("A non-admissible result requires an explicit reason")
        if any(not np.isfinite(value) or value < 0 for value in self.tolerances.values()):
            raise ValueError("Admissibility tolerances must be finite and non-negative")
        if (
            self.primary_endpoint_eligible
            and self.status is not MeasurementStatus.AVAILABLE
        ):
            raise ValueError("Primary endpoint eligibility requires available clinical evidence")
        if self.primary_endpoint_eligible and (
            self.preservation_evidence is None
            or not self.preservation_evidence.independent
        ):
            raise ValueError(
                "Primary endpoint eligibility requires independent target-specific "
                "preservation evidence"
            )


@dataclass(frozen=True, slots=True)
class ReleaseDecision(SerializableContract):
    defense: DefenseKind
    released: bool
    score: float
    threshold: float
    reasons: tuple[str, ...] = ()
    evidence_available: bool = True
    matched_clean_coverage: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "defense", DefenseKind(self.defense))
        if not np.isfinite(self.score) or not np.isfinite(self.threshold):
            raise ValueError("Release score and threshold must be finite")
        if not self.evidence_available and self.released:
            raise ValueError("A defense cannot release when required evidence is unavailable")
        if not self.released and not self.reasons:
            raise ValueError("A rejected release requires an explicit reason")
        if self.matched_clean_coverage is not None and not 0 <= self.matched_clean_coverage <= 1:
            raise ValueError("Matched clean coverage must be in [0, 1]")
