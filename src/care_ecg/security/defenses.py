from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .claims import DEFAULT_TARGET_ALIASES, ClaimOracle
from .contracts import (
    DefenseKind,
    Measurements,
    MeasurementStatus,
    ReleaseDecision,
    SerializableContract,
    VictimSnapshot,
)
from .measurements import MeasurementBackend, NumpyMeasurementBackend


class Verifier(Protocol):
    kind: DefenseKind
    threshold: float

    def decide(self, snapshot: VictimSnapshot, **kwargs: Any) -> ReleaseDecision: ...


def _finite_threshold(value: float, name: str = "threshold") -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(slots=True)
class ShippedVerifier:
    """V1: CARE-ECG's report/evidence risk gate with explicit release output."""

    threshold: float = 1.0
    hallucination_reference: float = 0.5
    unsupported_tolerance: float = 0.0
    kind: DefenseKind = field(default=DefenseKind.V1, init=False)

    def __post_init__(self) -> None:
        self.threshold = _finite_threshold(self.threshold)
        self.hallucination_reference = _finite_threshold(
            self.hallucination_reference, "hallucination_reference"
        )
        if self.hallucination_reference <= 0:
            raise ValueError("hallucination_reference must be positive")
        self.unsupported_tolerance = _finite_threshold(
            self.unsupported_tolerance, "unsupported_tolerance"
        )
        if self.unsupported_tolerance < 0:
            raise ValueError("unsupported_tolerance must be non-negative")

    def score(self, snapshot: VictimSnapshot) -> float:
        hallucination = snapshot.hallucination_score / self.hallucination_reference
        unsupported = 0.0
        if snapshot.unsupported_claim_score > self.unsupported_tolerance:
            unsupported = 1.0 + snapshot.unsupported_claim_score - self.unsupported_tolerance
        return float(max(hallucination, unsupported))

    def decide(self, snapshot: VictimSnapshot, **kwargs: Any) -> ReleaseDecision:
        del kwargs
        score = self.score(snapshot)
        reasons: list[str] = []
        if not snapshot.released:
            reasons.append("victim pipeline did not release the report")
        if snapshot.abstained:
            reasons.append("victim pipeline abstained")
        if score > self.threshold:
            reasons.append("V1 evidence-consistency risk exceeded threshold")
        return ReleaseDecision(
            defense=self.kind,
            released=not reasons,
            score=score,
            threshold=self.threshold,
            reasons=tuple(reasons),
            metadata={
                "hallucination_score": snapshot.hallucination_score,
                "unsupported_claim_score": snapshot.unsupported_claim_score,
                "used_fallback": snapshot.used_fallback,
            },
        )


_V2_REQUIRED = (
    "heart_rate_bpm",
    "rhythm_irregularity",
    "pr_ms",
    "qrs_ms",
    "qtc_ms",
    "st_deviation_mv",
)


def _measurement_consistency(diagnosis: str, measurements: Measurements) -> tuple[float, str]:
    values = measurements.values
    heart_rate = values["heart_rate_bpm"]
    irregularity = values["rhythm_irregularity"]
    pr_ms = values["pr_ms"]
    qrs_ms = values["qrs_ms"]
    qtc_ms = values["qtc_ms"]
    st_mv = values["st_deviation_mv"]
    conduction_abnormal = pr_ms > 220.0 or qrs_ms > 120.0
    repolarization_abnormal = not 340.0 <= qtc_ms <= 480.0 or abs(st_mv) > 0.10
    rhythm_abnormal = not 50.0 <= heart_rate <= 110.0 or irregularity > 0.16
    label = diagnosis.upper()
    if label in {"NORM", "NORMAL"}:
        consistent = not (conduction_abnormal or repolarization_abnormal or rhythm_abnormal)
        return (0.0, "measurements support a normal-range claim") if consistent else (
            2.0,
            "independent measurements conflict with a normal-range claim",
        )
    if label in {"CD", "CONDUCTION DISTURBANCE"}:
        return (0.0, "measurements support a conduction claim") if conduction_abnormal else (
            2.0,
            "independent intervals do not support a conduction claim",
        )
    if label in {"STTC", "ST/T CHANGE", "ST-T CHANGE"}:
        return (0.0, "measurements support a repolarization claim") if repolarization_abnormal else (
            2.0,
            "independent measurements do not support an ST/T claim",
        )
    # These interval/rhythm measurements cannot establish or refute MI by themselves.
    return 0.0, "measurement panel is non-dispositive for this diagnosis"


def _released_report_claims(snapshot: VictimSnapshot) -> tuple[str, ...]:
    labels = set(DEFAULT_TARGET_ALIASES)
    target_label = snapshot.metadata.get("target_label")
    if isinstance(target_label, str) and target_label.strip():
        labels.add(target_label.strip().upper())
    return tuple(
        label
        for label in sorted(labels)
        if ClaimOracle(label).contains_target_claim(snapshot.report)
    )


@dataclass(slots=True)
class MeasurementAwareVerifier:
    """V2: V1 plus a measurement path independent of the attacked latent state."""

    backend: MeasurementBackend = field(default_factory=NumpyMeasurementBackend)
    threshold: float = 1.0
    shipped: ShippedVerifier = field(default_factory=ShippedVerifier)
    required_measurements: tuple[str, ...] = _V2_REQUIRED
    kind: DefenseKind = field(default=DefenseKind.V2, init=False)

    def __post_init__(self) -> None:
        self.threshold = _finite_threshold(self.threshold)
        if not self.required_measurements:
            raise ValueError("V2 requires an explicit measurement panel")

    def _measure(
        self,
        measurements: Measurements | None,
        waveform_mv: np.ndarray | None,
        sample_rate_hz: float | None,
    ) -> Measurements:
        if measurements is not None:
            return measurements
        if waveform_mv is None or sample_rate_hz is None:
            return Measurements(
                backend=self.backend.name,
                status=MeasurementStatus.UNAVAILABLE,
                reason="V2 requires either measurements or a waveform and sample rate",
            )
        try:
            return self.backend.measure(waveform_mv, sample_rate_hz)
        # A configured backend is an isolation boundary; its exception cannot turn
        # missing evidence into an implicit pass.
        except Exception as exc:  # noqa: BLE001
            return Measurements(
                backend=self.backend.name,
                status=MeasurementStatus.UNAVAILABLE,
                reason=f"measurement backend error: {type(exc).__name__}: {exc}",
            )

    def decide(
        self,
        snapshot: VictimSnapshot,
        *,
        measurements: Measurements | None = None,
        waveform_mv: np.ndarray | None = None,
        sample_rate_hz: float | None = None,
        **kwargs: Any,
    ) -> ReleaseDecision:
        del kwargs
        v1 = self.shipped.decide(snapshot)
        measured = self._measure(measurements, waveform_mv, sample_rate_hz)
        missing = tuple(
            name for name in self.required_measurements if name not in measured.values
        )
        if not measured.available or missing:
            reasons = [
                f"V2 measurements unavailable: {measured.reason or measured.status.value}"
            ]
            if missing:
                reasons.append(f"missing required measurements: {', '.join(missing)}")
            return ReleaseDecision(
                defense=self.kind,
                released=False,
                score=max(v1.score, self.threshold + 1.0),
                threshold=self.threshold,
                reasons=tuple(reasons),
                evidence_available=False,
                metadata={"v1": v1.to_dict(), "measurements": measured.to_dict()},
            )
        report_claims = _released_report_claims(snapshot)
        assessments = []
        for claim in report_claims:
            claim_score, interpretation = _measurement_consistency(claim, measured)
            assessments.append(
                {
                    "claim": claim,
                    "score": claim_score,
                    "interpretation": interpretation,
                }
            )
        measurement_score = max(
            (float(item["score"]) for item in assessments), default=0.0
        )
        interpretations = tuple(
            dict.fromkeys(
                str(item["interpretation"])
                for item in assessments
                if float(item["score"]) > self.threshold
            )
        )
        score = max(v1.score, measurement_score)
        reasons = list(v1.reasons)
        reasons.extend(interpretations)
        if score > self.threshold and not reasons:
            reasons.append("V2 combined risk exceeded threshold")
        return ReleaseDecision(
            defense=self.kind,
            released=not reasons,
            score=score,
            threshold=self.threshold,
            reasons=tuple(reasons),
            metadata={
                "v1": v1.to_dict(),
                "measurements": measured.to_dict(),
                "evaluated_report_claims": list(report_claims),
                "report_claim_assessments": assessments,
                "target_claim_flag_without_label": bool(
                    snapshot.target_claim_present
                    and not isinstance(snapshot.metadata.get("target_label"), str)
                ),
                "measurement_interpretation": (
                    "; ".join(interpretations)
                    if interpretations
                    else "no measurement conflict in affirmed released-report claims"
                ),
                "measurement_score": measurement_score,
            },
        )


def hash_waveform(waveform_mv: np.ndarray) -> str:
    waveform = np.asarray(waveform_mv)
    if waveform.ndim != 2 or not np.isfinite(waveform).all():
        raise ValueError("Authenticated waveform must be a finite [leads, samples] array")
    canonical = np.ascontiguousarray(waveform, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(b"CARE-ECG/raw-mV/v1\0")
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AuthenticatedHistoryRecord(SerializableContract):
    patient_id: str
    record_id: str
    timestamp: float
    waveform_sha256: str
    measurements: Measurements
    signature: str
    algorithm: str = "HMAC-SHA256"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.patient_id.strip() or not self.record_id.strip():
            raise ValueError("Authenticated history identifiers must not be empty")
        if not np.isfinite(self.timestamp):
            raise ValueError("Authenticated history timestamp must be finite")
        if not re_full_sha256(self.waveform_sha256):
            raise ValueError("waveform_sha256 must be a lowercase SHA-256 digest")
        if not re_full_sha256(self.signature):
            raise ValueError("signature must be a lowercase HMAC-SHA256 digest")
        if self.algorithm != "HMAC-SHA256":
            raise ValueError("Only HMAC-SHA256 history records are supported")


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _history_payload(
    patient_id: str,
    record_id: str,
    timestamp: float,
    waveform_sha256: str,
    measurements: Measurements,
    metadata: Mapping[str, Any],
) -> bytes:
    payload = {
        "algorithm": "HMAC-SHA256",
        "patient_id": patient_id,
        "record_id": record_id,
        "timestamp": float(timestamp),
        "waveform_sha256": waveform_sha256,
        "measurements": measurements.to_dict(),
        "metadata": dict(metadata),
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("History metadata must be JSON-serializable") from exc
    return encoded.encode("utf-8")


class HistoryAuthenticator:
    def __init__(self, key: bytes | str):
        self._key = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        if len(self._key) < 16:
            raise ValueError("History HMAC key must contain at least 16 bytes")

    def seal(
        self,
        *,
        patient_id: str,
        record_id: str,
        timestamp: float,
        waveform_mv: np.ndarray,
        measurements: Measurements,
        metadata: Mapping[str, Any] | None = None,
    ) -> AuthenticatedHistoryRecord:
        waveform_sha256 = hash_waveform(waveform_mv)
        record_metadata = dict(metadata or {})
        payload = _history_payload(
            patient_id,
            record_id,
            timestamp,
            waveform_sha256,
            measurements,
            record_metadata,
        )
        signature = hmac.new(self._key, payload, hashlib.sha256).hexdigest()
        return AuthenticatedHistoryRecord(
            patient_id=patient_id,
            record_id=record_id,
            timestamp=timestamp,
            waveform_sha256=waveform_sha256,
            measurements=measurements,
            signature=signature,
            metadata=record_metadata,
        )

    def verify_detailed(
        self,
        record: AuthenticatedHistoryRecord,
        *,
        patient_id: str,
        waveform_mv: np.ndarray | None = None,
    ) -> tuple[bool, str]:
        if not hmac.compare_digest(record.patient_id, patient_id):
            return False, "authenticated history belongs to a different patient"
        try:
            payload = _history_payload(
                record.patient_id,
                record.record_id,
                record.timestamp,
                record.waveform_sha256,
                record.measurements,
                record.metadata,
            )
        except ValueError as exc:
            return False, str(exc)
        expected = hmac.new(self._key, payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(record.signature, expected):
            return False, "authenticated history signature mismatch"
        if waveform_mv is not None:
            try:
                observed_hash = hash_waveform(waveform_mv)
            except ValueError as exc:
                return False, str(exc)
            if not hmac.compare_digest(record.waveform_sha256, observed_hash):
                return False, "prior waveform hash mismatch"
        if not record.measurements.available:
            return False, "authenticated history measurements are unavailable"
        return True, "authenticated same-patient history verified"

    def verify(
        self,
        record: AuthenticatedHistoryRecord,
        *,
        patient_id: str,
        waveform_mv: np.ndarray | None = None,
    ) -> bool:
        verified, _ = self.verify_detailed(
            record, patient_id=patient_id, waveform_mv=waveform_mv
        )
        return verified


DEFAULT_LONGITUDINAL_TOLERANCES = {
    "heart_rate_bpm": 25.0,
    "rhythm_irregularity": 0.20,
    "pr_ms": 40.0,
    "qrs_ms": 25.0,
    "qtc_ms": 60.0,
    "st_deviation_mv": 0.20,
}


@dataclass(slots=True)
class LongitudinalVerifier:
    """V3: V2 plus HMAC-authenticated, same-patient prior measurements."""

    authenticator: HistoryAuthenticator
    measurement_verifier: MeasurementAwareVerifier = field(
        default_factory=MeasurementAwareVerifier
    )
    threshold: float = 1.0
    change_tolerances: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_LONGITUDINAL_TOLERANCES)
    )
    kind: DefenseKind = field(default=DefenseKind.V3, init=False)

    def __post_init__(self) -> None:
        self.threshold = _finite_threshold(self.threshold)
        if any(
            not np.isfinite(value) or value <= 0
            for value in self.change_tolerances.values()
        ):
            raise ValueError("Longitudinal tolerances must be finite and positive")

    def decide(
        self,
        snapshot: VictimSnapshot,
        *,
        patient_id: str,
        history: AuthenticatedHistoryRecord | None,
        prior_waveform_mv: np.ndarray | None = None,
        current_timestamp: float | None = None,
        measurements: Measurements | None = None,
        waveform_mv: np.ndarray | None = None,
        sample_rate_hz: float | None = None,
        **kwargs: Any,
    ) -> ReleaseDecision:
        del kwargs
        v2 = self.measurement_verifier.decide(
            snapshot,
            measurements=measurements,
            waveform_mv=waveform_mv,
            sample_rate_hz=sample_rate_hz,
        )
        if history is None:
            return ReleaseDecision(
                defense=self.kind,
                released=False,
                score=max(v2.score, self.threshold + 1.0),
                threshold=self.threshold,
                reasons=("authenticated same-patient history is unavailable",),
                evidence_available=False,
                metadata={"v2": v2.to_dict()},
            )
        verified, verification_reason = self.authenticator.verify_detailed(
            history,
            patient_id=patient_id,
            waveform_mv=prior_waveform_mv,
        )
        if not verified:
            return ReleaseDecision(
                defense=self.kind,
                released=False,
                score=max(v2.score, self.threshold + 1.0),
                threshold=self.threshold,
                reasons=(verification_reason,),
                evidence_available=False,
                metadata={"v2": v2.to_dict(), "history_record_id": history.record_id},
            )
        if current_timestamp is None:
            return ReleaseDecision(
                defense=self.kind,
                released=False,
                score=max(v2.score, self.threshold + 1.0),
                threshold=self.threshold,
                reasons=("current ECG timestamp is unavailable for history chronology",),
                evidence_available=False,
                metadata={"v2": v2.to_dict(), "history_record_id": history.record_id},
            )
        if not np.isfinite(current_timestamp):
            raise ValueError("current_timestamp must be finite when supplied")
        if history.timestamp >= current_timestamp:
            return ReleaseDecision(
                defense=self.kind,
                released=False,
                score=max(v2.score, self.threshold + 1.0),
                threshold=self.threshold,
                reasons=("authenticated history is not prior to the current ECG",),
                evidence_available=False,
                metadata={"v2": v2.to_dict(), "history_record_id": history.record_id},
            )
        current_data = v2.metadata.get("measurements", {})
        current_values = current_data.get("values", {})
        common = tuple(
            name
            for name in self.change_tolerances
            if name in current_values and name in history.measurements.values
        )
        if set(common) != set(self.change_tolerances):
            missing = sorted(set(self.change_tolerances) - set(common))
            return ReleaseDecision(
                defense=self.kind,
                released=False,
                score=max(v2.score, self.threshold + 1.0),
                threshold=self.threshold,
                reasons=(f"longitudinal measurements unavailable: {', '.join(missing)}",),
                evidence_available=False,
                metadata={"v2": v2.to_dict(), "history_record_id": history.record_id},
            )
        normalized_deltas = {
            name: abs(float(current_values[name]) - history.measurements.values[name])
            / self.change_tolerances[name]
            for name in common
        }
        longitudinal_score = max(normalized_deltas.values(), default=0.0)
        score = max(v2.score, longitudinal_score)
        reasons = list(v2.reasons)
        if longitudinal_score > self.threshold:
            reasons.append("current waveform differs materially from authenticated history")
        if score > self.threshold and not reasons:
            reasons.append("V3 combined risk exceeded threshold")
        return ReleaseDecision(
            defense=self.kind,
            released=not reasons,
            score=score,
            threshold=self.threshold,
            reasons=tuple(reasons),
            metadata={
                "v2": v2.to_dict(),
                "history_record_id": history.record_id,
                "history_verification": verification_reason,
                "history_timestamp": history.timestamp,
                "current_timestamp": current_timestamp,
                "normalized_longitudinal_deltas": normalized_deltas,
            },
        )


def _single_coverage_threshold(
    clean_scores: Sequence[float],
    target_coverage: float,
    eligible: Sequence[bool] | None,
) -> float:
    scores = np.asarray(clean_scores, dtype=float).reshape(-1)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Clean calibration scores must be non-empty and finite")
    if not 0 <= target_coverage <= 1:
        raise ValueError("target_coverage must be in [0, 1]")
    availability = (
        np.ones(len(scores), dtype=bool)
        if eligible is None
        else np.asarray(eligible, dtype=bool).reshape(-1)
    )
    if availability.shape != scores.shape:
        raise ValueError("eligible mask must match clean calibration scores")
    candidates = np.concatenate(
        ([np.nextafter(float(np.min(scores)), -np.inf)], np.unique(scores))
    )
    coverages = np.asarray(
        [np.mean(availability & (scores <= threshold)) for threshold in candidates]
    )
    differences = np.abs(coverages - target_coverage)
    # On an equal-distance tie, prefer lower coverage so abstention is not understated.
    best_difference = float(np.min(differences))
    tied = np.flatnonzero(np.isclose(differences, best_difference, atol=1e-12, rtol=0))
    index = int(tied[np.argmin(coverages[tied])])
    return float(candidates[index])


def calibrate_matched_coverage(
    clean_scores: Sequence[float] | Mapping[DefenseKind | str, Sequence[float]],
    target_coverage: float,
    *,
    eligible: Sequence[bool] | Mapping[DefenseKind | str, Sequence[bool]] | None = None,
) -> float | dict[str, float]:
    """Select lower-is-safer thresholds at a shared clean release coverage.

    Evidence-unavailable cases can be supplied through ``eligible`` and remain
    unreleasable at every threshold. Returned mapping keys use their string values.
    """

    if isinstance(clean_scores, Mapping):
        thresholds: dict[str, float] = {}
        for defense, scores in clean_scores.items():
            mask = eligible.get(defense) if isinstance(eligible, Mapping) else None
            key = defense.value if isinstance(defense, DefenseKind) else str(defense)
            thresholds[key] = _single_coverage_threshold(scores, target_coverage, mask)
        return thresholds
    if isinstance(eligible, Mapping):
        raise TypeError("A mapped eligible mask requires mapped clean_scores")
    return _single_coverage_threshold(clean_scores, target_coverage, eligible)


matched_coverage_threshold = calibrate_matched_coverage

AuthenticatedHistory = AuthenticatedHistoryRecord
