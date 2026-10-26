from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy import signal as scipy_signal

from .contracts import (
    AdmissibilityResult,
    ClinicalCaseContext,
    ClinicalTarget,
    Measurements,
    MeasurementSet,
    MeasurementStatus,
)

MEASUREMENT_UNITS = {
    "heart_rate_bpm": "bpm",
    "rr_ms": "ms",
    "rhythm_irregularity": "ratio",
    "pr_ms": "ms",
    "qrs_ms": "ms",
    "qt_ms": "ms",
    "qtc_ms": "ms",
    "st_deviation_mv": "mV",
}

DEFAULT_ADMISSIBILITY_TOLERANCES = {
    "heart_rate_bpm": 10.0,
    "rhythm_irregularity": 0.10,
    "pr_ms": 25.0,
    "qrs_ms": 15.0,
    "qtc_ms": 35.0,
    "st_deviation_mv": 0.10,
}


class MeasurementBackend(Protocol):
    name: str
    independence_group: str

    def measure(self, waveform_mv: np.ndarray, sample_rate_hz: float) -> Measurements: ...


IndependentMeasurementBackend = MeasurementBackend


class ClinicalPreservationChecker(Protocol):
    """Contract for target-aware clinical-preservation endpoint evaluators."""

    def evaluate(
        self,
        clean_waveform_mv: np.ndarray,
        attacked_waveform_mv: np.ndarray,
        sample_rate_hz: float,
        *,
        case: object | None = None,
        patient_id: str | None = None,
        record_id: str | None = None,
        target: ClinicalTarget | None = None,
    ) -> AdmissibilityResult: ...


def _unavailable(backend: str, reason: str, *, invalid: bool = False) -> Measurements:
    return Measurements(
        backend=backend,
        status=MeasurementStatus.INVALID if invalid else MeasurementStatus.UNAVAILABLE,
        reason=reason,
    )


def _validated_signal(
    waveform_mv: np.ndarray, sample_rate_hz: float, backend: str
) -> tuple[np.ndarray | None, Measurements | None]:
    waveform = np.asarray(waveform_mv, dtype=float)
    if waveform.ndim != 2 or waveform.shape[0] < 1 or waveform.shape[1] < 2:
        return None, _unavailable(backend, "expected [leads, samples] waveform", invalid=True)
    if not np.isfinite(waveform).all():
        return None, _unavailable(backend, "waveform contains non-finite values", invalid=True)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        return None, _unavailable(backend, "sample rate must be finite and positive", invalid=True)
    if waveform.shape[1] < int(2.0 * sample_rate_hz):
        return None, _unavailable(backend, "at least two seconds are required")
    return waveform, None


def _select_lead(waveform: np.ndarray) -> tuple[np.ndarray, int]:
    preferred = 1 if waveform.shape[0] > 1 else 0
    spread = np.ptp(waveform, axis=1)
    if spread[preferred] <= 1e-8:
        preferred = int(np.argmax(spread))
    return waveform[preferred], preferred


def _refractory_select(candidates: np.ndarray, score: np.ndarray, refractory: int) -> np.ndarray:
    selected: list[int] = []
    for index in candidates[np.argsort(score[candidates])[::-1]]:
        if all(abs(int(index) - prior) >= refractory for prior in selected):
            selected.append(int(index))
    return np.asarray(sorted(selected), dtype=int)


def _derive_intervals(
    lead: np.ndarray,
    r_peaks: np.ndarray,
    sample_rate_hz: float,
) -> dict[str, float] | None:
    if len(r_peaks) < 3:
        return None
    rr_samples = np.diff(r_peaks).astype(float)
    rr_seconds = float(np.median(rr_samples) / sample_rate_hz)
    if not 0.25 <= rr_seconds <= 2.5:
        return None
    centered = lead - np.median(lead)
    derivative = np.abs(np.gradient(centered))
    qrs_widths: list[float] = []
    pr_intervals: list[float] = []
    qt_intervals: list[float] = []
    st_offsets: list[float] = []
    for peak in r_peaks[1:-1]:
        left = max(0, peak - int(0.12 * sample_rate_hz))
        right = min(len(lead) - 1, peak + int(0.14 * sample_rate_hz))
        local = derivative[left : right + 1]
        if local.size < 3:
            continue
        threshold = max(float(np.max(local)) * 0.16, 1e-8)
        active = np.flatnonzero(local >= threshold)
        if not len(active):
            continue
        onset = left + int(active[0])
        offset = left + int(active[-1])
        qrs_widths.append((offset - onset + 1) * 1000.0 / sample_rate_hz)

        p_start = max(0, onset - int(0.28 * sample_rate_hz))
        p_end = max(p_start + 1, onset - int(0.06 * sample_rate_hz))
        p_segment = np.abs(centered[p_start:p_end])
        if len(p_segment):
            p_peak = p_start + int(np.argmax(p_segment))
            pr_intervals.append((onset - p_peak) * 1000.0 / sample_rate_hz)

        t_start = min(len(lead) - 1, offset + int(0.10 * sample_rate_hz))
        t_end_limit = min(len(lead), peak + int(0.55 * sample_rate_hz))
        t_segment = np.abs(centered[t_start:t_end_limit])
        if len(t_segment):
            t_peak = t_start + int(np.argmax(t_segment))
            tail = np.abs(centered[t_peak:t_end_limit])
            baseline_threshold = max(float(np.ptp(centered)) * 0.03, 1e-6)
            settled = np.flatnonzero(tail <= baseline_threshold)
            t_end = t_peak + (int(settled[0]) if len(settled) else len(tail) - 1)
            qt_intervals.append((t_end - onset) * 1000.0 / sample_rate_hz)

        baseline_start = max(0, onset - int(0.08 * sample_rate_hz))
        baseline_end = max(baseline_start + 1, onset - int(0.02 * sample_rate_hz))
        st_index = min(len(lead) - 1, offset + int(0.06 * sample_rate_hz))
        baseline = float(np.median(lead[baseline_start:baseline_end]))
        st_offsets.append(float(lead[st_index] - baseline))
    if not qrs_widths or not qt_intervals or not pr_intervals or not st_offsets:
        return None
    rr_ms = rr_seconds * 1000.0
    qt_ms = float(np.median(qt_intervals))
    return {
        "heart_rate_bpm": 60.0 / rr_seconds,
        "rr_ms": rr_ms,
        "rhythm_irregularity": float(np.std(rr_samples) / max(np.mean(rr_samples), 1e-12)),
        "pr_ms": float(np.median(pr_intervals)),
        "qrs_ms": float(np.median(qrs_widths)),
        "qt_ms": qt_ms,
        "qtc_ms": qt_ms / np.sqrt(rr_seconds),
        "st_deviation_mv": float(np.median(st_offsets)),
    }


@dataclass(frozen=True, slots=True)
class NumpyMeasurementBackend:
    """Dependency-light derivative detector used by the V2 verifier."""

    name: str = "numpy_derivative_v1"
    independence_group: str = "numpy_derivative"
    threshold_quantile: float = 0.92

    def __post_init__(self) -> None:
        if not 0.5 < self.threshold_quantile < 1:
            raise ValueError("threshold_quantile must be between 0.5 and 1")

    def measure(self, waveform_mv: np.ndarray, sample_rate_hz: float) -> Measurements:
        waveform, unavailable = _validated_signal(waveform_mv, sample_rate_hz, self.name)
        if unavailable is not None:
            return unavailable
        assert waveform is not None
        lead, lead_index = _select_lead(waveform)
        centered = lead - np.median(lead)
        derivative_energy = np.abs(np.gradient(centered))
        if float(np.ptp(centered)) <= 1e-8:
            return _unavailable(self.name, "no measurable ECG variation")
        threshold = float(np.quantile(derivative_energy, self.threshold_quantile))
        candidates = (
            np.flatnonzero(
                (derivative_energy[1:-1] >= derivative_energy[:-2])
                & (derivative_energy[1:-1] > derivative_energy[2:])
                & (derivative_energy[1:-1] >= threshold)
            )
            + 1
        )
        peaks = _refractory_select(
            candidates,
            derivative_energy,
            max(1, int(0.25 * sample_rate_hz)),
        )
        values = _derive_intervals(lead, peaks, sample_rate_hz)
        if values is None:
            return _unavailable(self.name, "QRS delineation did not yield three valid beats")
        return Measurements(
            backend=self.name,
            status=MeasurementStatus.AVAILABLE,
            values=values,
            units={key: MEASUREMENT_UNITS[key] for key in values},
            metadata={
                "lead_index": lead_index,
                "beat_count": len(peaks),
                "independence_group": self.independence_group,
            },
        )


@dataclass(frozen=True, slots=True)
class ScipyMeasurementBackend:
    """Bandpass/prominence detector kept independent from the V2 implementation."""

    name: str = "scipy_bandpass_v1"
    independence_group: str = "scipy_bandpass_prominence"
    low_hz: float = 5.0
    high_hz: float = 35.0

    def measure(self, waveform_mv: np.ndarray, sample_rate_hz: float) -> Measurements:
        waveform, unavailable = _validated_signal(waveform_mv, sample_rate_hz, self.name)
        if unavailable is not None:
            return unavailable
        assert waveform is not None
        if not 0 < self.low_hz < self.high_hz < sample_rate_hz / 2:
            return _unavailable(
                self.name, "bandpass frequencies are incompatible with sample rate", invalid=True
            )
        lead, lead_index = _select_lead(waveform)
        centered = lead - np.median(lead)
        if float(np.ptp(centered)) <= 1e-8:
            return _unavailable(self.name, "no measurable ECG variation")
        try:
            sos = scipy_signal.butter(
                3,
                [self.low_hz, self.high_hz],
                btype="bandpass",
                fs=sample_rate_hz,
                output="sos",
            )
            filtered = scipy_signal.sosfiltfilt(sos, centered)
        except ValueError as exc:
            return _unavailable(self.name, f"bandpass processing unavailable: {exc}")
        prominence = max(float(np.std(filtered)) * 1.5, 1e-6)
        positive, positive_properties = scipy_signal.find_peaks(
            filtered,
            distance=max(1, int(0.25 * sample_rate_hz)),
            prominence=prominence,
        )
        negative, negative_properties = scipy_signal.find_peaks(
            -filtered,
            distance=max(1, int(0.25 * sample_rate_hz)),
            prominence=prominence,
        )
        positive_strength = float(np.median(positive_properties.get("prominences", [0.0])))
        negative_strength = float(np.median(negative_properties.get("prominences", [0.0])))
        peaks = negative if negative_strength > positive_strength else positive
        values = _derive_intervals(lead, peaks, sample_rate_hz)
        if values is None:
            return _unavailable(self.name, "filtered delineation did not yield three valid beats")
        return Measurements(
            backend=self.name,
            status=MeasurementStatus.AVAILABLE,
            values=values,
            units={key: MEASUREMENT_UNITS[key] for key in values},
            metadata={
                "lead_index": lead_index,
                "beat_count": len(peaks),
                "independence_group": self.independence_group,
            },
        )


@dataclass(frozen=True, slots=True)
class UnavailableMeasurementBackend:
    """Explicit placeholder for a configured backend that cannot run."""

    reason: str
    name: str = "unavailable"
    independence_group: str = "unavailable"

    def measure(self, waveform_mv: np.ndarray, sample_rate_hz: float) -> Measurements:
        del waveform_mv, sample_rate_hz
        return _unavailable(self.name, self.reason)


def measure_with_backends(
    waveform_mv: np.ndarray,
    sample_rate_hz: float,
    backends: tuple[MeasurementBackend, ...] | list[MeasurementBackend],
) -> MeasurementSet:
    if not backends:
        raise ValueError("At least one measurement backend is required")
    results: dict[str, Measurements] = {}
    for backend in backends:
        if backend.name in results:
            raise ValueError(f"Duplicate measurement backend name: {backend.name}")
        try:
            result = backend.measure(waveform_mv, sample_rate_hz)
        # Third-party measurement backends are an isolation boundary. Any backend
        # exception becomes explicit evidence unavailability rather than a release.
        except Exception as exc:  # noqa: BLE001
            result = _unavailable(backend.name, f"backend error: {type(exc).__name__}: {exc}")
        if result.backend != backend.name:
            raise ValueError("Measurement backend returned a mismatched backend identifier")
        results[backend.name] = result
    return MeasurementSet(results)


def assert_independent_backends(
    defense_backend: MeasurementBackend,
    admissibility_backend: MeasurementBackend,
) -> None:
    if defense_backend is admissibility_backend:
        raise ValueError("Defense and admissibility must use separate backend instances")
    if defense_backend.name == admissibility_backend.name:
        raise ValueError("Defense and admissibility backends must have distinct names")
    if defense_backend.independence_group == admissibility_backend.independence_group:
        raise ValueError("Defense and admissibility backends must use independent implementations")


@dataclass(frozen=True, slots=True)
class ClinicalAdmissibilityChecker:
    """Exploratory local delta checker, not a primary-endpoint evaluator.

    Primary evaluation requires an external ``ClinicalPreservationChecker`` that
    returns target-specific ``ClinicalPreservationEvidence``.
    """

    backend: MeasurementBackend = field(default_factory=ScipyMeasurementBackend)
    tolerances: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ADMISSIBILITY_TOLERANCES)
    )
    required: tuple[str, ...] = (
        "heart_rate_bpm",
        "rhythm_irregularity",
        "pr_ms",
        "qrs_ms",
        "qtc_ms",
        "st_deviation_mv",
    )

    def __post_init__(self) -> None:
        if not self.required:
            raise ValueError("At least one admissibility measurement is required")
        missing_tolerances = set(self.required) - set(self.tolerances)
        if missing_tolerances:
            raise ValueError(f"Missing tolerances for {sorted(missing_tolerances)}")
        if any(not np.isfinite(value) or value < 0 for value in self.tolerances.values()):
            raise ValueError("Measurement tolerances must be finite and non-negative")

    def evaluate(
        self,
        clean_waveform_mv: np.ndarray,
        attacked_waveform_mv: np.ndarray,
        sample_rate_hz: float,
        *,
        case: object | None = None,
        patient_id: str | None = None,
        record_id: str | None = None,
        target: ClinicalTarget | None = None,
    ) -> AdmissibilityResult:
        if case is not None:
            patient_id = patient_id or getattr(case, "patient_id", None)
            record_id = record_id or getattr(case, "record_id", None)
            target = target or getattr(case, "target", None)
        context = None
        if patient_id is not None and record_id is not None and target is not None:
            context = ClinicalCaseContext(str(patient_id), str(record_id), target)
        endpoint_metadata = {
            "endpoint_classification": "exploratory",
            "primary_endpoint_eligible": False,
            "primary_ineligibility_reason": (
                "bundled local delta measurements do not provide independent "
                "target-specific preserved-finding evidence"
            ),
            "case_context": None if context is None else context.to_dict(),
        }
        clean = self.backend.measure(clean_waveform_mv, sample_rate_hz)
        attacked = self.backend.measure(attacked_waveform_mv, sample_rate_hz)
        unavailable: list[str] = []
        if not clean.available:
            unavailable.append(f"clean measurements {clean.status.value}: {clean.reason}")
        if not attacked.available:
            unavailable.append(f"attacked measurements {attacked.status.value}: {attacked.reason}")
        missing = [
            name
            for name in self.required
            if name not in clean.values or name not in attacked.values
        ]
        if missing:
            unavailable.append(f"required measurements unavailable: {', '.join(sorted(missing))}")
        if unavailable:
            return AdmissibilityResult(
                admissible=False,
                status=MeasurementStatus.UNAVAILABLE,
                reasons=tuple(unavailable),
                clean=clean,
                attacked=attacked,
                tolerances={name: self.tolerances[name] for name in self.required},
                primary_endpoint_eligible=False,
                metadata=endpoint_metadata,
            )
        deltas = {name: abs(attacked.values[name] - clean.values[name]) for name in self.required}
        exceeded = tuple(
            f"{name} delta {deltas[name]:.6g} exceeds {self.tolerances[name]:.6g}"
            for name in self.required
            if deltas[name] > self.tolerances[name]
        )
        return AdmissibilityResult(
            admissible=not exceeded,
            status=MeasurementStatus.AVAILABLE,
            reasons=exceeded,
            clean=clean,
            attacked=attacked,
            deltas=deltas,
            tolerances={name: self.tolerances[name] for name in self.required},
            primary_endpoint_eligible=False,
            metadata={"reference_backend": self.backend.name, **endpoint_metadata},
        )
