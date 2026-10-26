from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import numpy as np
import torch
from scipy import signal as scipy_signal

from .contracts import ConstraintKind

Waveform = TypeVar("Waveform", np.ndarray, torch.Tensor)


class WaveformConstraint(Protocol):
    """Allowed-set interface. Inputs and outputs are always raw millivolts."""

    kind: ConstraintKind

    def project(self, clean: Waveform, candidate: Waveform) -> Waveform: ...

    def contains(self, clean: np.ndarray, candidate: np.ndarray) -> bool: ...

    def distance(self, clean: np.ndarray, candidate: np.ndarray) -> float: ...

    def sample(self, clean: Waveform, generator: Any | None = None) -> Waveform: ...

    def violations(self, clean: np.ndarray, candidate: np.ndarray) -> tuple[str, ...]: ...


def _projection_limit(value: float, tolerance: float) -> float:
    """Keep projected boundary points stable through float32 serialization."""

    margin = max(float(tolerance), abs(float(value)) * 1e-7)
    return max(0.0, float(value) - margin)


def _positive_finite(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    valid = np.isfinite(result) and (result >= 0 if allow_zero else result > 0)
    if not valid:
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {comparison}")
    return result


def _validate_waveform(value: np.ndarray | torch.Tensor, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [leads, samples]")
    if value.shape[0] < 1 or value.shape[1] < 2:
        raise ValueError(f"{name} must contain at least one lead and two samples")
    finite = bool(torch.isfinite(value).all()) if isinstance(value, torch.Tensor) else bool(
        np.isfinite(value).all()
    )
    if not finite:
        raise ValueError(f"{name} must contain only finite raw-mV values")


def _validate_pair(
    clean: np.ndarray | torch.Tensor,
    candidate: np.ndarray | torch.Tensor,
) -> None:
    if type(clean) is not type(candidate):
        raise TypeError("clean and candidate must use the same array backend")
    _validate_waveform(clean, "clean")
    _validate_waveform(candidate, "candidate")
    if clean.shape != candidate.shape:
        raise ValueError("clean and candidate waveforms must have identical shapes")


def _numpy_pair(
    clean: np.ndarray | torch.Tensor,
    candidate: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_pair(clean, candidate)
    clean_array = (
        clean.detach().cpu().numpy() if isinstance(clean, torch.Tensor) else np.asarray(clean)
    ).astype(float, copy=False)
    candidate_array = (
        candidate.detach().cpu().numpy()
        if isinstance(candidate, torch.Tensor)
        else np.asarray(candidate)
    ).astype(float, copy=False)
    _validate_pair(clean_array, candidate_array)
    return clean_array, candidate_array


def _normal_like(
    clean: np.ndarray | torch.Tensor,
    generator: np.random.Generator | torch.Generator | None,
) -> np.ndarray | torch.Tensor:
    if isinstance(clean, torch.Tensor):
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("A tensor waveform requires a torch.Generator")
        return torch.randn(
            clean.shape,
            dtype=clean.dtype,
            device=clean.device,
            generator=generator,
        )
    if generator is not None and not isinstance(generator, np.random.Generator):
        raise TypeError("A NumPy waveform requires a numpy.random.Generator")
    rng = generator or np.random.default_rng()
    return rng.standard_normal(clean.shape).astype(clean.dtype, copy=False)


def _uniform_like(
    clean: np.ndarray | torch.Tensor,
    low: float,
    high: float,
    generator: np.random.Generator | torch.Generator | None,
) -> np.ndarray | torch.Tensor:
    if isinstance(clean, torch.Tensor):
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("A tensor waveform requires a torch.Generator")
        values = torch.rand(
            clean.shape,
            dtype=clean.dtype,
            device=clean.device,
            generator=generator,
        )
        return values * (high - low) + low
    if generator is not None and not isinstance(generator, np.random.Generator):
        raise TypeError("A NumPy waveform requires a numpy.random.Generator")
    rng = generator or np.random.default_rng()
    return rng.uniform(low, high, clean.shape).astype(clean.dtype, copy=False)


@dataclass(frozen=True, slots=True)
class SInfConstraint:
    """Per-sample L-infinity ball around a clean raw-mV waveform."""

    epsilon_mv: float
    tolerance: float = 1e-7
    kind: ConstraintKind = field(default=ConstraintKind.S_INF, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "epsilon_mv", _positive_finite(self.epsilon_mv, "epsilon_mv", allow_zero=True)
        )
        object.__setattr__(
            self, "tolerance", _positive_finite(self.tolerance, "tolerance", allow_zero=True)
        )

    def project(self, clean: Waveform, candidate: Waveform) -> Waveform:
        _validate_pair(clean, candidate)
        limit = _projection_limit(self.epsilon_mv, self.tolerance)
        if isinstance(clean, torch.Tensor):
            return clean + (candidate - clean).clamp(-limit, limit)
        return clean + np.clip(candidate - clean, -limit, limit)

    def distance(self, clean: np.ndarray, candidate: np.ndarray) -> float:
        clean_array, candidate_array = _numpy_pair(clean, candidate)
        return float(np.max(np.abs(candidate_array - clean_array)))

    def violations(self, clean: np.ndarray, candidate: np.ndarray) -> tuple[str, ...]:
        return (
            ("linf_bound",)
            if self.distance(clean, candidate) > self.epsilon_mv + self.tolerance
            else ()
        )

    def contains(self, clean: np.ndarray, candidate: np.ndarray) -> bool:
        return not self.violations(clean, candidate)

    def sample(
        self,
        clean: Waveform,
        generator: np.random.Generator | torch.Generator | None = None,
    ) -> Waveform:
        _validate_waveform(clean, "clean")
        return clean + _uniform_like(clean, -self.epsilon_mv, self.epsilon_mv, generator)


@dataclass(frozen=True, slots=True)
class S2Constraint:
    """Per-lead L2 ball around a clean raw-mV waveform."""

    epsilon_mv: float
    per_lead: bool = True
    tolerance: float = 1e-7
    kind: ConstraintKind = field(default=ConstraintKind.S2, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "epsilon_mv", _positive_finite(self.epsilon_mv, "epsilon_mv", allow_zero=True)
        )
        object.__setattr__(
            self, "tolerance", _positive_finite(self.tolerance, "tolerance", allow_zero=True)
        )

    def project(self, clean: Waveform, candidate: Waveform) -> Waveform:
        _validate_pair(clean, candidate)
        delta = candidate - clean
        limit = _projection_limit(self.epsilon_mv, self.tolerance)
        if isinstance(clean, torch.Tensor):
            if self.per_lead:
                norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            else:
                norm = torch.linalg.vector_norm(delta).reshape(1, 1)
            scale = torch.clamp(limit / norm.clamp_min(1e-30), max=1.0)
            return clean + delta * scale
        axis = -1 if self.per_lead else None
        norm = np.linalg.norm(delta, axis=axis, keepdims=True)
        scale = np.minimum(1.0, limit / np.maximum(norm, 1e-30))
        return clean + delta * scale

    def distance(self, clean: np.ndarray, candidate: np.ndarray) -> float:
        clean_array, candidate_array = _numpy_pair(clean, candidate)
        delta = candidate_array - clean_array
        if not self.per_lead:
            return float(np.linalg.norm(delta))
        return float(np.max(np.linalg.norm(delta, axis=-1)))

    def violations(self, clean: np.ndarray, candidate: np.ndarray) -> tuple[str, ...]:
        return (
            ("per_lead_l2_bound" if self.per_lead else "global_l2_bound",)
            if self.distance(clean, candidate) > self.epsilon_mv + self.tolerance
            else ()
        )

    def contains(self, clean: np.ndarray, candidate: np.ndarray) -> bool:
        return not self.violations(clean, candidate)

    def sample(
        self,
        clean: Waveform,
        generator: np.random.Generator | torch.Generator | None = None,
    ) -> Waveform:
        _validate_waveform(clean, "clean")
        direction = _normal_like(clean, generator)
        if isinstance(clean, torch.Tensor):
            if self.per_lead:
                norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-30)
                radii = torch.rand(
                    (clean.shape[0], 1),
                    dtype=clean.dtype,
                    device=clean.device,
                    generator=generator,
                ) ** (1.0 / clean.shape[-1])
            else:
                norm = torch.linalg.vector_norm(direction).reshape(1, 1).clamp_min(1e-30)
                radii = torch.rand(
                    (1, 1), dtype=clean.dtype, device=clean.device, generator=generator
                ) ** (1.0 / clean.numel())
            return clean + direction / norm * radii * self.epsilon_mv
        rng = generator or np.random.default_rng()
        if self.per_lead:
            norm = np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-30)
            radii = rng.random((clean.shape[0], 1)) ** (1.0 / clean.shape[-1])
        else:
            norm = np.maximum(np.linalg.norm(direction, keepdims=True), 1e-30)
            radii = rng.random((1, 1)) ** (1.0 / clean.size)
        return clean + direction / norm * radii * self.epsilon_mv


@dataclass(frozen=True, slots=True)
class SSpecConstraint:
    """Amplitude-bounded, smooth, low-frequency perturbations in raw mV."""

    epsilon_mv: float
    sample_rate_hz: float
    cutoff_hz: float = 40.0
    max_high_frequency_ratio: float = 0.05
    max_slew_mv: float | None = None
    tolerance: float = 1e-7
    kind: ConstraintKind = field(default=ConstraintKind.S_SPEC, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "epsilon_mv", _positive_finite(self.epsilon_mv, "epsilon_mv", allow_zero=True)
        )
        object.__setattr__(
            self, "sample_rate_hz", _positive_finite(self.sample_rate_hz, "sample_rate_hz")
        )
        object.__setattr__(self, "cutoff_hz", _positive_finite(self.cutoff_hz, "cutoff_hz"))
        if self.cutoff_hz >= self.sample_rate_hz / 2:
            raise ValueError("cutoff_hz must be below the Nyquist frequency")
        if not 0 <= self.max_high_frequency_ratio <= 1:
            raise ValueError("max_high_frequency_ratio must be in [0, 1]")
        if self.max_slew_mv is not None:
            object.__setattr__(
                self, "max_slew_mv", _positive_finite(self.max_slew_mv, "max_slew_mv")
            )
        object.__setattr__(
            self, "tolerance", _positive_finite(self.tolerance, "tolerance", allow_zero=True)
        )

    @property
    def _projection_high_frequency_ratio(self) -> float:
        if self.max_high_frequency_ratio >= 1.0:
            return 1.0
        return 0.9 * self.max_high_frequency_ratio

    def _project_numpy(self, delta: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(delta, axis=-1)
        frequencies = np.fft.rfftfreq(delta.shape[-1], d=1.0 / self.sample_rate_hz)
        high_mask = frequencies > self.cutoff_hz
        low_energy = np.sum(np.abs(spectrum[:, ~high_mask]) ** 2, axis=-1, keepdims=True)
        high_energy = np.sum(np.abs(spectrum[:, high_mask]) ** 2, axis=-1, keepdims=True)
        ratio_limit = self._projection_high_frequency_ratio
        if np.any(high_mask) and ratio_limit < 1:
            numerator = ratio_limit * low_energy
            denominator = (1.0 - ratio_limit) * high_energy
            scale = np.sqrt(np.minimum(1.0, numerator / np.maximum(denominator, 1e-30)))
            spectrum[:, high_mask] *= scale
        projected = np.fft.irfft(spectrum, n=delta.shape[-1], axis=-1)
        amplitude = np.max(np.abs(projected), axis=-1, keepdims=True)
        amplitude_limit = _projection_limit(self.epsilon_mv, self.tolerance)
        scale = np.minimum(1.0, amplitude_limit / np.maximum(amplitude, 1e-30))
        if self.max_slew_mv is not None:
            slew = np.max(np.abs(np.diff(projected, axis=-1)), axis=-1, keepdims=True)
            slew_limit = _projection_limit(self.max_slew_mv, self.tolerance)
            scale = np.minimum(scale, slew_limit / np.maximum(slew, 1e-30))
        return projected * scale

    def _project_torch(self, delta: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(delta, dim=-1)
        frequencies = torch.fft.rfftfreq(
            delta.shape[-1], d=1.0 / self.sample_rate_hz, device=delta.device
        )
        high_mask = frequencies > self.cutoff_hz
        low_energy = spectrum[:, ~high_mask].abs().square().sum(dim=-1, keepdim=True)
        high_energy = spectrum[:, high_mask].abs().square().sum(dim=-1, keepdim=True)
        ratio_limit = self._projection_high_frequency_ratio
        if bool(high_mask.any()) and ratio_limit < 1:
            numerator = ratio_limit * low_energy
            denominator = (1.0 - ratio_limit) * high_energy
            scale = torch.sqrt(
                torch.clamp(numerator / denominator.clamp_min(1e-30), max=1.0)
            )
            spectrum = spectrum.clone()
            spectrum[:, high_mask] = spectrum[:, high_mask] * scale
        projected = torch.fft.irfft(spectrum, n=delta.shape[-1], dim=-1)
        amplitude = projected.abs().amax(dim=-1, keepdim=True)
        amplitude_limit = _projection_limit(self.epsilon_mv, self.tolerance)
        scale = torch.clamp(amplitude_limit / amplitude.clamp_min(1e-30), max=1.0)
        if self.max_slew_mv is not None:
            slew = torch.diff(projected, dim=-1).abs().amax(dim=-1, keepdim=True)
            slew_limit = _projection_limit(self.max_slew_mv, self.tolerance)
            scale = torch.minimum(
                scale, slew_limit / slew.clamp_min(1e-30)
            )
        return projected * scale

    def project(self, clean: Waveform, candidate: Waveform) -> Waveform:
        _validate_pair(clean, candidate)
        delta = candidate - clean
        projected = (
            self._project_torch(delta)
            if isinstance(clean, torch.Tensor)
            else self._project_numpy(delta)
        )
        return clean + projected

    def _statistics(self, clean: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
        clean_array, candidate_array = _numpy_pair(clean, candidate)
        delta = candidate_array - clean_array
        amplitude = float(np.max(np.abs(delta)))
        spectrum = np.fft.rfft(delta, axis=-1)
        frequencies = np.fft.rfftfreq(delta.shape[-1], d=1.0 / self.sample_rate_hz)
        energy = np.abs(spectrum) ** 2
        total = np.sum(energy, axis=-1)
        high = np.sum(energy[:, frequencies > self.cutoff_hz], axis=-1)
        ratio = float(np.max(np.divide(high, total, out=np.zeros_like(high), where=total > 0)))
        slew = float(np.max(np.abs(np.diff(delta, axis=-1))))
        return amplitude, ratio, slew

    def distance(self, clean: np.ndarray, candidate: np.ndarray) -> float:
        amplitude, _, _ = self._statistics(clean, candidate)
        return amplitude

    def violations(self, clean: np.ndarray, candidate: np.ndarray) -> tuple[str, ...]:
        amplitude, ratio, slew = self._statistics(clean, candidate)
        violations: list[str] = []
        if amplitude > self.epsilon_mv + self.tolerance:
            violations.append("amplitude_bound")
        if ratio > self.max_high_frequency_ratio + self.tolerance:
            violations.append("high_frequency_energy")
        if self.max_slew_mv is not None and slew > self.max_slew_mv + self.tolerance:
            violations.append("temporal_slew")
        return tuple(violations)

    def contains(self, clean: np.ndarray, candidate: np.ndarray) -> bool:
        return not self.violations(clean, candidate)

    def sample(
        self,
        clean: Waveform,
        generator: np.random.Generator | torch.Generator | None = None,
    ) -> Waveform:
        _validate_waveform(clean, "clean")
        noise = _normal_like(clean, generator)
        return self.project(clean, clean + noise)


DEFAULT_PHYSIOLOGICAL_BOUNDS: dict[str, float] = {
    "p_amplitude_mv": 0.05,
    "qrs_amplitude_mv": 0.10,
    "t_amplitude_mv": 0.05,
    "p_timing_ms": 12.0,
    "qrs_timing_ms": 8.0,
    "t_timing_ms": 20.0,
    "baseline_drift_mv": 0.05,
    "rr_jitter_ms": 10.0,
}


@dataclass(frozen=True, slots=True)
class _PatientPhysiology:
    """Deterministic beat geometry inferred exclusively from the clean ECG."""

    source: str
    r_peaks: np.ndarray
    local_rr_seconds: np.ndarray
    rr_seconds: float
    p_centers: np.ndarray
    t_centers: np.ndarray
    p_window: np.ndarray
    qrs_window: np.ndarray
    t_window: np.ndarray
    beat_polarity: np.ndarray


@dataclass(frozen=True, slots=True)
class SPhysConstraint:
    """Patient-conditioned morphology, timing, drift, and RR-jitter edit set.

    Consensus R peaks, RR timing, wave windows, and lead morphology are inferred
    deterministically from the clean raw-mV waveform. ``nominal_rr_seconds`` is
    used only when the clean trace has insufficient beat evidence. Timing edits
    use a local first-order shift of that same clean waveform. Projection is
    therefore performed in the declared parameter space rather than by clipping
    individual samples.
    """

    sample_rate_hz: float
    bounds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PHYSIOLOGICAL_BOUNDS)
    )
    nominal_rr_seconds: float = 0.8
    manifold_tolerance_mv: float = 1e-5
    kind: ConstraintKind = field(default=ConstraintKind.S_PHYS, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sample_rate_hz", _positive_finite(self.sample_rate_hz, "sample_rate_hz")
        )
        object.__setattr__(
            self,
            "nominal_rr_seconds",
            _positive_finite(self.nominal_rr_seconds, "nominal_rr_seconds"),
        )
        object.__setattr__(
            self,
            "manifold_tolerance_mv",
            _positive_finite(
                self.manifold_tolerance_mv, "manifold_tolerance_mv", allow_zero=True
            ),
        )
        expected = set(DEFAULT_PHYSIOLOGICAL_BOUNDS)
        if set(self.bounds) != expected:
            raise ValueError(f"Physiological bounds must contain exactly {sorted(expected)}")
        clean_bounds = {
            name: _positive_finite(value, f"bounds[{name}]")
            for name, value in self.bounds.items()
        }
        object.__setattr__(self, "bounds", clean_bounds)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(DEFAULT_PHYSIOLOGICAL_BOUNDS)

    @staticmethod
    def _maximum_gaussian_window(
        samples: int,
        centers: np.ndarray,
        widths_samples: np.ndarray,
    ) -> np.ndarray:
        positions = np.arange(samples, dtype=np.float64)
        window = np.zeros(samples, dtype=np.float64)
        for center, width in zip(centers, widths_samples):
            if center < 0 or center > samples - 1:
                continue
            pulse = np.exp(-0.5 * ((positions - center) / max(float(width), 1.0)) ** 2)
            window = np.maximum(window, pulse)
        return window

    def _consensus_r_peaks(self, clean: np.ndarray) -> np.ndarray:
        """Locate QRS activity after robustly normalizing each non-flat lead."""

        samples = clean.shape[-1]
        if samples < max(8, round(0.4 * self.sample_rate_hz)):
            return np.empty(0, dtype=np.int64)

        centered = clean - np.median(clean, axis=-1, keepdims=True)
        lead_scales = np.quantile(np.abs(centered), 0.95, axis=-1)
        valid = lead_scales > 1e-7
        if not np.any(valid):
            return np.empty(0, dtype=np.int64)
        normalized = centered[valid] / lead_scales[valid, None]

        # The band-pass suppresses P/T waves before lead consensus is formed.
        # A derivative fallback keeps the constraint defined for short or very
        # low-rate inputs for which zero-phase filtering is unavailable.
        nyquist = 0.5 * self.sample_rate_hz
        low_hz = min(5.0, 0.20 * nyquist)
        high_hz = min(25.0, 0.85 * nyquist)
        filtered: np.ndarray
        if low_hz > 0 and high_hz > low_hz:
            try:
                sos = scipy_signal.butter(
                    2,
                    [low_hz, high_hz],
                    btype="bandpass",
                    fs=self.sample_rate_hz,
                    output="sos",
                )
                filtered = scipy_signal.sosfiltfilt(sos, normalized, axis=-1)
            except ValueError:
                filtered = np.gradient(normalized, axis=-1)
        else:
            filtered = np.gradient(normalized, axis=-1)

        envelope = np.sqrt(np.mean(np.square(filtered), axis=0))
        smoothing_samples = max(1, round(0.04 * self.sample_rate_hz))
        if smoothing_samples > 1:
            kernel = np.full(smoothing_samples, 1.0 / smoothing_samples)
            envelope = np.convolve(envelope, kernel, mode="same")
        spread = float(np.quantile(envelope, 0.90) - np.quantile(envelope, 0.25))
        if not np.isfinite(spread) or spread <= 1e-8:
            return np.empty(0, dtype=np.int64)
        peaks, _ = scipy_signal.find_peaks(
            envelope,
            distance=max(1, round(0.28 * self.sample_rate_hz)),
            height=float(np.quantile(envelope, 0.60)),
            prominence=max(0.15 * spread, 1e-8),
        )
        return peaks.astype(np.int64, copy=False)

    def _fallback_r_peaks(self, samples: int, anchor: int | None = None) -> np.ndarray:
        rr_samples = max(1, round(self.nominal_rr_seconds * self.sample_rate_hz))
        if anchor is None:
            anchor = min(samples - 1, round(0.40 * rr_samples))
        peaks = [int(anchor)]
        position = int(anchor) - rr_samples
        while position >= 0:
            peaks.append(position)
            position -= rr_samples
        position = int(anchor) + rr_samples
        while position < samples:
            peaks.append(position)
            position += rr_samples
        return np.asarray(sorted(set(peaks)), dtype=np.int64)

    def _patient_physiology(self, clean: np.ndarray) -> _PatientPhysiology:
        samples = clean.shape[-1]
        detected = self._consensus_r_peaks(clean)
        source = "consensus_r_peaks"
        if detected.size >= 2:
            intervals = np.diff(detected).astype(np.float64) / self.sample_rate_hz
            plausible = intervals[(intervals >= 0.28) & (intervals <= 2.0)]
            if plausible.size:
                rr_seconds = float(np.median(plausible))
                r_peaks = detected
            else:
                rr_seconds = self.nominal_rr_seconds
                r_peaks = self._fallback_r_peaks(samples, int(detected[0]))
                source = "nominal_rr_fallback"
        elif detected.size == 1:
            rr_seconds = self.nominal_rr_seconds
            r_peaks = self._fallback_r_peaks(samples, int(detected[0]))
            source = "detected_phase_nominal_rr_fallback"
        else:
            rr_seconds = self.nominal_rr_seconds
            r_peaks = self._fallback_r_peaks(samples)
            source = "nominal_rr_fallback"

        r_seconds = r_peaks.astype(np.float64) / self.sample_rate_hz
        if r_peaks.size == 1:
            local_rr = np.asarray([rr_seconds], dtype=np.float64)
        else:
            intervals = np.diff(r_seconds)
            local_rr = np.empty(r_peaks.size, dtype=np.float64)
            local_rr[0] = intervals[0]
            local_rr[-1] = intervals[-1]
            if r_peaks.size > 2:
                local_rr[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])
            local_rr = np.clip(local_rr, 0.28, 2.0)

        p_offsets = np.minimum(0.20, 0.25 * local_rr) * self.sample_rate_hz
        t_offsets = np.minimum(0.32, 0.40 * local_rr) * self.sample_rate_hz
        p_centers = r_peaks.astype(np.float64) - p_offsets
        t_centers = r_peaks.astype(np.float64) + t_offsets
        p_widths = np.minimum(0.05, 0.07 * local_rr) * self.sample_rate_hz
        qrs_widths = np.minimum(0.035, 0.045 * local_rr) * self.sample_rate_hz
        t_widths = np.minimum(0.10, 0.13 * local_rr) * self.sample_rate_hz

        positions = np.arange(samples, dtype=np.float64)
        closest_beat = np.argmin(
            np.abs(positions[:, None] - r_peaks.astype(np.float64)[None, :]), axis=1
        )
        beat_polarity = np.where(closest_beat % 2 == 0, 1.0, -1.0)
        return _PatientPhysiology(
            source=source,
            r_peaks=r_peaks,
            local_rr_seconds=local_rr,
            rr_seconds=rr_seconds,
            p_centers=p_centers,
            t_centers=t_centers,
            p_window=self._maximum_gaussian_window(samples, p_centers, p_widths),
            qrs_window=self._maximum_gaussian_window(
                samples, r_peaks.astype(np.float64), qrs_widths
            ),
            t_window=self._maximum_gaussian_window(samples, t_centers, t_widths),
            beat_polarity=beat_polarity,
        )

    @staticmethod
    def _amplitude_basis(centered: np.ndarray, window: np.ndarray) -> np.ndarray:
        localized = centered * window[None, :]
        scale = float(np.max(np.abs(localized)))
        if scale > 1e-8:
            return localized / scale
        return np.ones((centered.shape[0], 1), dtype=np.float64) * window

    def patient_profile(self, clean: Waveform) -> dict[str, Any]:
        """Return serializable clean-derived geometry used by the S-phys basis."""

        _validate_waveform(clean, "clean")
        clean_array = (
            clean.detach().cpu().numpy() if isinstance(clean, torch.Tensor) else np.asarray(clean)
        ).astype(np.float64, copy=False)
        profile = self._patient_physiology(clean_array)
        centered = clean_array - np.median(clean_array, axis=-1, keepdims=True)
        amplitude_bases = (
            self._amplitude_basis(centered, profile.p_window),
            self._amplitude_basis(centered, profile.qrs_window),
            self._amplitude_basis(centered, profile.t_window),
        )
        lead_weights = {
            wave: tuple(float(np.max(np.abs(basis[lead]))) for lead in range(clean.shape[0]))
            for wave, basis in zip(("p", "qrs", "t"), amplitude_bases)
        }
        return {
            "source": profile.source,
            "r_peak_indices": tuple(int(value) for value in profile.r_peaks),
            "rr_seconds": float(profile.rr_seconds),
            "p_center_indices": tuple(float(value) for value in profile.p_centers),
            "t_center_indices": tuple(float(value) for value in profile.t_centers),
            "lead_amplitude_weights": lead_weights,
        }

    def _numpy_basis(self, clean: np.ndarray) -> np.ndarray:
        clean_array = np.asarray(clean, dtype=np.float64)
        samples = clean_array.shape[-1]
        time = np.arange(samples, dtype=np.float64) / self.sample_rate_hz
        profile = self._patient_physiology(clean_array)
        centered = clean_array - np.median(clean_array, axis=-1, keepdims=True)
        derivative = np.gradient(clean_array, 1.0 / self.sample_rate_hz, axis=-1)
        lead_scale = np.ones((clean_array.shape[0], 1), dtype=np.float64)
        bases = [
            self._amplitude_basis(centered, profile.p_window),
            self._amplitude_basis(centered, profile.qrs_window),
            self._amplitude_basis(centered, profile.t_window),
            -derivative * profile.p_window * 0.001,
            -derivative * profile.qrs_window * 0.001,
            -derivative * profile.t_window * 0.001,
            lead_scale * np.sin(2.0 * np.pi * 0.33 * time),
            -derivative * profile.beat_polarity * 0.001,
        ]
        return np.stack(bases, axis=0)

    def _torch_basis(self, clean: torch.Tensor) -> torch.Tensor:
        # NumPy is the canonical clean-signal detector for both backends. This
        # prevents backend-specific peak choices from invalidating a float32
        # waveform when the independent membership oracle replays it.
        canonical = self._numpy_basis(
            clean.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        return torch.as_tensor(canonical, dtype=clean.dtype, device=clean.device)

    def project_parameters(
        self, parameters: dict[str, float] | np.ndarray | torch.Tensor
    ) -> dict[str, float] | np.ndarray | torch.Tensor:
        limits = np.asarray([self.bounds[name] for name in self.parameter_names], dtype=float)
        # Keep boundary parameters stable when the rendered raw-mV waveform is
        # serialized to float32 and independently fitted again by ``contains``.
        projected_limits = np.asarray(
            [_projection_limit(limit, limit * 1e-5) for limit in limits],
            dtype=float,
        )
        if isinstance(parameters, dict):
            unknown = set(parameters) - set(self.parameter_names)
            if unknown:
                raise ValueError(f"Unknown physiological parameters: {sorted(unknown)}")
            return {
                name: float(np.clip(float(parameters.get(name, 0.0)), -limit, limit))
                for name, limit in zip(self.parameter_names, projected_limits)
            }
        if isinstance(parameters, torch.Tensor):
            if parameters.shape != (len(self.parameter_names),):
                raise ValueError("Physiological parameter vector has the wrong shape")
            bound = torch.as_tensor(
                projected_limits,
                dtype=parameters.dtype,
                device=parameters.device,
            )
            return torch.maximum(torch.minimum(parameters, bound), -bound)
        array = np.asarray(parameters, dtype=float)
        if array.shape != (len(self.parameter_names),):
            raise ValueError("Physiological parameter vector has the wrong shape")
        return np.clip(array, -projected_limits, projected_limits)

    def synthesize(
        self,
        clean: Waveform,
        parameters: dict[str, float] | np.ndarray | torch.Tensor,
    ) -> Waveform:
        _validate_waveform(clean, "clean")
        if isinstance(parameters, dict):
            values = [float(parameters.get(name, 0.0)) for name in self.parameter_names]
            parameters = (
                torch.as_tensor(values, dtype=clean.dtype, device=clean.device)
                if isinstance(clean, torch.Tensor)
                else np.asarray(values, dtype=float)
            )
        projected = self.project_parameters(parameters)
        if isinstance(clean, torch.Tensor):
            if not isinstance(projected, torch.Tensor):
                projected = torch.as_tensor(projected, dtype=clean.dtype, device=clean.device)
            basis = self._torch_basis(clean)
            return clean + torch.einsum("p,pls->ls", projected, basis)
        if isinstance(projected, torch.Tensor):
            projected = projected.detach().cpu().numpy()
        basis = self._numpy_basis(np.asarray(clean, dtype=float))
        result = np.asarray(clean, dtype=float) + np.einsum("p,pls->ls", projected, basis)
        return result.astype(clean.dtype, copy=False)

    def _infer_numpy(self, clean: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, float]:
        clean_array, candidate_array = _numpy_pair(clean, candidate)
        basis = self._numpy_basis(clean_array)
        matrix = basis.reshape(len(self.parameter_names), -1).T
        parameters = np.linalg.lstsq(matrix, (candidate_array - clean_array).reshape(-1), rcond=None)[0]
        reconstruction = np.einsum("p,pls->ls", parameters, basis)
        residual = float(np.max(np.abs((candidate_array - clean_array) - reconstruction)))
        return parameters, residual

    def parameters(self, clean: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
        parameters, _ = self._infer_numpy(clean, candidate)
        return {name: float(value) for name, value in zip(self.parameter_names, parameters)}

    def project(self, clean: Waveform, candidate: Waveform) -> Waveform:
        _validate_pair(clean, candidate)
        if isinstance(clean, torch.Tensor):
            basis = self._torch_basis(clean)
            matrix = basis.reshape(len(self.parameter_names), -1).T
            delta = (candidate - clean).reshape(-1)
            parameters = torch.linalg.lstsq(matrix, delta).solution
            parameters = self.project_parameters(parameters)
            return clean + torch.einsum("p,pls->ls", parameters, basis)
        parameters, _ = self._infer_numpy(clean, candidate)
        return self.synthesize(clean, parameters)

    def parameter_distance(self, clean: np.ndarray, candidate: np.ndarray) -> float:
        parameters, _ = self._infer_numpy(clean, candidate)
        limits = np.asarray([self.bounds[name] for name in self.parameter_names])
        return float(np.max(np.abs(parameters) / limits))

    def distance(self, clean: np.ndarray, candidate: np.ndarray) -> float:
        return self.parameter_distance(clean, candidate)

    def violations(self, clean: np.ndarray, candidate: np.ndarray) -> tuple[str, ...]:
        parameters, residual = self._infer_numpy(clean, candidate)
        limits = np.asarray([self.bounds[name] for name in self.parameter_names])
        violations: list[str] = []
        parameter_tolerance = np.maximum(1e-7, limits * 1e-6)
        if np.any(np.abs(parameters) > limits + parameter_tolerance):
            violations.append("parameter_bounds")
        if residual > self.manifold_tolerance_mv:
            violations.append("off_physiological_parameter_manifold")
        return tuple(violations)

    def contains(self, clean: np.ndarray, candidate: np.ndarray) -> bool:
        return not self.violations(clean, candidate)

    def sample(
        self,
        clean: Waveform,
        generator: np.random.Generator | torch.Generator | None = None,
    ) -> Waveform:
        _validate_waveform(clean, "clean")
        count = len(self.parameter_names)
        if isinstance(clean, torch.Tensor):
            if generator is not None and not isinstance(generator, torch.Generator):
                raise TypeError("A tensor waveform requires a torch.Generator")
            unit = torch.rand(
                count, dtype=clean.dtype, device=clean.device, generator=generator
            ) * 2.0 - 1.0
            limits = torch.as_tensor(
                [self.bounds[name] for name in self.parameter_names],
                dtype=clean.dtype,
                device=clean.device,
            )
            return self.synthesize(clean, unit * limits)
        if generator is not None and not isinstance(generator, np.random.Generator):
            raise TypeError("A NumPy waveform requires a numpy.random.Generator")
        rng = generator or np.random.default_rng()
        limits = np.asarray([self.bounds[name] for name in self.parameter_names])
        return self.synthesize(clean, rng.uniform(-limits, limits))


LInfConstraint = SInfConstraint
SInfinityConstraint = SInfConstraint
L2Constraint = S2Constraint
SpectralConstraint = SSpecConstraint
PhysiologicalConstraint = SPhysConstraint


def make_constraint(kind: ConstraintKind | str, **kwargs: Any) -> WaveformConstraint:
    resolved = ConstraintKind(kind)
    implementations = {
        ConstraintKind.S_INF: SInfConstraint,
        ConstraintKind.S2: S2Constraint,
        ConstraintKind.S_SPEC: SSpecConstraint,
        ConstraintKind.S_PHYS: SPhysConstraint,
    }
    return implementations[resolved](**kwargs)


def scale_to_constraint_distance(
    constraint: WaveformConstraint,
    clean: Waveform,
    candidate: Waveform,
    distance: float,
) -> Waveform:
    """Scale a native constraint draw to a requested native distance.

    The distance unit is defined by ``constraint.distance``. The final project
    caps an out-of-range request at the constraint boundary while preserving the
    input array backend.
    """

    target = _positive_finite(distance, "distance", allow_zero=True)
    _validate_pair(clean, candidate)
    projected = constraint.project(clean, candidate)
    current = float(constraint.distance(*_numpy_pair(clean, projected)))
    if target == 0.0 or current <= 1e-30:
        return clean.clone() if isinstance(clean, torch.Tensor) else np.array(clean, copy=True)
    scaled = clean + (projected - clean) * (target / current)
    return constraint.project(clean, scaled)
