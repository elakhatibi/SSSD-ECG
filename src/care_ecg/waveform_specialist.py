"""Deterministic waveform/context features for a locked HYP specialist."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ensemble import CARE_LABELS, normalize_probability_rows

LEVEL_STATISTICS = (
    "mean",
    "std",
    "min",
    "max",
    "peak_to_peak",
    "rms",
    "median",
    "quantile_10",
    "quantile_25",
    "quantile_75",
    "quantile_90",
)
DIFFERENCE_STATISTICS = (
    "difference_mean",
    "difference_std",
    "difference_rms",
    "difference_max_abs",
)
FFT_BANDS_HZ = (
    (0.0, 0.5),
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 15.0),
    (15.0, 40.0),
    (40.0, 50.0),
)
DEMOGRAPHIC_COLUMNS = ("age", "sex", "height", "weight")
DEFAULT_HYP_INDEX = CARE_LABELS.index("HYP")


def waveform_feature_names(lead_count: int = 12) -> tuple[str, ...]:
    names: list[str] = []
    statistics = LEVEL_STATISTICS + DIFFERENCE_STATISTICS
    statistics += tuple(f"fft_fraction_{low:g}_{high:g}_hz" for low, high in FFT_BANDS_HZ)
    for statistic in statistics:
        names.extend(f"{statistic}:lead_{lead + 1}" for lead in range(lead_count))
    return tuple(names)


def extract_waveform_features(
    signals: np.ndarray,
    *,
    sampling_rate_hz: float = 100.0,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract 21 fixed statistics per lead without materializing a full FFT."""
    values = np.asarray(signals)
    if values.ndim != 3 or values.shape[1] != 12 or values.shape[2] < 2:
        raise ValueError("Signals must have shape (records, 12, samples>=2)")
    if not np.isfinite(values).all():
        raise ValueError("Signals must contain only finite values")
    if sampling_rate_hz <= 0.0 or batch_size <= 0:
        raise ValueError("Sampling rate and batch size must be positive")

    frequencies = np.fft.rfftfreq(values.shape[2], d=1.0 / sampling_rate_hz)
    nyquist = sampling_rate_hz / 2.0
    if nyquist < FFT_BANDS_HZ[-1][1]:
        raise ValueError("Sampling rate must provide a Nyquist frequency of at least 50 Hz")
    non_dc = frequencies > 0.0
    band_masks = [
        non_dc
        & (frequencies >= low)
        & (frequencies < high if high < nyquist else frequencies <= high)
        for low, high in FFT_BANDS_HZ
    ]
    output = np.empty(
        (len(values), len(waveform_feature_names(values.shape[1]))),
        dtype=np.float32,
    )
    epsilon = np.finfo(np.float64).eps

    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch = np.asarray(values[start:stop], dtype=np.float64)
        differences = np.diff(batch, axis=2)
        quantiles = np.quantile(batch, (0.1, 0.25, 0.75, 0.9), axis=2)
        parts = [
            batch.mean(axis=2),
            batch.std(axis=2),
            batch.min(axis=2),
            batch.max(axis=2),
            np.ptp(batch, axis=2),
            np.sqrt(np.mean(np.square(batch), axis=2)),
            np.median(batch, axis=2),
            quantiles[0],
            quantiles[1],
            quantiles[2],
            quantiles[3],
            differences.mean(axis=2),
            differences.std(axis=2),
            np.sqrt(np.mean(np.square(differences), axis=2)),
            np.max(np.abs(differences), axis=2),
        ]
        spectrum = np.fft.rfft(batch, axis=2)
        power = np.square(spectrum.real) + np.square(spectrum.imag)
        total_non_dc = power[:, :, non_dc].sum(axis=2) + epsilon
        parts.extend(power[:, :, mask].sum(axis=2) / total_non_dc for mask in band_masks)
        output[start:stop] = np.concatenate(parts, axis=1).astype(np.float32)
    return output


def _parse_sex(value: str) -> float:
    normalized = value.strip().lower()
    if normalized in {"f", "female"}:
        return 0.0
    if normalized in {"m", "male"}:
        return 1.0
    return float(normalized) if normalized else float("nan")


def read_demographics(database_path: str | Path, record_ids: np.ndarray) -> np.ndarray:
    """Read only inference-safe demographic columns under exact record alignment."""
    identifiers = np.asarray(record_ids, dtype=str)
    if identifiers.ndim != 1 or len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("Record IDs must be a one-dimensional unique sequence")
    rows_by_id: dict[str, tuple[float, ...]] = {}
    with Path(database_path).open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"ecg_id", *DEMOGRAPHIC_COLUMNS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Demographic database must contain {sorted(required)}")
        for row in reader:
            identifier = str(row["ecg_id"])
            values: list[float] = []
            for column in DEMOGRAPHIC_COLUMNS:
                raw = row[column]
                try:
                    value = _parse_sex(raw) if column == "sex" else float(raw)
                except (TypeError, ValueError):
                    value = float("nan")
                values.append(value)
            rows_by_id[identifier] = tuple(values)
    missing = [identifier for identifier in identifiers if identifier not in rows_by_id]
    if missing:
        raise ValueError(f"Demographics are missing records: {missing[:5]}")
    return np.asarray([rows_by_id[identifier] for identifier in identifiers], dtype=np.float64)


def load_inference_inputs(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only inference inputs, leaving any target member unopened."""
    with np.load(path, allow_pickle=False) as archive:
        signals = np.asarray(archive["signals"], dtype=np.float32)
        record_ids = np.asarray(archive["record_ids"], dtype=str)
    if signals.shape[0] != len(record_ids):
        raise ValueError("Dataset signals and record IDs are not aligned")
    return signals, record_ids


def fit_demographic_medians(raw_demographics: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_demographics, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(DEMOGRAPHIC_COLUMNS):
        raise ValueError("Raw demographics have an incompatible shape")
    medians = np.nanmedian(values, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("Every demographic column must have at least one finite value")
    return medians


def encode_demographics(
    raw_demographics: np.ndarray,
    medians: np.ndarray,
) -> np.ndarray:
    values = np.asarray(raw_demographics, dtype=np.float64)
    fill = np.asarray(medians, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(DEMOGRAPHIC_COLUMNS):
        raise ValueError("Raw demographics have an incompatible shape")
    if fill.shape != (len(DEMOGRAPHIC_COLUMNS),) or not np.isfinite(fill).all():
        raise ValueError("Demographic medians have an incompatible value or shape")
    missing = ~np.isfinite(values)
    imputed = np.where(missing, fill[None, :], values)
    return np.concatenate((imputed, missing.astype(np.float64)), axis=1).astype(np.float32)


def specialist_feature_names() -> tuple[str, ...]:
    demographic = tuple(f"demographic:{name}" for name in DEMOGRAPHIC_COLUMNS)
    missing = tuple(f"demographic_missing:{name}" for name in DEMOGRAPHIC_COLUMNS)
    return waveform_feature_names() + demographic + missing


def build_specialist_features(
    signals: np.ndarray,
    raw_demographics: np.ndarray,
    medians: np.ndarray,
    *,
    sampling_rate_hz: float = 100.0,
    batch_size: int = 256,
) -> np.ndarray:
    waveform = extract_waveform_features(
        signals,
        sampling_rate_hz=sampling_rate_hz,
        batch_size=batch_size,
    )
    demographics = encode_demographics(raw_demographics, medians)
    if len(waveform) != len(demographics):
        raise ValueError("Waveform and demographic rows are not aligned")
    features = np.concatenate((waveform, demographics), axis=1)
    if features.shape[1] != len(specialist_feature_names()):
        raise AssertionError("Specialist feature contract has an unexpected width")
    return features


def fit_hyp_specialist(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_weight_power: float = 0.5,
    random_state: int = 20261021,
) -> Any:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Install CARE-ECG with the 'ensemble' extra") from error

    matrix = np.asarray(features, dtype=np.float32)
    targets = (np.asarray(labels, dtype=np.int64) == DEFAULT_HYP_INDEX).astype(np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != len(targets):
        raise ValueError("Specialist features and labels are not aligned")
    counts = np.bincount(targets, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both HYP and non-HYP records are required")
    class_weights = (len(targets) / (2.0 * counts)) ** class_weight_power
    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.07,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=random_state,
    )
    model.fit(matrix, targets, sample_weight=class_weights[targets])
    return model


def fit_ovr_specialists(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_weight_power: float = 0.5,
    random_state: int = 20261021,
) -> tuple[Any, ...]:
    """Fit one deterministic binary waveform specialist per CARE class."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Install CARE-ECG with the 'ensemble' extra") from error

    matrix = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2 or targets.shape != (len(matrix),):
        raise ValueError("Specialist features and labels are not aligned")
    if np.any((targets < 0) | (targets >= len(CARE_LABELS))):
        raise ValueError("Specialist labels are outside the CARE label order")

    models: list[Any] = []
    for class_index in range(len(CARE_LABELS)):
        binary_targets = (targets == class_index).astype(np.int64)
        counts = np.bincount(binary_targets, minlength=2)
        if np.any(counts == 0):
            raise ValueError(f"Class {CARE_LABELS[class_index]} lacks a binary outcome")
        class_weights = (len(targets) / (2.0 * counts)) ** class_weight_power
        model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.07,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=3.0,
            early_stopping=False,
            random_state=random_state,
        )
        model.fit(
            matrix,
            binary_targets,
            sample_weight=class_weights[binary_targets],
        )
        models.append(model)
    return tuple(models)


def fit_alternate_hyp_specialist(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_weight_power: float,
    random_state: int = 20261021,
) -> Any:
    """Fit the predeclared lower-regularization HYP validation candidate."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Install CARE-ECG with the 'ensemble' extra") from error

    matrix = np.asarray(features, dtype=np.float32)
    targets = (np.asarray(labels, dtype=np.int64) == DEFAULT_HYP_INDEX).astype(np.int64)
    if matrix.ndim != 2 or targets.shape != (len(matrix),):
        raise ValueError("Specialist features and labels are not aligned")
    counts = np.bincount(targets, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both HYP and non-HYP records are required")
    class_weights = (len(targets) / (2.0 * counts)) ** class_weight_power
    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.2,
        early_stopping=True,
        n_iter_no_change=25,
        random_state=random_state,
    )
    model.fit(matrix, targets, sample_weight=class_weights[targets])
    return model


def predict_hyp_probability(model: Any, features: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_, dtype=np.int64)
    if not np.array_equal(classes, np.asarray((0, 1), dtype=np.int64)):
        raise ValueError(f"HYP specialist has unexpected classes: {classes.tolist()}")
    probabilities = np.asarray(
        model.predict_proba(np.asarray(features, dtype=np.float32)),
        dtype=np.float64,
    )[:, 1]
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("HYP specialist emitted invalid probabilities")
    return probabilities


def predict_ovr_probabilities(models: tuple[Any, ...], features: np.ndarray) -> np.ndarray:
    if len(models) != len(CARE_LABELS):
        raise ValueError(f"Expected {len(CARE_LABELS)} one-vs-rest specialists")
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Specialist features must be a matrix")
    columns: list[np.ndarray] = []
    for class_index, model in enumerate(models):
        classes = np.asarray(model.classes_, dtype=np.int64)
        if not np.array_equal(classes, np.asarray((0, 1), dtype=np.int64)):
            raise ValueError(
                f"{CARE_LABELS[class_index]} specialist has unexpected classes: "
                f"{classes.tolist()}"
            )
        columns.append(np.asarray(model.predict_proba(matrix), dtype=np.float64)[:, 1])
    probabilities = np.stack(columns, axis=1)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("One-vs-rest specialists emitted invalid probabilities")
    return probabilities


def specialist_logits(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CARE_LABELS):
        raise ValueError("Specialist probabilities have an incompatible shape")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Specialist probabilities must be finite and in [0, 1]")
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def apply_multiclass_logit_adjustment(
    baseline_probabilities: np.ndarray,
    specialist_probabilities: np.ndarray,
    *,
    beta: float,
    biases: np.ndarray,
    specialist_logit_center: np.ndarray,
    specialist_logit_scale: np.ndarray,
) -> np.ndarray:
    baseline = normalize_probability_rows(
        baseline_probabilities,
        expected_columns=len(CARE_LABELS),
        name="baseline probabilities",
    )
    evidence = specialist_logits(specialist_probabilities)
    center = np.asarray(specialist_logit_center, dtype=np.float64)
    scale = np.asarray(specialist_logit_scale, dtype=np.float64)
    offsets = np.asarray(biases, dtype=np.float64)
    expected = (len(CARE_LABELS),)
    if evidence.shape[0] != len(baseline):
        raise ValueError("Baseline and specialist probability rows are not aligned")
    if center.shape != expected or scale.shape != expected or offsets.shape != expected:
        raise ValueError("Multiclass specialist policy arrays have incompatible shapes")
    if not all(np.isfinite(values).all() for values in (center, scale, offsets)):
        raise ValueError("Multiclass specialist policy arrays must be finite")
    if np.any(scale <= 0.0) or not np.isfinite(beta):
        raise ValueError("Specialist scales and beta must be valid")
    logits = np.log(np.clip(baseline, 1e-8, 1.0))
    logits += beta * ((evidence - center[None, :]) / scale[None, :]) + offsets
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def apply_hyp_logit_adjustment(
    baseline_probabilities: np.ndarray,
    hyp_probability: np.ndarray,
    *,
    beta: float,
    bias: float,
) -> np.ndarray:
    baseline = normalize_probability_rows(
        baseline_probabilities,
        expected_columns=len(CARE_LABELS),
        name="baseline probabilities",
    )
    specialist = np.asarray(hyp_probability, dtype=np.float64)
    if specialist.shape != (len(baseline),) or not np.isfinite(specialist).all():
        raise ValueError("HYP probabilities must be a finite aligned vector")
    epsilon = 1e-6
    specialist = np.clip(specialist, epsilon, 1.0 - epsilon)
    logits = np.log(np.clip(baseline, 1e-8, 1.0))
    logits[:, DEFAULT_HYP_INDEX] += beta * np.log(specialist / (1.0 - specialist)) + bias
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class HypSpecialistPolicy:
    beta: float
    bias: float
    demographic_medians: np.ndarray
    model_sha256: str
    sampling_rate_hz: float = 100.0
    class_weight_power: float = 0.5
    random_state: int = 20261021

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            beta=np.asarray(self.beta, dtype=np.float64),
            bias=np.asarray(self.bias, dtype=np.float64),
            demographic_medians=np.asarray(self.demographic_medians, dtype=np.float64),
            model_sha256=np.asarray(self.model_sha256),
            sampling_rate_hz=np.asarray(self.sampling_rate_hz, dtype=np.float64),
            class_weight_power=np.asarray(self.class_weight_power, dtype=np.float64),
            random_state=np.asarray(self.random_state, dtype=np.int64),
            feature_names=np.asarray(specialist_feature_names()),
            label_order=np.asarray(CARE_LABELS),
            target_label=np.asarray("HYP"),
            adjustment=np.asarray("log(p_base) + beta*logit(p_hyp) + bias"),
        )

    @classmethod
    def load(cls, path: str | Path) -> HypSpecialistPolicy:
        with np.load(path, allow_pickle=False) as archive:
            expected_features = np.asarray(specialist_feature_names())
            if not np.array_equal(archive["feature_names"], expected_features):
                raise ValueError("HYP specialist feature contract does not match")
            if tuple(np.asarray(archive["label_order"], dtype=str)) != CARE_LABELS:
                raise ValueError("HYP specialist label order does not match")
            if str(archive["target_label"].item()) != "HYP":
                raise ValueError("HYP specialist target does not match")
            return cls(
                beta=float(archive["beta"].item()),
                bias=float(archive["bias"].item()),
                demographic_medians=np.asarray(
                    archive["demographic_medians"], dtype=np.float64
                ),
                model_sha256=str(archive["model_sha256"].item()),
                sampling_rate_hz=float(archive["sampling_rate_hz"].item()),
                class_weight_power=float(archive["class_weight_power"].item()),
                random_state=int(archive["random_state"].item()),
            )


@dataclass(frozen=True)
class MulticlassSpecialistPolicy:
    beta: float
    biases: np.ndarray
    specialist_logit_center: np.ndarray
    specialist_logit_scale: np.ndarray
    demographic_medians: np.ndarray
    model_sha256: str
    transform: str
    sampling_rate_hz: float = 100.0
    class_weight_power: float = 0.5
    random_state: int = 20261021

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            beta=np.asarray(self.beta, dtype=np.float64),
            biases=np.asarray(self.biases, dtype=np.float64),
            specialist_logit_center=np.asarray(
                self.specialist_logit_center, dtype=np.float64
            ),
            specialist_logit_scale=np.asarray(
                self.specialist_logit_scale, dtype=np.float64
            ),
            demographic_medians=np.asarray(self.demographic_medians, dtype=np.float64),
            model_sha256=np.asarray(self.model_sha256),
            transform=np.asarray(self.transform),
            sampling_rate_hz=np.asarray(self.sampling_rate_hz, dtype=np.float64),
            class_weight_power=np.asarray(self.class_weight_power, dtype=np.float64),
            random_state=np.asarray(self.random_state, dtype=np.int64),
            feature_names=np.asarray(specialist_feature_names()),
            label_order=np.asarray(CARE_LABELS),
            adjustment=np.asarray(
                "log(p_base) + beta*((logit(p_ovr)-center)/scale) + biases"
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> MulticlassSpecialistPolicy:
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(
                archive["feature_names"], np.asarray(specialist_feature_names())
            ):
                raise ValueError("Multiclass specialist feature contract does not match")
            if tuple(np.asarray(archive["label_order"], dtype=str)) != CARE_LABELS:
                raise ValueError("Multiclass specialist label order does not match")
            policy = cls(
                beta=float(archive["beta"].item()),
                biases=np.asarray(archive["biases"], dtype=np.float64),
                specialist_logit_center=np.asarray(
                    archive["specialist_logit_center"], dtype=np.float64
                ),
                specialist_logit_scale=np.asarray(
                    archive["specialist_logit_scale"], dtype=np.float64
                ),
                demographic_medians=np.asarray(
                    archive["demographic_medians"], dtype=np.float64
                ),
                model_sha256=str(archive["model_sha256"].item()),
                transform=str(archive["transform"].item()),
                sampling_rate_hz=float(archive["sampling_rate_hz"].item()),
                class_weight_power=float(archive["class_weight_power"].item()),
                random_state=int(archive["random_state"].item()),
            )
        expected = (len(CARE_LABELS),)
        if (
            policy.biases.shape != expected
            or policy.specialist_logit_center.shape != expected
            or policy.specialist_logit_scale.shape != expected
        ):
            raise ValueError("Multiclass specialist policy arrays have incompatible shapes")
        if np.any(policy.specialist_logit_scale <= 0.0):
            raise ValueError("Multiclass specialist logit scales must be positive")
        return policy
