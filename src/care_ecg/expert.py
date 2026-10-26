from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_PROBABILITY_ATOL = 1e-6
_PROBABILITY_RTOL = 1e-5


def _as_string_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return array.astype(str)


def validate_probability_vector(
    probabilities: Sequence[float] | np.ndarray,
    num_classes: int,
    name: str,
) -> np.ndarray:
    """Validate a single categorical distribution without silently repairing it."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (num_classes,):
        raise ValueError(f"{name} must have shape ({num_classes},)")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(values < 0):
        raise ValueError(f"{name} must be nonnegative")
    total = float(values.sum())
    if not np.isfinite(total) or not np.isclose(
        total,
        1.0,
        rtol=_PROBABILITY_RTOL,
        atol=_PROBABILITY_ATOL,
    ):
        raise ValueError(f"{name} must have probability mass equal to 1 (received {total:.8g})")
    return values / total


def validate_probability_matrix(
    probabilities: np.ndarray,
    rows: int,
    columns: int,
    name: str,
) -> np.ndarray:
    """Validate a matrix of categorical distributions row by row."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (rows, columns):
        raise ValueError(f"{name} must have shape ({rows}, {columns})")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(values < 0):
        raise ValueError(f"{name} must be nonnegative")
    totals = values.sum(axis=1)
    valid_mass = np.isclose(
        totals,
        1.0,
        rtol=_PROBABILITY_RTOL,
        atol=_PROBABILITY_ATOL,
    )
    if not np.all(valid_mass):
        index = int(np.flatnonzero(~valid_mass)[0])
        raise ValueError(
            f"{name} row {index} must have probability mass equal to 1 "
            f"(received {totals[index]:.8g})"
        )
    return values / totals[:, None]


def blend_external_expert(
    internal_probabilities: Sequence[float] | np.ndarray | None,
    expert_probabilities: Sequence[float] | np.ndarray | None,
    expert_weight: float | None,
    expert_source: str | None,
    label_names: Sequence[str],
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Blend a declared external expert into the neural stream before final fusion."""
    supplied = (
        expert_probabilities is not None,
        expert_weight is not None,
        expert_source is not None,
    )
    if not any(supplied):
        return (
            None
            if internal_probabilities is None
            else np.asarray(internal_probabilities, dtype=np.float64),
            None,
        )
    if not all(supplied):
        raise ValueError(
            "External expert probabilities, weight, and source must be supplied together"
        )
    if internal_probabilities is None:
        raise ValueError("External expert blending requires internal neural probabilities")

    if not isinstance(expert_source, str) or not expert_source.strip():
        raise ValueError("External expert source must be a non-empty string")
    source = expert_source.strip()
    try:
        weight = float(expert_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("External expert weight must be a finite number in [0, 1]") from exc
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("External expert weight must be a finite number in [0, 1]")

    class_count = len(label_names)
    internal = validate_probability_vector(
        internal_probabilities,
        class_count,
        "Internal neural probabilities",
    )
    expert = validate_probability_vector(
        expert_probabilities,
        class_count,
        "External expert probabilities",
    )
    blended = (1.0 - weight) * internal + weight * expert
    blended /= blended.sum()

    def labeled(values: np.ndarray) -> dict[str, float]:
        return {label: float(values[index]) for index, label in enumerate(label_names)}

    audit = {
        "source": source,
        "weight": weight,
        "weight_selection": "operator_supplied; select and lock on validation data",
        "stage": "before causal/neural fusion and class-bias calibration",
        "label_order": list(label_names),
        "internal_neural_probabilities": labeled(internal),
        "expert_probabilities": labeled(expert),
        "blended_neural_probabilities": labeled(blended),
    }
    return blended, audit


def load_aligned_expert_probabilities(
    path: str | Path,
    dataset_record_ids: Sequence[str] | np.ndarray | None,
    trained_label_names: Sequence[str],
) -> np.ndarray:
    """Load the strict expert NPZ contract and align columns to trained labels."""
    if dataset_record_ids is None:
        raise ValueError("External expert use requires record_ids in the prediction dataset")
    dataset_ids = _as_string_vector(np.asarray(dataset_record_ids), "Dataset record_ids")
    if len(set(dataset_ids.tolist())) != len(dataset_ids):
        raise ValueError("Prediction dataset record_ids must be unique for external expert use")

    required = {"probabilities", "record_ids", "label_order"}
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"External expert NPZ is missing required array(s): {names}")
        probabilities = np.asarray(data["probabilities"], dtype=np.float64)
        expert_ids = _as_string_vector(data["record_ids"], "External expert record_ids")
        expert_labels = _as_string_vector(data["label_order"], "External expert label_order")

    if len(expert_ids) != len(dataset_ids):
        raise ValueError(
            "External expert record_ids length does not match the prediction dataset "
            f"({len(expert_ids)} != {len(dataset_ids)})"
        )
    if len(set(expert_ids.tolist())) != len(expert_ids):
        raise ValueError("External expert record_ids must be unique")
    if not np.array_equal(expert_ids, dataset_ids):
        mismatches = np.flatnonzero(expert_ids != dataset_ids)
        index = int(mismatches[0])
        raise ValueError(
            "External expert record_ids are not exactly aligned with the prediction dataset; "
            f"first mismatch at row {index}: {expert_ids[index]!r} != {dataset_ids[index]!r}"
        )

    trained_labels = [str(label) for label in trained_label_names]
    if not trained_labels or len(set(trained_labels)) != len(trained_labels):
        raise ValueError("Trained label names must be non-empty and unique")
    labels = expert_labels.tolist()
    if len(set(labels)) != len(labels):
        raise ValueError("External expert label_order must contain unique labels")
    missing_labels = sorted(set(trained_labels) - set(labels))
    extra_labels = sorted(set(labels) - set(trained_labels))
    if missing_labels or extra_labels:
        raise ValueError(
            "External expert label_order must exactly match trained labels; "
            f"missing={missing_labels}, extra={extra_labels}"
        )

    validated = validate_probability_matrix(
        probabilities,
        len(dataset_ids),
        len(labels),
        "External expert probabilities",
    )
    label_to_column = {label: index for index, label in enumerate(labels)}
    return validated[:, [label_to_column[label] for label in trained_labels]]
