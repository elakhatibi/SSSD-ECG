"""Validated adapters and meta-models for reproducible ECG ensembles.

The helpers in this module keep row order, class order, and artifact provenance
explicit.  Legacy upstream prediction files are authenticated before pickle is
allowed to deserialize them.
"""

from __future__ import annotations

import csv
import hashlib
import pickle
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

CARE_LABELS = ("NORM", "MI", "STTC", "CD", "HYP")
UPSTREAM_BENCHMARK_COMMIT = "cdbf4e66d7e57d9b6a2657b6024716212b8d0afa"
UPSTREAM_BENCHMARK_REPOSITORY = "https://github.com/helme/ecg_ptbxl_benchmarking"
PINNED_XRESNET101_SHA256 = {
    "validation": "f8626f84dee06543d520dce605ef50f4b826db3146ba87a397081b3c1aee8a8e",
    "test": "8202066cda10db85a0f2b0af99aebce18763ce9fafe98ec1ab478b2f3672020a",
    "database": "cd9bf4e6445c5e022605857c4d6f90c90ff0e62df847da52efaebe94de786156",
    "statements": "ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f",
}
EPSILON = 1e-8
SCP_EPSILON = 1e-6


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: str | Path, expected_sha256: str) -> str:
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("Expected SHA-256 must be 64 lowercase hexadecimal characters")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"SHA-256 mismatch for {Path(path)}: expected {expected}, observed {observed}"
        )
    return observed


def load_hash_verified_legacy_array(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_columns: int | None = None,
) -> np.ndarray:
    """Load a legacy pickle only after its bytes match an expected digest."""
    source = Path(path)
    require_sha256(source, expected_sha256)
    with source.open("rb") as handle:
        values = pickle.load(handle, encoding="latin1")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Legacy prediction array must be two-dimensional, got {array.shape}")
    if expected_columns is not None and array.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} legacy prediction columns, got {array.shape[1]}"
        )
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("Legacy prediction values must be finite and in [0, 1]")
    return array


def normalize_probability_rows(
    values: np.ndarray,
    *,
    expected_columns: int | None = None,
    name: str = "probabilities",
) -> np.ndarray:
    """Normalize non-negative score rows into categorical distributions."""
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if expected_columns is not None and probabilities.shape[1] != expected_columns:
        raise ValueError(f"{name} must have {expected_columns} columns")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    mass = probabilities.sum(axis=1, keepdims=True)
    if np.any(mass <= 0.0):
        row = int(np.flatnonzero(mass[:, 0] <= 0.0)[0])
        raise ValueError(f"{name} row {row} has no positive probability mass")
    return probabilities / mass


def align_rows_by_record_id(
    values: np.ndarray,
    source_record_ids: Sequence[str] | np.ndarray,
    target_record_ids: Sequence[str] | np.ndarray,
) -> np.ndarray:
    """Align source rows to a target ID sequence, rejecting ambiguity and omissions."""
    matrix = np.asarray(values)
    source_ids = np.asarray(source_record_ids, dtype=str)
    target_ids = np.asarray(target_record_ids, dtype=str)
    if source_ids.ndim != 1 or len(source_ids) != len(matrix):
        raise ValueError("Source record IDs must be one-dimensional and match source rows")
    if target_ids.ndim != 1:
        raise ValueError("Target record IDs must be one-dimensional")
    if len(set(source_ids.tolist())) != len(source_ids):
        raise ValueError("Source record IDs must be unique")
    if len(set(target_ids.tolist())) != len(target_ids):
        raise ValueError("Target record IDs must be unique")
    row_by_id = {identifier: index for index, identifier in enumerate(source_ids)}
    missing = [identifier for identifier in target_ids if identifier not in row_by_id]
    if missing:
        raise ValueError(f"Target records are absent from the source: {missing[:5]}")
    return matrix[[row_by_id[identifier] for identifier in target_ids]]


def read_legacy_fold_ids(database_path: str | Path, folds: set[int]) -> np.ndarray:
    with Path(database_path).open(encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        identifiers = [str(row["ecg_id"]) for row in rows if int(row["strat_fold"]) in folds]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Legacy database contains duplicate ECG IDs in the selected folds")
    return np.asarray(identifiers, dtype=str)


def read_scp_statement_contract(
    statements_path: str | Path,
) -> tuple[tuple[str, ...], dict[str, str | None]]:
    """Return the official sorted SCP class order and diagnostic superclass mapping."""
    with Path(statements_path).open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("SCP statement table has no header")
        code_column = reader.fieldnames[0]
        rows = list(reader)
    rows.sort(key=lambda row: row[code_column])
    codes = tuple(row[code_column] for row in rows)
    if not codes or len(codes) != len(set(codes)):
        raise ValueError("SCP statement codes must be non-empty and unique")
    diagnostic_class: dict[str, str | None] = {}
    for row in rows:
        is_diagnostic = float(row.get("diagnostic") or 0.0) == 1.0
        value = row.get("diagnostic_class") if is_diagnostic else None
        diagnostic_class[row[code_column]] = value or None
    return codes, diagnostic_class


def collapse_scp_superclasses(
    scp_probabilities: np.ndarray,
    scp_labels: Sequence[str],
    diagnostic_class_by_code: Mapping[str, str | None],
    label_order: Sequence[str] = CARE_LABELS,
) -> np.ndarray:
    values = np.asarray(scp_probabilities, dtype=np.float64)
    labels = tuple(str(label) for label in scp_labels)
    if values.ndim != 2 or values.shape[1] != len(labels):
        raise ValueError("SCP probabilities and label order have incompatible dimensions")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("SCP probabilities must be finite and in [0, 1]")
    columns: list[np.ndarray] = []
    for superclass in label_order:
        indices = [
            index
            for index, code in enumerate(labels)
            if diagnostic_class_by_code.get(code) == superclass
        ]
        if not indices:
            raise ValueError(f"No diagnostic SCP code maps to superclass {superclass}")
        columns.append(values[:, indices].sum(axis=1))
    return normalize_probability_rows(
        np.stack(columns, axis=1),
        expected_columns=len(label_order),
        name="collapsed superclass scores",
    )


def build_scp_meta_features(
    scp_probabilities: np.ndarray,
    scp_labels: Sequence[str],
    diagnostic_class_by_code: Mapping[str, str | None],
    label_order: Sequence[str] = CARE_LABELS,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build a fixed 71-score plus ontology-aggregate feature contract."""
    labels = tuple(str(label) for label in scp_labels)
    values = np.asarray(scp_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(labels):
        raise ValueError("SCP probabilities and label order have incompatible dimensions")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("SCP probabilities must be finite and in [0, 1]")
    clipped = np.clip(values, SCP_EPSILON, 1.0 - SCP_EPSILON)
    parts = [clipped]
    feature_names = [f"scp_probability:{label}" for label in labels]
    for superclass in label_order:
        indices = [
            index
            for index, code in enumerate(labels)
            if diagnostic_class_by_code.get(code) == superclass
        ]
        if not indices:
            raise ValueError(f"No diagnostic SCP code maps to superclass {superclass}")
        members = clipped[:, indices]
        aggregates = (
            members.sum(axis=1),
            members.max(axis=1),
            members.mean(axis=1),
            1.0 - np.prod(1.0 - members, axis=1),
            np.log(members / (1.0 - members)).max(axis=1),
        )
        parts.extend(value[:, None] for value in aggregates)
        feature_names.extend(
            f"{superclass}:{name}"
            for name in ("sum", "max", "mean", "noisy_or", "max_logit")
        )
    return np.concatenate(parts, axis=1).astype(np.float32), tuple(feature_names)


def fit_scp_hist_gradient_booster(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int = 20261021,
) -> Any:
    """Fit the fixed SCP meta-model; scikit-learn is imported only when requested."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError("Install CARE-ECG with the 'ensemble' extra") from error

    matrix = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2 or targets.shape != (len(matrix),):
        raise ValueError("SCP features and labels have incompatible dimensions")
    counts = np.bincount(targets, minlength=len(CARE_LABELS))
    if np.any(counts == 0):
        raise ValueError("Every CARE class must occur when fitting the SCP meta-model")
    class_weights = (len(targets) / (len(CARE_LABELS) * counts)) ** 0.5
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


def predict_scp_hist_gradient_booster(model: Any, features: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_, dtype=np.int64)
    if not np.array_equal(classes, np.arange(len(CARE_LABELS))):
        raise ValueError(f"SCP meta-model has an unexpected class order: {classes.tolist()}")
    return normalize_probability_rows(
        model.predict_proba(np.asarray(features, dtype=np.float32)),
        expected_columns=len(CARE_LABELS),
        name="SCP meta-model probabilities",
    )


def dump_sklearn_model(model: Any, path: str | Path) -> str:
    try:
        import joblib
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError("Install CARE-ECG with the 'ensemble' extra") from error
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination, compress=3)
    return sha256_file(destination)


def load_hash_verified_sklearn_model(path: str | Path, expected_sha256: str) -> Any:
    """Authenticate a scikit-learn pickle artifact before joblib deserialization."""
    require_sha256(path, expected_sha256)
    try:
        import joblib
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError("Install CARE-ECG with the 'ensemble' extra") from error
    return joblib.load(path)


@dataclass(frozen=True)
class ProbabilityTable:
    probabilities: np.ndarray
    record_ids: np.ndarray
    label_order: tuple[str, ...]


def load_probability_table(
    path: str | Path,
    *,
    expected_labels: Sequence[str] = CARE_LABELS,
) -> ProbabilityTable:
    with np.load(path, allow_pickle=False) as archive:
        if "probabilities" in archive:
            values = np.asarray(archive["probabilities"], dtype=np.float64)
        elif "normalized_to_sum_scores" in archive:
            values = np.asarray(archive["normalized_to_sum_scores"], dtype=np.float64)
        else:
            raise ValueError(f"No supported probability member in {path}")
        if "record_ids" not in archive or "label_order" not in archive:
            raise ValueError(f"Probability archive lacks record_ids or label_order: {path}")
        record_ids = np.asarray(archive["record_ids"], dtype=str)
        source_labels = tuple(np.asarray(archive["label_order"], dtype=str).tolist())
    labels = tuple(str(label) for label in expected_labels)
    if len(set(source_labels)) != len(source_labels) or set(source_labels) != set(labels):
        raise ValueError(f"Probability labels do not exactly match {labels}: {source_labels}")
    if record_ids.ndim != 1 or len(record_ids) != len(values):
        raise ValueError("Probability record IDs must be one-dimensional and aligned")
    if len(set(record_ids.tolist())) != len(record_ids):
        raise ValueError("Probability record IDs must be unique")
    reordered = values[:, [source_labels.index(label) for label in labels]]
    probabilities = normalize_probability_rows(
        reordered,
        expected_columns=len(labels),
        name=f"probabilities from {path}",
    )
    return ProbabilityTable(probabilities, record_ids, labels)


def build_stacking_features(
    sources: Sequence[tuple[str, ProbabilityTable]],
    expected_record_ids: Sequence[str] | np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Concatenate log probabilities under an exact source/row/label contract."""
    if len(sources) < 2:
        raise ValueError("At least two probability sources are required")
    source_names = tuple(name for name, _ in sources)
    if any(not name.strip() for name in source_names) or len(set(source_names)) != len(source_names):
        raise ValueError("Probability source names must be non-empty and unique")
    expected_ids = np.asarray(expected_record_ids, dtype=str)
    if expected_ids.ndim != 1 or len(set(expected_ids.tolist())) != len(expected_ids):
        raise ValueError("Expected record IDs must be one-dimensional and unique")
    labels = sources[0][1].label_order
    parts: list[np.ndarray] = []
    feature_names: list[str] = []
    for name, table in sources:
        if table.label_order != labels:
            raise ValueError(f"Label order is inconsistent for source {name}")
        if not np.array_equal(table.record_ids, expected_ids):
            raise ValueError(f"Record IDs are not exactly aligned for source {name}")
        probabilities = normalize_probability_rows(
            table.probabilities,
            expected_columns=len(labels),
            name=f"probabilities from source {name}",
        )
        parts.append(np.log(np.clip(probabilities, EPSILON, 1.0)))
        feature_names.extend(f"log_probability:{name}:{label}" for label in labels)
    return np.concatenate(parts, axis=1), tuple(feature_names), source_names


def patient_group_folds(
    labels: np.ndarray,
    patient_ids: np.ndarray,
    fold_count: int = 5,
) -> np.ndarray:
    """Assign all records from a patient to a deterministic stratified fold."""
    targets = np.asarray(labels, dtype=np.int64)
    patients = np.asarray(patient_ids, dtype=str)
    if targets.shape != patients.shape or targets.ndim != 1:
        raise ValueError("Labels and patient IDs must be aligned vectors")
    groups, inverse = np.unique(patients, return_inverse=True)
    group_counts = np.zeros((len(groups), len(CARE_LABELS)), dtype=np.int64)
    np.add.at(group_counts, (inverse, targets), 1)
    order = sorted(
        range(len(groups)),
        key=lambda index: (
            -int(group_counts[index].max()),
            -int(group_counts[index].sum()),
            zlib.crc32(str(groups[index]).encode()),
        ),
    )
    target_counts = np.bincount(targets, minlength=len(CARE_LABELS)) / fold_count
    fold_counts = np.zeros((fold_count, len(CARE_LABELS)), dtype=np.float64)
    fold_sizes = np.zeros(fold_count, dtype=np.int64)
    group_fold = np.full(len(groups), -1, dtype=np.int64)
    for group_index in order:
        candidates: list[tuple[float, int, int]] = []
        for fold_index in range(fold_count):
            proposed = fold_counts.copy()
            proposed[fold_index] += group_counts[group_index]
            imbalance = np.var(
                (proposed - target_counts) / (target_counts + 1.0), axis=0
            ).sum()
            size = int(fold_sizes[fold_index] + group_counts[group_index].sum())
            candidates.append((float(imbalance + 1e-4 * size), size, fold_index))
        selected = min(candidates)[2]
        group_fold[group_index] = selected
        fold_counts[selected] += group_counts[group_index]
        fold_sizes[selected] += group_counts[group_index].sum()
    return group_fold[inverse]


def softmax(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class LinearProbabilityStacker:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    biases: np.ndarray
    class_biases: np.ndarray
    feature_names: tuple[str, ...]
    source_names: tuple[str, ...]
    label_order: tuple[str, ...] = CARE_LABELS

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Stack features must have {len(self.feature_names)} columns, got {matrix.shape}"
            )
        if self.mean.shape != (matrix.shape[1],) or self.scale.shape != (matrix.shape[1],):
            raise ValueError("Stacker normalization arrays do not match its feature contract")
        expected_weights = (matrix.shape[1], len(self.label_order))
        if self.weights.shape != expected_weights:
            raise ValueError(f"Stacker weights must have shape {expected_weights}")
        if self.biases.shape != (len(self.label_order),) or self.class_biases.shape != (
            len(self.label_order),
        ):
            raise ValueError("Stacker bias arrays do not match its label contract")
        stored_arrays = (
            self.mean,
            self.scale,
            self.weights,
            self.biases,
            self.class_biases,
        )
        if not np.isfinite(matrix).all() or not all(
            np.isfinite(values).all() for values in stored_arrays
        ):
            raise ValueError("Stacker features and parameters must be finite")
        if np.any(self.scale <= 0.0):
            raise ValueError("Stacker scales must be positive")
        logits = ((matrix - self.mean) / self.scale) @ self.weights
        logits = logits + self.biases + self.class_biases
        return softmax(logits)

    def predict_sources(
        self,
        sources: Sequence[tuple[str, ProbabilityTable]],
        expected_record_ids: Sequence[str] | np.ndarray,
    ) -> np.ndarray:
        features, feature_names, source_names = build_stacking_features(
            sources, expected_record_ids
        )
        if source_names != self.source_names:
            raise ValueError(
                f"Stack source order must be {self.source_names}, received {source_names}"
            )
        if feature_names != self.feature_names:
            raise ValueError("Stack feature order does not match the fitted estimator")
        return self.predict_features(features)

    def save(self, path: str | Path) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            mean=self.mean,
            scale=self.scale,
            weights=self.weights,
            biases=self.biases,
            class_biases=self.class_biases,
            feature_names=np.asarray(self.feature_names),
            source_names=np.asarray(self.source_names),
            label_order=np.asarray(self.label_order),
            feature_transform=np.asarray(
                "concatenate(log(clip(row_normalized_probability,1e-8,1)))"
            ),
        )
        return sha256_file(destination)

    @classmethod
    def load(cls, path: str | Path) -> LinearProbabilityStacker:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "mean",
                "scale",
                "weights",
                "biases",
                "class_biases",
                "feature_names",
                "source_names",
                "label_order",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"Stacker artifact is missing members: {sorted(missing)}")
            return cls(
                mean=np.asarray(archive["mean"], dtype=np.float64),
                scale=np.asarray(archive["scale"], dtype=np.float64),
                weights=np.asarray(archive["weights"], dtype=np.float64),
                biases=np.asarray(archive["biases"], dtype=np.float64),
                class_biases=np.asarray(archive["class_biases"], dtype=np.float64),
                feature_names=tuple(np.asarray(archive["feature_names"], dtype=str).tolist()),
                source_names=tuple(np.asarray(archive["source_names"], dtype=str).tolist()),
                label_order=tuple(np.asarray(archive["label_order"], dtype=str).tolist()),
            )


def fit_linear_probability_stacker(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    source_names: Sequence[str],
    *,
    regularization: float,
    class_weight_power: float,
) -> LinearProbabilityStacker:
    matrix = np.asarray(features, dtype=np.float64)
    targets_array = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2 or targets_array.shape != (len(matrix),):
        raise ValueError("Stacking features and labels have incompatible dimensions")
    if matrix.shape[1] != len(feature_names):
        raise ValueError("Stacking feature names do not match feature columns")
    counts = np.bincount(targets_array, minlength=len(CARE_LABELS))
    if np.any(counts == 0):
        raise ValueError("Every CARE class must occur when fitting a probability stacker")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0) + 1e-6
    normalized = torch.as_tensor((matrix - mean) / scale, dtype=torch.float64)
    targets = torch.as_tensor(targets_array, dtype=torch.long)
    class_weights = torch.as_tensor(
        (len(targets_array) / (len(CARE_LABELS) * counts)) ** class_weight_power,
        dtype=torch.float64,
    )
    torch.manual_seed(42)
    weights = torch.nn.Parameter(
        torch.zeros((matrix.shape[1], len(CARE_LABELS)), dtype=torch.float64)
    )
    biases = torch.nn.Parameter(
        torch.log(torch.as_tensor(counts / len(targets_array), dtype=torch.float64))
    )
    optimizer = torch.optim.LBFGS(
        [weights, biases],
        max_iter=300,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-11,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = normalized @ weights + biases
        loss = torch.nn.functional.cross_entropy(logits, targets, weight=class_weights)
        loss = loss + regularization * weights.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return LinearProbabilityStacker(
        mean=mean,
        scale=scale,
        weights=weights.detach().numpy(),
        biases=biases.detach().numpy(),
        class_biases=np.zeros(len(CARE_LABELS), dtype=np.float64),
        feature_names=tuple(feature_names),
        source_names=tuple(source_names),
    )


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    targets = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    size = len(CARE_LABELS)
    confusion = np.bincount(targets * size + predicted, minlength=size**2).reshape(size, size)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(np.mean(targets == predicted)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
    }
