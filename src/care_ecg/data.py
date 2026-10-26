from __future__ import annotations

import ast
import csv
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .types import ECGRecord

LEAD_ORDER = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
PTBXL_DIAGNOSTIC_SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def _as_channels_first(signal: np.ndarray, expected_leads: int | None = 12) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"ECG must have shape [leads, time] or [time, leads], got {x.shape}")
    if expected_leads is not None:
        if x.shape[0] == expected_leads:
            return x
        if x.shape[1] == expected_leads:
            return x.T
        raise ValueError(f"Expected {expected_leads} leads, got shape {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T


def _repair_nonfinite(x: np.ndarray) -> np.ndarray:
    repaired = x.copy()
    for lead in range(repaired.shape[0]):
        values = repaired[lead]
        good = np.flatnonzero(np.isfinite(values))
        if not len(good):
            repaired[lead] = 0.0
        elif len(good) != len(values):
            bad = np.flatnonzero(~np.isfinite(values))
            repaired[lead, bad] = np.interp(bad, good, values[good])
    return repaired


def preprocess_ecg(
    signal: np.ndarray,
    source_rate: int,
    target_rate: int = 500,
    seconds: int = 10,
    expected_leads: int = 12,
    normalization: str = "robust",
) -> np.ndarray:
    """Resample, center crop/pad, and per-lead normalize an ECG.

    Normalization is fitted per recording, so it never leaks information between
    train and test sets. The returned array is contiguous float32 [C, T].
    """
    if source_rate <= 0 or target_rate <= 0 or seconds <= 0:
        raise ValueError("Sampling rates and duration must be positive")
    x = _repair_nonfinite(_as_channels_first(signal, expected_leads))
    if source_rate != target_rate:
        divisor = int(np.gcd(source_rate, target_rate))
        x = resample_poly(x, target_rate // divisor, source_rate // divisor, axis=1)

    target_samples = target_rate * seconds
    if x.shape[1] > target_samples:
        start = (x.shape[1] - target_samples) // 2
        x = x[:, start : start + target_samples]
    elif x.shape[1] < target_samples:
        deficit = target_samples - x.shape[1]
        x = np.pad(x, ((0, 0), (deficit // 2, deficit - deficit // 2)))

    if normalization == "robust":
        center = np.median(x, axis=1, keepdims=True)
        q25, q75 = np.quantile(x, [0.25, 0.75], axis=1, keepdims=True)
        scale = q75 - q25
        std = np.std(x, axis=1, keepdims=True)
        scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0))
        x = (x - center) / scale
    elif normalization == "zscore":
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        x = (x - mean) / np.where(std > 1e-6, std, 1.0)
    elif normalization != "none":
        raise ValueError("normalization must be robust, zscore, or none")
    return np.ascontiguousarray(x, dtype=np.float32)


def fit_global_standardizer(signals: np.ndarray) -> tuple[float, float]:
    """Fit a scalar train-only standardizer, matching the PTB-XL benchmark policy."""
    values = np.asarray(signals, dtype=np.float32)
    if values.ndim != 3 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Standardizer input must be finite [records, leads, samples]")
    mean = float(np.mean(values, dtype=np.float64))
    scale = float(np.std(values, dtype=np.float64))
    return mean, scale if scale > 1e-8 else 1.0


def apply_global_standardizer(signals: np.ndarray, mean: float, scale: float) -> np.ndarray:
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Global normalization statistics must be finite with positive scale")
    values = np.asarray(signals, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("Signals must be finite before global normalization")
    return np.ascontiguousarray((values - mean) / scale, dtype=np.float32)


@dataclass(slots=True)
class ECGArrayBundle:
    signals: np.ndarray
    labels: np.ndarray | None
    patient_ids: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    record_ids: np.ndarray | None = None
    folds: np.ndarray | None = None

    def validate(self) -> None:
        if self.signals.ndim != 3:
            raise ValueError("signals must have shape [records, leads, samples]")
        n = len(self.signals)
        for name in ("labels", "patient_ids", "timestamps", "record_ids", "folds"):
            value = getattr(self, name)
            if value is not None and len(value) != n:
                raise ValueError(f"{name} length does not match signals")


def load_npz(path: str | Path, require_labels: bool = False) -> ECGArrayBundle:
    with np.load(path, allow_pickle=False) as data:
        if "signals" not in data:
            raise ValueError("NPZ must contain a 'signals' array")
        bundle = ECGArrayBundle(
            signals=np.asarray(data["signals"], dtype=np.float32),
            labels=np.asarray(data["labels"]) if "labels" in data else None,
            patient_ids=np.asarray(data["patient_ids"]).astype(str)
            if "patient_ids" in data
            else None,
            timestamps=np.asarray(data["timestamps"], dtype=float)
            if "timestamps" in data
            else None,
            record_ids=np.asarray(data["record_ids"]).astype(str) if "record_ids" in data else None,
            folds=np.asarray(data["folds"]) if "folds" in data else None,
        )
    bundle.validate()
    if require_labels and bundle.labels is None:
        raise ValueError("Training requires the NPZ 'labels' array")
    return bundle


class TorchECGDataset:
    """Small torch-compatible dataset without importing torch at module import time."""

    def __init__(self, signals: np.ndarray):
        if signals.ndim != 3:
            raise ValueError("signals must have shape [records, leads, samples]")
        self.signals = signals

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, index: int):
        import torch

        return torch.as_tensor(self.signals[index], dtype=torch.float32)


def iter_ptbxl(
    root: str | Path,
    sampling_rate: int = 500,
    min_likelihood: float = 0.0,
) -> Iterator[ECGRecord]:
    """Yield PTB-XL records from its standard extracted directory layout.

    This intentionally leaves label aggregation to the caller: ``metadata['scp_codes']``
    contains the original SCP statement/likelihood mapping.
    """
    try:
        import pandas as pd
        import wfdb
    except ImportError as exc:
        raise ImportError("PTB-XL loading requires `pip install 'care-ecg[data]'`") from exc

    root = Path(root)
    metadata = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    filename_field = "filename_hr" if sampling_rate == 500 else "filename_lr"
    for ecg_id, row in metadata.iterrows():
        waveform, _ = wfdb.rdsamp(str(root / row[filename_field]))
        codes = ast.literal_eval(row.scp_codes) if isinstance(row.scp_codes, str) else {}
        codes = {str(k): float(v) for k, v in codes.items() if float(v) >= min_likelihood}
        yield ECGRecord(
            signal=waveform.T.astype(np.float32),
            patient_id=str(row.patient_id),
            record_id=str(ecg_id),
            metadata={
                "scp_codes": codes,
                "sampling_rate": sampling_rate,
                "strat_fold": int(row.strat_fold),
            },
        )


def load_ptbxl_diagnostic_mapping(root: str | Path) -> dict[str, str]:
    """Load diagnostic SCP-code to superclass mappings from PTB-XL metadata."""
    path = Path(root) / "scp_statements.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not fields or "diagnostic" not in fields or "diagnostic_class" not in fields:
            raise ValueError(
                "scp_statements.csv must contain SCP code, diagnostic, and diagnostic_class columns"
            )
        code_field = fields[0]
        mapping: dict[str, str] = {}
        for row in reader:
            diagnostic = str(row.get("diagnostic", "")).strip().lower()
            if diagnostic not in {"1", "1.0", "true", "yes"}:
                continue
            code = str(row.get(code_field, "")).strip()
            superclass = str(row.get("diagnostic_class", "")).strip().upper()
            if code and superclass in PTBXL_DIAGNOSTIC_SUPERCLASSES:
                mapping[code] = superclass
    return mapping


def load_ptbxl_diagnostic_aliases(root: str | Path) -> dict[str, list[str]]:
    """Build superclass text aliases from PTB-XL's diagnostic SCP statements."""
    path = Path(root) / "scp_statements.csv"
    aliases = {name: [name] for name in PTBXL_DIAGNOSTIC_SUPERCLASSES}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not fields or "diagnostic" not in fields or "diagnostic_class" not in fields:
            raise ValueError(
                "scp_statements.csv must contain SCP code, diagnostic, and diagnostic_class columns"
            )
        code_field = fields[0]
        description_fields = [
            name for name in ("description", "SCP-ECG Statement Description") if name in fields
        ]
        for row in reader:
            diagnostic = str(row.get("diagnostic", "")).strip().lower()
            superclass = str(row.get("diagnostic_class", "")).strip().upper()
            if diagnostic not in {"1", "1.0", "true", "yes"}:
                continue
            if superclass not in aliases:
                continue
            values = [str(row.get(code_field, "")).strip()]
            values.extend(str(row.get(field, "")).strip() for field in description_fields)
            for value in values:
                if value and value not in aliases[superclass]:
                    aliases[superclass].append(value)
    return aliases


def map_scp_codes_to_diagnostic_superclasses(
    scp_codes: Mapping[str, float],
    code_to_superclass: Mapping[str, str],
) -> dict[str, float]:
    """Map SCP likelihoods to diagnostic superclasses using the maximum per class."""
    mapped: dict[str, float] = {}
    for code, likelihood in scp_codes.items():
        superclass = code_to_superclass.get(str(code))
        if superclass not in PTBXL_DIAGNOSTIC_SUPERCLASSES:
            continue
        score = float(likelihood)
        if superclass not in mapped or score > mapped[superclass]:
            mapped[superclass] = score
    return mapped


def iter_ptbxl_diagnostic_superclasses(
    root: str | Path,
    sampling_rate: int = 500,
    min_likelihood: float = 0.0,
) -> Iterator[ECGRecord]:
    """Yield PTB-XL records labeled with the five diagnostic superclasses."""
    code_to_superclass = load_ptbxl_diagnostic_mapping(root)
    for record in iter_ptbxl(root, sampling_rate=sampling_rate, min_likelihood=min_likelihood):
        raw_codes = record.metadata.get("scp_codes", {})
        superclass_codes = map_scp_codes_to_diagnostic_superclasses(raw_codes, code_to_superclass)
        if not superclass_codes:
            continue
        metadata = {
            **record.metadata,
            "raw_scp_codes": dict(raw_codes),
            "scp_codes": superclass_codes,
        }
        yield replace(record, metadata=metadata)


def records_to_arrays(
    records: Iterable[ECGRecord],
    label_order: Sequence[str],
    target_rate: int = 500,
    seconds: int = 10,
    normalization: str = "robust",
) -> ECGArrayBundle:
    """Convert records to arrays, selecting the highest-likelihood requested SCP code."""
    label_to_id = {name: i for i, name in enumerate(label_order)}
    signals: list[np.ndarray] = []
    labels: list[int] = []
    patient_ids: list[str] = []
    record_ids: list[str] = []
    folds: list[int] = []
    folds_complete = True
    for record in records:
        code_weights = record.metadata.get("scp_codes", {})
        eligible = [code for code in label_order if code in code_weights]
        if not eligible:
            continue
        label = max(eligible, key=lambda code: float(code_weights[code]))
        source_rate = int(record.metadata.get("sampling_rate", target_rate))
        signals.append(
            preprocess_ecg(
                record.signal,
                source_rate,
                target_rate,
                seconds,
                normalization=normalization,
            )
        )
        labels.append(label_to_id[label])
        patient_ids.append(record.patient_id or "")
        record_ids.append(record.record_id or str(len(record_ids)))
        if "strat_fold" in record.metadata:
            folds.append(int(record.metadata["strat_fold"]))
        else:
            folds_complete = False
    if not signals:
        raise ValueError("No records matched label_order")
    return ECGArrayBundle(
        signals=np.stack(signals),
        labels=np.asarray(labels, dtype=np.int64),
        patient_ids=np.asarray(patient_ids),
        record_ids=np.asarray(record_ids),
        folds=np.asarray(folds, dtype=np.int64) if folds_complete else None,
    )
