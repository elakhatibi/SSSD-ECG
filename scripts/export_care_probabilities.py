#!/usr/bin/env python3
"""Export label-free classifier probabilities from a saved CARE-ECG artifact."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from care_ecg.ensemble import sha256_file
from care_ecg.pipeline import CAREECGPipeline


def _is_test_named(path: Path) -> bool:
    return "test" in path.name.lower()


def _artifact_tree_sha256(directory: Path) -> str:
    """Hash artifact file names and contents under a stable, versioned contract."""
    files = sorted(
        (path for path in directory.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    if not files:
        raise ValueError(f"CARE artifact has no files: {directory}")
    digest = hashlib.sha256(b"CARE-ECG artifact tree v1\0")
    for path in files:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _load_label_free_split(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only waveform, identity, and fold members from a prepared archive."""
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in ("signals", "record_ids", "folds") if name not in archive]
        if missing:
            raise ValueError(f"Prepared NPZ lacks required members: {missing}")
        signals = np.asarray(archive["signals"], dtype=np.float32)
        record_ids = np.asarray(archive["record_ids"]).copy()
        folds = np.asarray(archive["folds"]).copy()

    if signals.ndim != 3 or not len(signals):
        raise ValueError("Prepared signals must be a non-empty [records, leads, samples] array")
    if record_ids.ndim != 1 or len(record_ids) != len(signals):
        raise ValueError("Prepared record IDs must be a one-dimensional vector aligned to signals")
    if record_ids.dtype.kind not in {"U", "S", "i", "u"}:
        raise ValueError("Prepared record IDs must use a non-object string or integer dtype")
    record_id_strings = np.asarray(record_ids, dtype=str)
    if np.any(np.char.strip(record_id_strings) == ""):
        raise ValueError("Prepared record IDs must be non-empty")
    if len(set(record_id_strings.tolist())) != len(record_ids):
        raise ValueError("Prepared record IDs must be unique")
    if folds.ndim != 1 or len(folds) != len(signals):
        raise ValueError("Prepared folds must be a one-dimensional vector aligned to signals")
    if not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("Prepared folds must use an integer dtype")
    return signals, record_ids, folds


def _validate_probabilities(
    probabilities: np.ndarray | None,
    record_count: int,
    label_order: tuple[str, ...],
) -> np.ndarray:
    if probabilities is None:
        raise ValueError("The CARE artifact has no classifier head probabilities")
    values = np.asarray(probabilities, dtype=np.float32)
    expected_shape = (record_count, len(label_order))
    if values.shape != expected_shape:
        raise ValueError(
            f"Classifier probabilities must have shape {expected_shape}, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Classifier probabilities must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Classifier probabilities must be in [0, 1]")
    row_mass = values.sum(axis=1, dtype=np.float64)
    if not np.allclose(row_mass, 1.0, rtol=1e-6, atol=1e-6):
        row = int(np.flatnonzero(~np.isclose(row_mass, 1.0, rtol=1e-6, atol=1e-6))[0])
        raise ValueError(
            f"Classifier probability row {row} has mass {row_mass[row]:.8g}, expected 1"
        )
    return values


def export_care_probabilities(
    *,
    artifact: str | Path,
    prepared_split: str | Path,
    expected_fold: int,
    output: str | Path,
    batch_size: int = 64,
    preprocessed: bool = False,
    source_rate: int | None = None,
    allow_test_after_lock: bool = False,
) -> Path:
    """Run deterministic classifier inference without opening prepared labels."""
    artifact_path = Path(artifact)
    split_path = Path(prepared_split)
    output_path = Path(output)

    if expected_fold < 1 or expected_fold > 10:
        raise ValueError("expected-fold must be in [1, 10]")
    guarded_test_path = any(
        _is_test_named(path) for path in (artifact_path, split_path, output_path)
    )
    if (expected_fold == 10 or guarded_test_path) and not allow_test_after_lock:
        raise PermissionError(
            "Refusing fold-10/test-named paths without --allow-test-after-lock"
        )
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    if preprocessed == (source_rate is not None):
        raise ValueError("Choose exactly one of --preprocessed or --source-rate")
    if source_rate is not None and source_rate < 1:
        raise ValueError("source-rate must be positive")
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Output must use the .npz extension")
    if output_path.resolve() == split_path.resolve():
        raise ValueError("Output must not overwrite the prepared split")
    try:
        output_path.resolve().relative_to(artifact_path.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("Output must be outside the CARE artifact directory")

    signals, record_ids, folds = _load_label_free_split(split_path)
    observed_folds = set(np.unique(folds).tolist())
    if observed_folds != {expected_fold}:
        raise ValueError(
            f"Prepared split must contain fold {expected_fold} only, "
            f"received {sorted(observed_folds)}"
        )

    config_path = artifact_path / "config.yaml"
    model_path = artifact_path / "causal_vae.pt"
    if not config_path.is_file() or not model_path.is_file():
        raise ValueError("CARE artifact must contain config.yaml and causal_vae.pt")
    artifact_sha256 = _artifact_tree_sha256(artifact_path)
    config_sha256 = sha256_file(config_path)
    model_sha256 = sha256_file(model_path)

    pipeline = CAREECGPipeline.load(artifact_path)
    label_order = tuple(str(label) for label in pipeline.causal_model.label_names)
    if not label_order or len(set(label_order)) != len(label_order):
        raise ValueError("CARE classifier label order must be non-empty and unique")
    if any(not label.strip() for label in label_order):
        raise ValueError("CARE classifier labels must be non-empty")

    _, raw_probabilities = pipeline.encode_batch_with_neural(
        signals,
        source_rate=None if preprocessed else source_rate,
        batch_size=batch_size,
    )
    probabilities = _validate_probabilities(raw_probabilities, len(signals), label_order)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        probabilities=probabilities,
        record_ids=record_ids,
        label_order=np.asarray(label_order),
        folds=folds,
        artifact_sha256=np.asarray(artifact_sha256),
        config_sha256=np.asarray(config_sha256),
        model_sha256=np.asarray(model_sha256),
        artifact_hash_contract=np.asarray("sha256:sorted-relative-path-and-file-sha256:v1"),
        deterministic_latent_sampling=np.asarray(True),
        encoder_batch_size=np.asarray(batch_size, dtype=np.int64),
        input_preprocessed=np.asarray(preprocessed),
        input_source_rate=np.asarray(-1 if source_rate is None else source_rate, dtype=np.int64),
        expected_fold=np.asarray(expected_fold, dtype=np.int64),
        inference_contract=np.asarray(
            "signals, record_ids, and folds read; prepared labels not opened"
        ),
    )
    return output_path.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--prepared-split", type=Path, required=True)
    parser.add_argument("--expected-fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    preprocessing = parser.add_mutually_exclusive_group(required=True)
    preprocessing.add_argument(
        "--preprocessed",
        action="store_true",
        help="Signals already follow the artifact's preprocessing contract.",
    )
    preprocessing.add_argument(
        "--source-rate",
        type=int,
        help="Sampling rate for raw signals; apply the artifact's preprocessing.",
    )
    parser.add_argument(
        "--allow-test-after-lock",
        action="store_true",
        help="Authorize fold-10/test-named input only after model selection is locked.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = export_care_probabilities(
        artifact=args.artifact,
        prepared_split=args.prepared_split,
        expected_fold=args.expected_fold,
        output=args.output,
        batch_size=args.batch_size,
        preprocessed=args.preprocessed,
        source_rate=args.source_rate,
        allow_test_after_lock=args.allow_test_after_lock,
    )
    print(f"Wrote label-free CARE probabilities to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
