#!/usr/bin/env python3
"""Build a fold-9 CARE superclass adapter for official XResNet1D-101 scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from care_ecg.ensemble import (
    CARE_LABELS,
    PINNED_XRESNET101_SHA256,
    UPSTREAM_BENCHMARK_COMMIT,
    UPSTREAM_BENCHMARK_REPOSITORY,
    align_rows_by_record_id,
    collapse_scp_superclasses,
    load_hash_verified_legacy_array,
    read_legacy_fold_ids,
    read_scp_statement_contract,
    require_sha256,
    sha256_file,
)


def validation_ids(path: Path) -> np.ndarray:
    if "test" in path.name.lower():
        raise ValueError("The validation adapter refuses a test-named prepared archive")
    with np.load(path, allow_pickle=False) as archive:
        record_ids = np.asarray(archive["record_ids"], dtype=str)
        folds = np.asarray(archive["folds"], dtype=np.int64)
    if record_ids.ndim != 1 or len(set(record_ids.tolist())) != len(record_ids):
        raise ValueError("Prepared validation record IDs must be one-dimensional and unique")
    if folds.shape != record_ids.shape or not np.all(folds == 9):
        raise ValueError("The prepared archive must contain official validation fold 9 only")
    return record_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--statements", type=Path, required=True)
    parser.add_argument("--prepared-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--predictions-sha256",
        default=PINNED_XRESNET101_SHA256["validation"],
    )
    parser.add_argument("--database-sha256", default=PINNED_XRESNET101_SHA256["database"])
    parser.add_argument("--statements-sha256", default=PINNED_XRESNET101_SHA256["statements"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.predictions, args.prepared_validation, args.output, args.report):
        if "test" in path.name.lower():
            raise ValueError("This command is validation-only and refuses test-named paths")

    require_sha256(args.database, args.database_sha256)
    require_sha256(args.statements, args.statements_sha256)
    scp_labels, diagnostic_classes = read_scp_statement_contract(args.statements)
    predictions = load_hash_verified_legacy_array(
        args.predictions,
        args.predictions_sha256,
        expected_columns=len(scp_labels),
    )
    legacy_ids = read_legacy_fold_ids(args.database, {9})
    current_ids = validation_ids(args.prepared_validation)
    aligned_scp = align_rows_by_record_id(predictions, legacy_ids, current_ids)
    probabilities = collapse_scp_superclasses(
        aligned_scp,
        scp_labels,
        diagnostic_classes,
    ).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        probabilities=probabilities,
        record_ids=current_ids,
        label_order=np.asarray(CARE_LABELS),
        scp_probabilities=aligned_scp.astype(np.float32),
        scp_label_order=np.asarray(scp_labels),
        folds=np.full(len(current_ids), 9, dtype=np.int64),
        source_model=np.asarray("official FastAI XResNet1D-101 exp0"),
        selection_scope=np.asarray("official PTB-XL validation fold 9 only"),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "official PTB-XL validation fold 9 only; fold 10 was not opened",
        "model": "official FastAI XResNet1D-101 exp0",
        "repository": UPSTREAM_BENCHMARK_REPOSITORY,
        "commit": UPSTREAM_BENCHMARK_COMMIT,
        "mapping": (
            "sum diagnostic SCP sigmoid scores per CARE superclass, then normalize "
            "the five superclass scores to unit mass"
        ),
        "records": len(current_ids),
        "label_order": list(CARE_LABELS),
        "scp_label_order": list(scp_labels),
        "inputs": {
            "predictions": {
                "path": str(args.predictions.resolve()),
                "sha256": sha256_file(args.predictions),
            },
            "database": {
                "path": str(args.database.resolve()),
                "sha256": sha256_file(args.database),
            },
            "statements": {
                "path": str(args.statements.resolve()),
                "sha256": sha256_file(args.statements),
            },
            "prepared_validation": str(args.prepared_validation.resolve()),
        },
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "prepared_labels_read": False,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
