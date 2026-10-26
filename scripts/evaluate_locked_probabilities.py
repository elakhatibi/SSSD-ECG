#!/usr/bin/env python3
"""Evaluate one already-locked probability artifact exactly once."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from care_ecg.ensemble import (
    CARE_LABELS,
    classification_metrics,
    load_probability_table,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"One-shot evaluation report already exists: {args.output}")
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "locked before fold-10 specialist inference":
        raise ValueError("Selection was not locked before inference")
    with np.load(args.dataset, allow_pickle=False) as archive:
        record_ids = np.asarray(archive["record_ids"], dtype=str)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        folds = np.asarray(archive["folds"], dtype=np.int64)
    if labels.shape != record_ids.shape or folds.shape != record_ids.shape:
        raise ValueError("Evaluation dataset arrays are not aligned")
    if set(np.unique(folds).tolist()) != {10}:
        raise ValueError("One-shot evaluation accepts only official fold 10")
    table = load_probability_table(args.probabilities, expected_labels=CARE_LABELS)
    if not np.array_equal(table.record_ids, record_ids):
        raise ValueError("Locked probabilities are not exactly row-aligned")
    metrics = classification_metrics(labels, table.probabilities.argmax(axis=1))
    report = {
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_count_for_this_artifact": 1,
        "scope": "prepared PTB-XL fold 10",
        "records": len(labels),
        "metrics": metrics,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "probabilities": str(args.probabilities.resolve()),
        "probabilities_sha256": sha256_file(args.probabilities),
        "selection_lock": str(args.selection_lock.resolve()),
        "selection_lock_sha256": sha256_file(args.selection_lock),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
