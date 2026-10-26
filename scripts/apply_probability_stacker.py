#!/usr/bin/env python3
"""Apply a locked probability stacker to aligned, label-free source scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from care_ecg.ensemble import (
    LinearProbabilityStacker,
    load_probability_table,
    require_sha256,
    sha256_file,
)


def parse_named_paths(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path:
            raise ValueError("Probability sources must use NAME=PATH syntax")
        result.append((name.strip(), Path(path)))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacker", type=Path, required=True)
    parser.add_argument("--stacker-sha256", required=True)
    parser.add_argument("--prepared-split", type=Path, required=True)
    parser.add_argument("--expected-fold", type=int, required=True)
    parser.add_argument("--source", action="append", default=[], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-test-after-lock",
        action="store_true",
        help="Authorize test-named probability sources only after model selection is locked.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    named_paths = parse_named_paths(args.source)
    if args.expected_fold < 1 or args.expected_fold > 10:
        raise ValueError("expected-fold must be in [1, 10]")
    test_named = "test" in args.prepared_split.name.lower() or args.expected_fold == 10
    test_named = test_named or any("test" in path.name.lower() for _, path in named_paths)
    test_named = test_named or "test" in args.output.name.lower()
    if test_named and not args.allow_test_after_lock:
        raise PermissionError(
            "Refusing test-named inputs without the explicit --allow-test-after-lock flag"
        )

    require_sha256(args.stacker, args.stacker_sha256)
    stacker = LinearProbabilityStacker.load(args.stacker)
    with np.load(args.prepared_split, allow_pickle=False) as archive:
        record_ids = np.asarray(archive["record_ids"], dtype=str)
        folds = np.asarray(archive["folds"], dtype=np.int64)
    if record_ids.ndim != 1 or folds.shape != record_ids.shape or not len(record_ids):
        raise ValueError("Prepared record IDs and folds must be non-empty aligned vectors")
    if len(set(record_ids.tolist())) != len(record_ids):
        raise ValueError("Prepared record IDs must be unique")
    observed_folds = set(np.unique(folds).tolist())
    if observed_folds != {args.expected_fold}:
        raise ValueError(
            f"Prepared split must contain fold {args.expected_fold} only, "
            f"received {sorted(observed_folds)}"
        )
    sources = [
        (name, load_probability_table(path, expected_labels=stacker.label_order))
        for name, path in named_paths
    ]
    if not sources:
        raise ValueError("At least one probability source is required")
    probabilities = stacker.predict_sources(sources, record_ids).astype(np.float32)
    source_hashes = [sha256_file(path) for _, path in named_paths]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        probabilities=probabilities,
        record_ids=record_ids,
        label_order=np.asarray(stacker.label_order),
        folds=folds,
        source_order=np.asarray(stacker.source_names),
        source_sha256=np.asarray(source_hashes),
        stacker_sha256=np.asarray(sha256_file(args.stacker)),
        inference_contract=np.asarray("label-free aligned probability sources"),
    )
    print(f"Wrote {len(record_ids)} stacked predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
