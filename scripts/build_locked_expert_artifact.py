#!/usr/bin/env python3
"""Build a CARE artifact locked to a validated external probability expert."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from care_ecg.finalization import (
    build_locked_expert_artifact,
    parse_named_probability_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--output-artifact", type=Path, required=True)
    parser.add_argument("--locked-stacker", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--expert-probabilities", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        required=True,
        help="Named probability artifact in NAME=PATH form; repeat in stacker order.",
    )
    parser.add_argument(
        "--paper-artifact-deviation",
        required=True,
        help="Explain how the locked expert differs from the paper's unavailable artifact.",
    )
    parser.add_argument(
        "--allow-test-after-lock",
        action="store_true",
        help="Authorize test-named, label-free expert paths after validation selection is locked.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output artifact directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_locked_expert_artifact(
        base_artifact=args.base_artifact,
        output_artifact=args.output_artifact,
        locked_stacker=args.locked_stacker,
        selection_report=args.selection_report,
        expert_probabilities=args.expert_probabilities,
        source_probabilities=parse_named_probability_paths(args.source),
        paper_artifact_deviation=args.paper_artifact_deviation,
        allow_test_after_lock=args.allow_test_after_lock,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
