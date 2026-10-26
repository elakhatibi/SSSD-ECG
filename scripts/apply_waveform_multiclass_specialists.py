#!/usr/bin/env python3
"""Apply the locked waveform specialists without reading target labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from care_ecg.ensemble import (
    CARE_LABELS,
    load_hash_verified_sklearn_model,
    load_probability_table,
    sha256_file,
)
from care_ecg.waveform_specialist import (
    MulticlassSpecialistPolicy,
    apply_multiclass_logit_adjustment,
    build_specialist_features,
    load_inference_inputs,
    predict_ovr_probabilities,
    read_demographics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--baseline-probabilities", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "locked before fold-10 specialist inference":
        raise ValueError("Selection lock does not have the required pre-inference status")

    policy = MulticlassSpecialistPolicy.load(args.policy)
    selection = lock.get("selection", {})
    if selection.get("selected_transform") != policy.transform:
        raise ValueError("Policy transform differs from the locked selection")
    if not np.isclose(float(selection.get("selected_beta")), policy.beta):
        raise ValueError("Policy beta differs from the locked selection")
    if not np.allclose(
        np.asarray(selection.get("selected_biases"), dtype=np.float64),
        policy.biases,
    ):
        raise ValueError("Policy biases differ from the locked selection")

    signals, record_ids = load_inference_inputs(args.dataset)
    baseline = load_probability_table(
        args.baseline_probabilities,
        expected_labels=CARE_LABELS,
    )
    if not np.array_equal(baseline.record_ids, record_ids):
        raise ValueError("Baseline probabilities are not exactly row-aligned")
    models = load_hash_verified_sklearn_model(args.model, policy.model_sha256)
    raw_demographics = read_demographics(args.database, record_ids)
    features = build_specialist_features(
        signals,
        raw_demographics,
        policy.demographic_medians,
        sampling_rate_hz=policy.sampling_rate_hz,
        batch_size=args.batch_size,
    )
    specialist_probabilities = predict_ovr_probabilities(models, features)
    probabilities = apply_multiclass_logit_adjustment(
        baseline.probabilities,
        specialist_probabilities,
        beta=policy.beta,
        biases=policy.biases,
        specialist_logit_center=policy.specialist_logit_center,
        specialist_logit_scale=policy.specialist_logit_scale,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        probabilities=probabilities.astype(np.float32),
        specialist_probabilities=specialist_probabilities.astype(np.float32),
        record_ids=record_ids,
        label_order=np.asarray(CARE_LABELS),
        baseline_probabilities_sha256=np.asarray(sha256_file(args.baseline_probabilities)),
        specialist_model_sha256=np.asarray(policy.model_sha256),
        policy_sha256=np.asarray(sha256_file(args.policy)),
        selection_lock_sha256=np.asarray(sha256_file(args.selection_lock)),
        labels_accessed=np.asarray(False),
    )
    report = {
        "records": len(record_ids),
        "labels_accessed": False,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "baseline_probabilities_sha256": sha256_file(args.baseline_probabilities),
        "specialist_model_sha256": policy.model_sha256,
        "policy_sha256": sha256_file(args.policy),
        "selection_lock_sha256": sha256_file(args.selection_lock),
        "model_family": selection["selected_model_family"],
        "transform": policy.transform,
        "beta": policy.beta,
        "biases": policy.biases.tolist(),
    }
    report_path = args.output.with_suffix(".inference.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
