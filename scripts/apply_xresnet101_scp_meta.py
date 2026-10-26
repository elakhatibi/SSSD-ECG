#!/usr/bin/env python3
"""Apply the locked SCP meta-model to an authenticated, label-free PTB-XL split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from care_ecg.ensemble import (
    CARE_LABELS,
    PINNED_XRESNET101_SHA256,
    align_rows_by_record_id,
    build_scp_meta_features,
    load_hash_verified_legacy_array,
    load_hash_verified_sklearn_model,
    predict_scp_hist_gradient_booster,
    read_legacy_fold_ids,
    read_scp_statement_contract,
    require_sha256,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--official-predictions", type=Path, required=True)
    parser.add_argument("--official-predictions-sha256", required=True)
    parser.add_argument("--official-database", type=Path, required=True)
    parser.add_argument("--database-sha256", default=PINNED_XRESNET101_SHA256["database"])
    parser.add_argument("--scp-statements", type=Path, required=True)
    parser.add_argument("--statements-sha256", default=PINNED_XRESNET101_SHA256["statements"])
    parser.add_argument("--prepared-split", type=Path, required=True)
    parser.add_argument("--expected-fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-test-after-lock",
        action="store_true",
        help="Authorize fold-10/test-named input only after model selection is locked.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_named = any(
        "test" in path.name.lower()
        for path in (args.official_predictions, args.prepared_split, args.output)
    )
    if (test_named or args.expected_fold == 10) and not args.allow_test_after_lock:
        raise PermissionError(
            "Refusing fold-10/test-named input without --allow-test-after-lock"
        )
    if args.expected_fold < 1 or args.expected_fold > 10:
        raise ValueError("expected-fold must be in [1, 10]")

    # Authenticate every legacy/pickle-bearing dependency before deserializing it.
    require_sha256(args.official_database, args.database_sha256)
    require_sha256(args.scp_statements, args.statements_sha256)
    scp_labels, diagnostic_classes = read_scp_statement_contract(args.scp_statements)
    raw_predictions = load_hash_verified_legacy_array(
        args.official_predictions,
        args.official_predictions_sha256,
        expected_columns=len(scp_labels),
    )

    # Deliberately access only identity and fold members, never labels.
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

    legacy_ids = read_legacy_fold_ids(args.official_database, {args.expected_fold})
    aligned_scp = align_rows_by_record_id(raw_predictions, legacy_ids, record_ids)
    features, feature_names = build_scp_meta_features(
        aligned_scp,
        scp_labels,
        diagnostic_classes,
    )
    model = load_hash_verified_sklearn_model(args.model, args.model_sha256)
    if getattr(model, "n_features_in_", None) != len(feature_names):
        raise ValueError(
            "SCP meta-model feature count does not match the deterministic feature contract"
        )
    probabilities = predict_scp_hist_gradient_booster(model, features).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        probabilities=probabilities,
        record_ids=record_ids,
        label_order=np.asarray(CARE_LABELS),
        folds=folds,
        source_model=np.asarray("official XResNet1D-101 exp0 plus SCP meta-model"),
        model_sha256=np.asarray(sha256_file(args.model)),
        official_predictions_sha256=np.asarray(sha256_file(args.official_predictions)),
        official_database_sha256=np.asarray(sha256_file(args.official_database)),
        scp_statements_sha256=np.asarray(sha256_file(args.scp_statements)),
        feature_order=np.asarray(feature_names),
        inference_contract=np.asarray("record_ids and folds read; prepared labels not opened"),
    )
    print(f"Wrote {len(record_ids)} SCP-meta predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
