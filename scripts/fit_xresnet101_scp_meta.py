#!/usr/bin/env python3
"""Fit an SCP-aware XResNet101 meta-model using folds 1-8 and fold 9 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from care_ecg.ensemble import (
    CARE_LABELS,
    PINNED_XRESNET101_SHA256,
    align_rows_by_record_id,
    build_scp_meta_features,
    classification_metrics,
    dump_sklearn_model,
    fit_scp_hist_gradient_booster,
    load_hash_verified_legacy_array,
    predict_scp_hist_gradient_booster,
    read_legacy_fold_ids,
    read_scp_statement_contract,
    require_sha256,
    sha256_file,
)


def prepared_split(path: Path, allowed_folds: set[int]) -> tuple[np.ndarray, np.ndarray]:
    if "test" in path.name.lower():
        raise ValueError("The validation-stage fitter refuses test-named inputs")
    with np.load(path, allow_pickle=False) as archive:
        record_ids = np.asarray(archive["record_ids"], dtype=str)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        folds = np.asarray(archive["folds"], dtype=np.int64)
    if record_ids.ndim != 1 or len(set(record_ids.tolist())) != len(record_ids):
        raise ValueError("Prepared record IDs must be one-dimensional and unique")
    if labels.shape != record_ids.shape or folds.shape != record_ids.shape:
        raise ValueError("Prepared IDs, labels, and folds are not aligned")
    observed_folds = set(np.unique(folds).tolist())
    if not len(record_ids) or observed_folds != allowed_folds:
        raise ValueError(
            f"Prepared split must contain exactly folds {sorted(allowed_folds)}, "
            f"received {sorted(observed_folds)}"
        )
    return record_ids, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--official-train-predictions", type=Path, required=True)
    parser.add_argument("--official-train-sha256", required=True)
    parser.add_argument("--official-validation-predictions", type=Path, required=True)
    parser.add_argument(
        "--official-validation-sha256",
        default=PINNED_XRESNET101_SHA256["validation"],
    )
    parser.add_argument("--official-database", type=Path, required=True)
    parser.add_argument("--database-sha256", default=PINNED_XRESNET101_SHA256["database"])
    parser.add_argument("--scp-statements", type=Path, required=True)
    parser.add_argument("--statements-sha256", default=PINNED_XRESNET101_SHA256["statements"])
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    guarded = (
        args.train_dataset,
        args.validation_dataset,
        args.official_train_predictions,
        args.official_validation_predictions,
        args.output_directory,
    )
    if any("test" in path.name.lower() for path in guarded):
        raise ValueError("This validation-stage fitter refuses test-named inputs")

    require_sha256(args.official_database, args.database_sha256)
    require_sha256(args.scp_statements, args.statements_sha256)
    scp_labels, diagnostic_classes = read_scp_statement_contract(args.scp_statements)
    train_raw = load_hash_verified_legacy_array(
        args.official_train_predictions,
        args.official_train_sha256,
        expected_columns=len(scp_labels),
    )
    validation_raw = load_hash_verified_legacy_array(
        args.official_validation_predictions,
        args.official_validation_sha256,
        expected_columns=len(scp_labels),
    )
    train_ids, train_labels = prepared_split(args.train_dataset, set(range(1, 9)))
    validation_ids, validation_labels = prepared_split(args.validation_dataset, {9})
    legacy_train_ids = read_legacy_fold_ids(args.official_database, set(range(1, 9)))
    legacy_validation_ids = read_legacy_fold_ids(args.official_database, {9})
    train_scp = align_rows_by_record_id(train_raw, legacy_train_ids, train_ids)
    validation_scp = align_rows_by_record_id(
        validation_raw, legacy_validation_ids, validation_ids
    )
    train_features, feature_names = build_scp_meta_features(
        train_scp, scp_labels, diagnostic_classes
    )
    validation_features, validation_feature_names = build_scp_meta_features(
        validation_scp, scp_labels, diagnostic_classes
    )
    if validation_feature_names != feature_names:
        raise AssertionError("Training and validation SCP feature contracts differ")

    model = fit_scp_hist_gradient_booster(train_features, train_labels)
    probabilities = predict_scp_hist_gradient_booster(model, validation_features)
    metrics = classification_metrics(validation_labels, probabilities.argmax(axis=1))

    args.output_directory.mkdir(parents=True, exist_ok=True)
    model_path = args.output_directory / "xresnet101_scp_hist_gradient_booster.joblib"
    model_sha256 = dump_sklearn_model(model, model_path)
    probabilities_path = args.output_directory / "validation_probabilities.npz"
    np.savez_compressed(
        probabilities_path,
        probabilities=probabilities.astype(np.float32),
        record_ids=validation_ids,
        label_order=np.asarray(CARE_LABELS),
        folds=np.full(len(validation_ids), 9, dtype=np.int64),
        source_model=np.asarray("official XResNet1D-101 exp0 plus SCP meta-model"),
        selection_scope=np.asarray("folds 1-8 fit; official fold 9 validation; fold 10 unopened"),
    )
    report = {
        "scope": "PTB-XL folds 1-8 fit and fold 9 validation; fold 10 was not opened",
        "method": (
            "HistGradientBoostingClassifier over official 71 SCP scores and "
            "25 fixed ontology aggregates"
        ),
        "feature_order": list(feature_names),
        "train_records": len(train_ids),
        "validation_records": len(validation_ids),
        "validation_metrics": metrics,
        "model": str(model_path.resolve()),
        "model_sha256": model_sha256,
        "validation_probabilities": str(probabilities_path.resolve()),
        "validation_probabilities_sha256": sha256_file(probabilities_path),
        "sources": {
            "official_train_predictions_sha256": sha256_file(
                args.official_train_predictions
            ),
            "official_validation_predictions_sha256": sha256_file(
                args.official_validation_predictions
            ),
            "official_database_sha256": sha256_file(args.official_database),
            "scp_statements_sha256": sha256_file(args.scp_statements),
        },
        "serialization": {
            "format": "joblib/pickle",
            "safe_loading": "verify model_sha256 before deserialization",
        },
    }
    report_path = args.output_directory / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
