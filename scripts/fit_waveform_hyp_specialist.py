#!/usr/bin/env python3
"""Select a validation-locked HYP rescue policy and refit it on folds 1-9."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from care_ecg.ensemble import (
    CARE_LABELS,
    classification_metrics,
    dump_sklearn_model,
    load_probability_table,
    sha256_file,
)
from care_ecg.waveform_specialist import (
    HypSpecialistPolicy,
    apply_hyp_logit_adjustment,
    build_specialist_features,
    fit_demographic_medians,
    fit_hyp_specialist,
    predict_hyp_probability,
    read_demographics,
    specialist_feature_names,
)

BETA_GRID = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
BIAS_GRID = (-0.25, -0.15, -0.05, 0.0, 0.05, 0.1, 0.15, 0.175, 0.2, 0.25, 0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--baseline-validation", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def load_training_split(
    path: Path,
    expected_folds: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        signals = np.asarray(archive["signals"], dtype=np.float32)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        record_ids = np.asarray(archive["record_ids"], dtype=str)
        folds = np.asarray(archive["folds"], dtype=np.int64)
    if signals.shape[0] != len(labels) or labels.shape != record_ids.shape:
        raise ValueError(f"Dataset arrays are not aligned in {path}")
    if folds.shape != labels.shape or set(np.unique(folds).tolist()) != expected_folds:
        raise ValueError(f"Dataset {path} must contain exactly folds {sorted(expected_folds)}")
    if len(set(record_ids.tolist())) != len(record_ids):
        raise ValueError(f"Dataset record IDs are not unique in {path}")
    return signals, labels, record_ids


def main() -> None:
    args = parse_args()
    guarded = (
        args.train_dataset,
        args.validation_dataset,
        args.baseline_validation,
        args.output_directory,
        args.selection_lock,
    )
    if any("test" in str(path).lower() for path in guarded):
        raise ValueError("Validation selection refuses test-named inputs and outputs")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")

    train_signals, train_labels, train_ids = load_training_split(
        args.train_dataset, set(range(1, 9))
    )
    validation_signals, validation_labels, validation_ids = load_training_split(
        args.validation_dataset, {9}
    )
    baseline = load_probability_table(
        args.baseline_validation,
        expected_labels=CARE_LABELS,
    )
    if not np.array_equal(baseline.record_ids, validation_ids):
        raise ValueError("Baseline validation probabilities are not exactly row-aligned")

    train_raw_demographics = read_demographics(args.database, train_ids)
    validation_raw_demographics = read_demographics(args.database, validation_ids)
    validation_medians = fit_demographic_medians(train_raw_demographics)
    train_features = build_specialist_features(
        train_signals,
        train_raw_demographics,
        validation_medians,
        batch_size=args.batch_size,
    )
    validation_features = build_specialist_features(
        validation_signals,
        validation_raw_demographics,
        validation_medians,
        batch_size=args.batch_size,
    )
    validation_model = fit_hyp_specialist(train_features, train_labels)
    specialist_probability = predict_hyp_probability(
        validation_model,
        validation_features,
    )

    baseline_metrics = classification_metrics(
        validation_labels,
        baseline.probabilities.argmax(axis=1),
    )
    search: list[dict[str, object]] = []
    eligible: list[tuple[tuple[float, ...], float, float, np.ndarray, dict[str, object]]] = []
    for beta in BETA_GRID:
        for bias in BIAS_GRID:
            probabilities = apply_hyp_logit_adjustment(
                baseline.probabilities,
                specialist_probability,
                beta=beta,
                bias=bias,
            )
            metrics = classification_metrics(
                validation_labels,
                probabilities.argmax(axis=1),
            )
            improved_recall = metrics["macro_recall"] > baseline_metrics["macro_recall"]
            improved_f1 = metrics["macro_f1"] > baseline_metrics["macro_f1"]
            row = {
                "beta": beta,
                "bias": bias,
                "improves_macro_recall": improved_recall,
                "improves_macro_f1": improved_f1,
                "metrics": metrics,
            }
            search.append(row)
            if improved_recall and improved_f1:
                key = (
                    float(metrics["macro_f1"]),
                    float(metrics["macro_recall"]),
                    float(metrics["accuracy"]),
                    -abs(beta),
                    -abs(bias),
                )
                eligible.append((key, beta, bias, probabilities, row))
    if not eligible:
        raise RuntimeError(
            "No predeclared HYP policy improves both macro recall and macro F1"
        )
    _, selected_beta, selected_bias, selected_probabilities, selected_row = max(eligible)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    validation_model_path = args.output_directory / "validation_hyp_specialist.joblib"
    validation_model_sha256 = dump_sklearn_model(
        validation_model,
        validation_model_path,
    )
    validation_probability_path = (
        args.output_directory / "validation_probabilities_hyp_specialist.npz"
    )
    np.savez_compressed(
        validation_probability_path,
        probabilities=selected_probabilities.astype(np.float32),
        hyp_specialist_probability=specialist_probability.astype(np.float32),
        record_ids=validation_ids,
        label_order=np.asarray(CARE_LABELS),
        folds=np.full(len(validation_ids), 9, dtype=np.int64),
        beta=np.asarray(selected_beta, dtype=np.float64),
        bias=np.asarray(selected_bias, dtype=np.float64),
        selection_scope=np.asarray("fold-9 predeclared-grid model selection"),
    )

    lock = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "locked before fold-10 specialist inference",
        "test_artifact_or_label_opened_by_this_script": False,
        "method": (
            "folds1-8 HistGradientBoosting HYP specialist over 252 deterministic "
            "waveform features and 8 inference-safe demographic features"
        ),
        "feature_count": len(specialist_feature_names()),
        "feature_contract": list(specialist_feature_names()),
        "hist_gradient_boosting": {
            "max_iter": 150,
            "learning_rate": 0.07,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 3.0,
            "early_stopping": False,
            "random_state": 20261021,
            "class_weight_power": 0.5,
        },
        "selection": {
            "beta_grid": list(BETA_GRID),
            "bias_grid": list(BIAS_GRID),
            "eligibility": "strict improvement in macro recall and macro F1",
            "ranking": "macro F1, macro recall, accuracy, smaller adjustment",
            "selected_beta": selected_beta,
            "selected_bias": selected_bias,
            "baseline_metrics": baseline_metrics,
            "selected_metrics": selected_row["metrics"],
            "metric_interpretation": (
                "fold-9 model-selection estimate; the fixed base stack probabilities "
                "are patient-group cross-fitted, while the specialist is fit on folds1-8"
            ),
        },
        "inputs": {
            "train_dataset": str(args.train_dataset.resolve()),
            "train_dataset_sha256": sha256_file(args.train_dataset),
            "validation_dataset": str(args.validation_dataset.resolve()),
            "validation_dataset_sha256": sha256_file(args.validation_dataset),
            "database": str(args.database.resolve()),
            "database_sha256": sha256_file(args.database),
            "baseline_validation": str(args.baseline_validation.resolve()),
            "baseline_validation_sha256": sha256_file(args.baseline_validation),
        },
        "validation_artifacts": {
            "model": str(validation_model_path.resolve()),
            "model_sha256": validation_model_sha256,
            "probabilities": str(validation_probability_path.resolve()),
            "probabilities_sha256": sha256_file(validation_probability_path),
        },
        "search": search,
    }
    args.selection_lock.parent.mkdir(parents=True, exist_ok=True)
    args.selection_lock.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    selection_lock_sha256 = sha256_file(args.selection_lock)

    final_labels = np.concatenate((train_labels, validation_labels))
    final_raw_demographics = np.concatenate(
        (train_raw_demographics, validation_raw_demographics), axis=0
    )
    final_medians = fit_demographic_medians(final_raw_demographics)
    final_features = np.concatenate(
        (
            build_specialist_features(
                train_signals,
                train_raw_demographics,
                final_medians,
                batch_size=args.batch_size,
            ),
            build_specialist_features(
                validation_signals,
                validation_raw_demographics,
                final_medians,
                batch_size=args.batch_size,
            ),
        ),
        axis=0,
    )
    final_model = fit_hyp_specialist(final_features, final_labels)
    final_model_path = args.output_directory / "hyp_specialist_refit_folds1_9.joblib"
    final_model_sha256 = dump_sklearn_model(final_model, final_model_path)
    policy = HypSpecialistPolicy(
        beta=selected_beta,
        bias=selected_bias,
        demographic_medians=final_medians,
        model_sha256=final_model_sha256,
    )
    policy_path = args.output_directory / "hyp_specialist_locked_policy.npz"
    policy.save(policy_path)
    report = {
        "selection_lock": str(args.selection_lock.resolve()),
        "selection_lock_sha256": selection_lock_sha256,
        "final_fit_scope": "prepared PTB-XL folds 1-9 after selection lock",
        "final_fit_records": len(final_labels),
        "final_model": str(final_model_path.resolve()),
        "final_model_sha256": final_model_sha256,
        "policy": str(policy_path.resolve()),
        "policy_sha256": sha256_file(policy_path),
        "validation_probabilities": str(validation_probability_path.resolve()),
        "validation_probabilities_sha256": sha256_file(validation_probability_path),
        "validation_metrics": selected_row["metrics"],
    }
    report_path = args.output_directory / "hyp_specialist_refit_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
