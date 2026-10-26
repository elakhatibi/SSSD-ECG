#!/usr/bin/env python3
"""Select a fold-9-locked waveform specialist policy and refit on folds 1-9."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from care_ecg.ensemble import (
    CARE_LABELS,
    classification_metrics,
    dump_sklearn_model,
    load_probability_table,
    sha256_file,
)
from care_ecg.waveform_specialist import (
    MulticlassSpecialistPolicy,
    apply_multiclass_logit_adjustment,
    build_specialist_features,
    encode_demographics,
    fit_alternate_hyp_specialist,
    fit_demographic_medians,
    fit_ovr_specialists,
    predict_ovr_probabilities,
    read_demographics,
    specialist_feature_names,
    specialist_logits,
    waveform_feature_names,
)

BETA_GRID = (0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3)
BIAS_LEVELS = (-0.3, -0.15, 0.0, 0.15, 0.3)
TRANSFORMS = ("raw", "mean_centered", "zscore")
ALTERNATE_HYP_WEIGHT_POWERS = (0.5, 0.75)
TARGET_MACRO_RECALL = 0.72
TARGET_MACRO_F1 = 0.70


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


def calibration_parameters(
    training_probabilities: np.ndarray,
    transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    logits = specialist_logits(training_probabilities)
    if transform == "raw":
        return np.zeros(len(CARE_LABELS)), np.ones(len(CARE_LABELS))
    center = logits.mean(axis=0)
    if transform == "mean_centered":
        return center, np.ones(len(CARE_LABELS))
    if transform == "zscore":
        return center, logits.std(axis=0) + 1e-6
    raise ValueError(f"Unsupported specialist logit transform: {transform}")


def replace_hyp_model(models: tuple[Any, ...], hyp_model: Any) -> tuple[Any, ...]:
    if len(models) != len(CARE_LABELS):
        raise ValueError("Fixed OVR model family has the wrong number of classes")
    return (*models[:-1], hyp_model)


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
    if args.selection_lock.exists():
        raise FileExistsError(f"Selection lock already exists: {args.selection_lock}")

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
    selection_medians = fit_demographic_medians(train_raw_demographics)
    train_features = build_specialist_features(
        train_signals,
        train_raw_demographics,
        selection_medians,
        batch_size=args.batch_size,
    )
    validation_features = build_specialist_features(
        validation_signals,
        validation_raw_demographics,
        selection_medians,
        batch_size=args.batch_size,
    )
    del train_signals, validation_signals

    fixed_models = fit_ovr_specialists(train_features, train_labels)
    model_families: list[tuple[str, tuple[Any, ...], dict[str, object]]] = [
        (
            "fixed_l2_3",
            fixed_models,
            {"all_classes": "fixed HistGradientBoosting contract"},
        )
    ]
    for weight_power in ALTERNATE_HYP_WEIGHT_POWERS:
        hyp_model = fit_alternate_hyp_specialist(
            train_features,
            train_labels,
            class_weight_power=weight_power,
        )
        model_families.append(
            (
                f"alternate_hyp_weight_{weight_power:g}",
                replace_hyp_model(fixed_models, hyp_model),
                {
                    "classes_0_to_3": "fixed HistGradientBoosting contract",
                    "HYP": "lower-regularization predeclared variant",
                    "HYP_class_weight_power": weight_power,
                },
            )
        )

    baseline_metrics = classification_metrics(
        validation_labels,
        baseline.probabilities.argmax(axis=1),
    )
    candidate_summaries: list[dict[str, object]] = []
    selected: tuple[
        tuple[float, ...],
        str,
        tuple[Any, ...],
        dict[str, object],
        str,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, object],
    ] | None = None
    eligible_count = 0
    target_feasible_count = 0
    for family_name, models, family_contract in model_families:
        validation_specialist = predict_ovr_probabilities(models, validation_features)
        training_specialist = predict_ovr_probabilities(models, train_features)
        for transform in TRANSFORMS:
            center, scale = calibration_parameters(training_specialist, transform)
            for beta in BETA_GRID:
                for other_biases in itertools.product(BIAS_LEVELS, repeat=4):
                    biases = np.asarray((0.0, *other_biases), dtype=np.float64)
                    probabilities = apply_multiclass_logit_adjustment(
                        baseline.probabilities,
                        validation_specialist,
                        beta=beta,
                        biases=biases,
                        specialist_logit_center=center,
                        specialist_logit_scale=scale,
                    )
                    metrics = classification_metrics(
                        validation_labels,
                        probabilities.argmax(axis=1),
                    )
                    improves_recall = (
                        metrics["macro_recall"] > baseline_metrics["macro_recall"]
                    )
                    improves_f1 = metrics["macro_f1"] > baseline_metrics["macro_f1"]
                    target_feasible = (
                        metrics["macro_recall"] >= TARGET_MACRO_RECALL
                        and metrics["macro_f1"] >= TARGET_MACRO_F1
                    )
                    candidate_summaries.append(
                        {
                            "model_family": family_name,
                            "transform": transform,
                            "beta": beta,
                            "biases": biases.tolist(),
                            "accuracy": metrics["accuracy"],
                            "macro_precision": metrics["macro_precision"],
                            "macro_recall": metrics["macro_recall"],
                            "macro_f1": metrics["macro_f1"],
                            "improves_both": improves_recall and improves_f1,
                            "meets_numeric_targets": target_feasible,
                        }
                    )
                    if not (improves_recall and improves_f1):
                        continue
                    eligible_count += 1
                    target_feasible_count += int(target_feasible)
                    key = (
                        float(target_feasible),
                        float(metrics["macro_f1"]),
                        float(metrics["macro_recall"]),
                        float(metrics["accuracy"]),
                        -abs(beta),
                        -float(np.abs(biases).sum()),
                    )
                    if selected is None or key > selected[0]:
                        selected = (
                            key,
                            family_name,
                            models,
                            family_contract,
                            transform,
                            beta,
                            biases.copy(),
                            center.copy(),
                            scale.copy(),
                            validation_specialist.copy(),
                            probabilities.copy(),
                            metrics,
                        )
    if selected is None:
        raise RuntimeError("No predeclared specialist policy improves recall and F1")
    (
        _,
        selected_family,
        selected_models,
        selected_family_contract,
        selected_transform,
        selected_beta,
        selected_biases,
        selected_center,
        selected_scale,
        selected_specialist_probability,
        selected_probabilities,
        selected_metrics,
    ) = selected

    args.output_directory.mkdir(parents=True, exist_ok=True)
    validation_model_path = args.output_directory / "validation_multiclass_specialists.joblib"
    validation_model_sha256 = dump_sklearn_model(selected_models, validation_model_path)
    validation_probability_path = (
        args.output_directory / "validation_probabilities_multiclass_specialists.npz"
    )
    np.savez_compressed(
        validation_probability_path,
        probabilities=selected_probabilities.astype(np.float32),
        specialist_probabilities=selected_specialist_probability.astype(np.float32),
        record_ids=validation_ids,
        label_order=np.asarray(CARE_LABELS),
        folds=np.full(len(validation_ids), 9, dtype=np.int64),
        model_family=np.asarray(selected_family),
        transform=np.asarray(selected_transform),
        beta=np.asarray(selected_beta, dtype=np.float64),
        biases=selected_biases,
        specialist_logit_center=selected_center,
        specialist_logit_scale=selected_scale,
        selection_scope=np.asarray("fold-9 predeclared-grid model selection"),
    )

    candidate_summaries.sort(
        key=lambda row: (
            bool(row["meets_numeric_targets"]),
            float(row["macro_f1"]),
            float(row["macro_recall"]),
            float(row["accuracy"]),
        ),
        reverse=True,
    )
    lock = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "locked before fold-10 specialist inference",
        "test_artifact_or_label_opened_by_this_script": False,
        "method": (
            "five one-vs-rest HistGradientBoosting specialists over 252 deterministic "
            "waveform and 8 inference-safe demographic features"
        ),
        "feature_count": len(specialist_feature_names()),
        "feature_contract": list(specialist_feature_names()),
        "spectral_contract": (
            "six rFFT band-energy fractions exclude DC in numerator and denominator"
        ),
        "fixed_hist_gradient_boosting": {
            "max_iter": 150,
            "learning_rate": 0.07,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 3.0,
            "early_stopping": False,
            "random_state": 20261021,
            "class_weight_power": 0.5,
        },
        "alternate_hyp_candidate": {
            "max_iter": 250,
            "learning_rate": 0.06,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 25,
            "l2_regularization": 0.2,
            "early_stopping": True,
            "n_iter_no_change": 25,
            "random_state": 20261021,
            "class_weight_power_grid": list(ALTERNATE_HYP_WEIGHT_POWERS),
        },
        "selection": {
            "model_families": [name for name, _, _ in model_families],
            "transform_grid": list(TRANSFORMS),
            "beta_grid": list(BETA_GRID),
            "bias_grid": list(BIAS_LEVELS),
            "norm_bias_fixed_to_zero": True,
            "candidate_count": len(candidate_summaries),
            "eligible_count": eligible_count,
            "target_feasible_count": target_feasible_count,
            "eligibility": "strict improvement in macro recall and macro F1",
            "ranking": (
                "recall>=0.72 and F1>=0.70 feasibility, then macro F1, macro recall, "
                "accuracy, and smaller adjustment"
            ),
            "selected_model_family": selected_family,
            "selected_model_family_contract": selected_family_contract,
            "selected_transform": selected_transform,
            "selected_beta": selected_beta,
            "selected_biases": selected_biases.tolist(),
            "baseline_metrics": baseline_metrics,
            "selected_metrics": selected_metrics,
            "numeric_targets": {
                "macro_recall": TARGET_MACRO_RECALL,
                "macro_f1": TARGET_MACRO_F1,
                "both_met_on_validation": bool(
                    selected_metrics["macro_recall"] >= TARGET_MACRO_RECALL
                    and selected_metrics["macro_f1"] >= TARGET_MACRO_F1
                ),
            },
            "metric_interpretation": (
                "fold-9 selection estimate, not an independent test estimate; the base "
                "probabilities are patient-group cross-fitted and specialists use folds1-8"
            ),
            "top_100_candidates": candidate_summaries[:100],
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
    }
    args.selection_lock.parent.mkdir(parents=True, exist_ok=True)
    args.selection_lock.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    selection_lock_sha256 = sha256_file(args.selection_lock)

    final_raw_demographics = np.concatenate(
        (train_raw_demographics, validation_raw_demographics), axis=0
    )
    final_medians = fit_demographic_medians(final_raw_demographics)
    waveform_width = len(waveform_feature_names())
    final_features = np.concatenate(
        (
            np.concatenate(
                (
                    train_features[:, :waveform_width],
                    encode_demographics(train_raw_demographics, final_medians),
                ),
                axis=1,
            ),
            np.concatenate(
                (
                    validation_features[:, :waveform_width],
                    encode_demographics(validation_raw_demographics, final_medians),
                ),
                axis=1,
            ),
        ),
        axis=0,
    )
    final_labels = np.concatenate((train_labels, validation_labels))
    final_models = fit_ovr_specialists(final_features, final_labels)
    if selected_family.startswith("alternate_hyp_weight_"):
        selected_weight_power = float(selected_family.rsplit("_", maxsplit=1)[1])
        final_hyp_model = fit_alternate_hyp_specialist(
            final_features,
            final_labels,
            class_weight_power=selected_weight_power,
        )
        final_models = replace_hyp_model(final_models, final_hyp_model)
    final_specialist_probability = predict_ovr_probabilities(final_models, final_features)
    final_center, final_scale = calibration_parameters(
        final_specialist_probability,
        selected_transform,
    )
    final_model_path = args.output_directory / "multiclass_specialists_refit_folds1_9.joblib"
    final_model_sha256 = dump_sklearn_model(final_models, final_model_path)
    policy = MulticlassSpecialistPolicy(
        beta=selected_beta,
        biases=selected_biases,
        specialist_logit_center=final_center,
        specialist_logit_scale=final_scale,
        demographic_medians=final_medians,
        model_sha256=final_model_sha256,
        transform=selected_transform,
    )
    policy_path = args.output_directory / "multiclass_specialists_locked_policy.npz"
    policy.save(policy_path)
    report = {
        "selection_lock": str(args.selection_lock.resolve()),
        "selection_lock_sha256": selection_lock_sha256,
        "final_fit_scope": "prepared PTB-XL folds 1-9 after selection lock",
        "final_fit_records": len(final_labels),
        "final_model_family": selected_family,
        "final_model": str(final_model_path.resolve()),
        "final_model_sha256": final_model_sha256,
        "policy": str(policy_path.resolve()),
        "policy_sha256": sha256_file(policy_path),
        "validation_probabilities": str(validation_probability_path.resolve()),
        "validation_probabilities_sha256": sha256_file(validation_probability_path),
        "baseline_validation_metrics": baseline_metrics,
        "selected_validation_metrics": selected_metrics,
    }
    report_path = args.output_directory / "multiclass_specialists_refit_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
