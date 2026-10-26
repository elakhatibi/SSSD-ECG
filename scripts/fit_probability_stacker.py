#!/usr/bin/env python3
"""Select and fit a probability stacker on PTB-XL validation fold 9 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from care_ecg.ensemble import (
    CARE_LABELS,
    build_stacking_features,
    classification_metrics,
    fit_linear_probability_stacker,
    load_probability_table,
    patient_group_folds,
    sha256_file,
)

REGULARIZATION_GRID = (0.001, 0.01, 0.1, 1.0)
CLASS_WEIGHT_POWER_GRID = (0.0, 0.125, 0.25, 0.375, 0.5)


def parse_named_paths(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path:
            raise ValueError("Probability sources must use NAME=PATH syntax")
        result.append((name.strip(), Path(path)))
    names = [name for name, _ in result]
    if len(result) < 2 or len(names) != len(set(names)):
        raise ValueError("At least two uniquely named probability sources are required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--validation-source", action="append", default=[], required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    named_paths = parse_named_paths(args.validation_source)
    guarded = [args.validation_dataset, args.output_directory]
    guarded.extend(path for _, path in named_paths)
    if any("test" in path.name.lower() for path in guarded):
        raise ValueError("Validation-only stacker fitting refuses test-named paths")

    with np.load(args.validation_dataset, allow_pickle=False) as archive:
        record_ids = np.asarray(archive["record_ids"], dtype=str)
        patient_ids = np.asarray(archive["patient_ids"], dtype=str)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        folds = np.asarray(archive["folds"], dtype=np.int64)
    if folds.shape != record_ids.shape or not np.all(folds == 9):
        raise ValueError("Probability-stacker selection accepts official fold 9 only")
    if labels.shape != record_ids.shape or patient_ids.shape != record_ids.shape:
        raise ValueError("Validation IDs, patients, labels, and folds must be aligned")
    if labels.min() < 0 or labels.max() >= len(CARE_LABELS):
        raise ValueError("Validation labels are outside the CARE label order")

    sources = [
        (name, load_probability_table(path, expected_labels=CARE_LABELS))
        for name, path in named_paths
    ]
    features, feature_names, source_names = build_stacking_features(sources, record_ids)
    grouped_folds = patient_group_folds(labels, patient_ids)

    search: list[dict[str, object]] = []
    selected: tuple[tuple[float, float, float, float], float, float, np.ndarray] | None = None
    for regularization in REGULARIZATION_GRID:
        for class_weight_power in CLASS_WEIGHT_POWER_GRID:
            oof_probabilities = np.zeros((len(labels), len(CARE_LABELS)), dtype=np.float64)
            for fold_index in range(5):
                held_out = grouped_folds == fold_index
                fitted = fit_linear_probability_stacker(
                    features[~held_out],
                    labels[~held_out],
                    feature_names,
                    source_names,
                    regularization=regularization,
                    class_weight_power=class_weight_power,
                )
                oof_probabilities[held_out] = fitted.predict_features(features[held_out])
            metrics = classification_metrics(labels, oof_probabilities.argmax(axis=1))
            search.append(
                {
                    "regularization": regularization,
                    "class_weight_power": class_weight_power,
                    "metrics": metrics,
                }
            )
            key = (
                float(metrics["macro_f1"]),
                float(metrics["accuracy"]),
                -class_weight_power,
                -regularization,
            )
            if selected is None or key > selected[0]:
                selected = (
                    key,
                    regularization,
                    class_weight_power,
                    oof_probabilities.copy(),
                )
    if selected is None:
        raise AssertionError("Stacker hyperparameter grid is empty")
    _, regularization, class_weight_power, selected_oof = selected
    selected_metrics = classification_metrics(labels, selected_oof.argmax(axis=1))
    final_model = fit_linear_probability_stacker(
        features,
        labels,
        feature_names,
        source_names,
        regularization=regularization,
        class_weight_power=class_weight_power,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    model_path = args.output_directory / "locked_probability_stacker.npz"
    model_sha256 = final_model.save(model_path)
    oof_path = args.output_directory / "validation_oof_probabilities.npz"
    np.savez_compressed(
        oof_path,
        probabilities=selected_oof.astype(np.float32),
        record_ids=record_ids,
        label_order=np.asarray(CARE_LABELS),
        patient_group_fold=grouped_folds,
        source_order=np.asarray(source_names),
        selection_scope=np.asarray("patient-grouped cross-fit on official fold 9 only"),
    )
    source_provenance = [
        {"name": name, "path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in named_paths
    ]
    report = {
        "scope": "official PTB-XL validation fold 9 only; fold 10 was not opened",
        "method": "regularized multinomial linear stack over concatenated log probabilities",
        "selection": (
            "five-fold patient-grouped cross-fit; hyperparameters selected by pooled OOF "
            "macro-F1 then accuracy"
        ),
        "metric_interpretation": (
            "selected_oof_metrics reuse the OOF predictions used for hyperparameter selection; "
            "they are a selection estimate, not an independent nested-validation estimate. "
            "Cross-fitting applies to the stacker only; source checkpoints retain their own "
            "documented validation histories."
        ),
        "validation_records": len(labels),
        "validation_patients": len(np.unique(patient_ids)),
        "group_fold_sizes": np.bincount(grouped_folds, minlength=5).tolist(),
        "feature_sources": source_provenance,
        "feature_order": list(feature_names),
        "selected_regularization": regularization,
        "selected_class_weight_power": class_weight_power,
        "selected_oof_metrics": selected_metrics,
        "candidate_search": search,
        "final_fit": "all fold-9 records after patient-grouped cross-fit selection",
        "model": str(model_path.resolve()),
        "model_sha256": model_sha256,
        "oof_probabilities": str(oof_path.resolve()),
        "oof_probabilities_sha256": sha256_file(oof_path),
    }
    report_path = args.output_directory / "probability_stacker_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
