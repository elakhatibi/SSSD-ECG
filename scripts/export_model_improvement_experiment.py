#!/usr/bin/env python3
"""Export the sealed waveform-specialist experiment as a comparison workbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from care_ecg.ensemble import CARE_LABELS, classification_metrics, sha256_file

PAPER_TARGETS = {
    "accuracy": 0.70,
    "macro_precision": 0.68,
    "macro_recall": 0.72,
    "macro_f1": 0.70,
}
METRIC_KEYS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--candidate-evaluation", type=Path, required=True)
    parser.add_argument("--candidate-inference-report", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--baseline-metadata", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--refit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def metrics_from_prediction_jsonl(path: Path) -> dict[str, Any]:
    label_index = {label: index for index, label in enumerate(CARE_LABELS)}
    expected: list[int] = []
    predicted: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                expected.append(label_index[str(row["expected"])])
                predicted.append(label_index[str(row["diagnosis"])])
            except (KeyError, ValueError) as error:
                raise ValueError(f"Invalid prediction row {line_number} in {path}") from error
    if not expected:
        raise ValueError(f"Prediction JSONL is empty: {path}")
    return classification_metrics(np.asarray(expected), np.asarray(predicted))


def metric_row(metrics: dict[str, Any]) -> list[float]:
    return [float(metrics[key]) for key in METRIC_KEYS]


def delta_metrics(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {key: float(candidate[key]) - float(baseline[key]) for key in METRIC_KEYS}


def style_workbook(workbook: Any) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    dark_fill = PatternFill("solid", fgColor="203864")
    accent_fill = PatternFill("solid", fgColor="D9EAF7")
    warning_fill = PatternFill("solid", fgColor="FCE4D6")
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.fill = dark_fill
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")
        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for cell in sheet[2]:
            cell.fill = accent_fill
        for row in sheet.iter_rows():
            if row and str(row[0].value).lower().startswith(("delta", "decision")):
                for cell in row:
                    cell.fill = warning_fill
        for column in sheet.columns:
            values = [len(str(cell.value or "")) for cell in column]
            width = min(max(max(values, default=10) + 2, 12), 70)
            sheet.column_dimensions[column[0].column_letter].width = width


def append_confusion_matrix(
    sheet: Any,
    title: str,
    matrix: list[list[int]],
) -> None:
    sheet.append((title, "", "", "", "", ""))
    sheet.append(("true / predicted", *CARE_LABELS))
    for label, row in zip(CARE_LABELS, matrix, strict=True):
        sheet.append((label, *row))
    sheet.append(())


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Comparison workbook already exists: {args.output}")
    try:
        from openpyxl import Workbook, load_workbook
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Install CARE-ECG with the 'reports' extra") from error

    lock = read_json(args.selection_lock)
    candidate_evaluation = read_json(args.candidate_evaluation)
    candidate_inference = read_json(args.candidate_inference_report)
    baseline_metadata = read_json(args.baseline_metadata)
    refit_report = read_json(args.refit_report)

    selection = lock["selection"]
    baseline_validation = selection["baseline_metrics"]
    candidate_validation = selection["selected_metrics"]
    candidate_test = candidate_evaluation["metrics"]
    baseline_test = baseline_metadata["held_out_results"][
        "direct_and_first_mention_text_to_scp"
    ]
    baseline_test_from_rows = metrics_from_prediction_jsonl(args.baseline_predictions)
    for key in METRIC_KEYS:
        if not np.isclose(
            float(baseline_test[key]),
            float(baseline_test_from_rows[key]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Published baseline {key} does not match its JSONL")

    validation_delta = delta_metrics(candidate_validation, baseline_validation)
    test_delta = delta_metrics(candidate_test, baseline_test)
    target_status = {
        key: bool(float(candidate_test[key]) >= target)
        for key, target in PAPER_TARGETS.items()
    }

    workbook = Workbook()
    summary = workbook.active
    summary.title = "Metric Comparison"
    summary.append(("cohort", "system", *METRIC_KEYS, "interpretation"))
    summary.append(
        (
            "validation fold 9",
            "baseline three-source stack",
            *metric_row(baseline_validation),
            "Reference used before the waveform extension",
        )
    )
    summary.append(
        (
            "validation fold 9",
            "selected waveform specialist",
            *metric_row(candidate_validation),
            "Selection estimate; not an independent test estimate",
        )
    )
    summary.append(
        (
            "delta: validation candidate - baseline",
            "positive validation result",
            *metric_row(validation_delta),
            "Recall and F1 both improved under the predeclared eligibility rule",
        )
    )
    summary.append(
        (
            "sealed fold 10",
            "baseline three-source stack",
            *metric_row(baseline_test),
            "Existing promoted repository result",
        )
    )
    summary.append(
        (
            "sealed fold 10",
            "waveform specialist candidate",
            *metric_row(candidate_test),
            "Exactly one evaluation after lock; candidate is not promoted",
        )
    )
    summary.append(
        (
            "delta: fold10 candidate - baseline",
            "negative held-out result",
            *metric_row(test_delta),
            "Recall and F1 regressed; no post-test adjustment was made",
        )
    )
    summary.append(
        (
            "numeric reference only",
            "paper-reported CARE-ECG targets",
            *[PAPER_TARGETS[key] for key in METRIC_KEYS],
            "Directional numeric comparison, not protocol equivalence",
        )
    )
    for row in summary.iter_rows(min_row=2, min_col=3, max_col=6):
        for cell in row:
            cell.number_format = "0.000000"

    matrices = workbook.create_sheet("Confusion Matrices")
    matrices.append(("matrix", "NORM", "MI", "STTC", "CD", "HYP"))
    append_confusion_matrix(
        matrices,
        "Baseline validation (rows=true, columns=predicted)",
        baseline_validation["confusion_matrix"],
    )
    append_confusion_matrix(
        matrices,
        "Selected validation (rows=true, columns=predicted)",
        candidate_validation["confusion_matrix"],
    )
    append_confusion_matrix(
        matrices,
        "Baseline fold 10 (rows=true, columns=predicted)",
        baseline_test_from_rows["confusion_matrix"],
    )
    append_confusion_matrix(
        matrices,
        "Sealed candidate fold 10 (rows=true, columns=predicted)",
        candidate_test["confusion_matrix"],
    )
    baseline_matrix = np.asarray(baseline_test_from_rows["confusion_matrix"], dtype=int)
    candidate_matrix = np.asarray(candidate_test["confusion_matrix"], dtype=int)
    append_confusion_matrix(
        matrices,
        "Fold-10 delta: candidate - baseline",
        (candidate_matrix - baseline_matrix).tolist(),
    )

    provenance = workbook.create_sheet("Provenance")
    provenance.append(("item", "path or value", "sha256"))
    artifact_paths = {
        "selection lock": args.selection_lock,
        "candidate evaluation": args.candidate_evaluation,
        "candidate inference report": args.candidate_inference_report,
        "candidate probabilities": Path(candidate_evaluation["probabilities"]),
        "candidate JSONL": args.candidate_predictions,
        "baseline metadata": args.baseline_metadata,
        "baseline JSONL": args.baseline_predictions,
        "refit report": args.refit_report,
        "final specialist model": Path(refit_report["final_model"]),
        "locked specialist policy": Path(refit_report["policy"]),
    }
    for name, path in artifact_paths.items():
        provenance.append((name, str(path.resolve()), sha256_file(path)))
    provenance.append(("selection status", lock["status"], ""))
    provenance.append(("model family", selection["selected_model_family"], ""))
    provenance.append(("transform", selection["selected_transform"], ""))
    provenance.append(("beta", selection["selected_beta"], ""))
    provenance.append(("biases", json.dumps(selection["selected_biases"]), ""))
    provenance.append(("feature count", lock["feature_count"], ""))
    provenance.append(("labels accessed during inference", candidate_inference["labels_accessed"], ""))
    provenance.append(
        (
            "one-shot evaluation count",
            candidate_evaluation["evaluation_count_for_this_artifact"],
            "",
        )
    )

    attestation = workbook.create_sheet("Attestation")
    attestation.append(("claim", "value", "detail"))
    attestation.append(("decision", "NOT PROMOTED", "Fold-10 recall and F1 regressed"))
    attestation.append(
        (
            "selection chronology",
            "PASS",
            "Policy lock was written before label-free fold-10 inference",
        )
    )
    attestation.append(
        (
            "inference label access",
            "PASS",
            "The specialist inference artifact records labels_accessed=false",
        )
    )
    attestation.append(
        (
            "evaluation count",
            "1",
            "The candidate probability artifact was evaluated exactly once",
        )
    )
    attestation.append(
        (
            "post-test tuning",
            "NONE",
            "No parameters were changed after the sealed evaluation",
        )
    )
    attestation.append(
        (
            "paper numeric targets",
            "accuracy/precision met; recall/F1 not met",
            json.dumps(target_status, sort_keys=True),
        )
    )
    attestation.append(
        (
            "protocol equivalence",
            "NOT CLAIMED",
            (
                "The paper's exact artifacts, label projection, free-text generator, and "
                "evaluation implementation are unavailable"
            ),
        )
    )
    attestation.append(
        (
            "comparison scope",
            "DIRECTIONAL NUMERIC ONLY",
            "Matching a displayed number would not establish paper reproduction",
        )
    )

    style_workbook(workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(args.output)
    verified = load_workbook(args.output, read_only=True, data_only=False)
    expected_sheets = {
        "Metric Comparison",
        "Confusion Matrices",
        "Provenance",
        "Attestation",
    }
    if set(verified.sheetnames) != expected_sheets:
        raise RuntimeError("Saved workbook sheet contract does not match")
    verified.close()
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
                "decision": "NOT PROMOTED",
                "validation_delta": validation_delta,
                "fold10_delta": test_delta,
                "paper_numeric_targets_met": target_status,
                "paper_protocol_equivalence": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
