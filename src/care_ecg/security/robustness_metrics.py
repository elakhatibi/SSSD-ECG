"""Automated adversarial-robustness metrics for CARE-ECG.

These model-level endpoints complement, but never replace, the independently
adjudicated clinical incorrect-release endpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .contracts import AttackAttempt, AttackResult, VictimSnapshot
from .metrics import patient_bootstrap


@dataclass(frozen=True, slots=True)
class WaveformDistortion:
    l1_mv: float
    l2_mv: float
    linf_mv: float
    mean_absolute_error_mv: float
    rmse_mv: float
    max_per_lead_l2_mv: float
    relative_l2: float | None
    prd_percent: float | None
    snr_db: float | None
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AutomatedAttackMetrics:
    expected_clean_label: str
    clean_diagnosis: str
    clean_correct: bool
    selected_diagnosis: str | None
    selected_correct: bool
    robust_correct: bool
    targeted_diagnosis_success: bool
    untargeted_diagnosis_success: bool
    released_target_claim_success: bool
    selected_constraint_satisfied: bool | None
    selected_query_index: int | None
    first_targeted_diagnosis_query: int | None
    first_untargeted_diagnosis_query: int | None
    first_released_target_claim_query: int | None
    clean_target_probability: float | None
    selected_target_probability: float | None
    target_probability_gain: float | None
    target_log_probability_gain: float | None
    first_targeted_distortion: WaveformDistortion | None
    distortion: WaveformDistortion | None
    endpoint_classification: str = "automated_nonclinical"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def waveform_distortion(
    clean_waveform: np.ndarray,
    adversarial_waveform: np.ndarray,
    *,
    zero_tolerance: float = 1e-12,
) -> WaveformDistortion:
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("zero_tolerance must be finite and non-negative")
    clean = np.asarray(clean_waveform, dtype=np.float64)
    adversarial = np.asarray(adversarial_waveform, dtype=np.float64)
    if clean.ndim != 2 or adversarial.shape != clean.shape or not clean.size:
        raise ValueError("Waveforms must be non-empty and aligned [leads, samples] arrays")
    if not np.isfinite(clean).all() or not np.isfinite(adversarial).all():
        raise ValueError("Waveforms must contain only finite values")

    delta = adversarial - clean
    signal_l2 = float(np.linalg.norm(clean))
    noise_l2 = float(np.linalg.norm(delta))
    relative_l2 = None if signal_l2 <= zero_tolerance else float(noise_l2 / signal_l2)
    snr_db = (
        None
        if signal_l2 <= zero_tolerance or noise_l2 <= zero_tolerance
        else float(20.0 * np.log10(signal_l2 / noise_l2))
    )
    return WaveformDistortion(
        l1_mv=float(np.linalg.norm(delta.ravel(), ord=1)),
        l2_mv=noise_l2,
        linf_mv=float(np.max(np.abs(delta))),
        mean_absolute_error_mv=float(np.mean(np.abs(delta))),
        rmse_mv=float(np.sqrt(np.mean(np.square(delta)))),
        max_per_lead_l2_mv=float(np.max(np.linalg.norm(delta, axis=1))),
        relative_l2=relative_l2,
        prd_percent=None if relative_l2 is None else float(100.0 * relative_l2),
        snr_db=snr_db,
        samples=int(delta.size),
    )


def _constraint_satisfied(attempt: AttackAttempt) -> bool:
    value = attempt.metadata.get("constraint_satisfied")
    return bool(attempt.admissible) if value is None else bool(value)


def _selected_attempt(result: AttackResult) -> AttackAttempt | None:
    selected_query = result.metadata.get("selected_attempt_query_index")
    if selected_query is not None:
        selected = next(
            (attempt for attempt in result.attempts if attempt.query_index == int(selected_query)),
            None,
        )
        if selected is not None:
            return selected
    if result.adversarial_signal is not None:
        matches = [
            attempt
            for attempt in result.attempts
            if np.array_equal(attempt.waveform, result.adversarial_signal)
        ]
        if matches:
            return matches[-1]
    return result.attempts[-1] if result.attempts else None


def automated_attack_metrics(
    result: AttackResult,
    clean_waveform: np.ndarray,
    *,
    expected_clean_label: str,
) -> AutomatedAttackMetrics:
    expected = str(expected_clean_label).strip()
    if not expected:
        raise ValueError("expected_clean_label must not be empty")
    if not isinstance(result.clean_snapshot, VictimSnapshot):
        raise TypeError("Automated robustness requires a typed clean VictimSnapshot")

    clean_diagnosis = result.clean_snapshot.diagnosis
    clean_correct = clean_diagnosis == expected
    valid_attempts = [attempt for attempt in result.attempts if _constraint_satisfied(attempt)]
    targeted_attempts = [
        attempt for attempt in valid_attempts if attempt.diagnosis == result.target_label
    ]
    untargeted_attempts = [attempt for attempt in valid_attempts if attempt.diagnosis != expected]
    released_claim_attempts = [
        attempt for attempt in valid_attempts if attempt.released and attempt.target_claim_present
    ]
    selected = _selected_attempt(result)
    selected_valid = None if selected is None else _constraint_satisfied(selected)
    clean_probability = result.clean_snapshot.probabilities.get(result.target_label)
    selected_probability = None if selected is None else float(selected.target_probability)
    return AutomatedAttackMetrics(
        expected_clean_label=expected,
        clean_diagnosis=clean_diagnosis,
        clean_correct=clean_correct,
        selected_diagnosis=None if selected is None else selected.diagnosis,
        selected_correct=bool(
            clean_correct
            if selected is None or not selected_valid
            else selected.diagnosis == expected
        ),
        robust_correct=bool(clean_correct and not untargeted_attempts),
        targeted_diagnosis_success=bool(targeted_attempts),
        untargeted_diagnosis_success=bool(untargeted_attempts),
        released_target_claim_success=bool(released_claim_attempts),
        selected_constraint_satisfied=selected_valid,
        selected_query_index=None if selected is None else selected.query_index,
        first_targeted_diagnosis_query=(
            None
            if not targeted_attempts
            else min(attempt.query_index for attempt in targeted_attempts)
        ),
        first_untargeted_diagnosis_query=(
            None
            if not untargeted_attempts
            else min(attempt.query_index for attempt in untargeted_attempts)
        ),
        first_released_target_claim_query=(
            None
            if not released_claim_attempts
            else min(attempt.query_index for attempt in released_claim_attempts)
        ),
        clean_target_probability=(None if clean_probability is None else float(clean_probability)),
        selected_target_probability=selected_probability,
        target_probability_gain=(
            None
            if clean_probability is None or selected_probability is None
            else selected_probability - float(clean_probability)
        ),
        target_log_probability_gain=(
            None
            if clean_probability is None or selected_probability is None
            else float(
                np.log(max(selected_probability, np.finfo(float).tiny))
                - np.log(max(float(clean_probability), np.finfo(float).tiny))
            )
        ),
        first_targeted_distortion=(
            None
            if not targeted_attempts
            else waveform_distortion(clean_waveform, targeted_attempts[0].waveform)
        ),
        distortion=(
            None if selected is None else waveform_distortion(clean_waveform, selected.waveform)
        ),
    )


def _bootstrap(
    values: Sequence[float | bool],
    patient_ids: Sequence[str],
    *,
    metric: str,
    n_resamples: int,
    confidence: float,
    seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return patient_bootstrap(
        values,
        patient_ids,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
        metadata={"metric": metric, **dict(metadata or {})},
    ).to_dict()


def _not_evaluable(
    metric: str,
    *,
    reason: str,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "estimate": None,
        "lower": None,
        "upper": None,
        "confidence": confidence,
        "n_patients": 0,
        "n_resamples": 0,
        "seed": seed,
        "metadata": {
            "metric": metric,
            "status": "not_evaluable",
            "reason": reason,
        },
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _distortion_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str = "distortion",
) -> dict[str, Any]:
    items = [row.get(field) for row in rows if isinstance(row.get(field), Mapping)]
    names = (
        "l1_mv",
        "l2_mv",
        "linf_mv",
        "mean_absolute_error_mv",
        "rmse_mv",
        "max_per_lead_l2_mv",
        "relative_l2",
        "prd_percent",
        "snr_db",
    )
    return {
        name: _numeric_summary(
            [float(item[name]) for item in items if item is not None and item.get(name) is not None]
        )
        for name in names
    }


def summarize_automated_robustness(
    metrics: Sequence[AutomatedAttackMetrics | Mapping[str, Any]],
    patient_ids: Sequence[str],
    *,
    query_budgets: Sequence[int] = (1, 10, 50, 100),
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    rows = [
        metric.to_dict() if isinstance(metric, AutomatedAttackMetrics) else dict(metric)
        for metric in metrics
    ]
    normalized_ids = [str(value).strip() for value in patient_ids]
    if not rows or len(rows) != len(normalized_ids) or any(not value for value in normalized_ids):
        raise ValueError("metrics and patient_ids must be non-empty and aligned")
    budgets = sorted({int(value) for value in query_budgets})
    if not budgets or budgets[0] < 1:
        raise ValueError("query_budgets must contain positive integers")

    clean = [bool(row["clean_correct"]) for row in rows]
    robust = [bool(row["robust_correct"]) for row in rows]
    clean_indices = [index for index, value in enumerate(clean) if value]
    clean_ids = [normalized_ids[index] for index in clean_indices]
    bootstrap_options = {
        "n_resamples": n_resamples,
        "confidence": confidence,
        "seed": seed,
    }

    def conditioned(field: str, name: str) -> dict[str, Any]:
        if not clean_indices:
            return _not_evaluable(
                name,
                reason="no clean-correct observations",
                confidence=confidence,
                seed=seed,
            )
        return _bootstrap(
            [bool(rows[index][field]) for index in clean_indices],
            clean_ids,
            metric=name,
            metadata={"conditioning": "clean_correct"},
            **bootstrap_options,
        )

    targeted_by_budget: dict[int, dict[str, Any]] = {}
    robust_by_budget: dict[int, dict[str, Any]] = {}
    for budget in budgets:
        if clean_indices:
            targeted_by_budget[budget] = _bootstrap(
                [
                    rows[index].get("first_targeted_diagnosis_query") is not None
                    and int(rows[index]["first_targeted_diagnosis_query"]) <= budget
                    for index in clean_indices
                ],
                clean_ids,
                metric="targeted_diagnosis_asr_at_budget",
                metadata={
                    "conditioning": "clean_correct",
                    "query_budget": budget,
                },
                **bootstrap_options,
            )
        else:
            targeted_by_budget[budget] = _not_evaluable(
                "targeted_diagnosis_asr_at_budget",
                reason="no clean-correct observations",
                confidence=confidence,
                seed=seed,
            )
        robust_by_budget[budget] = _bootstrap(
            [
                bool(row["clean_correct"])
                and not (
                    row.get("first_untargeted_diagnosis_query") is not None
                    and int(row["first_untargeted_diagnosis_query"]) <= budget
                )
                for row in rows
            ],
            normalized_ids,
            metric="robust_accuracy_at_budget",
            metadata={"query_budget": budget},
            **bootstrap_options,
        )

    targeted_successes = [
        row
        for row in rows
        if bool(row["clean_correct"])
        and bool(row["targeted_diagnosis_success"])
    ]
    valid_selected = [row for row in rows if bool(row.get("selected_constraint_satisfied", False))]
    return {
        "endpoint_classification": "automated_nonclinical",
        "observations": len(rows),
        "patients": len(set(normalized_ids)),
        "clean_correct_observations": len(clean_indices),
        "clean_accuracy": _bootstrap(
            clean,
            normalized_ids,
            metric="clean_accuracy",
            **bootstrap_options,
        ),
        "robust_accuracy": _bootstrap(
            robust,
            normalized_ids,
            metric="robust_accuracy",
            metadata={"definition": "clean-correct with no valid misclassification query"},
            **bootstrap_options,
        ),
        "clean_to_robust_accuracy_drop": _bootstrap(
            [
                float(clean_value) - float(robust_value)
                for clean_value, robust_value in zip(clean, robust, strict=True)
            ],
            normalized_ids,
            metric="clean_to_robust_accuracy_drop",
            **bootstrap_options,
        ),
        "targeted_diagnosis_attack_success_rate": conditioned(
            "targeted_diagnosis_success",
            "targeted_diagnosis_attack_success_rate",
        ),
        "untargeted_diagnosis_attack_success_rate": conditioned(
            "untargeted_diagnosis_success",
            "untargeted_diagnosis_attack_success_rate",
        ),
        "released_target_claim_attack_success_rate": conditioned(
            "released_target_claim_success",
            "released_target_claim_attack_success_rate",
        ),
        "valid_selected_perturbation_rate": _bootstrap(
            [bool(row.get("selected_constraint_satisfied", False)) for row in rows],
            normalized_ids,
            metric="valid_selected_perturbation_rate",
            **bootstrap_options,
        ),
        "targeted_diagnosis_asr_at_budget": targeted_by_budget,
        "robust_accuracy_at_budget": robust_by_budget,
        "queries_to_first_targeted_diagnosis": _numeric_summary(
            [
                float(row["first_targeted_diagnosis_query"])
                for row in targeted_successes
                if row.get("first_targeted_diagnosis_query") is not None
            ]
        ),
        "queries_to_first_released_target_claim": _numeric_summary(
            [
                float(row["first_released_target_claim_query"])
                for row in rows
                if row.get("first_released_target_claim_query") is not None
            ]
        ),
        "target_probability_gain": _numeric_summary(
            [
                float(row["target_probability_gain"])
                for row in rows
                if row.get("target_probability_gain") is not None
            ]
        ),
        "target_log_probability_gain": _numeric_summary(
            [
                float(row["target_log_probability_gain"])
                for row in rows
                if row.get("target_log_probability_gain") is not None
            ]
        ),
        "distortion_all_valid_selected": _distortion_summaries(valid_selected),
        "distortion_targeted_diagnosis_successes": _distortion_summaries(
            targeted_successes,
            field="first_targeted_distortion",
        ),
        "notes": {
            "asr_conditioning": "clean-correct observations only",
            "bootstrap_statistical_unit": ("patient; repeated rows collapse to a patient mean"),
            "descriptive_statistics_unit": "observation",
            "prd_definition": ("100 * ||adversarial-clean||_2 / ||clean||_2"),
            "snr_definition": ("20 * log10(||clean||_2 / ||adversarial-clean||_2)"),
            "clinical_primary_endpoint_unchanged": True,
        },
    }
