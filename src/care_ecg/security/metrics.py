from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .contracts import AttackResult, ReleaseDecision


@dataclass(frozen=True, slots=True)
class BootstrapEstimate:
    estimate: float
    lower: float
    upper: float
    confidence: float
    n_patients: int
    n_resamples: int
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoundsSummary:
    total_cases: int
    successful_cases: int
    median_rounds_to_pass: float | None
    mean_rounds_to_pass: float | None
    success_by_budget: dict[int, float]
    first_success_queries: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SelectiveMetrics:
    coverage: float
    selective_risk: float | None
    false_rejection_rate: float | None
    released: int
    total: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CounterfactualAlignment:
    cosine_similarity: float
    adversarial_l2: float
    counterfactual_l2: float
    magnitude_ratio: float | None
    factor_overlap_at_k: float
    support_match: bool
    k: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validated_confidence(confidence: float) -> float:
    result = float(confidence)
    if not 0 < result < 1:
        raise ValueError("confidence must be between zero and one")
    return result


def _patient_means(
    values: Sequence[float | bool], patient_ids: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    numeric = np.asarray(values, dtype=float).reshape(-1)
    identifiers = np.asarray(patient_ids, dtype=object).reshape(-1)
    if not len(numeric) or numeric.shape != identifiers.shape:
        raise ValueError("Values and patient_ids must be non-empty and have equal length")
    if not np.isfinite(numeric).all():
        raise ValueError("Metric values must be finite")
    grouped: dict[str, list[float]] = defaultdict(list)
    for patient_id, value in zip(identifiers, numeric):
        identifier = str(patient_id).strip()
        if not identifier:
            raise ValueError("patient_ids must not be empty")
        grouped[identifier].append(float(value))
    ordered = tuple(sorted(grouped))
    return np.asarray([np.mean(grouped[key]) for key in ordered]), ordered


def patient_bootstrap(
    values: Sequence[float | bool],
    patient_ids: Sequence[str],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    metadata: Mapping[str, Any] | None = None,
) -> BootstrapEstimate:
    """Bootstrap a mean after first collapsing repeated rows within patient."""

    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    confidence = _validated_confidence(confidence)
    patient_values, patients = _patient_means(values, patient_ids)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(patient_values), size=(n_resamples, len(patient_values)))
    samples = patient_values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return BootstrapEstimate(
        estimate=float(patient_values.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        n_patients=len(patients),
        n_resamples=n_resamples,
        seed=int(seed),
        metadata=dict(metadata or {}),
    )


def incorrect_release_at_budget(result: AttackResult, budget: int) -> bool:
    if budget < 1:
        raise ValueError("budget must be positive")
    return any(attempt.success and attempt.query_index <= budget for attempt in result.attempts)


def bootstrap_incorrect_release_rate(
    results: Sequence[AttackResult] | Sequence[bool],
    patient_ids: Sequence[str],
    *,
    budget: int | None = None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapEstimate:
    if len(results) == 0:
        raise ValueError("At least one attack result is required")
    first = results[0]
    if isinstance(first, AttackResult):
        typed_results = [result for result in results if isinstance(result, AttackResult)]
        if len(typed_results) != len(results):
            raise TypeError("results cannot mix AttackResult and boolean values")
        values = [
            result.success if budget is None else incorrect_release_at_budget(result, budget)
            for result in typed_results
        ]
    else:
        if budget is not None:
            raise ValueError("budget requires AttackResult inputs with query traces")
        values = [bool(value) for value in results]
    return patient_bootstrap(
        values,
        patient_ids,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
        metadata={"metric": "incorrect_release_rate", "budget": budget},
    )


def _as_patient_mapping(
    values: Mapping[str, float | bool] | Sequence[float | bool],
    patient_ids: Sequence[str] | None,
) -> dict[str, float]:
    if isinstance(values, Mapping):
        if not values:
            raise ValueError("Paired patient mappings must not be empty")
        result = {str(key): float(value) for key, value in values.items()}
        if any(not key or not np.isfinite(value) for key, value in result.items()):
            raise ValueError("Paired patient mappings must have valid IDs and finite values")
        return result
    if patient_ids is None:
        raise ValueError("patient_ids are required for sequence inputs")
    collapsed, ordered = _patient_means(values, patient_ids)
    return dict(zip(ordered, collapsed))


def bootstrap_paired_irr_difference(
    adaptive: Mapping[str, float | bool] | Sequence[float | bool],
    comparator: Mapping[str, float | bool] | Sequence[float | bool],
    patient_ids: Sequence[str] | None = None,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapEstimate:
    """Estimate adaptive-minus-comparator IRR with paired patient resampling."""

    adaptive_by_patient = _as_patient_mapping(adaptive, patient_ids)
    comparator_by_patient = _as_patient_mapping(comparator, patient_ids)
    if adaptive_by_patient.keys() != comparator_by_patient.keys():
        raise ValueError("Paired IRR inputs must contain identical patient IDs")
    ordered = sorted(adaptive_by_patient)
    differences = [
        adaptive_by_patient[patient] - comparator_by_patient[patient]
        for patient in ordered
    ]
    return patient_bootstrap(
        differences,
        ordered,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
        metadata={"metric": "paired_delta_irr", "direction": "adaptive_minus_comparator"},
    )


def irr_by_budget(
    results: Sequence[AttackResult], budgets: Sequence[int]
) -> dict[int, float]:
    if len(results) == 0:
        raise ValueError("At least one attack result is required")
    normalized = sorted({int(budget) for budget in budgets})
    if not normalized or normalized[0] < 1:
        raise ValueError("budgets must contain positive integers")
    return {
        budget: float(np.mean([incorrect_release_at_budget(result, budget) for result in results]))
        for budget in normalized
    }


def rounds_to_pass(
    results: Sequence[AttackResult], budgets: Sequence[int] = (1, 10, 50, 100)
) -> RoundsSummary:
    if len(results) == 0:
        raise ValueError("At least one attack result is required")
    rounds = [
        min(attempt.query_index for attempt in result.attempts if attempt.success)
        for result in results
        if any(attempt.success for attempt in result.attempts)
    ]
    return RoundsSummary(
        total_cases=len(results),
        successful_cases=len(rounds),
        median_rounds_to_pass=None if not rounds else float(np.median(rounds)),
        mean_rounds_to_pass=None if not rounds else float(np.mean(rounds)),
        success_by_budget=irr_by_budget(results, budgets),
        first_success_queries=tuple(int(value) for value in rounds),
    )


def coverage(decisions: Sequence[bool | ReleaseDecision]) -> float:
    if len(decisions) == 0:
        raise ValueError("At least one release decision is required")
    released = [
        decision.released if isinstance(decision, ReleaseDecision) else bool(decision)
        for decision in decisions
    ]
    return float(np.mean(released))


def selective_risk(
    correct: Sequence[bool], decisions: Sequence[bool | ReleaseDecision]
) -> float | None:
    if len(correct) == 0 or len(correct) != len(decisions):
        raise ValueError("correct and decisions must be non-empty and aligned")
    released = np.asarray(
        [
            decision.released if isinstance(decision, ReleaseDecision) else bool(decision)
            for decision in decisions
        ],
        dtype=bool,
    )
    correctness = np.asarray(correct, dtype=bool)
    if not released.any():
        return None
    return float(np.mean(~correctness[released]))


def selective_metrics(
    correct: Sequence[bool], decisions: Sequence[bool | ReleaseDecision]
) -> SelectiveMetrics:
    if len(correct) == 0 or len(correct) != len(decisions):
        raise ValueError("correct and decisions must be non-empty and aligned")
    released = np.asarray(
        [
            decision.released if isinstance(decision, ReleaseDecision) else bool(decision)
            for decision in decisions
        ],
        dtype=bool,
    )
    correctness = np.asarray(correct, dtype=bool)
    correct_count = int(correctness.sum())
    false_rejection = (
        None
        if not correct_count
        else float(np.sum(correctness & ~released) / correct_count)
    )
    return SelectiveMetrics(
        coverage=float(released.mean()),
        selective_risk=None if not released.any() else float(np.mean(~correctness[released])),
        false_rejection_rate=false_rejection,
        released=int(released.sum()),
        total=len(released),
    )


def false_rejection_rate(
    correct: Sequence[bool], decisions: Sequence[bool | ReleaseDecision]
) -> float | None:
    """Return the fraction of correct cases rejected or escalated."""

    return selective_metrics(correct, decisions).false_rejection_rate


def retrieval_overlap_at_k(
    clean_ids: Sequence[str], attacked_ids: Sequence[str], k: int
) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    clean = set(clean_ids[:k])
    attacked = set(attacked_ids[:k])
    union = clean | attacked
    return 1.0 if not union else float(len(clean & attacked) / len(union))


def latent_displacement(clean: Sequence[float], attacked: Sequence[float]) -> dict[str, Any]:
    clean_array = np.asarray(clean, dtype=float).reshape(-1)
    attacked_array = np.asarray(attacked, dtype=float).reshape(-1)
    if clean_array.shape != attacked_array.shape or not len(clean_array):
        raise ValueError("Latent vectors must be non-empty and aligned")
    if not np.isfinite(clean_array).all() or not np.isfinite(attacked_array).all():
        raise ValueError("Latent vectors must be finite")
    delta = attacked_array - clean_array
    return {
        "delta": delta.tolist(),
        "l1": float(np.linalg.norm(delta, ord=1)),
        "l2": float(np.linalg.norm(delta)),
        "linf": float(np.max(np.abs(delta))),
        "changed_factors": np.flatnonzero(np.abs(delta) > 1e-12).tolist(),
    }


def counterfactual_alignment(
    adversarial_delta: Sequence[float],
    counterfactual_delta: Sequence[float],
    *,
    k: int | None = None,
    support_tolerance: float = 1e-12,
) -> CounterfactualAlignment:
    adversarial = np.asarray(adversarial_delta, dtype=float).reshape(-1)
    counterfactual = np.asarray(counterfactual_delta, dtype=float).reshape(-1)
    if adversarial.shape != counterfactual.shape or not len(adversarial):
        raise ValueError("Adversarial and counterfactual deltas must be non-empty and aligned")
    if not np.isfinite(adversarial).all() or not np.isfinite(counterfactual).all():
        raise ValueError("Alignment deltas must be finite")
    if support_tolerance < 0 or not np.isfinite(support_tolerance):
        raise ValueError("support_tolerance must be finite and non-negative")
    selected_k = min(len(adversarial), 5 if k is None else int(k))
    if selected_k < 1:
        raise ValueError("k must be positive")
    adversarial_norm = float(np.linalg.norm(adversarial))
    counterfactual_norm = float(np.linalg.norm(counterfactual))
    if adversarial_norm == 0 and counterfactual_norm == 0:
        cosine = 1.0
    elif adversarial_norm == 0 or counterfactual_norm == 0:
        cosine = 0.0
    else:
        cosine = float(
            np.clip(
                np.dot(adversarial, counterfactual)
                / (adversarial_norm * counterfactual_norm),
                -1.0,
                1.0,
            )
        )
    # Stable descending order makes equal-magnitude factors reproducible.
    adversarial_top = set(
        np.argsort(-np.abs(adversarial), kind="stable")[:selected_k]
    )
    counterfactual_top = set(
        np.argsort(-np.abs(counterfactual), kind="stable")[:selected_k]
    )
    union = adversarial_top | counterfactual_top
    overlap = 1.0 if not union else len(adversarial_top & counterfactual_top) / len(union)
    adversarial_support = set(np.flatnonzero(np.abs(adversarial) > support_tolerance))
    counterfactual_support = set(np.flatnonzero(np.abs(counterfactual) > support_tolerance))
    return CounterfactualAlignment(
        cosine_similarity=cosine,
        adversarial_l2=adversarial_norm,
        counterfactual_l2=counterfactual_norm,
        magnitude_ratio=None
        if counterfactual_norm == 0
        else adversarial_norm / counterfactual_norm,
        factor_overlap_at_k=float(overlap),
        support_match=counterfactual_support.issubset(adversarial_support),
        k=selected_k,
    )


paired_delta_irr = bootstrap_paired_irr_difference
incorrect_release_rate = bootstrap_incorrect_release_rate
