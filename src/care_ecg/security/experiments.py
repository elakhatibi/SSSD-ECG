from __future__ import annotations

import copy
import hashlib
import hmac
import inspect
import json
import secrets
import weakref
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from time import perf_counter
from typing import Any

import numpy as np

from .attacks import F3AdaptiveAttack
from .contracts import (
    AdmissibilityResult,
    AttackFamily,
    AttackResult,
    ClinicalCaseContext,
    ClinicalPreservationEvidence,
    ClinicalTarget,
    ConstraintKind,
    DefenseKind,
    MeasurementStatus,
    PreservationVerificationReceipt,
    ReleaseDecision,
    ReportCorrectnessResult,
    VictimSnapshot,
)
from .defenses import (
    AuthenticatedHistoryRecord,
    calibrate_matched_coverage,
    hash_waveform,
)
from .logging import AppendOnlySecurityLogger
from .metrics import (
    bootstrap_incorrect_release_rate,
    bootstrap_paired_irr_difference,
    counterfactual_alignment,
    patient_bootstrap,
    retrieval_overlap_at_k,
    selective_metrics,
)
from .robustness_metrics import (
    automated_attack_metrics,
    summarize_automated_robustness,
)


def _plain_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return _plain_value(value.to_dict())
    return value


@dataclass(frozen=True, slots=True)
class CohortCase:
    patient_id: str
    record_id: str
    waveform_mv: np.ndarray
    target: ClinicalTarget
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        waveform = np.asarray(self.waveform_mv, dtype=np.float32)
        if waveform.ndim != 2 or not np.isfinite(waveform).all():
            raise ValueError("Cohort waveform must be finite with shape [leads, samples]")
        if not self.patient_id.strip() or not self.record_id.strip():
            raise ValueError("Cohort patient and record IDs must not be empty")
        object.__setattr__(self, "waveform_mv", waveform)

    @property
    def key(self) -> str:
        return f"{self.patient_id}:{self.record_id}:{self.target.target_id}"

    def to_dict(self, *, include_waveform: bool = False) -> dict[str, Any]:
        result = {
            "patient_id": self.patient_id,
            "record_id": self.record_id,
            "target": self.target.to_dict(),
            "metadata": _plain_value(self.metadata),
        }
        if include_waveform:
            result["waveform_mv"] = self.waveform_mv.tolist()
        return result


def _private_case_copy(case: CohortCase) -> CohortCase:
    """Copy mutable case inputs before exposing them to extension code."""

    return CohortCase(
        patient_id=case.patient_id,
        record_id=case.record_id,
        waveform_mv=np.ascontiguousarray(case.waveform_mv, dtype=np.float32).copy(),
        target=case.target,
        metadata=copy.deepcopy(case.metadata),
    )


@dataclass(frozen=True, slots=True)
class EligibleCase:
    case: CohortCase
    clean_snapshot: VictimSnapshot
    primary_endpoint_eligible: bool = False
    report_correctness: ReportCorrectnessResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "clean_snapshot": self.clean_snapshot.to_dict(),
            "primary_endpoint_eligible": self.primary_endpoint_eligible,
            "report_correctness": (
                None if self.report_correctness is None else self.report_correctness.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class E5CalibrationCase:
    """One shared clean calibration case evaluated by every E5 verifier.

    ``verifier_contexts`` carries only defense inputs that cannot be derived from
    the case itself, such as independent measurements or authenticated history.
    The orchestrator supplies the same case and snapshot to V1, V2, and V3.
    """

    case: CohortCase
    clean_snapshot: VictimSnapshot
    verifier_contexts: Mapping[DefenseKind | str, Mapping[str, Any]]
    report_correctness: ReportCorrectnessResult | None = None
    eligible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.case, CohortCase):
            raise TypeError("E5 calibration case must contain a CohortCase")
        if not isinstance(self.clean_snapshot, VictimSnapshot):
            raise TypeError("E5 calibration case must contain a VictimSnapshot")
        contexts: dict[DefenseKind, dict[str, Any]] = {}
        for raw_defense, raw_context in self.verifier_contexts.items():
            defense = DefenseKind(raw_defense)
            if defense in contexts:
                raise ValueError(
                    f"E5 calibration case contains duplicate context for {defense.value}"
                )
            if not isinstance(raw_context, Mapping):
                raise TypeError("E5 verifier calibration context must be a mapping")
            context = dict(raw_context)
            reserved = {
                "snapshot",
                "waveform_mv",
                "patient_id",
                "record_id",
                "sample_rate_hz",
                "current_timestamp",
            }
            overridden = sorted(reserved.intersection(context))
            if overridden:
                raise ValueError(
                    "E5 calibration verifier context cannot override shared case fields: "
                    + ", ".join(overridden)
                )
            contexts[defense] = context
        expected = {DefenseKind.V1, DefenseKind.V2, DefenseKind.V3}
        if set(contexts) != expected:
            raise ValueError("E5 calibration cases require contexts for exactly V1, V2, and V3")
        if self.report_correctness is not None:
            if not isinstance(self.report_correctness, ReportCorrectnessResult):
                raise TypeError(
                    "E5 calibration report correctness must be a ReportCorrectnessResult"
                )
            expected_context = ClinicalCaseContext(
                self.case.patient_id,
                self.case.record_id,
                self.case.target,
            )
            if self.report_correctness.context != expected_context:
                raise ValueError("E5 calibration report correctness does not match the case")
        object.__setattr__(self, "verifier_contexts", contexts)

    @property
    def key(self) -> str:
        return self.case.key


@dataclass(slots=True)
class ExperimentObservation:
    experiment_id: str
    arm: str
    patient_id: str
    record_id: str
    target_id: str
    success: bool
    released: bool
    seed: int | None = None
    budget: int | None = None
    defense: str | None = None
    queries_used: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    attack_result: AttackResult | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "experiment_id": self.experiment_id,
            "arm": self.arm,
            "patient_id": self.patient_id,
            "record_id": self.record_id,
            "target_id": self.target_id,
            "success": self.success,
            "released": self.released,
            "seed": self.seed,
            "budget": self.budget,
            "defense": self.defense,
            "queries_used": self.queries_used,
            "metrics": _plain_value(self.metrics),
            "metadata": _plain_value(self.metadata),
        }
        if self.attack_result is not None:
            constraint_distances: list[float | None] = []
            constraint_satisfied: list[bool | None] = []
            for attempt in self.attack_result.attempts:
                raw_distance = attempt.metadata.get("constraint_distance")
                distance = None if raw_distance is None else float(raw_distance)
                if distance is not None and (not np.isfinite(distance) or distance < 0):
                    raise ValueError(
                        "Serialized constraint distances must be finite and non-negative"
                    )
                constraint_distances.append(distance)
                raw_satisfied = attempt.metadata.get("constraint_satisfied")
                constraint_satisfied.append(None if raw_satisfied is None else bool(raw_satisfied))
            result["attack_result"] = {
                "family": str(self.attack_result.family),
                "target_label": self.attack_result.target_label,
                "success": self.attack_result.success,
                "queries_used": self.attack_result.queries_used,
                "stopped_reason": self.attack_result.stopped_reason,
                "constraint_distances": constraint_distances,
                "constraint_satisfied": constraint_satisfied,
                "metadata": _plain_value(self.attack_result.metadata),
            }
        return result


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    experiment_id: str
    observations: tuple[ExperimentObservation, ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "observations": [observation.to_dict() for observation in self.observations],
            "summary": _plain_value(self.summary),
        }


@dataclass(frozen=True, slots=True)
class CleanBaselineRun:
    run: ExperimentRun
    eligible_cases: tuple[EligibleCase, ...]

    def to_dict(self) -> dict[str, Any]:
        result = self.run.to_dict()
        result["eligible_case_keys"] = [item.case.key for item in self.eligible_cases]
        return result


def _snapshot_payload(snapshot: VictimSnapshot) -> str:
    """Canonicalize the complete clinical snapshot returned by a victim query."""

    if not isinstance(snapshot, VictimSnapshot):
        raise TypeError("Experiment snapshots must use the VictimSnapshot contract")
    return json.dumps(
        snapshot.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _snapshot_verifier_payload(snapshot: VictimSnapshot) -> str:
    """Canonicalize snapshot content that must remain stable across verification."""

    payload = snapshot.to_dict()
    for dynamic_field in ("released", "abstained", "defense_score"):
        payload.pop(dynamic_field, None)
    metadata = dict(payload.get("metadata", {}))
    metadata.pop("defense_decision", None)
    payload["metadata"] = metadata
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


class _ObservedVerifier:
    """Delegate while retaining orchestrator-owned verifier decisions."""

    __slots__ = (
        "__decide_identity",
        "__decision_contexts",
        "__decisions",
        "__snapshot_ids",
        "__snapshot_payloads",
        "__verifier",
        "_locked_threshold",
        "defense",
    )

    def __init__(self, verifier: Any, defense: DefenseKind):
        if not callable(getattr(verifier, "decide", None)):
            raise TypeError(f"E5 {defense.value} verifier must expose a callable decide method")
        try:
            kind = DefenseKind(verifier.kind)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(f"E5 {defense.value} verifier must expose its DefenseKind") from exc
        if kind is not defense:
            raise ValueError(f"E5 verifier kind {kind.value} does not match arm {defense.value}")
        try:
            threshold = float(verifier.threshold)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(f"E5 {defense.value} verifier must expose a finite threshold") from exc
        if not np.isfinite(threshold):
            raise ValueError(f"E5 {defense.value} verifier threshold must be finite")
        self.__verifier = verifier
        self.__decide_identity = self._callable_identity(verifier.decide)
        self.defense = defense
        self.__decisions: list[ReleaseDecision] = []
        self.__decision_contexts: list[dict[str, Any]] = []
        self.__snapshot_ids: list[int] = []
        self.__snapshot_payloads: list[str] = []
        self._locked_threshold: float | None = None

    @staticmethod
    def _callable_identity(function: Any) -> tuple[Any, Any]:
        return (
            getattr(function, "__self__", None),
            getattr(function, "__func__", function),
        )

    def _assert_decide_integrity(self) -> None:
        current = self._callable_identity(self.__verifier.decide)
        if any(
            observed is not expected
            for observed, expected in zip(current, self.__decide_identity, strict=True)
        ):
            raise ValueError("E5 verifier decision implementation changed during the run")

    @property
    def kind(self) -> DefenseKind:
        return self.defense

    @property
    def decisions(self) -> tuple[ReleaseDecision, ...]:
        return tuple(self.__decisions)

    @property
    def decision_contexts(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(value) for value in self.__decision_contexts)

    @property
    def snapshot_ids(self) -> tuple[int, ...]:
        return tuple(self.__snapshot_ids)

    @property
    def snapshot_payloads(self) -> tuple[str, ...]:
        return tuple(self.__snapshot_payloads)

    @property
    def threshold(self) -> float:
        return float(self.__verifier.threshold)

    @threshold.setter
    def threshold(self, value: float) -> None:
        threshold = float(value)
        if self._locked_threshold is not None and not np.isclose(
            threshold,
            self._locked_threshold,
            rtol=0.0,
            atol=1e-12,
        ):
            raise AttributeError("E5 bound verifier threshold is locked")
        _set_verifier_threshold(self.__verifier, threshold)

    def lock_threshold(self, expected: float) -> None:
        expected = float(expected)
        if not np.isclose(self.threshold, expected, rtol=0.0, atol=1e-12):
            raise ValueError("E5 bound verifier threshold changed before locking")
        self._locked_threshold = expected

    def assert_threshold_locked(self) -> None:
        if self._locked_threshold is None or not np.isclose(
            self.threshold,
            self._locked_threshold,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("E5 bound verifier threshold changed after calibration")

    def decide(self, snapshot: VictimSnapshot, **kwargs: Any) -> ReleaseDecision:
        if not isinstance(snapshot, VictimSnapshot):
            raise TypeError("E5 verifier decide must receive a VictimSnapshot")
        if self._locked_threshold is not None:
            self.assert_threshold_locked()
        self._assert_decide_integrity()
        snapshot_payload = _snapshot_verifier_payload(snapshot)
        decision = self.__verifier.decide(snapshot, **kwargs)
        if not isinstance(decision, ReleaseDecision):
            raise TypeError("E5 verifier decide must return a typed ReleaseDecision")
        if decision.defense is not self.defense:
            raise ValueError("E5 verifier returned a decision for a different defense arm")
        if not np.isclose(decision.threshold, self.threshold, rtol=0.0, atol=1e-12):
            raise ValueError("E5 verifier decision threshold does not match the bound verifier")
        if decision.released and decision.score > decision.threshold:
            raise ValueError("E5 verifier released a score above its threshold")
        if _snapshot_verifier_payload(snapshot) != snapshot_payload:
            raise ValueError("E5 verifier mutated its clinical snapshot input")
        self.__snapshot_ids.append(id(snapshot))
        self.__snapshot_payloads.append(snapshot_payload)
        self.__decision_contexts.append(copy.deepcopy(kwargs))
        self.__decisions.append(decision)
        return decision


class _CountingVictim:
    """Count complete-system calls made through an experiment boundary."""

    __slots__ = (
        "__audit_verifier",
        "__behaviorally_bound_queries",
        "__clean_waveform",
        "__constraint",
        "__queries_used",
        "__query_budget",
        "__query_snapshots",
        "__query_waveforms",
        "__verifier_observer",
        "__victim",
    )

    def __init__(
        self,
        victim: Any,
        query_budget: int,
        *,
        clean_waveform: np.ndarray,
        constraint: Any,
        verifier_observer: _ObservedVerifier | None = None,
        audit_verifier: _ObservedVerifier | None = None,
    ):
        self.__victim = victim
        self.__query_budget = int(query_budget)
        self.__queries_used = 0
        self.__clean_waveform = np.ascontiguousarray(clean_waveform, dtype=np.float32).copy()
        self.__constraint = constraint
        self.__query_waveforms: list[np.ndarray] = []
        self.__query_snapshots: list[VictimSnapshot] = []
        self.__verifier_observer = verifier_observer
        self.__audit_verifier = audit_verifier
        self.__behaviorally_bound_queries = 0

    def __getattr__(self, name: str) -> Any:
        allowed = {
            "counterfactual_guidance",
            "discretizer_thresholds",
            "label_names",
            "pipeline",
            "sample_rate_hz",
            "surrogate",
            "target_index",
        }
        if name not in allowed:
            raise AttributeError(f"Attack-facing victim does not expose {name!r}")
        return getattr(self.__victim, name)

    def verifier_binding_matches(self, verifier: _ObservedVerifier) -> bool:
        return self.__verifier_observer is verifier

    @property
    def queries_used(self) -> int:
        return self.__queries_used

    @property
    def query_waveforms(self) -> tuple[np.ndarray, ...]:
        return tuple(value.copy() for value in self.__query_waveforms)

    @property
    def query_snapshots(self) -> tuple[VictimSnapshot, ...]:
        return tuple(copy.deepcopy(value) for value in self.__query_snapshots)

    @property
    def behaviorally_bound_queries(self) -> int:
        return self.__behaviorally_bound_queries

    @property
    def independently_audited_verifier_queries(self) -> int:
        return 0 if self.__audit_verifier is None else len(self.__audit_verifier.decisions)

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        self.__queries_used += 1
        if self.__queries_used > self.__query_budget:
            raise RuntimeError(
                "Attack attempted a complete-system query beyond the experiment budget"
            )
        waveform = args[0] if args else kwargs.get("waveform_mv", kwargs.get("signal"))
        if waveform is None:
            raise TypeError("Complete-system queries must provide a waveform")
        candidate = np.asarray(waveform)
        if candidate.dtype != np.float32:
            raise TypeError(
                "Experiment victim queries must use the canonical float32 raw-mV waveform"
            )
        candidate = np.ascontiguousarray(candidate)
        if not self.__constraint.contains(self.__clean_waveform, candidate):
            raise ValueError(
                "Experiment boundary refused a victim query outside the exact constraint"
            )
        self.__query_waveforms.append(candidate.copy())
        prior_decisions = (
            0 if self.__verifier_observer is None else len(self.__verifier_observer.decisions)
        )
        snapshot = self.__victim.evaluate(*args, **kwargs)
        if not isinstance(snapshot, VictimSnapshot):
            raise TypeError("Complete-system queries must return VictimSnapshot")
        if self.__verifier_observer is not None:
            self.__verifier_observer.assert_threshold_locked()
            decision_count = len(self.__verifier_observer.decisions)
            if decision_count != prior_decisions + 1:
                raise ValueError("E5 victim must invoke the bound verifier exactly once per query")
            decision = self.__verifier_observer.decisions[-1]
            if id(snapshot) != self.__verifier_observer.snapshot_ids[-1] or (
                _snapshot_verifier_payload(snapshot)
                != self.__verifier_observer.snapshot_payloads[-1]
            ):
                raise ValueError("E5 victim returned a snapshot other than the one verified")
            if snapshot.released != decision.released or not np.isclose(
                snapshot.defense_score,
                decision.score,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("E5 victim output does not reflect its bound verifier decision")
            if _plain_value(snapshot.metadata.get("defense_decision")) != _plain_value(
                decision.to_dict()
            ):
                raise ValueError("E5 victim output does not retain its typed verifier decision")
            if self.__audit_verifier is None:
                raise RuntimeError("E5 verifier binding requires an audit verifier")
            audit_decision = self.__audit_verifier.decide(
                copy.deepcopy(snapshot),
                **self.__verifier_observer.decision_contexts[-1],
            )
            if _plain_value(audit_decision.to_dict()) != _plain_value(decision.to_dict()):
                raise ValueError("E5 bound verifier decision did not match independent replay")
            self.__behaviorally_bound_queries += 1
            self.__verifier_observer.assert_threshold_locked()
        self.__query_snapshots.append(copy.deepcopy(snapshot))
        return snapshot


def _call_with_supported(function: Callable[..., Any], **kwargs: Any) -> Any:
    signature = inspect.signature(function)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return function(**kwargs)
    supported = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return function(**supported)


def _require_context_factory(
    provider: Any,
    *,
    name: str,
    instance_method: str,
    required_context: Sequence[str],
) -> None:
    if hasattr(provider, instance_method) and not inspect.isclass(provider):
        raise TypeError(f"E5 {name} must be a factory, not an instantiated object")
    if not callable(provider):
        raise TypeError(f"E5 {name} must be callable")
    try:
        signature = inspect.signature(provider)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"E5 cannot inspect {name} factory context") from exc
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return
    missing = [item for item in required_context if item not in signature.parameters]
    if missing:
        raise TypeError(
            f"E5 {name} factory must accept reoptimization context: {', '.join(missing)}"
        )


def _victim(provider: Any, case: CohortCase, **context: Any) -> Any:
    if hasattr(provider, "evaluate") and not inspect.isclass(provider):
        return provider
    if not callable(provider):
        raise TypeError("victim provider must be a victim or callable factory")
    result = _call_with_supported(provider, case=case, **context)
    if not hasattr(result, "evaluate"):
        raise TypeError("victim factory must return an object exposing evaluate")
    return result


def _report_correct(
    snapshot: VictimSnapshot | None,
    case: CohortCase,
    evaluator: Callable[..., bool] | None,
) -> bool:
    if snapshot is None:
        return False
    if evaluator is None:
        return bool(
            snapshot.diagnosis == case.target.clean_label and not snapshot.target_claim_present
        )
    return bool(_call_with_supported(evaluator, snapshot=snapshot, case=case))


def _independent_report_correctness(
    snapshot: VictimSnapshot | None,
    case: CohortCase,
    evaluator: Callable[..., Any] | None,
) -> ReportCorrectnessResult | None:
    """Return primary-grade all-claims evidence, or ``None`` when not configured."""

    if evaluator is None:
        return None
    if snapshot is None:
        raise ValueError("E5 report correctness evaluation requires an attack snapshot")
    context = ClinicalCaseContext(case.patient_id, case.record_id, case.target)
    result = _call_with_supported(
        evaluator,
        snapshot=snapshot,
        report=snapshot.report,
        case=case,
        patient_id=case.patient_id,
        record_id=case.record_id,
        target=case.target,
        context=context,
    )
    if not isinstance(result, ReportCorrectnessResult):
        raise TypeError(
            "E5 report correctness evaluators must return ReportCorrectnessResult; "
            "boolean or target-only assessments cannot define primary selective risk"
        )
    if result.context != context:
        raise ValueError("E5 report correctness evidence does not match the attacked case")
    if not result.primary_endpoint_eligible:
        raise ValueError(
            "E5 selective risk requires independent all-claims report correctness evidence"
        )
    return result


def _e0_report_correctness(
    snapshot: VictimSnapshot,
    case: CohortCase,
    evaluator: Callable[..., Any] | None,
    *,
    experiment_id: str = "E0",
) -> tuple[bool, bool, ReportCorrectnessResult | None, dict[str, Any]]:
    context = ClinicalCaseContext(case.patient_id, case.record_id, case.target)
    if evaluator is None:
        correct = bool(
            snapshot.diagnosis == case.target.clean_label and not snapshot.target_claim_present
        )
        return (
            correct,
            False,
            None,
            {
                "source": "diagnosis_and_target_absence_heuristic",
                "endpoint_classification": "exploratory",
                "primary_ineligibility_reason": (
                    "no independent all-claims report correctness evaluator was configured"
                ),
                "case_context": context.to_dict(),
            },
        )
    result = _call_with_supported(
        evaluator,
        snapshot=snapshot,
        report=snapshot.report,
        case=case,
        patient_id=case.patient_id,
        record_id=case.record_id,
        target=case.target,
        context=context,
    )
    if isinstance(result, ReportCorrectnessResult):
        if result.context != context:
            raise ValueError(
                f"Report correctness evidence does not match the {experiment_id} case context"
            )
        primary_eligible = result.primary_endpoint_eligible
        return (
            bool(result.correct),
            primary_eligible,
            result,
            {
                "source": result.evaluator,
                "endpoint_classification": ("primary" if primary_eligible else "exploratory"),
                "assessment": result.to_dict(),
            },
        )
    if isinstance(result, (bool, np.bool_)):
        return (
            bool(result),
            False,
            None,
            {
                "source": "boolean_report_correctness_callback",
                "endpoint_classification": "exploratory",
                "primary_ineligibility_reason": (
                    "boolean report correctness does not establish independent all-claims review"
                ),
                "case_context": context.to_dict(),
            },
        )
    raise TypeError(
        f"{experiment_id} report correctness evaluators must return "
        "ReportCorrectnessResult or a boolean exploratory assessment"
    )


def _adult_eligibility(case: CohortCase) -> tuple[bool, dict[str, Any]]:
    """Resolve the proposal's adult-only primary-cohort requirement."""

    if "is_adult" in case.metadata:
        value = case.metadata["is_adult"]
        if not isinstance(value, (bool, np.bool_)):
            return False, {
                "adult_eligible": False,
                "adult_status": "not_evaluable",
                "adult_ineligibility_reason": "is_adult metadata is not boolean",
            }
        return bool(value), {
            "adult_eligible": bool(value),
            "adult_status": "adult" if value else "minor",
            "adult_source": "is_adult",
            **({} if value else {"adult_ineligibility_reason": "case is not an adult"}),
        }

    for key in ("age", "age_years", "patient_age"):
        if key not in case.metadata:
            continue
        raw_age = case.metadata[key]
        if isinstance(raw_age, (bool, np.bool_)):
            break
        try:
            age = float(raw_age)
        except (TypeError, ValueError):
            break
        if np.isfinite(age) and age >= 0:
            adult = age >= 18.0
            return adult, {
                "adult_eligible": adult,
                "adult_status": "adult" if adult else "minor",
                "adult_source": key,
                "age_years": age,
                **(
                    {} if adult else {"adult_ineligibility_reason": "case is younger than 18 years"}
                ),
            }
        break
    return False, {
        "adult_eligible": False,
        "adult_status": "not_evaluable",
        "adult_ineligibility_reason": (
            "adult eligibility cannot be established from case metadata"
        ),
    }


def _context_matches_case(value: Any, case: CohortCase) -> bool:
    expected = ClinicalCaseContext(case.patient_id, case.record_id, case.target)
    if isinstance(value, ClinicalCaseContext):
        return value == expected
    if not isinstance(value, Mapping):
        return False
    target = value.get("target")
    if isinstance(target, ClinicalTarget):
        target_matches = target == case.target
    elif isinstance(target, Mapping):
        target_matches = _plain_value(target) == _plain_value(case.target.to_dict())
    else:
        target_matches = False
    return bool(
        str(value.get("patient_id", "")) == case.patient_id
        and str(value.get("record_id", "")) == case.record_id
        and target_matches
    )


def _checker_sample_rate_hz(victim: Any, constraint: Any, case: CohortCase) -> float | None:
    candidates = [
        getattr(victim, "sample_rate_hz", None),
        getattr(constraint, "sample_rate_hz", None),
        case.metadata.get("sample_rate_hz"),
        case.metadata.get("sample_rate"),
    ]
    pipeline = getattr(victim, "pipeline", None)
    if pipeline is not None:
        config = getattr(pipeline, "config", None)
        data = getattr(config, "data", None)
        candidates.append(getattr(data, "sample_rate", None))
    for value in candidates:
        if value is None:
            continue
        rate = float(value)
        if np.isfinite(rate) and rate > 0:
            return rate
    return None


def _invoke_independent_admissibility_checker(
    checker: Any,
    clean_waveform: np.ndarray,
    candidate: np.ndarray,
    *,
    case: CohortCase,
    sample_rate_hz: float | None,
) -> tuple[Any | None, str | None]:
    """Invoke the experiment-owned checker, separate from attack execution."""

    context = ClinicalCaseContext(case.patient_id, case.record_id, case.target)
    context_kwargs = {
        "case": case,
        "patient_id": case.patient_id,
        "record_id": case.record_id,
        "target": case.target,
        "context": context,
    }
    check = getattr(checker, "check", None)
    evaluate = getattr(checker, "evaluate", None)
    if callable(check):
        function = check
        args = (clean_waveform, candidate)
    elif callable(evaluate):
        if sample_rate_hz is None:
            return None, "sample rate unavailable for independent checker invocation"
        function = evaluate
        args = (clean_waveform, candidate, sample_rate_hz)
    elif callable(checker):
        function = checker
        args = (clean_waveform, candidate)
    else:
        return None, "configured checker is not callable and exposes no check/evaluate method"

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(*args), None
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = (
        context_kwargs
        if accepts_kwargs
        else {name: value for name, value in context_kwargs.items() if name in signature.parameters}
    )
    return function(*args, **kwargs), None


@dataclass(frozen=True, slots=True)
class _PreservationReviewEntry:
    result: Any | None
    invocation_reason: str | None


class _AttackPreservationProxy:
    """Narrow attack-facing view of an orchestrator-owned query reviewer."""

    __slots__ = ("__reviewer",)

    def __init__(self, reviewer: _QueryBoundPreservationReviewer):
        self.__reviewer = reviewer

    def check(self, clean: np.ndarray, candidate: np.ndarray, **kwargs: Any) -> Any:
        del kwargs
        return self.__reviewer.review_next(clean, candidate)

    def evaluate(
        self,
        clean: np.ndarray,
        candidate: np.ndarray,
        sample_rate_hz: float,
        **kwargs: Any,
    ) -> Any:
        del sample_rate_hz, kwargs
        return self.__reviewer.review_next(clean, candidate)

    def __call__(self, clean: np.ndarray, candidate: np.ndarray, **kwargs: Any) -> Any:
        del kwargs
        return self.__reviewer.review_next(clean, candidate)


class _QueryBoundPreservationReviewer:
    """Invoke the primary reviewer at most once per exact recorded query."""

    __slots__ = (
        "__case",
        "__checker",
        "__clean",
        "__constraint",
        "__entries",
        "__proxy",
        "__sample_rate_hz",
        "__victim",
        "__victim_recorder",
    )

    def __init__(
        self,
        checker: Any,
        *,
        clean_waveform: np.ndarray,
        case: CohortCase,
        victim: Any,
        constraint: Any,
        victim_recorder: _CountingVictim,
    ):
        self.__checker = checker
        self.__clean = np.ascontiguousarray(clean_waveform, dtype=np.float32).copy()
        self.__case = case
        self.__victim = victim
        self.__constraint = constraint
        self.__victim_recorder = victim_recorder
        self.__sample_rate_hz = _checker_sample_rate_hz(victim, constraint, case)
        self.__entries: list[_PreservationReviewEntry] = []
        self.__proxy = _AttackPreservationProxy(self)

    @property
    def attack_proxy(self) -> _AttackPreservationProxy:
        return self.__proxy

    def _invoke(self, candidate: np.ndarray) -> _PreservationReviewEntry:
        result, reason = _invoke_independent_admissibility_checker(
            self.__checker,
            self.__clean.copy(),
            candidate.copy(),
            case=_private_case_copy(self.__case),
            sample_rate_hz=self.__sample_rate_hz,
        )
        try:
            cached = copy.deepcopy(result)
        except Exception as exc:
            raise TypeError("Independent clinical-preservation results must be copyable") from exc
        return _PreservationReviewEntry(cached, reason)

    def review_next(self, clean: np.ndarray, candidate: np.ndarray) -> Any:
        clean_array = np.asarray(clean)
        candidate_array = np.asarray(candidate)
        if clean_array.dtype != np.float32 or candidate_array.dtype != np.float32:
            raise TypeError("Attack-side preservation review requires float32 waveforms")
        if not np.array_equal(clean_array, self.__clean):
            raise ValueError("Attack-side preservation review clean waveform mismatch")
        query_index = len(self.__entries)
        queries = self.__victim_recorder.query_waveforms
        if query_index >= len(queries):
            raise RuntimeError("Preservation review requires a preceding unreviewed victim query")
        expected = queries[query_index]
        if not np.array_equal(candidate_array, expected):
            raise ValueError(
                "Preservation review must follow victim queries in exact submission order"
            )
        entry = self._invoke(np.ascontiguousarray(expected))
        self.__entries.append(entry)
        if entry.invocation_reason is not None:
            return False
        return copy.deepcopy(entry.result)

    def complete(self) -> tuple[_PreservationReviewEntry, ...]:
        queries = self.__victim_recorder.query_waveforms
        while len(self.__entries) < len(queries):
            entry = self._invoke(np.ascontiguousarray(queries[len(self.__entries)]))
            self.__entries.append(entry)
        if len(self.__entries) != len(queries):
            raise RuntimeError("Preservation review cache exceeded recorded victim queries")
        return tuple(copy.deepcopy(self.__entries))


def _independent_preservation_assessment(
    checker: Any | None,
    *,
    clean_waveform: np.ndarray,
    query_waveforms: Sequence[np.ndarray],
    query_snapshots: Sequence[VictimSnapshot],
    case: CohortCase,
    victim: Any,
    constraint: Any,
    review_entries: Sequence[_PreservationReviewEntry] | None = None,
) -> tuple[bool, bool, int | None, dict[str, Any]]:
    """Derive the primary endpoint only from query-bound checker invocations.

    The attack can request a result only for the next exact victim submission. The
    orchestrator owns the reviewer and cache, completes any omitted reviews, and never
    reads attack-authored preservation metadata.
    """

    if len(query_waveforms) != len(query_snapshots):
        raise ValueError("Independent preservation inputs do not align with victim queries")
    if not query_waveforms:
        return (
            False,
            False,
            None,
            {
                "checker_configured": checker is not None,
                "primary_endpoint_evaluable": False,
                "primary_success": False,
                "reason": "no submitted waveform was available for independent review",
                "evaluations": [],
            },
        )
    if checker is None:
        return (
            False,
            False,
            None,
            {
                "checker_configured": False,
                "primary_endpoint_evaluable": False,
                "primary_success": False,
                "reason": "no independent clinical-preservation checker was configured",
                "evaluations": [],
            },
        )
    if review_entries is not None and len(review_entries) != len(query_waveforms):
        raise ValueError("Independent review cache does not cover every victim query")

    expected_context = ClinicalCaseContext(case.patient_id, case.record_id, case.target)
    clean = np.ascontiguousarray(clean_waveform, dtype=np.float32)
    sample_rate_hz = _checker_sample_rate_hz(victim, constraint, case)
    evaluations: list[dict[str, Any]] = []
    primary_success_query: int | None = None
    claim_query_indices: list[int] = []
    for query_index, (waveform, snapshot) in enumerate(
        zip(query_waveforms, query_snapshots, strict=True), start=1
    ):
        candidate = np.asarray(waveform)
        if candidate.dtype != np.float32:
            raise TypeError("Independent preservation review requires float32 waveforms")
        candidate = np.ascontiguousarray(candidate)
        waveform_sha256 = hashlib.sha256(candidate.tobytes()).hexdigest()
        released_target_claim = bool(snapshot.released and snapshot.target_claim_present)
        if released_target_claim:
            claim_query_indices.append(query_index)
        if review_entries is None:
            result, invocation_reason = _invoke_independent_admissibility_checker(
                checker,
                clean,
                candidate,
                case=case,
                sample_rate_hz=sample_rate_hz,
            )
        else:
            entry = review_entries[query_index - 1]
            if not isinstance(entry, _PreservationReviewEntry):
                raise TypeError("Independent review cache entries must be typed")
            result = copy.deepcopy(entry.result)
            invocation_reason = entry.invocation_reason
        detail: dict[str, Any] = {
            "query_index": query_index,
            "waveform_sha256": waveform_sha256,
            "released_target_claim": released_target_claim,
            "typed_admissibility_result": isinstance(result, AdmissibilityResult),
            "primary_grade": False,
            "admissible": False,
        }
        if invocation_reason is not None:
            detail["reason"] = invocation_reason
            evaluations.append(detail)
            continue
        if not isinstance(result, AdmissibilityResult):
            detail["reason"] = "independent checker did not return AdmissibilityResult"
            detail["returned_type"] = type(result).__name__
            evaluations.append(detail)
            continue

        evidence = result.preservation_evidence
        evidence_valid = bool(
            isinstance(evidence, ClinicalPreservationEvidence)
            and evidence.context == expected_context
            and evidence.independent
        )
        available = result.status is MeasurementStatus.AVAILABLE
        primary_grade = bool(available and result.primary_endpoint_eligible and evidence_valid)
        admissible = bool(primary_grade and result.admissible)
        detail.update(
            {
                "status": result.status.value,
                "primary_endpoint_eligible": result.primary_endpoint_eligible,
                "evidence_context_matches": bool(
                    isinstance(evidence, ClinicalPreservationEvidence)
                    and evidence.context == expected_context
                ),
                "evidence_independent": bool(
                    isinstance(evidence, ClinicalPreservationEvidence) and evidence.independent
                ),
                "primary_grade": primary_grade,
                "admissible": admissible,
                "result": result.to_dict(),
            }
        )
        if not evidence_valid:
            detail["reason"] = (
                "missing, non-independent, or patient-record-target-mismatched "
                "ClinicalPreservationEvidence"
            )
        elif not available:
            detail["reason"] = "independent clinical evidence is unavailable"
        elif not result.primary_endpoint_eligible:
            detail["reason"] = "checker classified its evidence as non-primary"
        elif not result.admissible:
            detail["reason"] = "independent review found clinical non-preservation"
        if released_target_claim and admissible and primary_success_query is None:
            primary_success_query = query_index
        evaluations.append(detail)

    if primary_success_query is not None:
        evaluable = True
        success = True
        reason = "an independently admissible released target claim was observed"
    else:
        relevant = (
            [evaluations[index - 1] for index in claim_query_indices]
            if claim_query_indices
            else evaluations
        )
        evaluable = bool(relevant) and all(item["primary_grade"] for item in relevant)
        success = False
        if evaluable and claim_query_indices:
            reason = "released target claims were independently found non-preserving"
        elif evaluable:
            reason = "no released target claim was observed under primary-grade review"
        else:
            reason = (
                "submitted waveform evidence was missing, unavailable, untyped, "
                "mismatched, or non-primary"
            )
    return (
        evaluable,
        success,
        primary_success_query,
        {
            "checker_configured": True,
            "sample_rate_hz": sample_rate_hz,
            "submitted_queries": len(query_waveforms),
            "released_target_claim_queries": claim_query_indices,
            "primary_endpoint_evaluable": evaluable,
            "primary_success": success,
            "primary_success_query_index": primary_success_query,
            "reason": reason,
            "evaluations": evaluations,
        },
    )


_PRESERVATION_RECEIPT_SCHEMA = "care-ecg-preservation-receipt.v1"
_PRESERVATION_RECEIPT_KEY = secrets.token_bytes(32)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        _plain_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _waveform_sha256(waveform: np.ndarray) -> str:
    value = np.asarray(waveform)
    if value.dtype != np.float32:
        raise TypeError("Receipt waveform bindings require canonical float32 values")
    return hash_waveform(value)


def _snapshot_sha256(snapshot: VictimSnapshot) -> str:
    return hashlib.sha256(_snapshot_payload(snapshot).encode("utf-8")).hexdigest()


def _receipt_body(receipt: PreservationVerificationReceipt) -> dict[str, Any]:
    return {
        "schema_version": receipt.schema_version,
        "experiment_id": receipt.experiment_id,
        "context": receipt.context.to_dict(),
        "attack_family": receipt.attack_family.value,
        "target_label": receipt.target_label,
        "query_count": receipt.query_count,
        "clean_waveform_sha256": receipt.clean_waveform_sha256,
        "clean_snapshot_sha256": receipt.clean_snapshot_sha256,
        "query_waveform_sha256s": list(receipt.query_waveform_sha256s),
        "query_snapshot_sha256s": list(receipt.query_snapshot_sha256s),
        "primary_endpoint_eligible": receipt.primary_endpoint_eligible,
        "primary_success": receipt.primary_success,
        "primary_success_query_index": receipt.primary_success_query_index,
        "selected_waveform_sha256": receipt.selected_waveform_sha256,
        "selected_snapshot_sha256": receipt.selected_snapshot_sha256,
        "review_payload_sha256": receipt.review_payload_sha256,
    }


def _issue_preservation_receipt(
    result: AttackResult,
    *,
    experiment_id: str,
    case: CohortCase,
    clean_snapshot: VictimSnapshot,
    query_waveforms: Sequence[np.ndarray],
    query_snapshots: Sequence[VictimSnapshot],
    primary_endpoint_eligible: bool,
    primary_success: bool,
    primary_success_query_index: int | None,
    review_payload: Mapping[str, Any],
) -> PreservationVerificationReceipt | None:
    if result.verification_receipt is not None:
        raise ValueError("Attack result already carries a preservation receipt")
    if len(query_waveforms) != len(query_snapshots):
        raise ValueError("Receipt query waveforms and snapshots must align")
    if not query_waveforms:
        return None
    selected_index = (
        None
        if not primary_success or primary_success_query_index is None
        else primary_success_query_index - 1
    )
    query_waveform_hashes = tuple(_waveform_sha256(value) for value in query_waveforms)
    query_snapshot_hashes = tuple(_snapshot_sha256(value) for value in query_snapshots)
    if selected_index is not None:
        if not 0 <= selected_index < len(query_waveforms):
            raise ValueError("Primary success query is outside the recorded trace")
        selected_attempt = result.attempts[selected_index]
        selected_snapshot = query_snapshots[selected_index]
        if not (
            selected_snapshot.released
            and selected_snapshot.target_claim_present
            and selected_attempt.released
            and selected_attempt.target_claim_present
        ):
            raise ValueError("Primary success receipt requires a released target claim")
    placeholder = PreservationVerificationReceipt(
        schema_version=_PRESERVATION_RECEIPT_SCHEMA,
        experiment_id=experiment_id,
        context=ClinicalCaseContext(case.patient_id, case.record_id, case.target),
        attack_family=result.family,
        target_label=result.target_label,
        query_count=len(query_waveforms),
        clean_waveform_sha256=_waveform_sha256(case.waveform_mv),
        clean_snapshot_sha256=_snapshot_sha256(clean_snapshot),
        query_waveform_sha256s=query_waveform_hashes,
        query_snapshot_sha256s=query_snapshot_hashes,
        primary_endpoint_eligible=primary_endpoint_eligible,
        primary_success=primary_success,
        primary_success_query_index=(None if selected_index is None else selected_index + 1),
        selected_waveform_sha256=(
            None if selected_index is None else query_waveform_hashes[selected_index]
        ),
        selected_snapshot_sha256=(
            None if selected_index is None else query_snapshot_hashes[selected_index]
        ),
        review_payload_sha256=_sha256_json(review_payload),
        payload_sha256="0" * 64,
        authenticator="pending",
    )
    body = _receipt_body(placeholder)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    receipt = replace(
        placeholder,
        payload_sha256=hashlib.sha256(canonical).hexdigest(),
        authenticator=hmac.new(
            _PRESERVATION_RECEIPT_KEY,
            canonical,
            hashlib.sha256,
        ).hexdigest(),
    )
    result.verification_receipt = receipt
    valid, evidence = _validate_preservation_receipt_for_case(
        case,
        clean_snapshot,
        result,
    )
    if not valid:
        result.verification_receipt = None
        raise ValueError(
            "Issued preservation receipt failed self-validation: "
            + ", ".join(evidence["validation_reasons"])
        )
    return receipt


def _validate_preservation_receipt_for_case(
    case: CohortCase,
    clean_snapshot: VictimSnapshot,
    result: AttackResult,
) -> tuple[bool, dict[str, Any]]:
    receipt = result.verification_receipt
    if not isinstance(receipt, PreservationVerificationReceipt):
        return False, {"reason": "source attack lacks an orchestrator-issued receipt"}
    body = _receipt_body(receipt)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    payload_sha256 = hashlib.sha256(canonical).hexdigest()
    expected_authenticator = hmac.new(
        _PRESERVATION_RECEIPT_KEY,
        canonical,
        hashlib.sha256,
    ).hexdigest()
    reasons: list[str] = []
    if payload_sha256 != receipt.payload_sha256 or not hmac.compare_digest(
        expected_authenticator, receipt.authenticator
    ):
        reasons.append("receipt_authentication_invalid")
    expected_context = ClinicalCaseContext(
        case.patient_id,
        case.record_id,
        case.target,
    )
    if receipt.context != expected_context:
        reasons.append("receipt_context_mismatch")
    if receipt.attack_family is not result.family:
        reasons.append("receipt_attack_family_mismatch")
    if receipt.target_label != result.target_label:
        reasons.append("receipt_target_label_mismatch")
    if receipt.query_count != result.queries_used or len(result.attempts) != receipt.query_count:
        reasons.append("receipt_query_count_mismatch")
    if _waveform_sha256(case.waveform_mv) != receipt.clean_waveform_sha256:
        reasons.append("receipt_clean_waveform_mismatch")
    if not isinstance(result.clean_snapshot, VictimSnapshot) or (
        _snapshot_sha256(result.clean_snapshot) != receipt.clean_snapshot_sha256
    ):
        reasons.append("receipt_result_clean_snapshot_mismatch")
    if _snapshot_sha256(clean_snapshot) != receipt.clean_snapshot_sha256:
        reasons.append("receipt_cohort_clean_snapshot_mismatch")
    attempt_hashes = tuple(_waveform_sha256(item.waveform) for item in result.attempts)
    if attempt_hashes != receipt.query_waveform_sha256s:
        reasons.append("receipt_query_waveforms_mismatch")
    if receipt.primary_success:
        selected_index = int(receipt.primary_success_query_index or 0) - 1
        if not 0 <= selected_index < len(result.attempts):
            reasons.append("receipt_selected_query_invalid")
        else:
            selected_attempt = result.attempts[selected_index]
            if receipt.query_waveform_sha256s[selected_index] != (receipt.selected_waveform_sha256):
                reasons.append("receipt_selected_waveform_binding_invalid")
            if receipt.query_snapshot_sha256s[selected_index] != (receipt.selected_snapshot_sha256):
                reasons.append("receipt_selected_snapshot_binding_invalid")
            if result.adversarial_signal is None or (
                _waveform_sha256(result.adversarial_signal) != receipt.selected_waveform_sha256
            ):
                reasons.append("receipt_adversarial_waveform_mismatch")
            if not (selected_attempt.released and selected_attempt.target_claim_present):
                reasons.append("receipt_selected_attempt_is_not_target_release")
            if not isinstance(result.adversarial_snapshot, VictimSnapshot) or (
                _snapshot_sha256(result.adversarial_snapshot) != receipt.selected_snapshot_sha256
            ):
                reasons.append("receipt_adversarial_snapshot_mismatch")
            elif not (
                result.adversarial_snapshot.released
                and result.adversarial_snapshot.target_claim_present
            ):
                reasons.append("receipt_selected_snapshot_is_not_target_release")
    return not reasons, {
        "schema_version": receipt.schema_version,
        "experiment_id": receipt.experiment_id,
        "payload_sha256": receipt.payload_sha256,
        "primary_endpoint_eligible": receipt.primary_endpoint_eligible,
        "primary_success": receipt.primary_success,
        "primary_success_query_index": receipt.primary_success_query_index,
        "review_payload_sha256": receipt.review_payload_sha256,
        "valid": not reasons,
        "validation_reasons": reasons,
    }


def _validate_preservation_receipt(
    eligible: EligibleCase,
    result: AttackResult,
) -> tuple[bool, dict[str, Any]]:
    return _validate_preservation_receipt_for_case(
        eligible.case,
        eligible.clean_snapshot,
        result,
    )


def _source_primary_status(
    eligible: EligibleCase,
    result: AttackResult,
) -> tuple[bool, dict[str, Any]]:
    reasons: list[str] = []
    receipt_valid, receipt_evidence = _validate_preservation_receipt(eligible, result)
    receipt = result.verification_receipt
    receipt_success = bool(
        receipt_valid
        and isinstance(receipt, PreservationVerificationReceipt)
        and receipt.primary_success
    )
    receipt_eligible = bool(
        receipt_valid
        and isinstance(receipt, PreservationVerificationReceipt)
        and receipt.primary_endpoint_eligible
    )
    if not receipt_valid:
        reasons.append("source_attack_lacks_valid_orchestrator_receipt")
    if not receipt_success:
        reasons.append("source_attack_not_successful")
    if not receipt_eligible:
        reasons.append("source_attack_not_primary_endpoint_eligible")
    if not eligible.primary_endpoint_eligible:
        reasons.append("clean_case_not_primary_endpoint_eligible")
    adult_eligible, adult_evidence = _adult_eligibility(eligible.case)
    if not adult_eligible:
        reasons.append("adult_primary_cohort_eligibility_not_established")
    return not reasons, {
        "source_attack_success": receipt_success,
        "source_clean_primary_endpoint_eligible": eligible.primary_endpoint_eligible,
        "source_adult_primary_cohort_eligible": adult_eligible,
        "source_adult_evidence": adult_evidence,
        "source_preservation_evidence_valid": receipt_valid,
        "source_preservation_evidence": receipt_evidence,
        "source_selected_clinically_admissible": receipt_success,
        "source_primary_endpoint_eligible": not reasons,
        "source_ineligibility_reasons": reasons,
    }


def _validate_attack_boundary(
    result: AttackResult,
    *,
    case: CohortCase,
    clean_snapshot: VictimSnapshot,
    constraint: Any,
    query_budget: int,
    counted_queries: int,
    query_waveforms: Sequence[np.ndarray],
    query_snapshots: Sequence[VictimSnapshot],
    experiment_id: str,
) -> VictimSnapshot | None:
    """Recompute budget and allowed-set membership at the orchestrator boundary."""

    if result.verification_receipt is not None:
        raise ValueError(f"{experiment_id} attack returned an unauthorized preservation receipt")

    if not isinstance(result.clean_snapshot, VictimSnapshot) or (
        _snapshot_payload(result.clean_snapshot) != _snapshot_payload(clean_snapshot)
    ):
        raise ValueError(f"{experiment_id} attack clean snapshot does not match its E0 input")
    if result.queries_used > query_budget:
        raise ValueError(f"{experiment_id} attack result exceeded its query budget")
    if counted_queries != result.queries_used:
        raise ValueError(
            f"{experiment_id} counted {counted_queries} complete-system calls but the "
            f"attack reported {result.queries_used}"
        )
    if len(query_waveforms) != counted_queries:
        raise ValueError(f"{experiment_id} query recorder is internally inconsistent")
    if len(query_snapshots) != counted_queries:
        raise ValueError(f"{experiment_id} snapshot recorder is internally inconsistent")
    if len(result.attempts) != len(query_waveforms) or any(
        not np.array_equal(attempt.waveform, queried)
        for attempt, queried in zip(result.attempts, query_waveforms, strict=True)
    ):
        raise ValueError(
            f"{experiment_id} returned attack attempts do not match victim submissions"
        )
    for attempt, snapshot in zip(result.attempts, query_snapshots, strict=True):
        if (
            attempt.diagnosis != snapshot.diagnosis
            or attempt.released != snapshot.released
            or attempt.target_claim_present != snapshot.target_claim_present
        ):
            raise ValueError(f"{experiment_id} attack trace does not match recorded victim outputs")
    selected_snapshot: VictimSnapshot | None = None
    if result.adversarial_snapshot is not None:
        if not isinstance(result.adversarial_snapshot, VictimSnapshot):
            raise TypeError(f"{experiment_id} adversarial artifact must be a VictimSnapshot")
        if result.adversarial_signal is None:
            raise ValueError(f"{experiment_id} adversarial snapshot lacks a submitted signal")
        matching_queries = [
            index
            for index, waveform in enumerate(query_waveforms)
            if np.array_equal(result.adversarial_signal, waveform)
        ]
        selected_query = result.metadata.get("selected_attempt_query_index")
        if selected_query is not None:
            selected_index = int(selected_query) - 1
            if selected_index not in matching_queries:
                raise ValueError(
                    f"{experiment_id} selected artifact does not match its recorded query"
                )
            matching_queries = [selected_index]
        artifact_payload = _snapshot_payload(result.adversarial_snapshot)
        matching_snapshots = [
            index
            for index in matching_queries
            if _snapshot_payload(query_snapshots[index]) == artifact_payload
        ]
        if not matching_snapshots:
            raise ValueError(
                f"{experiment_id} adversarial snapshot does not match the full "
                "recorded victim output for its submitted signal"
            )
        selected_snapshot = copy.deepcopy(query_snapshots[matching_snapshots[0]])
    clean = np.asarray(case.waveform_mv, dtype=np.float32)
    for attempt in result.attempts:
        candidate = np.asarray(attempt.waveform, dtype=np.float32)
        try:
            contained = bool(constraint.contains(clean, candidate))
        except Exception as exc:
            raise ValueError(
                f"{experiment_id} could not verify returned attempt membership"
            ) from exc
        if not contained:
            raise ValueError(
                f"{experiment_id} attack returned an attempt outside the exact constraint"
            )
        distance = float(constraint.distance(clean, candidate))
        if not np.isfinite(distance) or distance < 0:
            raise ValueError(f"{experiment_id} constraint returned an invalid distance")
        attempt.metadata["orchestrator_constraint_satisfied"] = True
        attempt.metadata["orchestrator_constraint_distance"] = distance
    return selected_snapshot


def _e5_query_trace_matches(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
) -> bool:
    return len(left) == len(right) and all(
        np.array_equal(left_value, right_value)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _e5_decision_trace(decisions: Sequence[ReleaseDecision]) -> list[dict[str, Any]]:
    return [_plain_value(decision.to_dict()) for decision in decisions]


def _probe_e5_attack_verifier_causality(
    *,
    victim_provider: Any,
    attack_factory: Any,
    verifier_template: Any,
    case: CohortCase,
    clean_snapshot: VictimSnapshot,
    defense: DefenseKind,
    threshold: float,
    seed: int,
    query_budget: int,
    constraint_provider: Any,
    admissibility_checker: Any | None,
    reference_waveforms: Sequence[np.ndarray],
    reference_decisions: Sequence[ReleaseDecision],
) -> dict[str, Any]:
    """Diagnose causal use of verifier output with a controlled intervention.

    Custom factories execute in this Python process and can inspect frames, closures,
    and object internals. The probe is therefore diagnostic evidence only; it never
    attests a custom factory for the primary E5 endpoint.
    """

    probe_interventions: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()

    class DecisionProbeVerifier:
        """Use one object shape for both arms without module-global arm state."""

        __slots__ = ("__verifier", "__weakref__", "kind", "threshold")

        def __init__(self, verifier: Any, *, intervene: bool):
            self.__verifier = verifier
            self.kind = defense
            self.threshold = float(verifier.threshold)
            probe_interventions[self] = bool(intervene)

        def decide(
            self,
            snapshot: VictimSnapshot,
            **kwargs: Any,
        ) -> ReleaseDecision:
            _set_verifier_threshold(self.__verifier, self.threshold)
            factual = self.__verifier.decide(copy.deepcopy(snapshot), **kwargs)
            if not isinstance(factual, ReleaseDecision):
                raise TypeError("E5 intervention requires typed verifier decisions")
            intervene = probe_interventions.get(self)
            if intervene is None:
                raise RuntimeError("E5 decision probe lost its orchestrator-owned state")
            if not intervene:
                return factual
            released = not factual.released
            if released:
                score = min(float(factual.score), self.threshold)
                reasons: tuple[str, ...] = ()
            else:
                score = float(factual.score)
                reasons = factual.reasons or ("release policy rejected output",)
            return ReleaseDecision(
                defense=self.kind,
                released=released,
                score=score,
                threshold=self.threshold,
                reasons=reasons,
                evidence_available=True,
                matched_clean_coverage=factual.matched_clean_coverage,
                metadata=copy.deepcopy(factual.metadata),
            )

    def run_probe(
        *, intervene: bool
    ) -> tuple[
        tuple[np.ndarray, ...],
        tuple[ReleaseDecision, ...],
    ]:
        bound = _ObservedVerifier(copy.deepcopy(verifier_template), defense)
        _set_verifier_threshold(bound, threshold)
        bound.lock_threshold(threshold)
        audit = _ObservedVerifier(copy.deepcopy(verifier_template), defense)
        _set_verifier_threshold(audit, threshold)
        audit.lock_threshold(threshold)

        attack_delegate = DecisionProbeVerifier(
            copy.deepcopy(verifier_template),
            intervene=intervene,
        )
        _set_verifier_threshold(attack_delegate, threshold)
        attack_verifier = _ObservedVerifier(attack_delegate, defense)
        _set_verifier_threshold(attack_verifier, threshold)
        attack_verifier.lock_threshold(threshold)

        victim = _victim(
            victim_provider,
            case,
            defense=defense,
            verifier=bound,
        )
        attack_case = _private_case_copy(case)
        attack_clean = case.waveform_mv.copy()
        attack = _factory(
            attack_factory,
            seed=int(seed),
            defense=defense,
            verifier=attack_verifier,
            case=attack_case,
        )
        probe_constraint = _constraint(
            constraint_provider,
            case=case,
            seed=int(seed),
        )
        _require_e5_sphys(probe_constraint)
        counted = _CountingVictim(
            victim,
            query_budget,
            clean_waveform=case.waveform_mv,
            constraint=probe_constraint,
            verifier_observer=bound,
            audit_verifier=audit,
        )
        preservation_reviewer = (
            None
            if admissibility_checker is None
            else _QueryBoundPreservationReviewer(
                admissibility_checker,
                clean_waveform=case.waveform_mv,
                case=case,
                victim=victim,
                constraint=probe_constraint,
                victim_recorder=counted,
            )
        )
        result = attack.run(
            counted,
            attack_clean,
            case.target.target_label,
            query_budget=query_budget,
            constraint=probe_constraint,
            admissibility_checker=(
                None if preservation_reviewer is None else preservation_reviewer.attack_proxy
            ),
            case=attack_case,
            defense=defense,
            clean_snapshot=copy.deepcopy(clean_snapshot),
        )
        if not isinstance(result, AttackResult):
            raise TypeError("E5 causal probe attack must return AttackResult")
        if result.family not in {AttackFamily.F3, AttackFamily.F3_DEFENSE}:
            raise ValueError("E5 causal probe requires an F3 re-attack")
        if result.target_label != case.target.target_label:
            raise ValueError("E5 causal probe target does not match the cohort target")
        _validate_attack_boundary(
            result,
            case=case,
            clean_snapshot=clean_snapshot,
            constraint=probe_constraint,
            query_budget=query_budget,
            counted_queries=counted.queries_used,
            query_waveforms=counted.query_waveforms,
            query_snapshots=counted.query_snapshots,
            experiment_id="E5 causal probe",
        )
        _assert_e5_verifier_configuration(
            bound,
            defense,
            threshold,
            role="causal-probe bound victim",
        )
        _assert_e5_verifier_configuration(
            attack_verifier,
            defense,
            threshold,
            role="causal-probe attack-context",
        )
        return counted.query_waveforms, attack_verifier.decisions

    factual_waveforms, factual_decisions = run_probe(intervene=False)
    intervened_waveforms, intervened_decisions = run_probe(intervene=True)
    reference_decision_trace = _e5_decision_trace(reference_decisions)
    factual_decision_trace = _e5_decision_trace(factual_decisions)
    intervened_decision_trace = _e5_decision_trace(intervened_decisions)
    factual_replay_matches = bool(
        _e5_query_trace_matches(reference_waveforms, factual_waveforms)
        and reference_decision_trace == factual_decision_trace
    )
    intervention_changed_decision = bool(
        factual_decision_trace
        and intervened_decision_trace
        and factual_decision_trace != intervened_decision_trace
    )
    intervention_changed_queries = not _e5_query_trace_matches(
        factual_waveforms,
        intervened_waveforms,
    )
    return {
        "valid": bool(
            factual_replay_matches
            and intervention_changed_decision
            and intervention_changed_queries
        ),
        "method": "orchestrator_counterfactual_attack_verifier_intervention",
        "certification_role": "diagnostic_only_for_custom_factories",
        "isolated_execution_attested": False,
        "threat_boundary": (
            "custom attack code executes in-process and may inspect Python runtime "
            "state; this probe cannot establish primary E5 eligibility"
        ),
        "factual_replay_matches": factual_replay_matches,
        "intervention_changed_attack_verifier_decisions": (intervention_changed_decision),
        "intervention_changed_submitted_queries": intervention_changed_queries,
        "reference_queries": len(reference_waveforms),
        "factual_replay_queries": len(factual_waveforms),
        "intervened_queries": len(intervened_waveforms),
    }


@dataclass(frozen=True, slots=True)
class _E5BuiltinImplementationAttestation:
    expected_class: str
    expected_method: str
    captured_before_attack_factory: bool
    pre_run_exact_type: bool
    pre_run_class_method_unchanged: bool
    pre_run_bound_method_exact: bool
    post_run_exact_type: bool
    post_run_class_method_unchanged: bool
    post_run_bound_method_exact: bool

    @property
    def valid(self) -> bool:
        return bool(
            self.captured_before_attack_factory
            and self.pre_run_exact_type
            and self.pre_run_class_method_unchanged
            and self.pre_run_bound_method_exact
            and self.post_run_exact_type
            and self.post_run_class_method_unchanged
            and self.post_run_bound_method_exact
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_class": self.expected_class,
            "expected_method": self.expected_method,
            "captured_before_attack_factory": self.captured_before_attack_factory,
            "pre_run_exact_type": self.pre_run_exact_type,
            "pre_run_class_method_unchanged": self.pre_run_class_method_unchanged,
            "pre_run_bound_method_exact": self.pre_run_bound_method_exact,
            "post_run_exact_type": self.post_run_exact_type,
            "post_run_class_method_unchanged": self.post_run_class_method_unchanged,
            "post_run_bound_method_exact": self.post_run_bound_method_exact,
            "valid": self.valid,
        }


def _e5_builtin_identity_check(
    attack: Any,
    expected_class: type,
    expected_run_method: Callable[..., Any],
) -> tuple[bool, bool, bool]:
    return (
        type(attack) is expected_class,
        getattr(expected_class, "run", None) is expected_run_method,
        getattr(getattr(attack, "run", None), "__func__", None) is expected_run_method,
    )


def _trusted_builtin_e5_conditioning(
    result: AttackResult,
    defense: DefenseKind,
    implementation_attestation: _E5BuiltinImplementationAttestation,
) -> bool:
    """Recognize the repository's audited defense-conditioned F3 implementation."""

    if not implementation_attestation.valid:
        return False
    evidence = result.metadata.get("reoptimization_evidence")
    if not isinstance(evidence, Mapping):
        return False
    return bool(
        result.metadata.get("attack_reoptimized_for_defense") is True
        and result.metadata.get("reoptimized_defense") == defense.value
        and evidence.get("defense") == defense.value
        and evidence.get("fresh_search_from_clean") is True
        and (
            evidence.get("verifier_surrogate_defense_specific") is True
            or evidence.get("defense_surrogate_applied") is True
            or (
                result.metadata.get("knowledge_level") == "k1"
                and evidence.get("complete_system_feedback") is True
            )
        )
    )


def _reoptimization_evidence(
    validator: Callable[..., bool | Mapping[str, Any]] | None,
    *,
    case: CohortCase,
    defense: DefenseKind,
    verifier: Any,
    attack_verifier: Any,
    victim: Any,
    attack: Any,
    result: AttackResult,
    implementation_attestation: _E5BuiltinImplementationAttestation,
    counted_victim: _CountingVictim,
    causal_probe: Mapping[str, Any] | None,
    seed: int,
    query_budget: int,
) -> dict[str, Any]:
    observed_queries = counted_victim.queries_used
    observed_decisions = counted_victim.behaviorally_bound_queries
    audited_decisions = counted_victim.independently_audited_verifier_queries
    attack_verifier_calls = len(attack_verifier.decisions)
    trusted_builtin_conditioning = _trusted_builtin_e5_conditioning(
        result,
        defense,
        implementation_attestation,
    )
    causal_verifier_use = bool(causal_probe and causal_probe.get("valid") is True)
    native = {
        "family_valid": result.family in {AttackFamily.F3, AttackFamily.F3_DEFENSE},
        "fresh_victim_factory_instance": True,
        "fresh_attack_factory_instance": True,
        "factory_context": {
            "case_key": case.key,
            "defense": defense.value,
            "verifier_kind": verifier.kind.value,
            "seed": int(seed),
            "query_budget": int(query_budget),
        },
        "bound_verifier_not_exposed_to_attack_factory": (attack_verifier is not verifier),
        "attack_verifier_configuration_matches_bound_defense": bool(
            DefenseKind(attack_verifier.kind) is defense
            and np.isclose(
                float(attack_verifier.threshold),
                float(verifier.threshold),
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "observed_attack_verifier_calls": attack_verifier_calls,
        "causal_attack_verifier_probe": _plain_value(causal_probe),
        "causal_probe_certification_role": "diagnostic_only_for_custom_factories",
        "trusted_builtin_defense_conditioning": trusted_builtin_conditioning,
        "builtin_implementation_attestation": (implementation_attestation.to_dict()),
        "isolated_custom_factory_execution_attested": False,
        "primary_defense_conditioning_attested": trusted_builtin_conditioning,
        "defense_conditioning_observed": bool(causal_verifier_use or trusted_builtin_conditioning),
        "clean_start_supplied_by_orchestrator": True,
        "observed_complete_system_queries": observed_queries,
        "observed_bound_verifier_decisions": observed_decisions,
        "independently_replayed_verifier_decisions": audited_decisions,
        "attempts_match_observed_queries": bool(
            observed_queries == result.queries_used == len(result.attempts)
        ),
        "verifier_behaviorally_bound": bool(
            counted_victim.verifier_binding_matches(verifier)
            and observed_queries > 0
            and observed_decisions == observed_queries
            and audited_decisions == observed_queries
        ),
        "attack_authored_metadata_used_for_validity": False,
        "attack_reported_provenance": _plain_value(result.metadata.get("reoptimization_evidence")),
    }
    native_execution_valid = bool(
        native["family_valid"]
        and native["bound_verifier_not_exposed_to_attack_factory"]
        and native["attack_verifier_configuration_matches_bound_defense"]
        and native["attempts_match_observed_queries"]
        and native["verifier_behaviorally_bound"]
        and (trusted_builtin_conditioning or attack_verifier_calls > 0)
    )

    external: dict[str, Any] | None = None
    external_valid = True
    if validator is not None:
        value = _call_with_supported(
            validator,
            case=case,
            defense=defense,
            verifier=verifier,
            attack_verifier=attack_verifier,
            victim=victim,
            attack=attack,
            result=result,
            seed=int(seed),
            query_budget=int(query_budget),
        )
        if isinstance(value, Mapping):
            external = dict(value)
            if "valid" not in external:
                raise ValueError("E5 reoptimization evidence mapping requires a valid field")
        else:
            external = {"valid": bool(value)}
        external.setdefault("source", "explicit_validator")
        external_valid = bool(external["valid"])
    diagnostic_valid = bool(
        native_execution_valid and native["defense_conditioning_observed"] and external_valid
    )
    primary_eligible = bool(diagnostic_valid and native["primary_defense_conditioning_attested"])
    evidence = {
        "valid": diagnostic_valid,
        "diagnostic_valid": diagnostic_valid,
        "primary_eligible": primary_eligible,
        "endpoint_classification": ("primary" if primary_eligible else "exploratory"),
        "source": (
            "audited_builtin_f3"
            if trusted_builtin_conditioning
            else "in_process_custom_factory_diagnostic"
        ),
        "native": native,
        "external": external,
    }
    if not native_execution_valid:
        raise ValueError(
            "E5 could not establish that the calibrated verifier was installed and F3 "
            "was freshly run against it; an external bare valid flag or attack-authored "
            "metadata cannot replace orchestrator-observed verifier interactions"
        )
    if not external_valid:
        raise ValueError("E5 external reoptimization validation vetoed the run")
    return evidence


_E5_PRIMARY_DEFENSES = frozenset({DefenseKind.V1.value, DefenseKind.V2.value, DefenseKind.V3.value})


def _normalize_e5_defense_mapping(
    values: Mapping[DefenseKind | str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in values.items():
        defense = DefenseKind(raw_key).value
        if defense in normalized:
            raise ValueError(f"E5 {name} contains duplicate aliases for {defense}")
        normalized[defense] = value
    observed = frozenset(normalized)
    if observed != _E5_PRIMARY_DEFENSES:
        missing = sorted(_E5_PRIMARY_DEFENSES - observed)
        extra = sorted(observed - _E5_PRIMARY_DEFENSES)
        raise ValueError(
            f"E5 primary comparison requires exactly V1, V2, and V3 in {name}; "
            f"missing={missing}, extra={extra}"
        )
    return normalized


def _require_e5_sphys(constraint: Any) -> None:
    try:
        kind = ConstraintKind(constraint.kind)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("E5 primary comparison requires a typed SPhys constraint") from exc
    if kind is not ConstraintKind.S_PHYS:
        raise ValueError("E5 primary comparison requires the SPhys constraint")


def _assert_e5_verifier_configuration(
    verifier: Any,
    defense: DefenseKind,
    threshold: float,
    *,
    role: str,
) -> None:
    if isinstance(verifier, _ObservedVerifier):
        verifier.assert_threshold_locked()
        verifier._assert_decide_integrity()
    try:
        kind = DefenseKind(verifier.kind)
        observed_threshold = float(verifier.threshold)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f"E5 {role} verifier configuration is unavailable") from exc
    if kind is not defense or not np.isclose(
        observed_threshold,
        float(threshold),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"E5 {role} verifier configuration changed after matched calibration")


def _calibration_correctness_values(
    assessments: Sequence[ReportCorrectnessResult],
    *,
    defense_name: str,
    expected_length: int,
    expected_case_keys: Sequence[str] | None = None,
) -> np.ndarray:
    if len(assessments) != expected_length:
        raise ValueError("Calibration report correctness evidence must align with scores")
    values: list[bool] = []
    for index, assessment in enumerate(assessments):
        if not isinstance(assessment, ReportCorrectnessResult):
            raise TypeError(
                "E5 calibration correctness must contain ReportCorrectnessResult values"
            )
        if not assessment.primary_endpoint_eligible:
            raise ValueError(
                "E5 calibration selective risk requires independent all-claims evidence "
                f"for {defense_name}"
            )
        if expected_case_keys is not None:
            context = assessment.context
            observed_key = f"{context.patient_id}:{context.record_id}:{context.target.target_id}"
            if observed_key != expected_case_keys[index]:
                raise ValueError(
                    "E5 calibration correctness context does not match its keyed cohort "
                    f"for {defense_name} at index {index}"
                )
        values.append(bool(assessment.correct))
    return np.asarray(values, dtype=bool)


def _normalize_calibration_case_keys(
    values: Mapping[DefenseKind | str, Sequence[str]] | None,
    score_lengths: Mapping[str, int],
) -> tuple[dict[str, tuple[str, ...]] | None, bool]:
    """Validate that all defense scores are aligned to one explicit clean cohort."""

    if values is None:
        return None, False
    normalized = _normalize_e5_defense_mapping(values, name="calibration_case_keys")
    keys_by_defense: dict[str, tuple[str, ...]] = {}
    for defense, raw_keys in normalized.items():
        keys = tuple(str(value).strip() for value in raw_keys)
        if len(keys) != score_lengths[defense]:
            raise ValueError("E5 calibration case keys must align with calibration scores")
        if any(not value for value in keys) or len(set(keys)) != len(keys):
            raise ValueError("E5 calibration case keys must be non-empty and unique")
        keys_by_defense[defense] = keys
    reference = keys_by_defense[DefenseKind.V1.value]
    for defense, keys in keys_by_defense.items():
        if keys != reference:
            raise ValueError(
                "E5 matched coverage requires the same ordered keyed calibration cohort "
                f"for V1/V2/V3; {defense} differs"
            )
    return keys_by_defense, True


def _e5_calibration_context(
    item: E5CalibrationCase,
    defense: DefenseKind,
) -> dict[str, Any]:
    context = copy.deepcopy(item.verifier_contexts[defense])
    context["waveform_mv"] = item.case.waveform_mv.copy()
    context["patient_id"] = item.case.patient_id
    context["record_id"] = item.case.record_id
    sample_rate_hz = item.case.metadata.get("sample_rate_hz")
    if sample_rate_hz is not None:
        context["sample_rate_hz"] = sample_rate_hz
    current_timestamp = item.case.metadata.get("timestamp")
    if current_timestamp is not None:
        context["current_timestamp"] = current_timestamp
    return context


def _evaluate_e5_calibration_cases(
    cases: Sequence[E5CalibrationCase],
    verifier_templates: Mapping[str, Any],
) -> tuple[
    dict[str, list[float]],
    dict[str, list[bool]],
    dict[str, tuple[str, ...]],
    dict[str, list[ReportCorrectnessResult]] | None,
    dict[str, _ObservedVerifier],
]:
    if not cases:
        raise ValueError("E5 typed calibration cases must not be empty")
    if any(not isinstance(item, E5CalibrationCase) for item in cases):
        raise TypeError("E5 calibration_cases must contain E5CalibrationCase values")
    keys = tuple(item.key for item in cases)
    if len(set(keys)) != len(keys):
        raise ValueError("E5 typed calibration case keys must be unique")
    correctness_present = [item.report_correctness is not None for item in cases]
    if any(correctness_present) and not all(correctness_present):
        raise ValueError("E5 typed calibration correctness must be present for every case or none")

    scores: dict[str, list[float]] = {}
    eligible: dict[str, list[bool]] = {}
    case_keys: dict[str, tuple[str, ...]] = {}
    observers: dict[str, _ObservedVerifier] = {}
    correctness: dict[str, list[ReportCorrectnessResult]] | None = (
        {} if all(correctness_present) else None
    )
    for defense_name in sorted(verifier_templates):
        defense = DefenseKind(defense_name)
        observer = _ObservedVerifier(
            copy.deepcopy(verifier_templates[defense_name]),
            defense,
        )
        arm_scores: list[float] = []
        arm_eligible: list[bool] = []
        for item in cases:
            decision = observer.decide(
                copy.deepcopy(item.clean_snapshot),
                **_e5_calibration_context(item, defense),
            )
            arm_scores.append(float(decision.score))
            arm_eligible.append(bool(item.eligible and decision.evidence_available))
        scores[defense_name] = arm_scores
        eligible[defense_name] = arm_eligible
        case_keys[defense_name] = keys
        observers[defense_name] = observer
        if correctness is not None:
            correctness[defense_name] = [
                item.report_correctness for item in cases if item.report_correctness is not None
            ]
    return scores, eligible, case_keys, correctness, observers


def _recompute_e5_calibration_releases(
    cases: Sequence[E5CalibrationCase],
    observers: Mapping[str, _ObservedVerifier],
    scores: Mapping[str, Sequence[float]],
    thresholds: Mapping[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    releases: dict[str, np.ndarray] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for defense_name in sorted(observers):
        defense = DefenseKind(defense_name)
        observer = observers[defense_name]
        _set_verifier_threshold(observer, thresholds[defense_name])
        arm_releases: list[bool] = []
        arm_scores: list[float] = []
        for item in cases:
            decision = observer.decide(
                copy.deepcopy(item.clean_snapshot),
                **_e5_calibration_context(item, defense),
            )
            arm_scores.append(float(decision.score))
            arm_releases.append(
                bool(item.eligible and decision.evidence_available and decision.released)
            )
        expected_scores = np.asarray(scores[defense_name], dtype=float)
        observed_scores = np.asarray(arm_scores, dtype=float)
        if not np.allclose(expected_scores, observed_scores, rtol=0.0, atol=1e-12):
            raise ValueError("E5 verifier score changed when applying its calibrated threshold")
        releases[defense_name] = np.asarray(arm_releases, dtype=bool)
        provenance[defense_name] = {
            "case_keys": [item.key for item in cases],
            "score_calls": len(cases),
            "thresholded_release_calls": len(cases),
            "typed_release_decisions": len(observer.decisions),
            "verifier_kind": observer.kind.value,
            "threshold": observer.threshold,
            "score_stable_across_threshold_application": True,
        }
    return releases, provenance


def _e5_matched_coverage_validation(
    releases: Mapping[str, np.ndarray],
    target_coverage: float,
) -> dict[str, Any]:
    counts = {
        defense: int(np.sum(np.asarray(values, dtype=bool))) for defense, values in releases.items()
    }
    totals = {defense: len(np.asarray(values).reshape(-1)) for defense, values in releases.items()}
    if not totals or len(set(totals.values())) != 1:
        raise ValueError("E5 calibration release vectors must share one cohort size")
    total = next(iter(totals.values()))
    if total < 1:
        raise ValueError("E5 calibration release vectors must not be empty")
    coverages = {defense: counts[defense] / total for defense in sorted(counts)}
    counts_match = len(set(counts.values())) == 1
    attained = next(iter(coverages.values())) if counts_match else None
    quantization_tolerance = 0.5 / total + np.finfo(float).eps * 8
    target_attained = bool(
        attained is not None and abs(attained - float(target_coverage)) <= quantization_tolerance
    )
    valid = bool(counts_match and target_attained)
    reasons: list[str] = []
    if not counts_match:
        reasons.append("recomputed release counts differ across V1/V2/V3")
    if not target_attained:
        reasons.append("recomputed coverage is not the nearest cohort-quantized target coverage")
    return {
        "valid": valid,
        "target_coverage": float(target_coverage),
        "cohort_size": total,
        "released_counts": counts,
        "actual_coverages": coverages,
        "counts_match_across_defenses": counts_match,
        "target_attained_within_quantization": target_attained,
        "quantization_tolerance": quantization_tolerance,
        "reasons": reasons,
    }


def _factory(provider: Any, **context: Any) -> Any:
    if hasattr(provider, "run") and not inspect.isclass(provider):
        result = copy.deepcopy(provider)
        if "seed" in context and hasattr(result, "seed"):
            result.seed = int(context["seed"])
        return result
    if not callable(provider):
        raise TypeError("attack provider must be an attack or callable factory")
    result = _call_with_supported(provider, **context)
    if not hasattr(result, "run"):
        raise TypeError("attack factory must return an object exposing run")
    return result


def _constraint(provider: Any, **context: Any) -> Any:
    if hasattr(provider, "contains") and not inspect.isclass(provider):
        return copy.deepcopy(provider)
    if callable(provider):
        return _call_with_supported(provider, **context)
    raise TypeError("constraint provider must expose contains or be callable")


def _evaluate(victim: Any, case: CohortCase, defense: DefenseKind) -> VictimSnapshot:
    result = victim.evaluate(
        case.waveform_mv,
        case.target.target_label,
        defense=defense,
    )
    if not isinstance(result, VictimSnapshot):
        raise TypeError("victim.evaluate must return VictimSnapshot")
    return result


def run_e0_clean_baseline(
    cases: Sequence[CohortCase],
    victim_provider: Any,
    *,
    defense: DefenseKind | str = DefenseKind.V1,
    report_correctness_evaluator: Callable[..., Any] | None = None,
    logger: AppendOnlySecurityLogger | None = None,
) -> CleanBaselineRun:
    """E0 clean eligibility, with an explicit exploratory fallback."""

    defense = DefenseKind(defense)
    observations: list[ExperimentObservation] = []
    eligible: list[EligibleCase] = []
    for case in cases:
        snapshot = _evaluate(_victim(victim_provider, case, defense=defense), case, defense)
        correct = snapshot.diagnosis == case.target.clean_label
        report_correct, report_primary, report_assessment, correctness_metadata = (
            _e0_report_correctness(snapshot, case, report_correctness_evaluator)
        )
        adult_eligible, adult_metadata = _adult_eligibility(case)
        analysis_eligible = bool(report_correct and snapshot.released)
        primary_eligible = bool(analysis_eligible and report_primary and adult_eligible)
        if analysis_eligible:
            eligible.append(
                EligibleCase(
                    case,
                    snapshot,
                    primary_endpoint_eligible=primary_eligible,
                    report_correctness=report_assessment,
                )
            )
        observation = ExperimentObservation(
            experiment_id="E0",
            arm="clean",
            patient_id=case.patient_id,
            record_id=case.record_id,
            target_id=case.target.target_id,
            success=primary_eligible,
            released=snapshot.released,
            defense=defense.value,
            metrics={
                "diagnosis_correct": correct,
                "report_correct": report_correct,
                "target_claim_present": snapshot.target_claim_present,
                "analysis_eligible": analysis_eligible,
                "eligible": primary_eligible,
                "primary_endpoint_eligible": primary_eligible,
            },
            metadata={
                "diagnosis": snapshot.diagnosis,
                **correctness_metadata,
                **adult_metadata,
                "endpoint_classification": ("primary" if primary_eligible else "exploratory"),
                "primary_ineligibility_reason": (
                    None
                    if primary_eligible
                    else adult_metadata.get("adult_ineligibility_reason")
                    or correctness_metadata.get("primary_ineligibility_reason")
                    or "clean case is not released and independently all-claims correct"
                ),
            },
        )
        observations.append(observation)
        if logger is not None:
            logger.append("e0_clean", observation.to_dict(), waveform_mv=case.waveform_mv)
    correct_values = [bool(item.metrics["diagnosis_correct"]) for item in observations]
    report_correct_values = [bool(item.metrics["report_correct"]) for item in observations]
    release_values = [item.released for item in observations]
    selective = (
        None
        if not observations
        else selective_metrics(report_correct_values, release_values).to_dict()
    )
    endpoint_classes = {
        str(item.metadata.get("endpoint_classification", "exploratory")) for item in observations
    }
    endpoint_classification = (
        "primary"
        if endpoint_classes == {"primary"}
        else "mixed"
        if "primary" in endpoint_classes
        else "exploratory"
    )
    summary = {
        "cases": len(cases),
        "eligible": sum(item.primary_endpoint_eligible for item in eligible),
        "analysis_eligible": len(eligible),
        "exploratory_eligible": sum(not item.primary_endpoint_eligible for item in eligible),
        "endpoint_classification": endpoint_classification,
        "report_correctness_evaluator_configured": (report_correctness_evaluator is not None),
        "adult_primary_eligible": sum(
            bool(item.metadata.get("adult_eligible", False)) for item in observations
        ),
        "adult_metadata_missing_or_invalid": sum(
            item.metadata.get("adult_status") == "not_evaluable" for item in observations
        ),
        "diagnosis_accuracy": 0.0 if not cases else float(np.mean(correct_values)),
        "report_accuracy": 0.0 if not cases else float(np.mean(report_correct_values)),
        "false_flag_rate": (None if selective is None else selective["false_rejection_rate"]),
        "clean_rejection_rate": (
            0.0 if not cases else float(np.mean(np.logical_not(release_values)))
        ),
        "clean_coverage": 0.0 if not cases else float(np.mean(release_values)),
        "selective_metrics": selective,
    }
    return CleanBaselineRun(ExperimentRun("E0", tuple(observations), summary), tuple(eligible))


def _attack_summary(observations: Sequence[ExperimentObservation]) -> dict[str, Any]:
    groups: dict[tuple[str, str, int | None, str | None], list[ExperimentObservation]] = (
        defaultdict(list)
    )
    for observation in observations:
        groups[
            (
                observation.target_id,
                observation.arm,
                observation.budget,
                observation.defense,
            )
        ].append(observation)
    summary: dict[str, Any] = {}
    for (target_id, arm, budget, defense), rows in groups.items():
        result_rows = [row for row in rows if row.attack_result is not None]
        results = [row.attack_result for row in result_rows]
        patient_ids = [row.patient_id for row in result_rows]
        if not results:
            continue
        automated_rows = [row.metrics.get("automated_robustness") for row in result_rows]
        automated_summary = (
            summarize_automated_robustness(
                automated_rows,
                patient_ids,
                query_budgets=sorted({1, int(budget or 1)}),
                n_resamples=10_000,
                seed=42,
            )
            if all(isinstance(item, Mapping) for item in automated_rows)
            else None
        )
        primary_rows = [
            row for row in result_rows if bool(row.metrics.get("primary_endpoint_eligible", False))
        ]
        if primary_rows:
            estimate = bootstrap_incorrect_release_rate(
                [row.success for row in primary_rows],
                [row.patient_id for row in primary_rows],
                n_resamples=500,
                seed=42,
            ).to_dict()
            endpoint_classification = (
                "primary" if len(primary_rows) == len(result_rows) else "mixed"
            )
        else:
            estimate = {
                "estimate": None,
                "lower": None,
                "upper": None,
                "confidence": 0.95,
                "n_patients": 0,
                "n_resamples": 0,
                "seed": 42,
                "metadata": {
                    "metric": "incorrect_release_rate",
                    "status": "not_evaluable",
                    "reason": "no primary-endpoint-eligible observations",
                },
            }
            endpoint_classification = "not_evaluable"
        success_queries = [
            int(query_index)
            for row in primary_rows
            if row.success
            and (query_index := row.metrics.get("primary_success_query_index")) is not None
        ]
        exploratory_values = [
            bool(row.metrics.get("exploratory_incorrect_release", False)) for row in result_rows
        ]
        exploratory_estimate = bootstrap_incorrect_release_rate(
            exploratory_values,
            patient_ids,
            n_resamples=500,
            seed=42,
        )
        key = f"target={target_id}|{arm}|B={budget}|{defense or 'none'}"
        summary[key] = {
            **estimate,
            "endpoint_classification": endpoint_classification,
            "primary_endpoint_evaluable_observations": len(primary_rows),
            "exploratory_incorrect_release_rate": exploratory_estimate.to_dict(),
            "rounds_to_pass": {
                "total_cases": len(primary_rows),
                "successful_cases": len(success_queries),
                "median_rounds_to_pass": (
                    None if not success_queries else float(np.median(success_queries))
                ),
                "mean_rounds_to_pass": (
                    None if not success_queries else float(np.mean(success_queries))
                ),
                "success_by_budget": {
                    budget or 1: (
                        None
                        if not primary_rows
                        else float(np.mean([row.success for row in primary_rows]))
                    )
                },
                "first_success_queries": success_queries,
            },
            "observations": len(results),
            "seeds": sorted({row.seed for row in rows if row.seed is not None}),
            "statistical_unit": "patient; repeated records and seeds collapsed within patient",
            "automated_robustness": automated_summary,
        }
    return summary


def _paired_f3_f1_summary(
    observations: Sequence[ExperimentObservation],
) -> dict[str, dict[str, Any]]:
    strata: dict[
        tuple[str, str, int | None, str | None],
        dict[str, dict[tuple[str, str, int | None], ExperimentObservation]],
    ] = defaultdict(lambda: defaultdict(dict))
    for row in observations:
        family, _, constraint = row.arm.partition(":")
        stratum = (
            row.target_id,
            constraint,
            row.budget,
            row.defense,
        )
        pair_key = (
            row.patient_id,
            row.record_id,
            row.seed,
        )
        if pair_key in strata[stratum][family]:
            raise ValueError(
                "E1 paired comparison has duplicate family rows for the same case and seed"
            )
        strata[stratum][family][pair_key] = row

    summary: dict[str, dict[str, Any]] = {}
    for (target_id, constraint, budget, defense), families in strata.items():
        comparator = families.get(AttackFamily.F1.value)
        if comparator is None:
            continue
        for adaptive_family in (
            AttackFamily.F3.value,
            AttackFamily.F3_DEFENSE.value,
        ):
            adaptive = families.get(adaptive_family)
            if adaptive is None:
                continue
            comparator_keys = set(comparator)
            adaptive_keys = set(adaptive)
            if comparator_keys != adaptive_keys:
                missing_adaptive = len(comparator_keys - adaptive_keys)
                missing_comparator = len(adaptive_keys - comparator_keys)
                raise ValueError(
                    "E1 paired F3-F1 comparison requires complete case/seed pairing "
                    f"(missing adaptive={missing_adaptive}, "
                    f"missing comparator={missing_comparator})"
                )
            eligible_keys = {
                pair_key
                for pair_key in comparator_keys
                if bool(comparator[pair_key].metrics.get("primary_endpoint_eligible", False))
                and bool(adaptive[pair_key].metrics.get("primary_endpoint_eligible", False))
            }
            by_patient_adaptive: dict[str, list[float]] = defaultdict(list)
            by_patient_comparator: dict[str, list[float]] = defaultdict(list)
            for pair_key in sorted(eligible_keys):
                patient_id = pair_key[0]
                by_patient_adaptive[patient_id].append(float(adaptive[pair_key].success))
                by_patient_comparator[patient_id].append(float(comparator[pair_key].success))
            adaptive_patient = {
                patient: float(np.mean(values)) for patient, values in by_patient_adaptive.items()
            }
            comparator_patient = {
                patient: float(np.mean(values)) for patient, values in by_patient_comparator.items()
            }
            if eligible_keys:
                estimate = bootstrap_paired_irr_difference(
                    adaptive_patient,
                    comparator_patient,
                    n_resamples=500,
                    seed=42,
                ).to_dict()
                endpoint_classification = (
                    "primary" if len(eligible_keys) == len(comparator_keys) else "mixed"
                )
            else:
                estimate = {
                    "estimate": None,
                    "lower": None,
                    "upper": None,
                    "confidence": 0.95,
                    "n_patients": 0,
                    "n_resamples": 0,
                    "seed": 42,
                    "metadata": {
                        "metric": "paired_delta_irr",
                        "status": "not_evaluable",
                        "reason": "no primary-endpoint-eligible pairs",
                    },
                }
                endpoint_classification = "not_evaluable"
            key = (
                f"target={target_id}|{adaptive_family}-f1:{constraint}"
                f"|B={budget}|{defense or 'none'}"
            )
            summary[key] = {
                **estimate,
                "endpoint_classification": endpoint_classification,
                "pairs": len(eligible_keys),
                "total_matched_pairs": len(comparator_keys),
                "excluded_non_primary_pairs": len(comparator_keys - eligible_keys),
                "seeds": sorted(
                    {pair_key[2] for pair_key in eligible_keys if pair_key[2] is not None}
                ),
                "pairing_unit": "patient, record, target, and seed",
            }
    return summary


def run_e1_attack_matrix(
    eligible_cases: Sequence[EligibleCase],
    victim_provider: Any,
    attack_factories: Mapping[AttackFamily | str, Any],
    constraints: Mapping[str, Any],
    *,
    budgets: Sequence[int] = (1, 10, 50, 100),
    seeds: Sequence[int] = (42,),
    defense: DefenseKind | str = DefenseKind.V1,
    admissibility_checker: Any | None = None,
    logger: AppendOnlySecurityLogger | None = None,
) -> ExperimentRun:
    """E1: matched family/constraint/budget matrix with first-success stopping."""

    defense = DefenseKind(defense)
    normalized_budgets = tuple(sorted({int(value) for value in budgets}))
    if not normalized_budgets or normalized_budgets[0] < 1 or not seeds:
        raise ValueError("E1 requires positive budgets and at least one seed")
    observations: list[ExperimentObservation] = []
    for eligible in eligible_cases:
        case = _private_case_copy(eligible.case)
        clean_snapshot = copy.deepcopy(eligible.clean_snapshot)
        for family_key, attack_provider in attack_factories.items():
            family = AttackFamily(family_key)
            for constraint_name, constraint_provider in constraints.items():
                for budget in normalized_budgets:
                    for seed in seeds:
                        victim = _victim(victim_provider, case, defense=defense)
                        attack_case = _private_case_copy(case)
                        attack_clean = case.waveform_mv.copy()
                        attack = _factory(
                            attack_provider,
                            seed=int(seed),
                            budget=budget,
                            defense=defense,
                            case=attack_case,
                        )
                        constraint = _constraint(constraint_provider, seed=int(seed), case=case)
                        counted_victim = _CountingVictim(
                            victim,
                            budget,
                            clean_waveform=case.waveform_mv,
                            constraint=constraint,
                        )
                        preservation_reviewer = (
                            None
                            if admissibility_checker is None
                            else _QueryBoundPreservationReviewer(
                                admissibility_checker,
                                clean_waveform=case.waveform_mv,
                                case=case,
                                victim=victim,
                                constraint=constraint,
                                victim_recorder=counted_victim,
                            )
                        )
                        attack_checker = (
                            None
                            if preservation_reviewer is None
                            else preservation_reviewer.attack_proxy
                        )
                        result = attack.run(
                            counted_victim,
                            attack_clean,
                            case.target.target_label,
                            query_budget=budget,
                            constraint=constraint,
                            admissibility_checker=attack_checker,
                            case=attack_case,
                            defense=defense,
                            clean_snapshot=copy.deepcopy(clean_snapshot),
                        )
                        if not isinstance(result, AttackResult):
                            raise TypeError("Attack runner must return AttackResult")
                        allowed_families = {family}
                        if family is AttackFamily.F3:
                            allowed_families.add(AttackFamily.F3_DEFENSE)
                        if result.family not in allowed_families:
                            raise ValueError(
                                f"E1 attack factory for {family.value} returned "
                                f"{result.family.value}"
                            )
                        if result.target_label != case.target.target_label:
                            raise ValueError(
                                "E1 attack result target does not match the cohort target"
                            )
                        _validate_attack_boundary(
                            result,
                            case=case,
                            clean_snapshot=clean_snapshot,
                            constraint=constraint,
                            query_budget=budget,
                            counted_queries=counted_victim.queries_used,
                            query_waveforms=counted_victim.query_waveforms,
                            query_snapshots=counted_victim.query_snapshots,
                            experiment_id="E1",
                        )
                        arm = f"{family.value}:{constraint_name}"
                        (
                            preservation_evaluable,
                            independently_successful,
                            primary_success_query,
                            preservation_evidence,
                        ) = _independent_preservation_assessment(
                            admissibility_checker,
                            clean_waveform=case.waveform_mv,
                            query_waveforms=counted_victim.query_waveforms,
                            query_snapshots=counted_victim.query_snapshots,
                            case=case,
                            victim=victim,
                            constraint=constraint,
                            review_entries=(
                                None
                                if preservation_reviewer is None
                                else preservation_reviewer.complete()
                            ),
                        )
                        primary_endpoint_eligible = bool(
                            eligible.primary_endpoint_eligible
                            and _adult_eligibility(case)[0]
                            and preservation_evaluable
                        )
                        primary_success = bool(
                            independently_successful and primary_endpoint_eligible
                        )
                        receipt = _issue_preservation_receipt(
                            result,
                            experiment_id="E1",
                            case=case,
                            clean_snapshot=clean_snapshot,
                            query_waveforms=counted_victim.query_waveforms,
                            query_snapshots=counted_victim.query_snapshots,
                            primary_endpoint_eligible=primary_endpoint_eligible,
                            primary_success=primary_success,
                            primary_success_query_index=primary_success_query,
                            review_payload=preservation_evidence,
                        )
                        exploratory_success = bool(
                            result.metadata.get("exploratory_success", result.success)
                        )
                        automated = automated_attack_metrics(
                            result,
                            case.waveform_mv,
                            expected_clean_label=case.target.clean_label,
                        ).to_dict()
                        observation = ExperimentObservation(
                            experiment_id="E1",
                            arm=arm,
                            patient_id=case.patient_id,
                            record_id=case.record_id,
                            target_id=case.target.target_id,
                            success=primary_success,
                            released=bool(
                                result.adversarial_snapshot and result.adversarial_snapshot.released
                            ),
                            seed=int(seed),
                            budget=budget,
                            defense=defense.value,
                            queries_used=result.queries_used,
                            metrics={
                                "incorrect_release": primary_success,
                                "exploratory_incorrect_release": exploratory_success,
                                "primary_endpoint_eligible": primary_endpoint_eligible,
                                "selected_primary_preservation_evidence": (
                                    independently_successful
                                ),
                                "primary_success_query_index": primary_success_query,
                                "automated_robustness": automated,
                            },
                            metadata={
                                "constraint": constraint_name,
                                "counted_complete_system_queries": (counted_victim.queries_used),
                                "selected_preservation_evidence": preservation_evidence,
                                "independent_preservation_assessment": (preservation_evidence),
                                "preservation_verification_receipt": (
                                    None if receipt is None else receipt.to_dict()
                                ),
                                "endpoint_classification": (
                                    "primary" if primary_endpoint_eligible else "exploratory"
                                ),
                            },
                            attack_result=result,
                        )
                        observations.append(observation)
                        if logger is not None:
                            logger.append_attack_result(
                                result,
                                patient_id=case.patient_id,
                                record_id=case.record_id,
                                target_id=case.target.target_id,
                                experiment_id="E1",
                                context={
                                    "preservation_verification_receipt": (
                                        None if receipt is None else receipt.to_dict()
                                    ),
                                    "independent_preservation_assessment": (preservation_evidence),
                                },
                            )
    summary = _attack_summary(observations)
    summary["paired_f3_minus_f1"] = _paired_f3_f1_summary(observations)
    return ExperimentRun("E1", tuple(observations), summary)


def _selected_result(selected: Mapping[str, AttackResult], eligible: EligibleCase) -> AttackResult:
    for key in (eligible.case.key, eligible.case.record_id):
        if key in selected:
            return selected[key]
    raise KeyError(f"No selected attack for {eligible.case.key}")


def run_e1b_staged_ablations(
    eligible_cases: Sequence[EligibleCase],
    selected_attacks: Mapping[str, AttackResult],
    variant_victims: Mapping[str, Any],
    *,
    defense: DefenseKind | str = DefenseKind.V1,
) -> ExperimentRun:
    """E1b: replay the selected perturbation through staged victim variants."""

    defense = DefenseKind(defense)
    observations: list[ExperimentObservation] = []
    for eligible in eligible_cases:
        result = _selected_result(selected_attacks, eligible)
        if result.adversarial_signal is None:
            continue
        source_primary, source_evidence = _source_primary_status(eligible, result)
        for variant, provider in variant_victims.items():
            victim = _victim(provider, eligible.case, variant=variant)
            snapshot = victim.evaluate(
                result.adversarial_signal,
                eligible.case.target.target_label,
                defense=defense,
            )
            exploratory_success = bool(snapshot.released and snapshot.target_claim_present)
            success = bool(exploratory_success and source_primary)
            observations.append(
                ExperimentObservation(
                    "E1b",
                    variant,
                    eligible.case.patient_id,
                    eligible.case.record_id,
                    eligible.case.target.target_id,
                    success,
                    snapshot.released,
                    defense=defense.value,
                    metrics={
                        "incorrect_release": success,
                        "exploratory_incorrect_release": exploratory_success,
                        "primary_endpoint_eligible": source_primary,
                        **source_evidence,
                    },
                    metadata={
                        "diagnosis": snapshot.diagnosis,
                        "endpoint_classification": ("primary" if source_primary else "exploratory"),
                    },
                )
            )
    return ExperimentRun(
        "E1b",
        tuple(observations),
        _endpoint_aware_arm_rates(
            observations,
            exploratory_metric="exploratory_incorrect_release",
        ),
    )


def _normalize_replay(value: Any) -> tuple[bool, bool, dict[str, Any]]:
    if isinstance(value, VictimSnapshot):
        return value.released, value.target_claim_present, {"diagnosis": value.diagnosis}
    if isinstance(value, ReleaseDecision):
        raise TypeError(
            "A ReleaseDecision does not establish target-claim presence; replay callbacks "
            "must return VictimSnapshot or an explicit released/target_claim_present mapping"
        )
    if isinstance(value, Mapping):
        missing = {"released", "target_claim_present"} - set(value)
        if missing:
            raise ValueError(
                "Replay mappings must explicitly provide released and "
                f"target_claim_present; missing={sorted(missing)}"
            )
        return bool(value["released"]), bool(value["target_claim_present"]), dict(value)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            "A boolean replay result cannot distinguish release from target-claim presence"
        )
    raise TypeError(
        "Replay callback must return VictimSnapshot or an explicit "
        "released/target_claim_present mapping"
    )


def run_e2_retrieval_mediation(
    pairs: Sequence[tuple[EligibleCase, AttackResult]],
    replay: Callable[..., Any],
) -> ExperimentRun:
    """E2 interventions over successful, primary-eligible preserved source attacks."""

    observations: list[ExperimentObservation] = []
    eligible_pairs = 0
    exclusion_counts: dict[str, int] = defaultdict(int)
    excluded_case_keys: list[str] = []
    for eligible, result in pairs:
        source_primary_eligible, source_evidence = _source_primary_status(eligible, result)
        exclusion_reasons = list(source_evidence["source_ineligibility_reasons"])
        if (
            not source_primary_eligible
            and "source_attack_not_primary_endpoint_eligible" not in exclusion_reasons
        ):
            exclusion_reasons.append("source_attack_not_primary_endpoint_eligible")
        if exclusion_reasons:
            excluded_case_keys.append(eligible.case.key)
            for reason in exclusion_reasons:
                exclusion_counts[reason] += 1
            continue
        attacked = result.adversarial_snapshot
        if attacked is None:
            raise ValueError("An E2-eligible source attack must carry its attacked snapshot")
        eligible_pairs += 1
        source_evidence = {
            **source_evidence,
            "source_selected_clinically_admissible": True,
            "primary_endpoint_eligible": True,
        }
        snapshots = {"clean": eligible.clean_snapshot, "attacked": attacked}
        overlap = retrieval_overlap_at_k(
            eligible.clean_snapshot.retrieved_fact_ids,
            attacked.retrieved_fact_ids,
            max(
                1,
                min(
                    len(eligible.clean_snapshot.retrieved_fact_ids),
                    len(attacked.retrieved_fact_ids),
                ),
            ),
        )
        for context_name, context in snapshots.items():
            for retrieval_name, retrieval in snapshots.items():
                outcome = _call_with_supported(
                    replay,
                    case=eligible.case,
                    context_snapshot=context,
                    retrieval_snapshot=retrieval,
                    draft_report=None,
                    frozen_draft=False,
                )
                released, claim, metadata = _normalize_replay(outcome)
                arm = f"context_{context_name}:retrieval_{retrieval_name}"
                observations.append(
                    _observation(
                        "E2",
                        arm,
                        eligible.case,
                        released and claim,
                        released,
                        {
                            **metadata,
                            **source_evidence,
                            "retrieval_overlap_at_k": overlap,
                        },
                    )
                )
        for retrieval_name, retrieval in snapshots.items():
            outcome = _call_with_supported(
                replay,
                case=eligible.case,
                context_snapshot=attacked,
                retrieval_snapshot=retrieval,
                draft_report=attacked.report,
                frozen_draft=True,
            )
            released, claim, metadata = _normalize_replay(outcome)
            arm = f"frozen_wrong_draft:retrieval_{retrieval_name}"
            observations.append(
                _observation(
                    "E2",
                    arm,
                    eligible.case,
                    released and claim,
                    released,
                    {**metadata, **source_evidence},
                )
            )
    summary = _simple_arm_rates(observations)
    summary["source_pair_eligibility"] = {
        "submitted": len(pairs),
        "eligible": eligible_pairs,
        "excluded": len(pairs) - eligible_pairs,
        "excluded_case_keys": excluded_case_keys,
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "criterion": (
            "authenticated E1/E5 preservation receipt establishing source success, "
            "clean/source primary-endpoint eligibility, and selected-artifact binding"
        ),
    }
    factorial_arms = (
        "context_clean:retrieval_clean",
        "context_clean:retrieval_attacked",
        "context_attacked:retrieval_clean",
        "context_attacked:retrieval_attacked",
    )
    if all(arm in summary for arm in factorial_arms):
        clean_clean, clean_attacked, attacked_clean, attacked_attacked = (
            summary[arm] for arm in factorial_arms
        )
        summary["factorial_interaction"] = {
            "acceptance_difference_in_differences": (
                attacked_attacked["release_rate"]
                - attacked_clean["release_rate"]
                - clean_attacked["release_rate"]
                + clean_clean["release_rate"]
            ),
            "incorrect_release_difference_in_differences": (
                attacked_attacked["rate"]
                - attacked_clean["rate"]
                - clean_attacked["rate"]
                + clean_clean["rate"]
            ),
        }
    return ExperimentRun("E2", tuple(observations), summary)


def run_e3_latent_restoration(
    pairs: Sequence[tuple[EligibleCase, AttackResult]],
    latent_replay: Callable[..., Any],
    *,
    top_k: Sequence[int] = (1, 3, 5),
    random_repeats: int = 10,
    seed: int = 42,
) -> ExperimentRun:
    """E3: restore top-k displaced latent factors and matched random-k controls."""

    if random_repeats < 1:
        raise ValueError("random_repeats must be positive")
    rng = np.random.default_rng(seed)
    observations: list[ExperimentObservation] = []
    for eligible, result in pairs:
        attacked = result.adversarial_snapshot
        if attacked is None:
            continue
        source_primary, source_evidence = _source_primary_status(eligible, result)
        clean = eligible.clean_snapshot.latent
        adversarial = attacked.latent
        if clean.shape != adversarial.shape:
            raise ValueError("Clean and attacked latent vectors must align")
        ranking = np.argsort(np.abs(adversarial - clean))[::-1]
        for requested_k in sorted({int(value) for value in top_k}):
            if requested_k < 1:
                raise ValueError("top_k values must be positive")
            k = min(requested_k, len(clean))
            arms: list[tuple[str, np.ndarray, list[int], int | None]] = []
            top_indices = ranking[:k]
            restored = adversarial.copy()
            restored[top_indices] = clean[top_indices]
            arms.append((f"top_{k}", restored, top_indices.tolist(), None))
            for repeat in range(random_repeats):
                indices = np.sort(rng.choice(len(clean), size=k, replace=False))
                control = adversarial.copy()
                control[indices] = clean[indices]
                arms.append((f"random_{k}", control, indices.tolist(), repeat))
            for arm, latent, indices, repeat in arms:
                outcome = _call_with_supported(
                    latent_replay,
                    case=eligible.case,
                    latent=latent,
                    target_label=eligible.case.target.target_label,
                )
                released, claim, metadata = _normalize_replay(outcome)
                diagnosis = metadata.get("diagnosis")
                diagnosis_recovered = diagnosis == eligible.clean_snapshot.diagnosis
                claim_recovered = not claim
                release_recovered = released == eligible.clean_snapshot.released
                recovered = bool(diagnosis_recovered and claim_recovered and release_recovered)
                primary_recovered = bool(recovered and source_primary)
                observations.append(
                    _observation(
                        "E3",
                        arm,
                        eligible.case,
                        primary_recovered,
                        released,
                        {
                            **metadata,
                            "restored_indices": indices,
                            "repeat": repeat,
                            "target_claim_present": claim,
                            "diagnosis_recovered": diagnosis_recovered,
                            "claim_recovered": claim_recovered,
                            "release_recovered": release_recovered,
                            "full_recovery": recovered,
                            "exploratory_recovery": recovered,
                            "primary_endpoint_eligible": source_primary,
                            **source_evidence,
                        },
                    )
                )
    return ExperimentRun(
        "E3",
        tuple(observations),
        _endpoint_aware_arm_rates(
            observations,
            exploratory_metric="exploratory_recovery",
        ),
    )


def run_e4_counterfactual_alignment(
    pairs: Sequence[tuple[EligibleCase, AttackResult]],
    counterfactual_provider: Callable[..., Sequence[float]] | None = None,
    *,
    k: int = 5,
) -> ExperimentRun:
    """E4: compare adversarial latent displacement with the minimal CARE counterfactual."""

    observations: list[ExperimentObservation] = []
    for eligible, result in pairs:
        attacked = result.adversarial_snapshot
        if attacked is None:
            continue
        source_primary, source_evidence = _source_primary_status(eligible, result)
        if counterfactual_provider is None:
            value = eligible.clean_snapshot.metadata.get("counterfactual_delta")
            if value is None:
                raise ValueError("E4 requires a counterfactual delta provider or metadata")
            counterfactual_delta = np.asarray(value, dtype=float)
        else:
            counterfactual_delta = np.asarray(
                _call_with_supported(
                    counterfactual_provider,
                    case=eligible.case,
                    clean_snapshot=eligible.clean_snapshot,
                    attack_result=result,
                ),
                dtype=float,
            )
        adversarial_delta = attacked.latent - eligible.clean_snapshot.latent
        alignment = counterfactual_alignment(adversarial_delta, counterfactual_delta, k=k)
        observations.append(
            _observation(
                "E4",
                "counterfactual_alignment",
                eligible.case,
                source_primary,
                attacked.released,
                {
                    **alignment.to_dict(),
                    "primary_endpoint_eligible": source_primary,
                    **source_evidence,
                },
            )
        )
    primary = [row for row in observations if row.metrics.get("primary_endpoint_eligible", False)]
    summary = {
        "cases": len(observations),
        "primary_endpoint_evaluable_cases": len(primary),
        "endpoint_classification": (
            "not_evaluable"
            if not primary
            else "primary"
            if len(primary) == len(observations)
            else "mixed"
        ),
        "mean_cosine_similarity": None
        if not primary
        else float(np.mean([row.metrics["cosine_similarity"] for row in primary])),
        "mean_factor_overlap_at_k": None
        if not primary
        else float(np.mean([row.metrics["factor_overlap_at_k"] for row in primary])),
        "exploratory_mean_cosine_similarity": None
        if not observations
        else float(np.mean([row.metrics["cosine_similarity"] for row in observations])),
        "exploratory_mean_factor_overlap_at_k": None
        if not observations
        else float(np.mean([row.metrics["factor_overlap_at_k"] for row in observations])),
    }
    return ExperimentRun("E4", tuple(observations), summary)


def run_e5_matched_coverage_reattack(
    eligible_cases: Sequence[EligibleCase],
    victim_provider: Any,
    attack_factory: Any,
    verifiers: Mapping[DefenseKind | str, Any],
    calibration_scores: Mapping[DefenseKind | str, Sequence[float]] | None = None,
    *,
    target_coverage: float,
    constraint: Any,
    seeds: Sequence[int] = (42,),
    query_budget: int = 100,
    calibration_cases: Sequence[E5CalibrationCase] | None = None,
    calibration_eligible: Mapping[DefenseKind | str, Sequence[bool]] | None = None,
    calibration_case_keys: Mapping[DefenseKind | str, Sequence[str]] | None = None,
    calibration_report_correctness: Mapping[DefenseKind | str, Sequence[ReportCorrectnessResult]]
    | None = None,
    admissibility_checker: Any | None = None,
    report_correctness_evaluator: Callable[..., ReportCorrectnessResult] | None = None,
    reoptimization_validator: Callable[..., bool | Mapping[str, Any]] | None = None,
    logger: AppendOnlySecurityLogger | None = None,
) -> ExperimentRun:
    """E5: behaviorally calibrate V1/V2/V3, then freshly run F3 in SPhys.

    Primary matched-coverage estimates require ``calibration_cases``. Precomputed
    ``calibration_scores`` remain supported solely for exploratory compatibility.
    """

    audited_f3_class = F3AdaptiveAttack
    audited_f3_run_method = audited_f3_class.run
    audited_f3_class_name = f"{audited_f3_class.__module__}.{audited_f3_class.__qualname__}"
    audited_f3_method_name = (
        f"{audited_f3_run_method.__module__}.{audited_f3_run_method.__qualname__}"
    )

    _require_context_factory(
        victim_provider,
        name="victim_provider",
        instance_method="evaluate",
        required_context=("case", "defense", "verifier"),
    )
    _require_context_factory(
        attack_factory,
        name="attack_factory",
        instance_method="run",
        required_context=("seed", "defense", "verifier", "case"),
    )
    if query_budget < 1 or not seeds:
        raise ValueError("E5 requires a positive query budget and at least one seed")
    if any(
        not item.primary_endpoint_eligible or not _adult_eligibility(item.case)[0]
        for item in eligible_cases
    ):
        raise ValueError("E5 primary comparison requires E0 primary-eligible cases")
    normalized_verifiers = _normalize_e5_defense_mapping(verifiers, name="verifiers")
    for defense_name, verifier_template in normalized_verifiers.items():
        _ObservedVerifier(copy.deepcopy(verifier_template), DefenseKind(defense_name))

    typed_calibration = calibration_cases is not None
    calibration_releases: dict[str, np.ndarray] | None = None
    calibration_provenance: dict[str, dict[str, Any]] | None = None
    calibration_coverage_validation: dict[str, Any] | None = None
    legacy_keyed_common_cohort = False
    if typed_calibration:
        if calibration_scores is not None or any(
            value is not None
            for value in (
                calibration_eligible,
                calibration_case_keys,
                calibration_report_correctness,
            )
        ):
            raise ValueError(
                "E5 typed calibration_cases cannot be combined with legacy calibration arrays"
            )
        typed_cases = tuple(calibration_cases or ())
        (
            normalized_scores,
            normalized_eligible,
            normalized_case_keys,
            normalized_correctness,
            calibration_observers,
        ) = _evaluate_e5_calibration_cases(typed_cases, normalized_verifiers)
        calibration_cohort_primary = False
        calibration_source = "orchestrator_recomputed_typed_shared_cases"
    else:
        if calibration_scores is None:
            raise ValueError("E5 requires typed calibration_cases or legacy calibration_scores")
        typed_cases = ()
        normalized_scores = _normalize_e5_defense_mapping(
            calibration_scores,
            name="calibration_scores",
        )
        normalized_eligible = (
            None
            if calibration_eligible is None
            else _normalize_e5_defense_mapping(
                calibration_eligible,
                name="calibration_eligible",
            )
        )
        normalized_correctness = (
            None
            if calibration_report_correctness is None
            else _normalize_e5_defense_mapping(
                calibration_report_correctness,
                name="calibration_report_correctness",
            )
        )
        score_lengths = {
            defense_name: len(np.asarray(values).reshape(-1))
            for defense_name, values in normalized_scores.items()
        }
        normalized_case_keys, legacy_keyed_common_cohort = _normalize_calibration_case_keys(
            calibration_case_keys, score_lengths
        )
        calibration_cohort_primary = False
        calibration_source = "legacy_precomputed_arrays_unverified"
    if hasattr(constraint, "contains") and not inspect.isclass(constraint):
        _require_e5_sphys(constraint)
    thresholds = calibrate_matched_coverage(
        normalized_scores,
        target_coverage,
        eligible=normalized_eligible,
    )
    assert isinstance(thresholds, dict)
    if typed_calibration:
        calibration_releases, calibration_provenance = _recompute_e5_calibration_releases(
            typed_cases,
            calibration_observers,
            normalized_scores,
            thresholds,
        )
        calibration_coverage_validation = _e5_matched_coverage_validation(
            calibration_releases,
            target_coverage,
        )
        calibration_cohort_primary = bool(calibration_coverage_validation["valid"])
    clean_operating_points: dict[str, dict[str, Any]] = {}
    for defense_name, values in normalized_scores.items():
        scores = np.asarray(values, dtype=float).reshape(-1)
        available = (
            np.ones(len(scores), dtype=bool)
            if normalized_eligible is None or defense_name not in normalized_eligible
            else np.asarray(normalized_eligible[defense_name], dtype=bool).reshape(-1)
        )
        if available.shape != scores.shape:
            raise ValueError("Calibration eligibility masks must align with scores")
        released = (
            available & (scores <= thresholds[defense_name])
            if calibration_releases is None
            else calibration_releases[defense_name]
        )
        operating_point: dict[str, Any] = {
            "threshold": thresholds[defense_name],
            "coverage": float(np.mean(released)),
            "released": int(np.sum(released)),
            "total": len(scores),
            "selective_risk": None,
            "false_rejection_rate": None,
            "correctness_available": False,
            "calibration_source": calibration_source,
            "verifier_behavior_recomputed": typed_calibration,
        }
        if normalized_correctness is not None:
            correct = _calibration_correctness_values(
                normalized_correctness[defense_name],
                defense_name=defense_name,
                expected_length=len(scores),
                expected_case_keys=(
                    None if normalized_case_keys is None else normalized_case_keys[defense_name]
                ),
            )
            operating_point.update(selective_metrics(correct, released).to_dict())
            operating_point["correctness_available"] = True
            operating_point["correctness_contract"] = "independent_all_claims"
        operating_point["keyed_common_cohort"] = bool(
            typed_calibration or legacy_keyed_common_cohort
        )
        operating_point["coverage_endpoint_classification"] = (
            "primary" if calibration_cohort_primary else "exploratory"
        )
        operating_point["selective_risk_endpoint_classification"] = (
            "primary"
            if calibration_cohort_primary and operating_point["correctness_available"]
            else "not_evaluable"
            if not operating_point["correctness_available"]
            else "exploratory"
        )
        if calibration_provenance is not None:
            operating_point["behavioral_calibration_provenance"] = calibration_provenance[
                defense_name
            ]
        clean_operating_points[defense_name] = operating_point
    observations: list[ExperimentObservation] = []
    victim_instances: list[Any] = []
    attack_instances: list[Any] = []
    for defense_name in sorted(normalized_verifiers):
        defense = DefenseKind(defense_name)
        verifier_template = normalized_verifiers[defense_name]
        for eligible in eligible_cases:
            for seed in seeds:
                clean_snapshot = copy.deepcopy(eligible.clean_snapshot)
                attack_case = _private_case_copy(eligible.case)
                attack_clean = eligible.case.waveform_mv.copy()
                verifier = _ObservedVerifier(copy.deepcopy(verifier_template), defense)
                _set_verifier_threshold(verifier, thresholds[defense.value])
                verifier.lock_threshold(thresholds[defense.value])
                audit_verifier = _ObservedVerifier(
                    copy.deepcopy(verifier_template),
                    defense,
                )
                _set_verifier_threshold(
                    audit_verifier,
                    thresholds[defense.value],
                )
                audit_verifier.lock_threshold(thresholds[defense.value])
                attack_verifier = _ObservedVerifier(
                    copy.deepcopy(verifier_template),
                    defense,
                )
                _set_verifier_threshold(
                    attack_verifier,
                    thresholds[defense.value],
                )
                attack_verifier.lock_threshold(thresholds[defense.value])
                victim = _victim(
                    victim_provider,
                    eligible.case,
                    defense=defense,
                    verifier=verifier,
                )
                _assert_e5_verifier_configuration(
                    verifier,
                    defense,
                    thresholds[defense.value],
                    role="bound victim",
                )
                _assert_e5_verifier_configuration(
                    audit_verifier,
                    defense,
                    thresholds[defense.value],
                    role="independent replay",
                )
                attack = _factory(
                    attack_factory,
                    seed=int(seed),
                    defense=defense,
                    verifier=attack_verifier,
                    case=attack_case,
                )
                _assert_e5_verifier_configuration(
                    attack_verifier,
                    defense,
                    thresholds[defense.value],
                    role="attack-context",
                )
                if any(victim is prior for prior in victim_instances):
                    raise ValueError("E5 victim factory must return a fresh configured instance")
                if any(attack is prior for prior in attack_instances):
                    raise ValueError("E5 attack factory must return a fresh reoptimized instance")
                victim_instances.append(victim)
                attack_instances.append(attack)
                started = perf_counter()
                attack_constraint = _constraint(
                    constraint,
                    case=eligible.case,
                    seed=int(seed),
                )
                _require_e5_sphys(attack_constraint)
                counted_victim = _CountingVictim(
                    victim,
                    query_budget,
                    clean_waveform=eligible.case.waveform_mv,
                    constraint=attack_constraint,
                    verifier_observer=verifier,
                    audit_verifier=audit_verifier,
                )
                preservation_reviewer = (
                    None
                    if admissibility_checker is None
                    else _QueryBoundPreservationReviewer(
                        admissibility_checker,
                        clean_waveform=eligible.case.waveform_mv,
                        case=eligible.case,
                        victim=victim,
                        constraint=attack_constraint,
                        victim_recorder=counted_victim,
                    )
                )
                attack_checker = (
                    None if preservation_reviewer is None else preservation_reviewer.attack_proxy
                )
                (
                    pre_run_exact_type,
                    pre_run_class_method_unchanged,
                    pre_run_bound_method_exact,
                ) = _e5_builtin_identity_check(
                    attack,
                    audited_f3_class,
                    audited_f3_run_method,
                )
                result = attack.run(
                    counted_victim,
                    attack_clean,
                    eligible.case.target.target_label,
                    query_budget=query_budget,
                    constraint=attack_constraint,
                    admissibility_checker=attack_checker,
                    case=attack_case,
                    defense=defense,
                    clean_snapshot=copy.deepcopy(clean_snapshot),
                )
                (
                    post_run_exact_type,
                    post_run_class_method_unchanged,
                    post_run_bound_method_exact,
                ) = _e5_builtin_identity_check(
                    attack,
                    audited_f3_class,
                    audited_f3_run_method,
                )
                implementation_attestation = _E5BuiltinImplementationAttestation(
                    expected_class=audited_f3_class_name,
                    expected_method=audited_f3_method_name,
                    captured_before_attack_factory=True,
                    pre_run_exact_type=pre_run_exact_type,
                    pre_run_class_method_unchanged=(pre_run_class_method_unchanged),
                    pre_run_bound_method_exact=pre_run_bound_method_exact,
                    post_run_exact_type=post_run_exact_type,
                    post_run_class_method_unchanged=(post_run_class_method_unchanged),
                    post_run_bound_method_exact=post_run_bound_method_exact,
                )
                _assert_e5_verifier_configuration(
                    verifier,
                    defense,
                    thresholds[defense.value],
                    role="bound victim",
                )
                _assert_e5_verifier_configuration(
                    audit_verifier,
                    defense,
                    thresholds[defense.value],
                    role="independent replay",
                )
                _assert_e5_verifier_configuration(
                    attack_verifier,
                    defense,
                    thresholds[defense.value],
                    role="attack-context",
                )
                elapsed_seconds = perf_counter() - started
                if not isinstance(result, AttackResult):
                    raise TypeError("Attack runner must return AttackResult")
                if result.family not in {AttackFamily.F3, AttackFamily.F3_DEFENSE}:
                    raise ValueError("E5 primary comparison requires an F3 re-attack")
                if result.target_label != eligible.case.target.target_label:
                    raise ValueError("E5 attack result target does not match the cohort target")
                attacked = _validate_attack_boundary(
                    result,
                    case=eligible.case,
                    clean_snapshot=clean_snapshot,
                    constraint=attack_constraint,
                    query_budget=query_budget,
                    counted_queries=counted_victim.queries_used,
                    query_waveforms=counted_victim.query_waveforms,
                    query_snapshots=counted_victim.query_snapshots,
                    experiment_id="E5",
                )
                causal_probe = None
                if attack_verifier.decisions and not _trusted_builtin_e5_conditioning(
                    result,
                    defense,
                    implementation_attestation,
                ):
                    causal_probe = _probe_e5_attack_verifier_causality(
                        victim_provider=victim_provider,
                        attack_factory=attack_factory,
                        verifier_template=verifier_template,
                        case=eligible.case,
                        clean_snapshot=clean_snapshot,
                        defense=defense,
                        threshold=thresholds[defense.value],
                        seed=int(seed),
                        query_budget=query_budget,
                        constraint_provider=constraint,
                        admissibility_checker=None,
                        reference_waveforms=counted_victim.query_waveforms,
                        reference_decisions=attack_verifier.decisions,
                    )
                reoptimization = _reoptimization_evidence(
                    reoptimization_validator,
                    case=eligible.case,
                    defense=defense,
                    verifier=verifier,
                    attack_verifier=attack_verifier,
                    victim=victim,
                    attack=attack,
                    result=result,
                    implementation_attestation=implementation_attestation,
                    counted_victim=counted_victim,
                    causal_probe=causal_probe,
                    seed=int(seed),
                    query_budget=query_budget,
                )
                _assert_e5_verifier_configuration(
                    verifier,
                    defense,
                    thresholds[defense.value],
                    role="bound victim after provenance validation",
                )
                _assert_e5_verifier_configuration(
                    audit_verifier,
                    defense,
                    thresholds[defense.value],
                    role="independent replay after provenance validation",
                )
                _assert_e5_verifier_configuration(
                    attack_verifier,
                    defense,
                    thresholds[defense.value],
                    role="attack-context after provenance validation",
                )
                attacked = _validate_attack_boundary(
                    result,
                    case=eligible.case,
                    clean_snapshot=clean_snapshot,
                    constraint=attack_constraint,
                    query_budget=query_budget,
                    counted_queries=counted_victim.queries_used,
                    query_waveforms=counted_victim.query_waveforms,
                    query_snapshots=counted_victim.query_snapshots,
                    experiment_id="E5 post-provenance validation",
                )
                result.adversarial_snapshot = copy.deepcopy(attacked)
                report_assessment = _independent_report_correctness(
                    attacked,
                    eligible.case,
                    report_correctness_evaluator,
                )
                report_correct = (
                    None if report_assessment is None else bool(report_assessment.correct)
                )
                (
                    preservation_evaluable,
                    independently_successful,
                    primary_success_query,
                    preservation_evidence,
                ) = _independent_preservation_assessment(
                    admissibility_checker,
                    clean_waveform=eligible.case.waveform_mv,
                    query_waveforms=counted_victim.query_waveforms,
                    query_snapshots=counted_victim.query_snapshots,
                    case=eligible.case,
                    victim=victim,
                    constraint=attack_constraint,
                    review_entries=(
                        None if preservation_reviewer is None else preservation_reviewer.complete()
                    ),
                )
                primary_endpoint_eligible = bool(
                    calibration_cohort_primary
                    and eligible.primary_endpoint_eligible
                    and preservation_evaluable
                    and reoptimization["primary_eligible"]
                )
                primary_success = bool(independently_successful and primary_endpoint_eligible)
                receipt = _issue_preservation_receipt(
                    result,
                    experiment_id="E5",
                    case=eligible.case,
                    clean_snapshot=clean_snapshot,
                    query_waveforms=counted_victim.query_waveforms,
                    query_snapshots=counted_victim.query_snapshots,
                    primary_endpoint_eligible=primary_endpoint_eligible,
                    primary_success=primary_success,
                    primary_success_query_index=primary_success_query,
                    review_payload=preservation_evidence,
                )
                audit_receipts = (
                    []
                    if logger is None
                    else logger.append_attack_result(
                        result,
                        patient_id=eligible.case.patient_id,
                        record_id=eligible.case.record_id,
                        target_id=eligible.case.target.target_id,
                        experiment_id="E5",
                        context={
                            "defense": defense.value,
                            "seed": int(seed),
                            "query_budget": query_budget,
                            "constraint": ConstraintKind.S_PHYS.value,
                            "verifier_threshold": verifier.threshold,
                            "counted_complete_system_queries": (counted_victim.queries_used),
                            "preservation_verification_receipt": (
                                None if receipt is None else receipt.to_dict()
                            ),
                            "independent_preservation_assessment": (preservation_evidence),
                        },
                    )
                )
                observations.append(
                    ExperimentObservation(
                        "E5",
                        f"reattack_{defense.value}",
                        eligible.case.patient_id,
                        eligible.case.record_id,
                        eligible.case.target.target_id,
                        primary_success,
                        bool(attacked and attacked.released),
                        seed=int(seed),
                        budget=query_budget,
                        defense=defense.value,
                        queries_used=result.queries_used,
                        metrics={
                            "incorrect_release": primary_success,
                            "exploratory_incorrect_release": bool(
                                result.metadata.get("exploratory_success", result.success)
                            ),
                            "primary_endpoint_eligible": primary_endpoint_eligible,
                            "selected_primary_preservation_evidence": (independently_successful),
                            "primary_success_query_index": primary_success_query,
                            "report_correct": report_correct,
                            "report_correctness_primary_eligible": (
                                report_assessment is not None
                                and report_assessment.primary_endpoint_eligible
                            ),
                            "elapsed_seconds": elapsed_seconds,
                            "seconds_per_query": elapsed_seconds / max(result.queries_used, 1),
                        },
                        metadata={
                            "threshold": verifier.threshold,
                            "target_clean_coverage": target_coverage,
                            "calibration_cohort_keyed_and_matched": (calibration_cohort_primary),
                            "selected_preservation_evidence": preservation_evidence,
                            "independent_preservation_assessment": (preservation_evidence),
                            "preservation_verification_receipt": (
                                None if receipt is None else receipt.to_dict()
                            ),
                            "counted_complete_system_queries": (counted_victim.queries_used),
                            "behaviorally_bound_verifier_queries": (
                                counted_victim.behaviorally_bound_queries
                            ),
                            "independently_replayed_verifier_queries": (
                                counted_victim.independently_audited_verifier_queries
                            ),
                            "reoptimized": bool(reoptimization["valid"]),
                            "reoptimization_endpoint_classification": (
                                reoptimization["endpoint_classification"]
                            ),
                            "reoptimization_evidence": reoptimization,
                            "report_correctness_source": (
                                None if report_assessment is None else report_assessment.evaluator
                            ),
                            "report_correctness_assessment": (
                                None if report_assessment is None else report_assessment.to_dict()
                            ),
                            "audit_event_ids": [item.event_id for item in audit_receipts],
                        },
                        attack_result=result,
                    )
                )
    attacked_selective: dict[str, dict[str, Any]] = {}
    for defense in sorted({row.defense for row in observations if row.defense is not None}):
        rows = [row for row in observations if row.defense == defense]
        correctness_available = bool(rows) and all(
            row.metrics["report_correct"] is not None
            and row.metrics["report_correctness_primary_eligible"]
            for row in rows
        )
        risk = (
            selective_metrics(
                [bool(row.metrics["report_correct"]) for row in rows],
                [row.released for row in rows],
            ).to_dict()
            if correctness_available
            else {
                "coverage": None,
                "selective_risk": None,
                "false_rejection_rate": None,
                "released": None,
                "total": len(rows),
            }
        )
        attacked_selective[defense] = {
            **risk,
            "correctness_available": correctness_available,
            "correctness_contract": ("independent_all_claims" if correctness_available else None),
            "endpoint_classification": (
                "primary"
                if calibration_cohort_primary
                and correctness_available
                and all(bool(row.metrics["primary_endpoint_eligible"]) for row in rows)
                else "not_evaluable"
                if not correctness_available
                else "exploratory"
            ),
            "mean_elapsed_seconds": float(
                np.mean([row.metrics["elapsed_seconds"] for row in rows])
            ),
            "mean_seconds_per_query": float(
                np.mean([row.metrics["seconds_per_query"] for row in rows])
            ),
        }
    summary = {
        "thresholds": thresholds,
        "target_clean_coverage": target_coverage,
        "endpoint_classification": (
            "not_evaluable"
            if not observations
            else "primary"
            if all(bool(row.metrics["primary_endpoint_eligible"]) for row in observations)
            else "mixed"
            if any(bool(row.metrics["primary_endpoint_eligible"]) for row in observations)
            else "not_evaluable"
        ),
        "primary_constraint": ConstraintKind.S_PHYS.value,
        "primary_defenses": sorted(_E5_PRIMARY_DEFENSES),
        "calibration_cohort_keyed_and_matched": calibration_cohort_primary,
        "legacy_calibration_keys_aligned": legacy_keyed_common_cohort,
        "calibration_source": calibration_source,
        "calibration_verifier_behavior_recomputed": typed_calibration,
        "calibration_behavioral_provenance": calibration_provenance,
        "calibration_coverage_validation": calibration_coverage_validation,
        "calibration_endpoint_classification": (
            "primary" if calibration_cohort_primary else "exploratory"
        ),
        "calibration_case_keys": (
            None
            if normalized_case_keys is None
            else list(normalized_case_keys[DefenseKind.V1.value])
        ),
        "audit_logging_configured": logger is not None,
        "clean_operating_points": clean_operating_points,
        "attacked_selective_metrics": attacked_selective,
    }
    summary["attack_metrics"] = _attack_summary(observations)
    return ExperimentRun("E5", tuple(observations), summary)


def run_e6_transfer_replay(
    pairs: Sequence[tuple[EligibleCase, AttackResult]],
    transfer_victim_provider: Any,
    *,
    defense: DefenseKind | str = DefenseKind.V1,
    successful_only: bool = True,
    report_correctness_evaluator: Callable[..., Any] | None = None,
    clean_correctness: Callable[..., bool] | None = None,
) -> ExperimentRun:
    """E6: replay attacks on independently reviewed clean-correct transfer cases.

    ``clean_correctness`` is retained as an exploratory compatibility callback. Primary
    E6 estimates require ``report_correctness_evaluator`` to return an independent,
    all-claims :class:`ReportCorrectnessResult` for the exact patient/record/target.
    """

    if report_correctness_evaluator is not None and clean_correctness is not None:
        raise ValueError(
            "Configure report_correctness_evaluator or legacy clean_correctness, not both"
        )

    defense = DefenseKind(defense)
    observations: list[ExperimentObservation] = []
    clean_evaluated = 0
    clean_analysis_eligible = 0
    clean_primary_eligible = 0
    excluded_keys: list[str] = []
    primary_ineligible_keys: list[str] = []
    for eligible, result in pairs:
        source_primary_eligible, source_evidence = _source_primary_status(eligible, result)
        if successful_only and not source_evidence["source_attack_success"]:
            continue
        if result.adversarial_signal is None:
            continue
        victim = _victim(transfer_victim_provider, eligible.case, defense=defense)
        clean_snapshot = _evaluate(victim, eligible.case, defense)
        clean_evaluated += 1
        evaluator = (
            report_correctness_evaluator
            if report_correctness_evaluator is not None
            else clean_correctness
        )
        (
            report_correct,
            report_primary,
            report_assessment,
            correctness_metadata,
        ) = _e0_report_correctness(
            clean_snapshot,
            eligible.case,
            evaluator,
            experiment_id="E6",
        )
        if clean_correctness is not None:
            report_primary = False
            correctness_metadata = {
                **correctness_metadata,
                "endpoint_classification": "exploratory",
                "legacy_parameter": "clean_correctness",
                "primary_ineligibility_reason": (
                    "legacy clean_correctness callbacks are exploratory; configure "
                    "report_correctness_evaluator with typed independent all-claims evidence"
                ),
            }
        transfer_analysis_eligible = bool(clean_snapshot.released and report_correct)
        transfer_primary_eligible = bool(transfer_analysis_eligible and report_primary)
        if not transfer_analysis_eligible:
            excluded_keys.append(eligible.case.key)
            continue
        clean_analysis_eligible += 1
        if transfer_primary_eligible:
            clean_primary_eligible += 1
        else:
            primary_ineligible_keys.append(eligible.case.key)
        snapshot = victim.evaluate(
            result.adversarial_signal,
            eligible.case.target.target_label,
            defense=defense,
        )
        if not isinstance(snapshot, VictimSnapshot):
            raise TypeError("Transfer victim must return VictimSnapshot")
        primary_endpoint_eligible = bool(source_primary_eligible and transfer_primary_eligible)
        exploratory_success = bool(snapshot.released and snapshot.target_claim_present)
        success = bool(exploratory_success and primary_endpoint_eligible)
        observations.append(
            _observation(
                "E6",
                "transfer_replay",
                eligible.case,
                success,
                snapshot.released,
                {
                    "diagnosis": snapshot.diagnosis,
                    "incorrect_release": success,
                    "exploratory_incorrect_release": exploratory_success,
                    "primary_endpoint_eligible": primary_endpoint_eligible,
                    "source_attack_success": source_evidence["source_attack_success"],
                    "source_primary_endpoint_eligible": source_primary_eligible,
                    "source_evidence": source_evidence,
                    "transfer_clean_diagnosis": clean_snapshot.diagnosis,
                    "transfer_clean_released": clean_snapshot.released,
                    "transfer_clean_report_correct": report_correct,
                    "transfer_clean_primary_endpoint_eligible": transfer_primary_eligible,
                    "transfer_clean_report_correctness": (
                        None if report_assessment is None else report_assessment.to_dict()
                    ),
                    "report_correctness_evidence": correctness_metadata,
                },
            )
        )
    summary = _endpoint_aware_arm_rates(
        observations,
        exploratory_metric="exploratory_incorrect_release",
    )
    summary["endpoint_classification"] = summary.get("transfer_replay", {}).get(
        "endpoint_classification",
        "not_evaluable",
    )
    arm_by_target = summary.get("transfer_replay", {}).get("by_target", {})
    by_target = {
        target_id: {
            "cases": target_summary["cases"],
            "transfer_irr": target_summary["rate"],
            "transfer_irr_ci": target_summary["rate_ci"],
            "endpoint_classification": target_summary["endpoint_classification"],
            "exploratory_transfer_irr": target_summary["exploratory_rate"],
            "exploratory_transfer_irr_ci": target_summary["exploratory_rate_ci"],
        }
        for target_id, target_summary in arm_by_target.items()
    }
    summary["by_target"] = by_target
    summary["clean_transfer_eligibility"] = {
        "evaluated": clean_evaluated,
        "eligible": clean_primary_eligible,
        "analysis_eligible": clean_analysis_eligible,
        "exploratory_eligible": clean_analysis_eligible - clean_primary_eligible,
        "excluded": clean_evaluated - clean_analysis_eligible,
        "excluded_case_keys": excluded_keys,
        "primary_ineligible_case_keys": primary_ineligible_keys,
        "endpoint_classification": (
            "primary"
            if clean_primary_eligible == clean_analysis_eligible and clean_analysis_eligible > 0
            else "mixed"
            if clean_primary_eligible > 0
            else "not_evaluable"
        ),
        "report_correctness_evaluator_configured": (report_correctness_evaluator is not None),
        "legacy_clean_correctness_configured": clean_correctness is not None,
        "criterion": (
            "released and independently reviewed all-claims-correct clean report on "
            "the transfer victim"
        ),
    }
    return ExperimentRun("E6", tuple(observations), summary)


def run_e7_authenticated_history_stress(
    pairs: Sequence[tuple[EligibleCase, AttackResult]],
    longitudinal_verifier: Any,
    history_context: Callable[..., Mapping[str, Any]],
    *,
    tamperer: Callable[[AuthenticatedHistoryRecord], AuthenticatedHistoryRecord] | None = None,
    pre_defense_replay: Callable[..., VictimSnapshot] | None = None,
) -> ExperimentRun:
    """E7: replay one common pre-defense output through V2 and V3 stress arms."""

    if tamperer is None:
        tamperer = lambda record: replace(record, record_id=f"{record.record_id}-tampered")
    observations: list[ExperimentObservation] = []
    for eligible, result in pairs:
        source_snapshot = result.adversarial_snapshot
        if source_snapshot is None or result.adversarial_signal is None:
            continue
        source_primary, source_evidence = _source_primary_status(eligible, result)
        if pre_defense_replay is None:
            snapshot = source_snapshot
            common_replay_primary = False
            replay_source = "legacy_source_attack_snapshot"
        else:
            snapshot = _call_with_supported(
                pre_defense_replay,
                case=eligible.case,
                waveform_mv=result.adversarial_signal,
                target_label=eligible.case.target.target_label,
                source_snapshot=source_snapshot,
                pre_defense=True,
            )
            if not isinstance(snapshot, VictimSnapshot):
                raise TypeError("E7 pre_defense_replay must return VictimSnapshot")
            common_replay_primary = True
            replay_source = "explicit_common_pre_defense_replay"
        context = dict(_call_with_supported(history_context, case=eligible.case))
        history = context.get("history")
        if not isinstance(history, AuthenticatedHistoryRecord):
            raise TypeError("history_context must provide an AuthenticatedHistoryRecord")
        raw_current_timestamp = context.get(
            "current_timestamp", eligible.case.metadata.get("timestamp")
        )
        if raw_current_timestamp is None:
            raise ValueError("E7 requires a current_timestamp for V3 chronology")
        current_timestamp = float(raw_current_timestamp)
        if not np.isfinite(current_timestamp):
            raise ValueError("E7 current_timestamp must be finite")
        if history.timestamp >= current_timestamp:
            raise ValueError("E7 authenticated history must predate the current ECG")
        measurement_verifier = getattr(longitudinal_verifier, "measurement_verifier", None)
        if measurement_verifier is None or not hasattr(measurement_verifier, "decide"):
            raise TypeError("E7 requires a longitudinal verifier exposing its V2 baseline")
        decision_context = {
            key: value
            for key, value in context.items()
            if key not in {"history", "patient_id", "waveform_mv", "current_timestamp"}
        }
        no_history = measurement_verifier.decide(
            snapshot,
            waveform_mv=result.adversarial_signal,
            **decision_context,
        )
        decisions = {"no_history_v2": no_history}
        for arm, record in {
            "trusted_history": history,
            "tampered_history": tamperer(history),
        }.items():
            decisions[arm] = longitudinal_verifier.decide(
                snapshot,
                patient_id=eligible.case.patient_id,
                history=record,
                waveform_mv=result.adversarial_signal,
                current_timestamp=current_timestamp,
                **decision_context,
            )
        for arm, decision in decisions.items():
            primary_endpoint_eligible = bool(source_primary and common_replay_primary)
            exploratory_success = bool(decision.released and snapshot.target_claim_present)
            success = bool(exploratory_success and primary_endpoint_eligible)
            observations.append(
                _observation(
                    "E7",
                    arm,
                    eligible.case,
                    success,
                    decision.released,
                    {
                        "decision": decision.to_dict(),
                        "stronger_threat_model": arm == "tampered_history",
                        "uses_authenticated_history": arm == "trusted_history",
                        "primary_endpoint_eligible": primary_endpoint_eligible,
                        "exploratory_incorrect_release": exploratory_success,
                        "common_pre_defense_replay": common_replay_primary,
                        "pre_defense_replay_source": replay_source,
                        "current_timestamp": current_timestamp,
                        "history_timestamp": history.timestamp,
                        **source_evidence,
                    },
                )
            )
    summary = _endpoint_aware_arm_rates(
        observations,
        exploratory_metric="exploratory_incorrect_release",
    )
    if "trusted_history" in summary and "no_history_v2" in summary:
        trusted = summary["trusted_history"]
        baseline = summary["no_history_v2"]
        if trusted["rate"] is not None and baseline["rate"] is not None:
            summary["trusted_history_gain"] = {
                "irr_reduction_vs_v2": baseline["rate"] - trusted["rate"],
                "release_rate_change_vs_v2": (trusted["release_rate"] - baseline["release_rate"]),
            }
    if "trusted_history" in summary and "tampered_history" in summary:
        trusted = summary["trusted_history"]
        tampered = summary["tampered_history"]
        if trusted["rate"] is not None and tampered["rate"] is not None:
            summary["authenticated_history_effect"] = {
                "irr_reduction_trusted_vs_tampered": tampered["rate"] - trusted["rate"],
                "release_rate_change_trusted_vs_tampered": (
                    trusted["release_rate"] - tampered["release_rate"]
                ),
            }
    return ExperimentRun("E7", tuple(observations), summary)


def _observation(
    experiment_id: str,
    arm: str,
    case: CohortCase,
    success: bool,
    released: bool,
    metrics: Mapping[str, Any],
) -> ExperimentObservation:
    return ExperimentObservation(
        experiment_id,
        arm,
        case.patient_id,
        case.record_id,
        case.target.target_id,
        bool(success),
        bool(released),
        metrics=dict(metrics),
    )


def _set_verifier_threshold(verifier: Any, threshold: float) -> None:
    if not hasattr(verifier, "threshold"):
        raise TypeError("E5 verifier must expose a threshold")
    verifier.threshold = float(threshold)
    shipped = getattr(verifier, "shipped", None)
    if shipped is not None and hasattr(shipped, "threshold"):
        shipped.threshold = float(threshold)
    measurement_verifier = getattr(verifier, "measurement_verifier", None)
    if measurement_verifier is not None:
        _set_verifier_threshold(measurement_verifier, threshold)


def _simple_arm_rates(observations: Sequence[ExperimentObservation]) -> dict[str, Any]:
    groups: dict[str, list[ExperimentObservation]] = defaultdict(list)
    for observation in observations:
        groups[observation.arm].append(observation)

    def summarize(rows: Sequence[ExperimentObservation]) -> dict[str, Any]:
        patients = [row.patient_id for row in rows]
        rate = patient_bootstrap(
            [row.success for row in rows],
            patients,
            n_resamples=500,
            seed=42,
            metadata={"metric": "rate"},
        )
        release_rate = patient_bootstrap(
            [row.released for row in rows],
            patients,
            n_resamples=500,
            seed=42,
            metadata={"metric": "release_rate"},
        )
        return {
            "cases": len(rows),
            "patients": rate.n_patients,
            "rate": rate.estimate,
            "rate_ci": rate.to_dict(),
            "release_rate": release_rate.estimate,
            "release_rate_ci": release_rate.to_dict(),
            "statistical_unit": "patient; repeated rows collapsed within patient",
        }

    summary: dict[str, Any] = {}
    for arm, rows in groups.items():
        arm_summary = summarize(rows)
        arm_summary["by_target"] = {
            target_id: summarize([row for row in rows if row.target_id == target_id])
            for target_id in sorted({row.target_id for row in rows})
        }
        summary[arm] = arm_summary
    return summary


def _endpoint_aware_arm_rates(
    observations: Sequence[ExperimentObservation],
    *,
    exploratory_metric: str,
) -> dict[str, Any]:
    """Summarize primary rows without promoting exploratory evidence to an endpoint."""

    groups: dict[str, list[ExperimentObservation]] = defaultdict(list)
    for observation in observations:
        groups[observation.arm].append(observation)

    def unavailable(metric: str) -> dict[str, Any]:
        return {
            "estimate": None,
            "lower": None,
            "upper": None,
            "confidence": 0.95,
            "n_patients": 0,
            "n_resamples": 0,
            "seed": 42,
            "metadata": {
                "metric": metric,
                "status": "not_evaluable",
                "reason": "no primary-endpoint-eligible observations",
            },
        }

    def summarize(rows: Sequence[ExperimentObservation]) -> dict[str, Any]:
        primary = [row for row in rows if bool(row.metrics.get("primary_endpoint_eligible", False))]
        if primary:
            rate_ci = patient_bootstrap(
                [row.success for row in primary],
                [row.patient_id for row in primary],
                n_resamples=500,
                seed=42,
                metadata={"metric": "rate"},
            ).to_dict()
            release_ci = patient_bootstrap(
                [row.released for row in primary],
                [row.patient_id for row in primary],
                n_resamples=500,
                seed=42,
                metadata={"metric": "release_rate"},
            ).to_dict()
            endpoint_classification = "primary" if len(primary) == len(rows) else "mixed"
        else:
            rate_ci = unavailable("rate")
            release_ci = unavailable("release_rate")
            endpoint_classification = "not_evaluable"
        exploratory_ci = patient_bootstrap(
            [bool(row.metrics.get(exploratory_metric, False)) for row in rows],
            [row.patient_id for row in rows],
            n_resamples=500,
            seed=42,
            metadata={"metric": exploratory_metric},
        ).to_dict()
        return {
            "cases": len(rows),
            "patients": rate_ci["n_patients"],
            "rate": rate_ci["estimate"],
            "rate_ci": rate_ci,
            "release_rate": release_ci["estimate"],
            "release_rate_ci": release_ci,
            "endpoint_classification": endpoint_classification,
            "primary_endpoint_evaluable_observations": len(primary),
            "exploratory_rate": exploratory_ci["estimate"],
            "exploratory_rate_ci": exploratory_ci,
            "statistical_unit": "patient; repeated rows collapsed within patient",
        }

    summary: dict[str, Any] = {}
    for arm, rows in groups.items():
        arm_summary = summarize(rows)
        arm_summary["by_target"] = {
            target_id: summarize([row for row in rows if row.target_id == target_id])
            for target_id in sorted({row.target_id for row in rows})
        }
        summary[arm] = arm_summary
    return summary


run_e0 = run_e0_clean_baseline
run_e1 = run_e1_attack_matrix
run_e1b = run_e1b_staged_ablations
run_e2 = run_e2_retrieval_mediation
run_e3 = run_e3_latent_restoration
run_e4 = run_e4_counterfactual_alignment
run_e5 = run_e5_matched_coverage_reattack
run_e6 = run_e6_transfer_replay
run_e7 = run_e7_authenticated_history_stress
