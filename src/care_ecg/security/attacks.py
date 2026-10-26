from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .constraints import scale_to_constraint_distance
from .contracts import (
    AdmissibilityResult,
    AttackAttempt,
    AttackFamily,
    AttackResult,
    ClinicalCaseContext,
    ClinicalPreservationEvidence,
    ClinicalTarget,
    ConstraintKind,
    DefenseKind,
    KnowledgeLevel,
    MeasurementStatus,
    VictimSnapshot,
)
from .objectives import adaptive_objective, latent_bin_margin_loss
from .victim import BudgetedVictim


def _waveform(value: Tensor | np.ndarray) -> np.ndarray:
    result = np.asarray(
        value.detach().cpu().numpy() if isinstance(value, Tensor) else value,
        dtype=np.float32,
    )
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("Attack signals must be finite [leads, samples] arrays")
    return np.ascontiguousarray(result)


def _target_name(victim: Any, target_label: str | int) -> str:
    index = victim.target_index(target_label)
    return str(victim.label_names[index])


def _target_probability(snapshot: VictimSnapshot, target_name: str) -> float:
    if target_name not in snapshot.probabilities:
        raise ValueError("Victim snapshot omits the requested target probability")
    value = float(snapshot.probabilities[target_name])
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Victim target probability is invalid")
    return value


def _snapshot_loss(
    snapshot: VictimSnapshot,
    target_name: str,
    *,
    verifier_aware: bool,
    verifier_weight: float = 1.0,
    decision_only: bool = False,
) -> float:
    if decision_only:
        visible_claim = bool(snapshot.released and snapshot.target_claim_present)
        return float(not visible_claim) + (
            verifier_weight * float(not snapshot.released) if verifier_aware else 0.0
        )
    probability = max(_target_probability(snapshot, target_name), 1e-12)
    loss = -math.log(probability)
    if verifier_aware:
        loss += verifier_weight * max(float(snapshot.defense_score), 0.0)
        loss += float(not snapshot.released)
        loss += float(not snapshot.target_claim_present)
    return float(loss)


def _lp_project(clean: Tensor, candidate: Tensor, epsilon: float, norm: str) -> Tensor:
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and non-negative")
    delta = candidate - clean
    if norm == "linf":
        delta = delta.clamp(-epsilon, epsilon)
    elif norm == "l2":
        flat = delta.reshape(delta.shape[0], -1)
        length = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True)
        scale = torch.clamp(float(epsilon) / length.clamp_min(1e-12), max=1.0)
        delta = (flat * scale).reshape_as(delta)
    else:
        raise ValueError("norm must be 'linf' or 'l2'")
    return clean + delta


def _constraint_project(constraint: Any | None, clean: Tensor, candidate: Tensor) -> Tensor:
    if constraint is None:
        return candidate
    if clean.ndim == 3:
        if candidate.shape != clean.shape:
            raise ValueError("Clean and candidate attack batches must have identical shapes")
        return torch.stack(
            [_constraint_project(constraint, left, right) for left, right in zip(clean, candidate)]
        )
    if clean.ndim != 2 or candidate.ndim != 2:
        raise ValueError("Constraint projection expects [leads, samples] or batched waveforms")
    projected = _feasible_waveform(constraint, _waveform(clean), _waveform(candidate))
    return torch.as_tensor(projected, dtype=candidate.dtype, device=candidate.device)


def _uses_builtin_lp_projection(norm: str, constraint: Any | None) -> bool:
    if constraint is None:
        return True
    try:
        kind = ConstraintKind(getattr(constraint, "kind", None))
    except (TypeError, ValueError):
        return False
    if norm == "linf":
        return kind is ConstraintKind.S_INF
    return kind is ConstraintKind.S2 and getattr(constraint, "per_lead", None) is False


def _project_candidate(
    clean: Tensor,
    candidate: Tensor,
    epsilon: float,
    norm: str,
    constraint: Any | None,
) -> Tensor:
    if _uses_builtin_lp_projection(norm, constraint):
        candidate = _lp_project(clean, candidate, epsilon, norm)
    return _constraint_project(constraint, clean, candidate)


def _constraint_contains(constraint: Any | None, clean: np.ndarray, candidate: np.ndarray) -> bool:
    if constraint is None:
        return True
    try:
        return bool(constraint.contains(clean, candidate))
    except (TypeError, AttributeError):
        return bool(constraint.contains(torch.as_tensor(clean), torch.as_tensor(candidate)))


def _constraint_project_numpy(
    constraint: Any,
    clean: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    try:
        projected = constraint.project(clean, candidate)
    except (TypeError, AttributeError):
        projected = constraint.project(torch.as_tensor(clean), torch.as_tensor(candidate))
    return _waveform(projected)


def _feasible_waveform(
    constraint: Any | None,
    clean: np.ndarray,
    candidate: Tensor | np.ndarray,
) -> np.ndarray:
    """Project and validate the exact float32 waveform sent to the victim."""

    clean_waveform = _waveform(clean)
    waveform = _waveform(candidate)
    if waveform.shape != clean_waveform.shape:
        raise ValueError("Clean and candidate attack waveforms must have identical shapes")
    if constraint is None:
        return waveform

    # Projection and membership use the same NumPy backend and serialized
    # float32 representation. Reprojection absorbs small FFT/lstsq roundoff.
    for _ in range(8):
        waveform = _constraint_project_numpy(constraint, clean_waveform, waveform)
        if _constraint_contains(constraint, clean_waveform, waveform):
            return waveform

    # Extremely small spectral edits can be dominated by float32 quantization.
    # Contracting toward the clean point retains as much of the edit as the
    # serialized allowed set can represent.
    delta = np.asarray(waveform - clean_waveform, dtype=np.float64)
    for scale in (0.75, 0.5, 0.25, 0.125, 0.0):
        trial = _waveform(clean_waveform.astype(np.float64) + scale * delta)
        trial = _constraint_project_numpy(constraint, clean_waveform, trial)
        if _constraint_contains(constraint, clean_waveform, trial):
            return trial
    raise RuntimeError("Constraint projection could not produce a feasible float32 waveform")


def _admissibility(
    checker: Any | None,
    clean: np.ndarray,
    candidate: np.ndarray,
    sample_rate_hz: float | None,
    *,
    case: Any | None,
) -> tuple[bool, dict[str, Any]]:
    if checker is None:
        return False, {
            "available": False,
            "admissible": False,
            "clinical_preservation": False,
            "primary_endpoint_eligible": False,
            "policy": "missing_independent_clinical_evidence",
            "reason": "no clinical-preservation checker was configured",
        }
    context = None
    if case is not None:
        patient_id = getattr(case, "patient_id", None)
        record_id = getattr(case, "record_id", None)
        target = getattr(case, "target", None)
        if patient_id is None or record_id is None or not isinstance(target, ClinicalTarget):
            raise TypeError(
                "Attack case context must expose patient_id, record_id, and ClinicalTarget"
            )
        context = ClinicalCaseContext(str(patient_id), str(record_id), target)
    context_kwargs = {
        "case": case,
        "patient_id": None if context is None else context.patient_id,
        "record_id": None if context is None else context.record_id,
        "target": None if context is None else context.target,
        "context": context,
    }

    def invoke(function: Any, *args: Any) -> Any:
        signature = inspect.signature(function)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        kwargs = (
            context_kwargs
            if accepts_kwargs
            else {
                name: value
                for name, value in context_kwargs.items()
                if name in signature.parameters
            }
        )
        return function(*args, **kwargs)

    if hasattr(checker, "check"):
        result = invoke(checker.check, clean, candidate)
    elif hasattr(checker, "evaluate"):
        if sample_rate_hz is None:
            raise ValueError("Clinical admissibility evaluation requires a sample rate")
        result = invoke(checker.evaluate, clean, candidate, sample_rate_hz)
    elif callable(checker):
        result = invoke(checker, clean, candidate)
    else:
        raise TypeError("Admissibility checker must be callable or expose check/evaluate")
    if isinstance(result, AdmissibilityResult):
        available = result.status is MeasurementStatus.AVAILABLE
        admissible = bool(available and result.admissible)
        primary_eligible = bool(result.primary_endpoint_eligible)
        evidence = result.preservation_evidence
        if primary_eligible:
            if context is None:
                raise ValueError(
                    "Primary clinical-preservation evidence requires patient-record-target context"
                )
            if not isinstance(evidence, ClinicalPreservationEvidence):
                raise ValueError(
                    "Primary clinical-preservation evidence must use the structured contract"
                )
            if evidence.context != context:
                raise ValueError(
                    "Clinical-preservation evidence does not match the submitted case context"
                )
        metadata = result.to_dict()
        metadata.update(
            {
                "available": available,
                "admissible": admissible,
                "clinical_preservation": admissible,
                "primary_endpoint_eligible": primary_eligible,
                "endpoint_classification": ("primary" if primary_eligible else "exploratory"),
                "case_context": None if context is None else context.to_dict(),
            }
        )
        return admissible, metadata
    if isinstance(result, dict):
        if "admissible" not in result:
            raise ValueError("Admissibility mapping must contain admissible")
        metadata = dict(result)
        if metadata.get("primary_endpoint_eligible"):
            raise TypeError(
                "Primary clinical evidence must be returned as AdmissibilityResult with "
                "ClinicalPreservationEvidence"
            )
        if "available" in metadata:
            available = bool(metadata["available"])
        elif "status" in metadata:
            try:
                available = MeasurementStatus(metadata["status"]) is MeasurementStatus.AVAILABLE
            except ValueError:
                available = False
        else:
            available = True
        admissible = bool(available and metadata["admissible"])
        metadata.update(
            {
                "available": available,
                "admissible": admissible,
                "clinical_preservation": admissible,
                "primary_endpoint_eligible": False,
                "endpoint_classification": "exploratory",
                "case_context": None if context is None else context.to_dict(),
            }
        )
        return admissible, metadata
    if isinstance(result, (bool, np.bool_)):
        admissible = bool(result)
        return admissible, {
            "available": True,
            "admissible": admissible,
            "clinical_preservation": admissible,
            "primary_endpoint_eligible": False,
            "endpoint_classification": "exploratory",
            "case_context": None if context is None else context.to_dict(),
            "primary_ineligibility_reason": (
                "boolean checker output has no target-specific preserved-finding evidence"
            ),
        }
    raise TypeError("Unsupported admissibility result type")


def _distance_metadata(clean: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(candidate - clean, dtype=np.float64)
    digest = hashlib.sha256(np.ascontiguousarray(candidate, dtype=np.float32).tobytes()).hexdigest()
    return {
        "waveform_sha256": digest,
        "linf_mv": float(np.max(np.abs(delta))),
        "l2_mv": float(np.linalg.norm(delta)),
        "rms_mv": float(np.sqrt(np.mean(delta**2))),
        "per_lead_linf_mv": np.max(np.abs(delta), axis=1).tolist(),
    }


@dataclass(slots=True)
class _Submission:
    """The optimization-facing response, deliberately excluding audit-only state."""

    loss: float


class _AttackSession:
    def __init__(
        self,
        *,
        family: AttackFamily,
        victim: Any,
        clean_signal: np.ndarray,
        target_label: str | int,
        query_budget: int,
        constraint: Any | None,
        admissibility_checker: Any | None,
        case: Any | None,
        defense: str | DefenseKind,
        clean_snapshot: VictimSnapshot | None,
        verifier_aware: bool,
        verifier_weight: float,
        knowledge_level: str | KnowledgeLevel,
    ):
        self.family = family
        self.victim = victim
        self.clean = _waveform(clean_signal)
        self.target_name = _target_name(victim, target_label)
        self.constraint = constraint
        self.admissibility_checker = admissibility_checker
        self.case = case
        self.sample_rate_hz = getattr(victim, "sample_rate_hz", None)
        if self.sample_rate_hz is None and hasattr(victim, "pipeline"):
            self.sample_rate_hz = victim.pipeline.config.data.sample_rate
        self.defense = DefenseKind(defense)
        self.oracle = BudgetedVictim(victim, query_budget)
        self.clean_snapshot = clean_snapshot or victim.evaluate(
            self.clean, self.target_name, defense=self.defense
        )
        self.verifier_aware = verifier_aware
        self.verifier_weight = verifier_weight
        self.knowledge_level = (
            knowledge_level.value
            if isinstance(knowledge_level, KnowledgeLevel)
            else str(knowledge_level)
        )
        self.decision_only = self.knowledge_level in {"none", KnowledgeLevel.K1.value}
        self.attempts: list[AttackAttempt] = []
        self.snapshots: list[VictimSnapshot] = []
        self.losses: list[float] = []

    @property
    def remaining(self) -> int:
        return self.oracle.remaining

    @property
    def succeeded(self) -> bool:
        return bool(self.attempts and self.attempts[-1].success)

    def submit(
        self,
        candidate: Tensor | np.ndarray,
        *,
        objective: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> _Submission:
        proposed_waveform = _waveform(candidate)
        waveform = _feasible_waveform(self.constraint, self.clean, proposed_waveform)
        allowed = _constraint_contains(self.constraint, self.clean, waveform)
        if not allowed:
            raise RuntimeError("Refusing to query the victim outside the declared constraint")
        snapshot = self.oracle.submit(
            waveform,
            self.target_name,
            defense=self.defense,
            clean_snapshot=self.clean_snapshot,
        )
        clinical, admissibility_metadata = _admissibility(
            self.admissibility_checker,
            self.clean,
            waveform,
            self.sample_rate_hz,
            case=self.case,
        )
        primary_endpoint_eligible = bool(
            admissibility_metadata.get("primary_endpoint_eligible", False)
        )
        exploratory_admissible = bool(allowed and clinical)
        admissible = bool(exploratory_admissible and primary_endpoint_eligible)
        loss = (
            _snapshot_loss(
                snapshot,
                self.target_name,
                verifier_aware=self.verifier_aware,
                verifier_weight=self.verifier_weight,
                decision_only=self.decision_only,
            )
            if objective is None
            else float(objective)
        )
        if not np.isfinite(loss):
            raise ValueError("Logged attack objective must be finite")
        target_probability = _target_probability(snapshot, self.target_name)
        success = bool(snapshot.released and snapshot.target_claim_present and admissible)
        exploratory_success = bool(
            snapshot.released and snapshot.target_claim_present and exploratory_admissible
        )
        attempt_metadata = {
            **(metadata or {}),
            **_distance_metadata(self.clean, waveform),
            "attack_family": self.family.value,
            "knowledge_level": self.knowledge_level,
            "target_label": self.target_name,
            "patient_id": getattr(self.victim, "patient_id", None),
            "interface_compliant": True,
            "constraint_satisfied": allowed,
            "constraint_projection_applied": not np.array_equal(proposed_waveform, waveform),
            "constraint_kind": str(getattr(self.constraint, "kind", "unspecified")),
            "admissibility": admissibility_metadata,
            "clinical_preservation": clinical,
            "exploratory_admissible": exploratory_admissible,
            "exploratory_success": exploratory_success,
            "primary_endpoint_eligible": primary_endpoint_eligible,
            "defense": self.defense.value,
            "probabilities": dict(snapshot.probabilities),
            "latent": snapshot.latent.tolist(),
            "discrete_latent": snapshot.discrete_latent.tolist(),
            "retrieved_fact_ids": list(snapshot.retrieved_fact_ids),
            "report": snapshot.report,
            "verifier_score": snapshot.defense_score,
            "verifier_decision": snapshot.metadata.get("defense_decision"),
            "hallucination_score": snapshot.hallucination_score,
            "unsupported_claim_score": snapshot.unsupported_claim_score,
            "used_fallback": snapshot.used_fallback,
            "abstained": snapshot.abstained,
            "attacker_feedback_fields": (
                ["released_report_target_claim", "release_decision"]
                if self.decision_only
                else ["surrogate_objective", "complete_system_result"]
            ),
            "audit_only_fields": (
                ["probabilities", "latent", "verifier_score"] if self.decision_only else []
            ),
        }
        if self.constraint is not None:
            actual_distance = float(self.constraint.distance(self.clean, waveform))
            attempt_metadata["constraint_distance"] = actual_distance
            if "achieved_constraint_distance" in attempt_metadata:
                attempt_metadata["achieved_constraint_distance"] = actual_distance
        attempt = AttackAttempt(
            query_index=self.oracle.queries_used,
            waveform=waveform,
            objective=loss,
            target_probability=target_probability,
            diagnosis=snapshot.diagnosis,
            released=bool(snapshot.released),
            target_claim_present=bool(snapshot.target_claim_present),
            admissible=admissible,
            success=success,
            metadata=attempt_metadata,
        )
        self.attempts.append(attempt)
        self.snapshots.append(snapshot)
        self.losses.append(loss)
        return _Submission(loss)

    def result(self, stopped_reason: str, metadata: dict[str, Any] | None = None) -> AttackResult:
        if self.succeeded:
            selected = len(self.attempts) - 1
            stopped_reason = "incorrect_release"
        elif self.attempts:
            if self.decision_only:
                # K1 may use only the released report and release/reject decision.
                # ``min`` is stable, so tied outcomes retain the first submitted
                # candidate without consulting hidden diagnosis probabilities.
                selected = min(range(len(self.attempts)), key=self.losses.__getitem__)
            else:
                selected = min(
                    range(len(self.attempts)),
                    key=lambda index: (
                        self.losses[index],
                        -self.attempts[index].target_probability,
                    ),
                )
        else:
            selected = None
        if self.oracle.queries_used != len(self.attempts):
            raise RuntimeError(
                "Every consumed complete-system query must have an attack-attempt trace"
            )
        selected_attempt = None if selected is None else self.attempts[selected]
        selected_primary_eligible = bool(
            selected_attempt is not None
            and selected_attempt.metadata.get("primary_endpoint_eligible", False)
        )
        successful_attempt_primary_eligible = selected_primary_eligible if self.succeeded else None
        # Clean eligibility fixes the attack-rate denominator. An unsuccessful
        # search remains a primary zero; when an event occurs, the selected
        # first-success query alone must carry independent preservation evidence.
        primary_endpoint_eligible = bool(not self.succeeded or successful_attempt_primary_eligible)
        artifact_selection_policy = (
            "first_success"
            if self.succeeded
            else "observable_decision_loss_first_query_tiebreak"
            if self.decision_only and selected is not None
            else "surrogate_loss_then_target_probability"
            if selected is not None
            else "no_submitted_query"
        )
        return AttackResult(
            family=self.family,
            target_label=self.target_name,
            clean_snapshot=self.clean_snapshot,
            adversarial_snapshot=None if selected is None else self.snapshots[selected],
            adversarial_signal=None if selected is None else self.attempts[selected].waveform,
            attempts=self.attempts,
            success=self.succeeded,
            queries_used=self.oracle.queries_used,
            stopped_reason=stopped_reason,
            metadata={
                **(metadata or {}),
                "query_budget": self.oracle.budget,
                "first_success_stopping": True,
                "success_definition": (
                    "released target claim within waveform and independently verified clinical "
                    "preservation constraints"
                ),
                "clinical_preservation_required": True,
                "clinical_preservation_checker_configured": (
                    self.admissibility_checker is not None
                ),
                "primary_endpoint_eligible": primary_endpoint_eligible,
                "primary_endpoint_eligibility_basis": (
                    "selected_first_success_independent_evidence"
                    if self.succeeded
                    else "fixed_clean_eligible_cohort_no_primary_event"
                ),
                "successful_attempt_primary_eligible": (successful_attempt_primary_eligible),
                "selected_attempt_primary_endpoint_eligible": (
                    selected_primary_eligible if selected is not None else None
                ),
                "selected_attempt_query_index": (
                    None if selected_attempt is None else selected_attempt.query_index
                ),
                "artifact_selection_policy": artifact_selection_policy,
                "attacker_feedback_contract": (
                    "released_report_and_release_decision_only"
                    if self.decision_only
                    else "declared_surrogate_and_complete_system_feedback"
                ),
                "query_accounting_policy": ("every complete-system submission consumes one query"),
                "complete_system_queries_consumed": self.oracle.queries_used,
                "inadmissible_queries": sum(not attempt.admissible for attempt in self.attempts),
                "attempt_evidence_coverage": (
                    0.0
                    if not self.attempts
                    else float(
                        np.mean(
                            [
                                bool(attempt.metadata.get("primary_endpoint_eligible", False))
                                for attempt in self.attempts
                            ]
                        )
                    )
                ),
                "endpoint_classification": (
                    "primary" if primary_endpoint_eligible else "exploratory"
                ),
                "exploratory_success": any(
                    bool(attempt.metadata.get("exploratory_success", False))
                    for attempt in self.attempts
                ),
            },
        )


class F0RandomAttack:
    family = AttackFamily.F0

    def __init__(
        self,
        *,
        norm: str = "linf",
        epsilon_mv: float = 0.01,
        matched_magnitude: float | None = None,
        seed: int = 42,
    ):
        if norm not in {"linf", "l2"}:
            raise ValueError("norm must be 'linf' or 'l2'")
        if epsilon_mv < 0 or not np.isfinite(epsilon_mv):
            raise ValueError("epsilon_mv must be finite and non-negative")
        if matched_magnitude is not None and (
            matched_magnitude < 0 or not np.isfinite(matched_magnitude)
        ):
            raise ValueError("matched_magnitude must be finite and non-negative")
        self.norm = norm
        self.epsilon_mv = float(epsilon_mv)
        self.matched_magnitude = matched_magnitude
        self.seed = int(seed)

    def run(
        self,
        victim: Any,
        clean_signal: np.ndarray,
        target_label: str | int,
        *,
        query_budget: int,
        constraint: Any | None = None,
        admissibility_checker: Any | None = None,
        case: Any | None = None,
        defense: str | DefenseKind = DefenseKind.V1,
        clean_snapshot: VictimSnapshot | None = None,
    ) -> AttackResult:
        session = _AttackSession(
            family=self.family,
            victim=victim,
            clean_signal=clean_signal,
            target_label=target_label,
            query_budget=query_budget,
            constraint=constraint,
            admissibility_checker=admissibility_checker,
            case=case,
            defense=defense,
            clean_snapshot=clean_snapshot,
            verifier_aware=False,
            verifier_weight=0.0,
            knowledge_level="none",
        )
        clean = torch.as_tensor(session.clean)
        generator = torch.Generator(device=clean.device).manual_seed(self.seed)
        native_generator = np.random.default_rng(self.seed)
        while session.remaining and not session.succeeded:
            if constraint is not None:
                try:
                    native_candidate = constraint.sample(session.clean, native_generator)
                except (TypeError, AttributeError):
                    native_candidate = constraint.sample(clean, generator)
                if self.matched_magnitude is not None:
                    native_candidate = scale_to_constraint_distance(
                        constraint,
                        session.clean,
                        _waveform(native_candidate),
                        self.matched_magnitude,
                    )
                candidate = _constraint_project(
                    constraint,
                    clean.unsqueeze(0),
                    torch.as_tensor(native_candidate, dtype=clean.dtype).unsqueeze(0),
                )[0]
            else:
                magnitude = (
                    self.epsilon_mv if self.matched_magnitude is None else self.matched_magnitude
                )
                if self.norm == "linf":
                    delta = torch.empty_like(clean).uniform_(-1.0, 1.0, generator=generator)
                    delta = delta * float(magnitude)
                else:
                    delta = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
                    length = torch.linalg.vector_norm(delta).clamp_min(1e-12)
                    delta = delta * (float(magnitude) / length)
                candidate = _project_candidate(
                    clean.unsqueeze(0),
                    (clean + delta).unsqueeze(0),
                    self.epsilon_mv,
                    self.norm,
                    constraint,
                )[0]
            candidate_waveform = _waveform(candidate)
            achieved_distance = (
                float(constraint.distance(session.clean, candidate_waveform))
                if constraint is not None
                else float(
                    np.max(np.abs(candidate_waveform - session.clean))
                    if self.norm == "linf"
                    else np.linalg.norm(candidate_waveform - session.clean)
                )
            )
            session.submit(
                candidate,
                metadata={
                    "random_draw": len(session.attempts) + 1,
                    "native_constraint_sampling": constraint is not None,
                    "achieved_constraint_distance": achieved_distance,
                },
            )
        reason = "query_budget_exhausted" if not session.succeeded else "incorrect_release"
        return session.result(
            reason,
            {
                "norm": self.norm,
                "epsilon_mv": self.epsilon_mv,
                "matched_magnitude": self.matched_magnitude,
                "seed": self.seed,
                "builtin_lp_projection": _uses_builtin_lp_projection(self.norm, constraint),
            },
        )


class F1PGDAttack:
    family = AttackFamily.F1

    def __init__(
        self,
        *,
        norm: str = "linf",
        epsilon_mv: float = 0.01,
        step_size_mv: float = 0.001,
        steps: int = 40,
        restarts: int = 1,
        random_start: bool = True,
        confidence: float = 0.0,
        seed: int = 42,
    ):
        if norm not in {"linf", "l2"}:
            raise ValueError("norm must be 'linf' or 'l2'")
        if any(
            not np.isfinite(value) or value < 0 for value in (epsilon_mv, step_size_mv, confidence)
        ):
            raise ValueError("Attack magnitudes must be finite and non-negative")
        if steps < 1 or restarts < 1:
            raise ValueError("steps and restarts must be positive")
        self.norm = norm
        self.epsilon_mv = float(epsilon_mv)
        self.step_size_mv = float(step_size_mv)
        self.steps = int(steps)
        self.restarts = int(restarts)
        self.random_start = bool(random_start)
        self.confidence = float(confidence)
        self.seed = int(seed)

    def _objective(
        self,
        victim: Any,
        candidate: Tensor,
        target_name: str,
        defense: DefenseKind,
        guidance: tuple[int, int] | None,
        weights: dict[str, float],
    ):
        surrogate = victim.surrogate(candidate, target_name, defense=defense)
        latent_loss = None
        if guidance is not None:
            feature_index, target_state = guidance
            thresholds = torch.as_tensor(
                victim.discretizer_thresholds,
                dtype=surrogate.latent.dtype,
                device=surrogate.latent.device,
            )
            latent_loss = latent_bin_margin_loss(
                surrogate.latent,
                thresholds,
                feature_index,
                target_state,
                margin=weights.get("latent_margin", 1e-3),
            )
        terms = adaptive_objective(
            surrogate.probabilities,
            victim.target_index(target_name),
            diagnosis_probabilities=True,
            retrieval_loss=surrogate.retrieval_loss,
            verifier_loss=surrogate.verifier_loss,
            defense_loss=surrogate.defense_loss,
            latent_loss=latent_loss,
            diagnosis_weight=weights.get("diagnosis", 1.0),
            retrieval_weight=weights.get("retrieval", 0.0),
            verifier_weight=weights.get("verifier", 0.0),
            defense_weight=weights.get("defense", 0.0),
            latent_weight=weights.get("latent", 0.0),
            confidence=self.confidence,
        )
        return terms, surrogate

    def _run_gradient(
        self,
        session: _AttackSession,
        *,
        constraint: Any | None,
        guidance: tuple[int, int] | None,
        weights: dict[str, float],
        family_metadata: dict[str, Any],
    ) -> AttackResult:
        device = getattr(session.victim, "device", torch.device("cpu"))
        clean = torch.as_tensor(session.clean, dtype=torch.float32, device=device).unsqueeze(0)
        generator = torch.Generator(device=device).manual_seed(self.seed)
        iterations = 0
        for restart in range(self.restarts):
            if session.remaining <= 0 or session.succeeded:
                break
            if self.random_start and self.epsilon_mv:
                if self.norm == "linf":
                    delta = torch.empty_like(clean).uniform_(
                        -self.epsilon_mv, self.epsilon_mv, generator=generator
                    )
                else:
                    delta = torch.randn(
                        clean.shape,
                        generator=generator,
                        dtype=clean.dtype,
                        device=clean.device,
                    )
                    flat = delta.reshape(1, -1)
                    radius = torch.rand((), generator=generator, device=device)
                    delta = (
                        flat
                        / torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
                        * (radius * self.epsilon_mv)
                    ).reshape_as(clean)
                candidate = clean + delta
            else:
                candidate = clean.clone()
            candidate = _project_candidate(
                clean, candidate, self.epsilon_mv, self.norm, constraint
            ).detach()
            best_candidate = candidate.clone()
            best_terms, best_surrogate = self._objective(
                session.victim,
                best_candidate,
                session.target_name,
                session.defense,
                guidance,
                weights,
            )
            best_loss = float(best_terms.total.detach().cpu())
            best_step = 0
            restart_iterations = 0
            for step in range(self.steps):
                candidate.requires_grad_(True)
                terms, _ = self._objective(
                    session.victim,
                    candidate,
                    session.target_name,
                    session.defense,
                    guidance,
                    weights,
                )
                gradient = torch.autograd.grad(terms.total, candidate)[0]
                with torch.no_grad():
                    if self.norm == "linf":
                        candidate = candidate - self.step_size_mv * gradient.sign()
                    else:
                        flat = gradient.reshape(len(gradient), -1)
                        normalized = flat / torch.linalg.vector_norm(
                            flat, dim=1, keepdim=True
                        ).clamp_min(1e-12)
                        candidate = candidate - self.step_size_mv * normalized.reshape_as(gradient)
                    candidate = _project_candidate(
                        clean, candidate, self.epsilon_mv, self.norm, constraint
                    ).detach()
                logged_terms, surrogate = self._objective(
                    session.victim,
                    candidate,
                    session.target_name,
                    session.defense,
                    guidance,
                    weights,
                )
                iterations += 1
                restart_iterations += 1
                logged_loss = float(logged_terms.total.detach().cpu())
                if logged_loss < best_loss:
                    best_candidate = candidate.clone()
                    best_terms = logged_terms
                    best_surrogate = surrogate
                    best_loss = logged_loss
                    best_step = step + 1
            session.submit(
                best_candidate[0],
                objective=best_loss,
                metadata={
                    "restart": restart + 1,
                    "gradient_step": best_step,
                    "surrogate_steps_in_restart": restart_iterations,
                    "objective_terms": best_terms.detached(),
                    "surrogate": best_surrogate.metadata,
                    "surrogate_selection_policy": "minimum_objective_over_restart",
                },
            )
        reason = (
            "incorrect_release"
            if session.succeeded
            else "query_budget_exhausted"
            if session.remaining <= 0
            else "optimization_steps_exhausted"
        )
        return session.result(
            reason,
            {
                "norm": self.norm,
                "epsilon_mv": self.epsilon_mv,
                "step_size_mv": self.step_size_mv,
                "steps": self.steps,
                "restarts": self.restarts,
                "random_start": self.random_start,
                "confidence": self.confidence,
                "surrogate_iterations": iterations,
                "complete_system_query_schedule": "one_best_candidate_per_restart",
                "builtin_lp_projection": _uses_builtin_lp_projection(self.norm, constraint),
                **family_metadata,
            },
        )

    def run(
        self,
        victim: Any,
        clean_signal: np.ndarray,
        target_label: str | int,
        *,
        query_budget: int,
        constraint: Any | None = None,
        admissibility_checker: Any | None = None,
        case: Any | None = None,
        defense: str | DefenseKind = DefenseKind.V1,
        clean_snapshot: VictimSnapshot | None = None,
    ) -> AttackResult:
        session = _AttackSession(
            family=self.family,
            victim=victim,
            clean_signal=clean_signal,
            target_label=target_label,
            query_budget=query_budget,
            constraint=constraint,
            admissibility_checker=admissibility_checker,
            case=case,
            defense=defense,
            clean_snapshot=clean_snapshot,
            verifier_aware=False,
            verifier_weight=0.0,
            knowledge_level=KnowledgeLevel.K2,
        )
        return self._run_gradient(
            session,
            constraint=constraint,
            guidance=None,
            weights={"diagnosis": 1.0},
            family_metadata={"objective": "targeted_diagnosis_margin"},
        )


class F1CounterfactualAttack(F1PGDAttack):
    family = AttackFamily.F1_CF

    def __init__(
        self,
        *,
        latent_weight: float = 1.0,
        latent_margin: float = 1e-3,
        feature_index: int | None = None,
        target_state: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if latent_weight <= 0 or not np.isfinite(latent_weight):
            raise ValueError("latent_weight must be finite and positive")
        if latent_margin < 0 or not np.isfinite(latent_margin):
            raise ValueError("latent_margin must be finite and non-negative")
        if (feature_index is None) != (target_state is None):
            raise ValueError("feature_index and target_state must be supplied together")
        self.latent_weight = float(latent_weight)
        self.latent_margin = float(latent_margin)
        self.feature_index = feature_index
        self.target_state = target_state

    def run(
        self,
        victim: Any,
        clean_signal: np.ndarray,
        target_label: str | int,
        *,
        query_budget: int,
        constraint: Any | None = None,
        admissibility_checker: Any | None = None,
        case: Any | None = None,
        defense: str | DefenseKind = DefenseKind.V1,
        clean_snapshot: VictimSnapshot | None = None,
    ) -> AttackResult:
        session = _AttackSession(
            family=self.family,
            victim=victim,
            clean_signal=clean_signal,
            target_label=target_label,
            query_budget=query_budget,
            constraint=constraint,
            admissibility_checker=admissibility_checker,
            case=case,
            defense=defense,
            clean_snapshot=clean_snapshot,
            verifier_aware=False,
            verifier_weight=0.0,
            knowledge_level=KnowledgeLevel.K2,
        )
        guidance_source = "explicit"
        guidance_reason = "explicit_feature_state"
        if self.feature_index is not None and self.target_state is not None:
            guidance = (int(self.feature_index), int(self.target_state))
        else:
            guidance_source = "victim_clean_snapshot"
            try:
                guidance = victim.counterfactual_guidance(session.clean_snapshot)
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                guidance = None
                guidance_reason = f"guidance_error:{type(exc).__name__}"
            else:
                guidance_reason = (
                    "target_achieving_feature_edit"
                    if guidance is not None
                    else "guidance_unavailable"
                )
            if guidance is not None:
                try:
                    feature_index, target_state = int(guidance[0]), int(guidance[1])
                    thresholds = np.asarray(victim.discretizer_thresholds)
                    if (
                        thresholds.ndim != 2
                        or not 0 <= feature_index < thresholds.shape[0]
                        or not 0 <= target_state <= thresholds.shape[1]
                    ):
                        raise ValueError("guidance is outside the discretizer contract")
                    guidance = (feature_index, target_state)
                except (IndexError, TypeError, ValueError):
                    guidance = None
                    guidance_reason = "invalid_counterfactual_guidance"
        guidance_available = guidance is not None
        return self._run_gradient(
            session,
            constraint=constraint,
            guidance=guidance,
            weights={
                "diagnosis": 1.0,
                "latent": self.latent_weight if guidance_available else 0.0,
                "latent_margin": self.latent_margin,
            },
            family_metadata={
                "objective": (
                    "targeted_diagnosis_plus_counterfactual_bin_margin"
                    if guidance_available
                    else "targeted_diagnosis_fallback_counterfactual_unavailable"
                ),
                "counterfactual_guidance_available": guidance_available,
                "counterfactual_guidance_source": guidance_source,
                "counterfactual_guidance_reason": guidance_reason,
                "counterfactual_feature_index": None if guidance is None else guidance[0],
                "counterfactual_target_state": None if guidance is None else guidance[1],
                "latent_weight": self.latent_weight,
                "effective_latent_weight": self.latent_weight if guidance_available else 0.0,
                "latent_margin": self.latent_margin,
            },
        )


class _ParameterSearchSpace:
    def __init__(self, clean: Tensor, parameterization: Any | None):
        self.clean = clean
        self.parameterization = parameterization
        self.scales: Tensor | None = None
        if parameterization is None:
            self.initial = clean.clone()
        elif hasattr(parameterization, "initial_parameters"):
            self.initial = torch.as_tensor(
                parameterization.initial_parameters(clean),
                dtype=clean.dtype,
                device=clean.device,
            )
        elif hasattr(parameterization, "initial"):
            self.initial = torch.as_tensor(
                parameterization.initial(clean), dtype=clean.dtype, device=clean.device
            )
        elif hasattr(parameterization, "parameter_names"):
            self.initial = torch.zeros(
                len(parameterization.parameter_names),
                dtype=clean.dtype,
                device=clean.device,
            )
            bounds = getattr(parameterization, "bounds", None)
            if isinstance(bounds, dict) and all(
                name in bounds for name in parameterization.parameter_names
            ):
                self.scales = torch.as_tensor(
                    [bounds[name] for name in parameterization.parameter_names],
                    dtype=clean.dtype,
                    device=clean.device,
                )
        else:
            raise TypeError("Physiological parameterization requires initial parameters")

    @property
    def normalized(self) -> bool:
        return self.scales is not None

    def project(self, values: Tensor) -> Tensor:
        if self.parameterization is None:
            return values
        if self.normalized:
            return values.clamp(-1.0, 1.0)
        if hasattr(self.parameterization, "project_parameters"):
            return torch.as_tensor(
                self.parameterization.project_parameters(values),
                dtype=values.dtype,
                device=values.device,
            )
        if hasattr(self.parameterization, "project"):
            return torch.as_tensor(
                self.parameterization.project(values), dtype=values.dtype, device=values.device
            )
        return values

    def render(self, values: Tensor) -> Tensor:
        if self.parameterization is None:
            return values
        physical_values = values * self.scales if self.normalized else values
        if hasattr(self.parameterization, "render"):
            return torch.as_tensor(
                self.parameterization.render(self.clean, physical_values),
                dtype=self.clean.dtype,
                device=self.clean.device,
            )
        if hasattr(self.parameterization, "synthesize"):
            return torch.as_tensor(
                self.parameterization.synthesize(self.clean, physical_values),
                dtype=self.clean.dtype,
                device=self.clean.device,
            )
        raise TypeError(
            "Physiological parameterization requires render or synthesize(clean, parameters)"
        )


class F2NESAttack:
    family = AttackFamily.F2

    def __init__(
        self,
        *,
        sigma: float = 0.001,
        learning_rate: float = 0.002,
        samples_per_step: int = 4,
        epsilon_mv: float = 0.01,
        norm: str = "linf",
        seed: int = 42,
        parameterization: Any | None = None,
        verifier_aware: bool = False,
        verifier_weight: float = 1.0,
    ):
        if (
            not np.isfinite(sigma)
            or not np.isfinite(learning_rate)
            or sigma <= 0
            or learning_rate <= 0
            or samples_per_step < 1
        ):
            raise ValueError("NES sigma, learning rate, and samples_per_step must be positive")
        if norm not in {"linf", "l2"}:
            raise ValueError("norm must be 'linf' or 'l2'")
        if (
            not np.isfinite(epsilon_mv)
            or epsilon_mv < 0
            or not np.isfinite(verifier_weight)
            or verifier_weight < 0
        ):
            raise ValueError("NES epsilon and verifier weight must be finite and non-negative")
        self.sigma = float(sigma)
        self.learning_rate = float(learning_rate)
        self.samples_per_step = int(samples_per_step)
        self.epsilon_mv = float(epsilon_mv)
        self.norm = norm
        self.seed = int(seed)
        self.parameterization = parameterization
        self.verifier_aware = bool(verifier_aware)
        self.verifier_weight = float(verifier_weight)

    def run(
        self,
        victim: Any,
        clean_signal: np.ndarray,
        target_label: str | int,
        *,
        query_budget: int,
        constraint: Any | None = None,
        admissibility_checker: Any | None = None,
        case: Any | None = None,
        defense: str | DefenseKind = DefenseKind.V1,
        clean_snapshot: VictimSnapshot | None = None,
        family: AttackFamily | None = None,
    ) -> AttackResult:
        selected_family = family or self.family
        session = _AttackSession(
            family=selected_family,
            victim=victim,
            clean_signal=clean_signal,
            target_label=target_label,
            query_budget=query_budget,
            constraint=constraint,
            admissibility_checker=admissibility_checker,
            case=case,
            defense=defense,
            clean_snapshot=clean_snapshot,
            verifier_aware=self.verifier_aware,
            verifier_weight=self.verifier_weight,
            knowledge_level=KnowledgeLevel.K1,
        )
        clean = torch.as_tensor(session.clean, dtype=torch.float32)
        space = _ParameterSearchSpace(clean, self.parameterization)
        parameters = space.initial.clone()
        generator = torch.Generator(device=parameters.device).manual_seed(self.seed)
        updates = 0
        while session.remaining and not session.succeeded:
            gradient = torch.zeros_like(parameters)
            pairs = 0
            for _ in range(self.samples_per_step):
                if session.remaining <= 0 or session.succeeded:
                    break
                direction = torch.randn(
                    parameters.shape,
                    generator=generator,
                    dtype=parameters.dtype,
                    device=parameters.device,
                )
                direction /= torch.linalg.vector_norm(direction).clamp_min(1e-12)
                plus_parameters = space.project(parameters + self.sigma * direction)
                plus = space.render(plus_parameters)
                if self.parameterization is None:
                    plus = _project_candidate(
                        clean.unsqueeze(0),
                        plus.unsqueeze(0),
                        self.epsilon_mv,
                        self.norm,
                        constraint,
                    )[0]
                else:
                    plus = _constraint_project(constraint, clean.unsqueeze(0), plus.unsqueeze(0))[0]
                plus_submission = session.submit(
                    plus,
                    metadata={"nes_update": updates + 1, "nes_side": "plus"},
                )
                if session.succeeded or session.remaining <= 0:
                    break
                minus_parameters = space.project(parameters - self.sigma * direction)
                minus = space.render(minus_parameters)
                if self.parameterization is None:
                    minus = _project_candidate(
                        clean.unsqueeze(0),
                        minus.unsqueeze(0),
                        self.epsilon_mv,
                        self.norm,
                        constraint,
                    )[0]
                else:
                    minus = _constraint_project(constraint, clean.unsqueeze(0), minus.unsqueeze(0))[
                        0
                    ]
                minus_submission = session.submit(
                    minus,
                    metadata={"nes_update": updates + 1, "nes_side": "minus"},
                )
                gradient += (
                    (plus_submission.loss - minus_submission.loss) / (2.0 * self.sigma) * direction
                )
                pairs += 1
            if not pairs or session.succeeded:
                break
            parameters = space.project(parameters - self.learning_rate * gradient / pairs)
            updates += 1
        reason = "incorrect_release" if session.succeeded else "query_budget_exhausted"
        return session.result(
            reason,
            {
                "method": "antithetic_nes",
                "knowledge_level": KnowledgeLevel.K1.value,
                "sigma": self.sigma,
                "learning_rate": self.learning_rate,
                "samples_per_step": self.samples_per_step,
                "updates": updates,
                "parameterized": self.parameterization is not None,
                "normalized_parameter_coordinates": space.normalized,
                "verifier_aware": self.verifier_aware,
                "verifier_weight": self.verifier_weight,
                "builtin_lp_projection": self.parameterization is None
                and _uses_builtin_lp_projection(self.norm, constraint),
            },
        )


class F2SimBAAttack:
    family = AttackFamily.F2

    def __init__(
        self,
        *,
        step_size_mv: float = 0.002,
        epsilon_mv: float = 0.01,
        norm: str = "linf",
        seed: int = 42,
        parameterization: Any | None = None,
        verifier_aware: bool = False,
        verifier_weight: float = 1.0,
    ):
        if (
            not np.isfinite(step_size_mv)
            or not np.isfinite(epsilon_mv)
            or step_size_mv <= 0
            or epsilon_mv < 0
        ):
            raise ValueError("SimBA step size must be positive and epsilon non-negative")
        if norm not in {"linf", "l2"}:
            raise ValueError("norm must be 'linf' or 'l2'")
        if not np.isfinite(verifier_weight) or verifier_weight < 0:
            raise ValueError("verifier_weight must be finite and non-negative")
        self.step_size_mv = float(step_size_mv)
        self.epsilon_mv = float(epsilon_mv)
        self.norm = norm
        self.seed = int(seed)
        self.parameterization = parameterization
        self.verifier_aware = bool(verifier_aware)
        self.verifier_weight = float(verifier_weight)

    def run(
        self,
        victim: Any,
        clean_signal: np.ndarray,
        target_label: str | int,
        *,
        query_budget: int,
        constraint: Any | None = None,
        admissibility_checker: Any | None = None,
        case: Any | None = None,
        defense: str | DefenseKind = DefenseKind.V1,
        clean_snapshot: VictimSnapshot | None = None,
        family: AttackFamily | None = None,
    ) -> AttackResult:
        session = _AttackSession(
            family=family or self.family,
            victim=victim,
            clean_signal=clean_signal,
            target_label=target_label,
            query_budget=query_budget,
            constraint=constraint,
            admissibility_checker=admissibility_checker,
            case=case,
            defense=defense,
            clean_snapshot=clean_snapshot,
            verifier_aware=self.verifier_aware,
            verifier_weight=self.verifier_weight,
            knowledge_level=KnowledgeLevel.K1,
        )
        clean = torch.as_tensor(session.clean, dtype=torch.float32)
        space = _ParameterSearchSpace(clean, self.parameterization)
        parameters = space.initial.clone()
        best_loss = _snapshot_loss(
            session.clean_snapshot,
            session.target_name,
            verifier_aware=self.verifier_aware,
            verifier_weight=self.verifier_weight,
            decision_only=session.decision_only,
        )
        rng = np.random.default_rng(self.seed)
        coordinates = rng.permutation(parameters.numel())
        passes = 0
        while session.remaining and not session.succeeded:
            for coordinate in coordinates:
                if session.remaining <= 0 or session.succeeded:
                    break
                direction = torch.zeros_like(parameters).reshape(-1)
                direction[int(coordinate)] = self.step_size_mv
                direction = direction.reshape_as(parameters)
                plus_parameters = space.project(parameters + direction)
                plus = space.render(plus_parameters)
                if self.parameterization is None:
                    plus = _project_candidate(
                        clean.unsqueeze(0),
                        plus.unsqueeze(0),
                        self.epsilon_mv,
                        self.norm,
                        constraint,
                    )[0]
                else:
                    plus = _constraint_project(constraint, clean.unsqueeze(0), plus.unsqueeze(0))[0]
                plus_submission = session.submit(
                    plus,
                    metadata={"simba_pass": passes + 1, "simba_side": "plus"},
                )
                if plus_submission.loss < best_loss:
                    parameters, best_loss = plus_parameters, plus_submission.loss
                    continue
                if session.remaining <= 0 or session.succeeded:
                    break
                minus_parameters = space.project(parameters - direction)
                minus = space.render(minus_parameters)
                if self.parameterization is None:
                    minus = _project_candidate(
                        clean.unsqueeze(0),
                        minus.unsqueeze(0),
                        self.epsilon_mv,
                        self.norm,
                        constraint,
                    )[0]
                else:
                    minus = _constraint_project(constraint, clean.unsqueeze(0), minus.unsqueeze(0))[
                        0
                    ]
                minus_submission = session.submit(
                    minus,
                    metadata={"simba_pass": passes + 1, "simba_side": "minus"},
                )
                if minus_submission.loss < best_loss:
                    parameters, best_loss = minus_parameters, minus_submission.loss
            passes += 1
            coordinates = rng.permutation(parameters.numel())
        reason = "incorrect_release" if session.succeeded else "query_budget_exhausted"
        return session.result(
            reason,
            {
                "method": "simba_coordinate_search",
                "step_size_mv": self.step_size_mv,
                "epsilon_mv": self.epsilon_mv,
                "norm": self.norm,
                "passes": passes,
                "parameterized": self.parameterization is not None,
                "normalized_parameter_coordinates": space.normalized,
                "verifier_aware": self.verifier_aware,
                "verifier_weight": self.verifier_weight,
                "builtin_lp_projection": self.parameterization is None
                and _uses_builtin_lp_projection(self.norm, constraint),
            },
        )


class F2SquareAttack:
    family = AttackFamily.F2

    def __init__(
        self,
        *,
        epsilon_mv: float = 0.01,
        initial_fraction: float = 0.2,
        norm: str = "linf",
        seed: int = 42,
        parameterization: Any | None = None,
        parameter_amplitude: float = 1.0,
        verifier_aware: bool = False,
        verifier_weight: float = 1.0,
    ):
        if (
            not np.isfinite(epsilon_mv)
            or not np.isfinite(initial_fraction)
            or not np.isfinite(parameter_amplitude)
            or epsilon_mv < 0
            or not 0 < initial_fraction <= 1
            or parameter_amplitude <= 0
        ):
            raise ValueError("Square epsilon, fraction, or parameter amplitude is invalid")
        if norm not in {"linf", "l2"}:
            raise ValueError("norm must be 'linf' or 'l2'")
        if not np.isfinite(verifier_weight) or verifier_weight < 0:
            raise ValueError("verifier_weight must be finite and non-negative")
        self.epsilon_mv = float(epsilon_mv)
        self.initial_fraction = float(initial_fraction)
        self.norm = norm
        self.seed = int(seed)
        self.parameterization = parameterization
        self.parameter_amplitude = float(parameter_amplitude)
        self.verifier_aware = bool(verifier_aware)
        self.verifier_weight = float(verifier_weight)

    def run(
        self,
        victim: Any,
        clean_signal: np.ndarray,
        target_label: str | int,
        *,
        query_budget: int,
        constraint: Any | None = None,
        admissibility_checker: Any | None = None,
        case: Any | None = None,
        defense: str | DefenseKind = DefenseKind.V1,
        clean_snapshot: VictimSnapshot | None = None,
        family: AttackFamily | None = None,
    ) -> AttackResult:
        session = _AttackSession(
            family=family or self.family,
            victim=victim,
            clean_signal=clean_signal,
            target_label=target_label,
            query_budget=query_budget,
            constraint=constraint,
            admissibility_checker=admissibility_checker,
            case=case,
            defense=defense,
            clean_snapshot=clean_snapshot,
            verifier_aware=self.verifier_aware,
            verifier_weight=self.verifier_weight,
            knowledge_level=KnowledgeLevel.K1,
        )
        clean = torch.as_tensor(session.clean, dtype=torch.float32)
        space = _ParameterSearchSpace(clean, self.parameterization)
        parameters = space.initial.clone()
        candidate = space.render(parameters)
        best_loss = _snapshot_loss(
            session.clean_snapshot,
            session.target_name,
            verifier_aware=self.verifier_aware,
            verifier_weight=self.verifier_weight,
            decision_only=session.decision_only,
        )
        rng = np.random.default_rng(self.seed)
        while session.remaining and not session.succeeded:
            progress = len(session.attempts) / max(session.oracle.budget, 1)
            sign = float(rng.choice((-1.0, 1.0)))
            if self.parameterization is None:
                fraction = max(1.0 / clean.shape[1], self.initial_fraction * (1.0 - progress))
                width = max(1, round(fraction * clean.shape[1]))
                start = int(rng.integers(0, clean.shape[1] - width + 1))
                lead_count = int(rng.integers(1, clean.shape[0] + 1))
                leads = rng.choice(clean.shape[0], size=lead_count, replace=False)
                proposal_parameters = candidate.clone()
                proposal_parameters[torch.as_tensor(leads), start : start + width] = (
                    clean[torch.as_tensor(leads), start : start + width] + sign * self.epsilon_mv
                )
                proposal = _project_candidate(
                    clean.unsqueeze(0),
                    proposal_parameters.unsqueeze(0),
                    self.epsilon_mv,
                    self.norm,
                    constraint,
                )[0]
                proposal_metadata = {
                    "square_start": start,
                    "square_width": width,
                    "square_leads": [int(value) for value in leads],
                }
            else:
                count = parameters.numel()
                fraction = max(1.0 / count, self.initial_fraction * (1.0 - progress))
                width = max(1, round(fraction * count))
                start = int(rng.integers(0, count - width + 1))
                proposal_parameters = parameters.clone()
                proposal_parameters.reshape(-1)[start : start + width] = (
                    sign * self.parameter_amplitude
                )
                proposal_parameters = space.project(proposal_parameters)
                proposal = space.render(proposal_parameters)
                proposal_metadata = {
                    "parameter_start": start,
                    "parameter_width": width,
                }
            if self.parameterization is not None:
                proposal = _constraint_project(
                    constraint, clean.unsqueeze(0), proposal.unsqueeze(0)
                )[0]
            submission = session.submit(
                proposal,
                metadata={
                    **proposal_metadata,
                    "square_sign": sign,
                },
            )
            if submission.loss < best_loss:
                candidate, best_loss = proposal, submission.loss
                parameters = proposal_parameters
        reason = "incorrect_release" if session.succeeded else "query_budget_exhausted"
        return session.result(
            reason,
            {
                "method": "square_style_block_search",
                "epsilon_mv": self.epsilon_mv,
                "initial_fraction": self.initial_fraction,
                "norm": self.norm,
                "parameterized": self.parameterization is not None,
                "parameter_amplitude": self.parameter_amplitude,
                "normalized_parameter_coordinates": space.normalized,
                "verifier_aware": self.verifier_aware,
                "verifier_weight": self.verifier_weight,
                "builtin_lp_projection": self.parameterization is None
                and _uses_builtin_lp_projection(self.norm, constraint),
            },
        )


class F3AdaptiveAttack(F1PGDAttack):
    family = AttackFamily.F3

    def __init__(
        self,
        *,
        knowledge_level: str | KnowledgeLevel = KnowledgeLevel.K3,
        diagnosis_weight: float = 1.0,
        retrieval_weight: float = 1.0,
        verifier_weight: float = 1.0,
        defense_weight: float = 0.0,
        black_box_method: str = "nes",
        black_box_parameterization: Any | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.knowledge_level = KnowledgeLevel(knowledge_level)
        weights = (diagnosis_weight, retrieval_weight, verifier_weight, defense_weight)
        if any(not np.isfinite(value) or value < 0 for value in weights):
            raise ValueError("F3 objective weights must be finite and non-negative")
        if not any(weights):
            raise ValueError("At least one F3 objective weight must be positive")
        if black_box_method not in {"nes", "simba", "square"}:
            raise ValueError("black_box_method must be nes, simba, or square")
        self.diagnosis_weight = float(diagnosis_weight)
        self.retrieval_weight = float(retrieval_weight)
        self.verifier_weight = float(verifier_weight)
        self.defense_weight = float(defense_weight)
        self.black_box_method = black_box_method
        self.black_box_parameterization = black_box_parameterization

    def run(
        self,
        victim: Any,
        clean_signal: np.ndarray,
        target_label: str | int,
        *,
        query_budget: int,
        constraint: Any | None = None,
        admissibility_checker: Any | None = None,
        case: Any | None = None,
        defense: str | DefenseKind = DefenseKind.V1,
        clean_snapshot: VictimSnapshot | None = None,
    ) -> AttackResult:
        defense_kind = DefenseKind(defense)
        selected_family = (
            AttackFamily.F3_DEFENSE
            if defense_kind in {DefenseKind.V2, DefenseKind.V3}
            else self.family
        )
        if self.knowledge_level is KnowledgeLevel.K1:
            common = {
                "epsilon_mv": self.epsilon_mv,
                "norm": self.norm,
                "seed": self.seed,
                "parameterization": self.black_box_parameterization,
                "verifier_aware": True,
                "verifier_weight": self.verifier_weight,
            }
            if self.black_box_method == "nes":
                attack = F2NESAttack(
                    sigma=max(self.step_size_mv, 1e-6),
                    learning_rate=max(self.step_size_mv, 1e-6),
                    samples_per_step=2,
                    **common,
                )
            elif self.black_box_method == "simba":
                attack = F2SimBAAttack(step_size_mv=max(self.step_size_mv, 1e-6), **common)
            else:
                attack = F2SquareAttack(**common)
            result = attack.run(
                victim,
                clean_signal,
                target_label,
                query_budget=query_budget,
                constraint=constraint,
                admissibility_checker=admissibility_checker,
                case=case,
                defense=defense_kind,
                clean_snapshot=clean_snapshot,
                family=selected_family,
            )
            result.metadata.update(
                {
                    "knowledge_level": self.knowledge_level.value,
                    "adaptive_black_box_method": self.black_box_method,
                    "retrieval_weight": self.retrieval_weight,
                    "verifier_weight": self.verifier_weight,
                    "defense_weight": self.defense_weight,
                    "defense_adaptation_status": "live_decision_feedback_applied",
                    "attack_reoptimized_for_defense": True,
                    "reoptimized_defense": defense_kind.value,
                    "re_adapted_to_defense": defense_kind.value,
                    "reoptimization_method": ("fresh_black_box_complete_system_feedback"),
                    "reoptimization_evidence": {
                        "defense": defense_kind.value,
                        "fresh_search_from_clean": True,
                        "complete_system_feedback": True,
                        "feedback_fields": [
                            "released_report_target_claim",
                            "release_decision",
                        ],
                        "hidden_target_probability_used": False,
                    },
                }
            )
            return result

        session = _AttackSession(
            family=selected_family,
            victim=victim,
            clean_signal=clean_signal,
            target_label=target_label,
            query_budget=query_budget,
            constraint=constraint,
            admissibility_checker=admissibility_checker,
            case=case,
            defense=defense_kind,
            clean_snapshot=clean_snapshot,
            verifier_aware=True,
            verifier_weight=self.verifier_weight,
            knowledge_level=self.knowledge_level,
        )
        device = getattr(session.victim, "device", torch.device("cpu"))
        clean_tensor = torch.as_tensor(session.clean, dtype=torch.float32, device=device).unsqueeze(
            0
        )
        clean_surrogate = session.victim.surrogate(
            clean_tensor, session.target_name, defense=defense_kind
        )
        retrieval_available = clean_surrogate.retrieval_loss is not None
        verifier_available = clean_surrogate.verifier_loss is not None
        verifier_distinct = (
            verifier_available
            and clean_surrogate.verifier_loss is not clean_surrogate.retrieval_loss
        )
        defense_available = clean_surrogate.defense_loss is not None

        if self.knowledge_level is KnowledgeLevel.K2:
            effective_retrieval_weight = self.retrieval_weight if retrieval_available else 0.0
            effective_verifier_weight = 0.0
            effective_defense_weight = 0.0
        else:
            effective_retrieval_weight = self.retrieval_weight
            effective_verifier_weight = self.verifier_weight
            effective_defense_weight = self.defense_weight
            if effective_retrieval_weight > 0 and not retrieval_available:
                raise ValueError(
                    "K3 retrieval_weight is positive but no retrieval surrogate is available"
                )
            if effective_verifier_weight > 0 and not verifier_available:
                raise ValueError(
                    "K3 verifier_weight is positive but no verifier surrogate is available"
                )
            if effective_verifier_weight > 0 and not verifier_distinct:
                raise ValueError(
                    "K3 requires a verifier surrogate distinct from the retrieval loss"
                )
            if effective_defense_weight > 0 and not defense_available:
                raise ValueError(
                    "F3 defense_weight is positive but no configured defense surrogate is available"
                )

        if not any(
            (
                self.diagnosis_weight,
                effective_retrieval_weight,
                effective_verifier_weight,
                effective_defense_weight,
            )
        ):
            raise ValueError("No available differentiable objective remains for F3")
        objective_parts = []
        if self.diagnosis_weight > 0:
            objective_parts.append("diagnosis")
        if effective_retrieval_weight > 0:
            objective_parts.append("retrieval")
        if effective_verifier_weight > 0:
            objective_parts.append("verifier")
        if effective_defense_weight > 0:
            objective_parts.append("defense")
        defense_adaptation_applied = bool(effective_defense_weight > 0)
        verifier_surrogate = clean_surrogate.metadata.get("verifier_surrogate")
        verifier_is_defense_specific = bool(
            effective_verifier_weight > 0
            and (
                defense_kind is DefenseKind.V1
                or verifier_surrogate == "configured_defense_surrogate"
            )
        )
        reoptimization_applied = bool(
            self.knowledge_level is KnowledgeLevel.K3
            and (verifier_is_defense_specific or defense_adaptation_applied)
        )
        if defense_kind is DefenseKind.V1:
            defense_adaptation_status = (
                "shipped_verifier_surrogate_applied"
                if reoptimization_applied
                else "shipped_verifier_surrogate_unavailable_or_unweighted"
            )
        elif defense_adaptation_applied:
            defense_adaptation_status = "applied"
        elif defense_available:
            defense_adaptation_status = "surrogate_available_not_weighted"
        else:
            defense_adaptation_status = "surrogate_unavailable"
        reoptimization_method = (
            "fresh_defense_conditioned_surrogate_optimization"
            if reoptimization_applied
            else "fresh_optimization_without_defense_specific_objective"
        )
        return self._run_gradient(
            session,
            constraint=constraint,
            guidance=None,
            weights={
                "diagnosis": self.diagnosis_weight,
                "retrieval": effective_retrieval_weight,
                "verifier": effective_verifier_weight,
                "defense": effective_defense_weight,
            },
            family_metadata={
                "objective": "_plus_".join(objective_parts),
                "knowledge_level": self.knowledge_level.value,
                "diagnosis_weight": self.diagnosis_weight,
                "requested_retrieval_weight": self.retrieval_weight,
                "requested_verifier_weight": self.verifier_weight,
                "requested_defense_weight": self.defense_weight,
                "retrieval_weight": effective_retrieval_weight,
                "verifier_weight": effective_verifier_weight,
                "defense_weight": effective_defense_weight,
                "retrieval_objective_available": retrieval_available,
                "retrieval_objective_applied": effective_retrieval_weight > 0,
                "verifier_objective_available": verifier_available,
                "verifier_objective_distinct_from_retrieval": verifier_distinct,
                "verifier_objective_permitted": (self.knowledge_level is KnowledgeLevel.K3),
                "verifier_objective_applied": effective_verifier_weight > 0,
                "verifier_surrogate": verifier_surrogate,
                "defense_objective_available": defense_available,
                "defense_objective_applied": effective_defense_weight > 0,
                "defense_adaptation_status": defense_adaptation_status,
                "attack_reoptimized_for_defense": reoptimization_applied,
                "reoptimized_defense": (defense_kind.value if reoptimization_applied else None),
                "re_adapted_to_defense": (defense_kind.value if reoptimization_applied else None),
                "reoptimization_method": reoptimization_method,
                "reoptimization_evidence": {
                    "defense": defense_kind.value,
                    "fresh_search_from_clean": True,
                    "surrogate_called_with_defense": defense_kind.value,
                    "complete_system_feedback": True,
                    "verifier_surrogate_applied": effective_verifier_weight > 0,
                    "verifier_surrogate_defense_specific": (verifier_is_defense_specific),
                    "defense_surrogate_applied": defense_adaptation_applied,
                    "objective_parts": objective_parts,
                },
            },
        )


# Concise aliases matching the proposal notation.
RandomAttack = F0RandomAttack
PGDAttack = F1PGDAttack
CounterfactualGuidedAttack = F1CounterfactualAttack
NESAttack = F2NESAttack
SimBAAttack = F2SimBAAttack
SquareAttack = F2SquareAttack
VerifierAwareAttack = F3AdaptiveAttack
