from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ObjectiveTerms:
    """Individual minimization terms used by an adaptive waveform attack."""

    total: Tensor
    diagnosis: Tensor
    retrieval: Tensor
    verifier: Tensor
    defense: Tensor
    latent: Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "diagnosis": float(self.diagnosis.detach().cpu()),
            "retrieval": float(self.retrieval.detach().cpu()),
            "verifier": float(self.verifier.detach().cpu()),
            "defense": float(self.defense.detach().cpu()),
            "latent": float(self.latent.detach().cpu()),
        }


def _class_matrix(values: Tensor) -> Tensor:
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Class scores must have shape [batch, classes] with at least two classes")
    if not torch.isfinite(values).all():
        raise ValueError("Class scores must be finite")
    return values


def targeted_margin_loss(
    scores: Tensor,
    target_index: int,
    *,
    confidence: float = 0.0,
    probabilities: bool = False,
) -> Tensor:
    """Carlini-Wagner style targeted margin loss.

    The result is minimized. It reaches zero once the target score exceeds every
    competing score by ``confidence``. Probability inputs are converted to log space
    so that small target probabilities retain useful gradients.
    """

    matrix = _class_matrix(scores)
    target_index = int(target_index)
    if not 0 <= target_index < matrix.shape[1]:
        raise ValueError("Target class index is out of range")
    if confidence < 0 or not np.isfinite(confidence):
        raise ValueError("confidence must be finite and non-negative")
    if probabilities:
        if torch.any(matrix < 0):
            raise ValueError("Probabilities must be non-negative")
        matrix = torch.log(matrix.clamp_min(torch.finfo(matrix.dtype).tiny))
    target = matrix[:, target_index]
    mask = F.one_hot(
        torch.full(
            (len(matrix),), target_index, dtype=torch.long, device=matrix.device
        ),
        num_classes=matrix.shape[1],
    ).bool()
    competing = matrix.masked_fill(mask, -torch.inf).amax(dim=1)
    return torch.relu(competing - target + float(confidence)).mean()


def targeted_nll(probabilities: Tensor, target_index: int) -> Tensor:
    matrix = _class_matrix(probabilities)
    target_index = int(target_index)
    if not 0 <= target_index < matrix.shape[1]:
        raise ValueError("Target class index is out of range")
    if torch.any(matrix < 0):
        raise ValueError("Probabilities must be non-negative")
    mass = matrix.sum(dim=1, keepdim=True).clamp_min(torch.finfo(matrix.dtype).tiny)
    normalized = matrix / mass
    return -torch.log(normalized[:, target_index].clamp_min(torch.finfo(matrix.dtype).tiny)).mean()


def soft_bin_memberships(latent: Tensor, thresholds: Tensor, temperature: float) -> Tensor:
    """Differentiable quantile-bin memberships for BPDA/soft-bin objectives.

    Returns ``[batch, latent_features, bins]``. The hard pipeline remains the success
    oracle; these memberships are only optimization guidance.
    """

    if latent.ndim == 1:
        latent = latent.unsqueeze(0)
    if (
        thresholds.ndim != 2
        or thresholds.shape[1] < 1
        or latent.ndim != 2
        or latent.shape[1] != thresholds.shape[0]
    ):
        raise ValueError("Latent values and thresholds have incompatible dimensions")
    if not bool(torch.isfinite(thresholds).all()) or bool(
        torch.any(thresholds[:, 1:] <= thresholds[:, :-1])
    ):
        raise ValueError("Bin thresholds must be finite and strictly increasing")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    boundaries = torch.sigmoid(
        (latent.unsqueeze(-1) - thresholds.to(device=latent.device, dtype=latent.dtype))
        / float(temperature)
    )
    pieces = [1.0 - boundaries[..., :1]]
    if boundaries.shape[-1] > 1:
        pieces.append(boundaries[..., :-1] - boundaries[..., 1:])
    pieces.append(boundaries[..., -1:])
    memberships = torch.cat(pieces, dim=-1).clamp_min(0.0)
    return memberships / memberships.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def latent_bin_margin_loss(
    latent: Tensor,
    thresholds: Tensor,
    feature_index: int,
    target_state: int,
    *,
    margin: float = 1e-3,
) -> Tensor:
    """Push one latent coordinate into the interior of a requested quantile bin."""

    if latent.ndim == 1:
        latent = latent.unsqueeze(0)
    if (
        thresholds.ndim != 2
        or thresholds.shape[1] < 1
        or latent.ndim != 2
        or latent.shape[1] != thresholds.shape[0]
    ):
        raise ValueError("Latent values and thresholds have incompatible dimensions")
    if not bool(torch.isfinite(thresholds).all()) or bool(
        torch.any(thresholds[:, 1:] <= thresholds[:, :-1])
    ):
        raise ValueError("Bin thresholds must be finite and strictly increasing")
    feature_index = int(feature_index)
    if not 0 <= feature_index < latent.shape[1]:
        raise ValueError("Counterfactual feature index is out of range")
    bins = thresholds.shape[1] + 1
    target_state = int(target_state)
    if not 0 <= target_state < bins:
        raise ValueError("Counterfactual target state is out of range")
    if margin < 0 or not np.isfinite(margin):
        raise ValueError("margin must be finite and non-negative")
    value = latent[:, feature_index]
    boundary = thresholds.to(device=latent.device, dtype=latent.dtype)[feature_index]
    loss = torch.zeros_like(value)
    if target_state > 0:
        loss = loss + torch.relu(boundary[target_state - 1] + margin - value)
    if target_state < bins - 1:
        loss = loss + torch.relu(value - boundary[target_state] + margin)
    return loss.mean()


def counterfactual_feature_state(
    counterfactual: Any,
    feature_names: list[str] | tuple[str, ...],
) -> tuple[int, int]:
    feature = getattr(counterfactual, "feature", None)
    state = getattr(counterfactual, "new_state", None)
    if feature is None or state is None:
        raise ValueError("Counterfactual guidance requires a feature-changing edit")
    try:
        index = list(feature_names).index(str(feature))
    except ValueError as exc:
        raise ValueError(f"Counterfactual feature is not in the latent contract: {feature}") from exc
    return index, int(state)


def target_evidence_loss(retrieval_scores: Tensor, target_mask: Tensor) -> Tensor:
    """Maximize soft retrieval mass assigned to target-consistent evidence."""

    if retrieval_scores.ndim == 1:
        retrieval_scores = retrieval_scores.unsqueeze(0)
    if retrieval_scores.ndim != 2:
        raise ValueError("Retrieval scores must have shape [batch, documents]")
    if not bool(torch.isfinite(retrieval_scores).all()):
        raise ValueError("Retrieval scores must be finite")
    mask = target_mask.to(device=retrieval_scores.device, dtype=torch.bool).reshape(-1)
    if mask.shape != (retrieval_scores.shape[1],) or not bool(mask.any()):
        raise ValueError("target_mask must select at least one retrieval document")
    all_mass = torch.logsumexp(retrieval_scores, dim=1)
    target_mass = torch.logsumexp(retrieval_scores[:, mask], dim=1)
    return (all_mass - target_mass).mean()


def adaptive_objective(
    diagnosis_scores: Tensor,
    target_index: int,
    *,
    diagnosis_probabilities: bool = False,
    retrieval_loss: Tensor | None = None,
    verifier_loss: Tensor | None = None,
    defense_loss: Tensor | None = None,
    latent_loss: Tensor | None = None,
    diagnosis_weight: float = 1.0,
    retrieval_weight: float = 0.0,
    verifier_weight: float = 0.0,
    defense_weight: float = 0.0,
    latent_weight: float = 0.0,
    confidence: float = 0.0,
) -> ObjectiveTerms:
    """Compose F1/F1-CF/F3 objectives while retaining auditable term values."""

    weights = (
        diagnosis_weight,
        retrieval_weight,
        verifier_weight,
        defense_weight,
        latent_weight,
    )
    if any(not np.isfinite(value) or value < 0 for value in weights):
        raise ValueError("Adaptive objective weights must be finite and non-negative")
    if not any(weights):
        raise ValueError("At least one adaptive objective weight must be positive")
    diagnosis = targeted_margin_loss(
        diagnosis_scores,
        target_index,
        confidence=confidence,
        probabilities=diagnosis_probabilities,
    )
    zero = diagnosis.new_zeros(())
    retrieval = zero if retrieval_loss is None else retrieval_loss
    verifier = zero if verifier_loss is None else verifier_loss
    defense = zero if defense_loss is None else defense_loss
    latent = zero if latent_loss is None else latent_loss
    for name, value, original, weight in (
        ("retrieval", retrieval, retrieval_loss, retrieval_weight),
        ("verifier", verifier, verifier_loss, verifier_weight),
        ("defense", defense, defense_loss, defense_weight),
        ("latent", latent, latent_loss, latent_weight),
    ):
        if weight > 0 and original is None:
            raise ValueError(f"{name} loss is required when its weight is positive")
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise ValueError(f"{name} loss must be a finite scalar tensor")
    total = (
        diagnosis_weight * diagnosis
        + retrieval_weight * retrieval
        + verifier_weight * verifier
        + defense_weight * defense
        + latent_weight * latent
    )
    return ObjectiveTerms(total, diagnosis, retrieval, verifier, defense, latent)
