from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..agents import HistoryEntry
from ..pipeline import CAREECGPipeline
from .claims import ClaimOracle
from .contracts import DefenseKind, ReleaseDecision, VictimSnapshot
from .objectives import target_evidence_loss


@dataclass(slots=True)
class SurrogateSnapshot:
    """Differentiable view used for optimization, never for declaring success."""

    logits: Tensor
    probabilities: Tensor
    latent: Tensor
    retrieval_scores: Tensor | None = None
    retrieval_loss: Tensor | None = None
    verifier_loss: Tensor | None = None
    defense_loss: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryBudgetExceeded(RuntimeError):
    pass


class InternalCAREECGVictim:
    """Waveform-callable CARE victim with separate exact and surrogate paths.

    ``evaluate`` always executes the real CARE pipeline. ``surrogate`` exposes a
    differentiable approximation whose forward diagnostic probabilities exactly match
    CARE whenever its saved neural fusion weight is one. For mixed Bayesian/neural
    fusion, the hard Bayesian component is preserved in the forward value while the
    neural stream supplies a straight-through gradient.
    """

    def __init__(
        self,
        pipeline: CAREECGPipeline,
        query: str = "Which diagnosis is most supported, and why?",
        *,
        source_rate: int | None = None,
        history: Sequence[HistoryEntry] = (),
        patient_id: str | None = None,
        claim_extractor: Any | None = None,
        defense_evaluators: Mapping[str | DefenseKind, Any] | None = None,
        defense_surrogates: Mapping[
            str | DefenseKind, Callable[[Tensor, Tensor, Tensor], Tensor]
        ]
        | None = None,
        defense_context: Mapping[str, Any] | None = None,
    ):
        if not query.strip():
            raise ValueError("Victim query must not be empty")
        self.pipeline = pipeline
        self.query = query
        self.source_rate = source_rate
        self.history = tuple(history)
        self.patient_id = patient_id
        self.claim_extractor = claim_extractor
        self.defense_evaluators = {
            DefenseKind(key): value for key, value in (defense_evaluators or {}).items()
        }
        self.defense_surrogates = {
            DefenseKind(key): value for key, value in (defense_surrogates or {}).items()
        }
        self.defense_context = dict(defense_context or {})
        self.label_names = tuple(str(label) for label in pipeline.causal_model.label_names)
        if len(self.label_names) < 2 or len(set(self.label_names)) != len(self.label_names):
            raise ValueError("Victim labels must be unique and contain at least two classes")

    @property
    def device(self) -> torch.device:
        return self.pipeline.device

    @property
    def sample_rate_hz(self) -> float:
        return float(self.pipeline.config.data.sample_rate)

    def target_index(self, target_label: str | int) -> int:
        if isinstance(target_label, (int, np.integer)):
            index = int(target_label)
        else:
            try:
                index = self.label_names.index(str(target_label))
            except ValueError as exc:
                raise ValueError(f"Unknown target diagnosis: {target_label}") from exc
        if not 0 <= index < len(self.label_names):
            raise ValueError("Target diagnosis index is out of range")
        return index

    def _as_waveform_batch(self, signal: Tensor | np.ndarray) -> Tensor:
        value = torch.as_tensor(signal, dtype=torch.float32, device=self.device)
        if value.ndim == 2:
            value = value.unsqueeze(0)
        expected = (self.pipeline.config.data.leads, self.pipeline.config.data.samples)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                f"Security victim expects preprocessed raw waveforms [batch,{expected[0]},"
                f"{expected[1]}], received {tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Security victim waveform must be finite")
        if self.source_rate not in (None, self.pipeline.config.data.sample_rate):
            raise ValueError(
                "Differentiable attacks require waveforms at the configured sample rate; "
                "resample before constructing the victim"
            )
        return value

    def _normalize(self, raw: Tensor) -> Tensor:
        policy = self.pipeline.config.data.normalization
        if policy == "global":
            if self.pipeline.normalization_stats is None:
                raise ValueError("Global normalization statistics are unavailable")
            mean, scale = self.pipeline.normalization_stats
            return (raw - float(mean)) / float(scale)
        if policy == "none":
            return raw
        if policy == "zscore":
            mean = raw.mean(dim=-1, keepdim=True)
            scale = raw.std(dim=-1, keepdim=True, unbiased=False)
            return (raw - mean) / torch.where(scale > 1e-6, scale, torch.ones_like(scale))
        if policy == "robust":
            center = torch.quantile(raw, 0.5, dim=-1, keepdim=True)
            q25 = torch.quantile(raw, 0.25, dim=-1, keepdim=True)
            q75 = torch.quantile(raw, 0.75, dim=-1, keepdim=True)
            iqr = q75 - q25
            standard_deviation = raw.std(dim=-1, keepdim=True, unbiased=False)
            scale = torch.where(
                iqr > 1e-6,
                iqr,
                torch.where(
                    standard_deviation > 1e-6,
                    standard_deviation,
                    torch.ones_like(standard_deviation),
                ),
            )
            return (raw - center) / scale
        raise ValueError(f"Unsupported normalization policy: {policy}")

    def _fused_probabilities(self, latent: Tensor, neural: Tensor) -> Tensor:
        causal_rows = []
        hard_states = self.pipeline.discretizer.transform(latent.detach().cpu().numpy())
        for states in np.asarray(hard_states):
            causal_rows.append(self.pipeline.causal_model.posterior_array(states))
        causal = torch.as_tensor(
            np.stack(causal_rows), dtype=neural.dtype, device=neural.device
        )
        weight = float(self.pipeline.causal_model.neural_weight)
        fused = (1.0 - weight) * causal + weight * neural
        biases = torch.as_tensor(
            self.pipeline.causal_model.class_biases,
            dtype=fused.dtype,
            device=fused.device,
        )
        if bool(torch.any(biases != 0)):
            fused = torch.softmax(torch.log(fused.clamp_min(1e-12)) + biases, dim=1)
        return fused / fused.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def _retrieval_surrogate(
        self, probabilities: Tensor, target_index: int
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        retriever = self.pipeline.retriever
        if retriever.embeddings is None or retriever._encoder is None or not retriever.documents:
            return None, None, None
        if retriever.backend == "tfidf":
            query_vector = retriever._encoder.transform([self.query])[0]
        else:
            query_vector = np.asarray(
                retriever._encoder.encode([self.query], normalize_embeddings=True)[0],
                dtype=np.float32,
            )
        lexical = torch.as_tensor(
            retriever.embeddings @ query_vector,
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        structured_columns = []
        target_mask = []
        target_label = self.label_names[target_index]
        for document in retriever.documents:
            labels = retriever._document_diagnosis_labels(document, self.label_names)
            indices = [self.label_names.index(label) for label in labels]
            if indices:
                structured_columns.append(probabilities[:, indices].amax(dim=1))
            else:
                structured_columns.append(probabilities.new_zeros(len(probabilities)))
            target_mask.append(target_label in labels)
        structured = torch.stack(structured_columns, dim=1)
        scores = (
            (1.0 - retriever.metadata_weight) * lexical.unsqueeze(0)
            + retriever.metadata_weight * structured
        )
        mask = torch.as_tensor(target_mask, dtype=torch.bool, device=scores.device)
        if not bool(mask.any()):
            return scores, None, None
        retrieval_loss = target_evidence_loss(scores, mask)
        target_support = torch.softmax(scores, dim=1)[:, mask].sum(dim=1)
        target_probability = probabilities[:, target_index]
        # A differentiable proxy for the shipped verifier's unsupported-claim gate:
        # target confidence should not exceed the retrieved target-support mass.
        verifier_loss = torch.relu(target_probability - target_support).mean()
        return scores, retrieval_loss, verifier_loss

    def surrogate(
        self,
        signal: Tensor | np.ndarray,
        target_label: str | int,
        *,
        defense: str | DefenseKind = DefenseKind.V1,
    ) -> SurrogateSnapshot:
        raw = self._as_waveform_batch(signal)
        normalized = self._normalize(raw)
        latent, _, _, logits = self.pipeline.model.encode_and_classify(normalized, sample=False)
        if logits is None:
            raise RuntimeError("The CARE victim has no differentiable classification head")
        neural = torch.softmax(logits, dim=1)
        probabilities = self._fused_probabilities(latent, neural)
        target_index = self.target_index(target_label)
        retrieval_scores, retrieval_loss, verifier_loss = self._retrieval_surrogate(
            probabilities, target_index
        )
        defense_kind = DefenseKind(defense)
        defense_loss = None
        hook = self.defense_surrogates.get(defense_kind)
        if hook is not None:
            defense_loss = hook(raw, probabilities, latent)
            if defense_loss.ndim != 0:
                defense_loss = defense_loss.mean()
        verifier_source = "target_claim_support_gap" if verifier_loss is not None else None
        if verifier_loss is None and defense_loss is not None:
            verifier_loss = defense_loss
            verifier_source = "configured_defense_surrogate"
        return SurrogateSnapshot(
            logits=torch.log(probabilities.clamp_min(1e-12)),
            probabilities=probabilities,
            latent=latent,
            retrieval_scores=retrieval_scores,
            retrieval_loss=retrieval_loss,
            verifier_loss=verifier_loss,
            defense_loss=defense_loss,
            metadata={
                "posterior_scope": "internal_care_fused",
                "hard_causal_forward": True,
                "causal_component_gradient": False,
                "neural_weight": float(self.pipeline.causal_model.neural_weight),
                "retrieval_loss_available": retrieval_loss is not None,
                "verifier_loss_available": verifier_loss is not None,
                "verifier_surrogate": verifier_source,
                "defense_loss_available": defense_loss is not None,
            },
        )

    def _target_counterfactual_guidance(self, output: Any, target_name: str) -> dict[str, Any]:
        guidance: dict[str, Any] = {
            "available": False,
            "target_label": target_name,
            "found": False,
            "feature": None,
            "old_state": None,
            "new_state": None,
            "edit_distance": None,
            "scope": None,
            "reason": "counterfactual_search_unavailable",
        }
        try:
            feature_names = tuple(self.pipeline.causal_model.feature_names)
            states = [int(output.discrete_biomarkers[name]) for name in feature_names]
            fusion = output.metadata.get("diagnosis_fusion", {})
            neural_mapping = (
                fusion.get("neural_probabilities", {})
                if isinstance(fusion, Mapping)
                else {}
            )
            neural_probabilities = None
            if isinstance(neural_mapping, Mapping) and all(
                label in neural_mapping for label in self.label_names
            ):
                neural_probabilities = np.asarray(
                    [float(neural_mapping[label]) for label in self.label_names],
                    dtype=float,
                )
            scope = (
                "bayesian_component"
                if self.pipeline.causal_model.neural_weight >= 1.0 - 1e-12
                and neural_probabilities is not None
                else "reported_fused_posterior"
            )
            result = self.pipeline.diagnosis_agent.counterfactual_engine.search(
                states,
                target_name,
                neural_probabilities=neural_probabilities,
                posterior_scope=scope,
            )
        except Exception as exc:  # noqa: BLE001 - auxiliary guidance must not abort a case
            guidance["reason"] = f"counterfactual_search_error:{type(exc).__name__}"
            guidance["error"] = str(exc)
            return guidance

        guidance.update(
            {
                "found": bool(result.found),
                "feature": result.feature,
                "old_state": result.old_state,
                "new_state": result.new_state,
                "edit_distance": int(result.edit_distance),
                "scope": result.scope,
                "message": result.message,
            }
        )
        feature_changing = (
            bool(result.found)
            and result.feature is not None
            and result.new_state is not None
            and int(result.edit_distance) > 0
        )
        if feature_changing:
            guidance["available"] = True
            guidance["reason"] = "target_achieving_feature_edit"
        elif result.found:
            guidance["reason"] = "target_already_satisfied_no_edit"
        else:
            guidance["reason"] = "no_target_achieving_single_feature_edit"
        return guidance

    def _contains_target_claim(self, report: str, target_label: str) -> bool:
        extractor = self.claim_extractor
        if extractor is None:
            return ClaimOracle(target_label).contains_target_claim(report)
        if hasattr(extractor, "contains_target_claim"):
            return bool(extractor.contains_target_claim(report))
        if hasattr(extractor, "contains"):
            return bool(extractor.contains(report, target_label))
        if hasattr(extractor, "extract"):
            claims = extractor.extract(report)
            if hasattr(claims, "target_claim_present"):
                return bool(claims.target_claim_present)
            if isinstance(claims, Mapping):
                claims = claims.keys()
            return str(target_label) in {str(value) for value in claims}
        if callable(extractor):
            return bool(extractor(report, target_label))
        raise TypeError("claim_extractor must be callable or expose contains/extract")

    @staticmethod
    def _apply_release_decision(
        baseline_released: bool,
        evaluator: Any,
        signal: np.ndarray,
        output: Any,
        snapshot: VictimSnapshot,
        sample_rate_hz: float,
        clean_snapshot: VictimSnapshot | None,
        context: Mapping[str, Any],
    ) -> tuple[bool, float, dict[str, Any]]:
        if hasattr(evaluator, "decide"):
            decision_context = {
                **context,
                "waveform_mv": signal,
                "sample_rate_hz": sample_rate_hz,
                "clean_snapshot": clean_snapshot,
            }
            decision = evaluator.decide(snapshot, **decision_context)
        elif hasattr(evaluator, "evaluate"):
            decision = evaluator.evaluate(signal, output, clean_snapshot=clean_snapshot)
        elif callable(evaluator):
            decision = evaluator(signal, output, clean_snapshot)
        else:
            raise TypeError("Defense evaluator must be callable or expose decide/evaluate")
        if isinstance(decision, ReleaseDecision):
            return decision.released, float(decision.score), decision.to_dict()
        if isinstance(decision, Mapping):
            if "released" not in decision:
                raise ValueError("Defense decision mapping must contain released")
            return (
                bool(decision["released"]),
                float(decision.get("score", 0.0)),
                dict(decision),
            )
        if isinstance(decision, (bool, np.bool_)):
            return bool(decision), 0.0, {"released": bool(decision)}
        if decision is None:
            return baseline_released, 0.0, {"released": baseline_released}
        raise TypeError("Unsupported defense decision type")

    def evaluate(
        self,
        signal: Tensor | np.ndarray,
        target_label: str | int,
        *,
        defense: str | DefenseKind = DefenseKind.V1,
        clean_snapshot: VictimSnapshot | None = None,
    ) -> VictimSnapshot:
        waveform = self._as_waveform_batch(signal)[0].detach().cpu().numpy()
        target_index = self.target_index(target_label)
        target_name = self.label_names[target_index]
        defense_kind = DefenseKind(defense)
        evaluator = self.defense_evaluators.get(defense_kind)
        if defense_kind is not DefenseKind.V1 and evaluator is None:
            raise ValueError(
                f"Defense {defense_kind.value} requires a configured release verifier"
            )
        output = self.pipeline.run(
            waveform,
            self.query,
            source_rate=self.source_rate,
            history=self.history,
            patient_id=self.patient_id,
            counterfactual_target=None,
        )
        counterfactual_guidance = self._target_counterfactual_guidance(output, target_name)
        released = not bool(output.abstained)
        defense_score = max(
            float(output.hallucination_score), float(output.unsupported_claim_score)
        )
        decision_metadata: dict[str, Any] = {
            "defense": defense_kind.value,
            "released": released,
            "source": "response_agent",
        }
        report = str(output.rationale).strip()
        target_claim_present = self._contains_target_claim(report, target_name)
        snapshot = VictimSnapshot(
            diagnosis=str(output.diagnosis),
            probabilities={str(key): float(value) for key, value in output.probabilities.items()},
            latent=np.asarray(list(output.continuous_biomarkers.values()), dtype=float),
            discrete_latent=np.asarray(list(output.discrete_biomarkers.values()), dtype=int),
            retrieved_fact_ids=tuple(fact.fact_id for fact in output.retrieved_facts),
            report=report,
            hallucination_score=float(output.hallucination_score),
            unsupported_claim_score=float(output.unsupported_claim_score),
            used_fallback=bool(output.used_fallback),
            abstained=bool(output.abstained),
            released=bool(released),
            target_claim_present=target_claim_present,
            defense_score=float(defense_score),
            metadata={
                "defense_decision": decision_metadata,
                "pipeline_metadata": output.metadata,
                "target_label": target_name,
                "counterfactual_guidance": counterfactual_guidance,
                "counterfactual": None
                if output.counterfactual is None
                else {
                    "found": output.counterfactual.found,
                    "feature": output.counterfactual.feature,
                    "old_state": output.counterfactual.old_state,
                    "new_state": output.counterfactual.new_state,
                    "scope": output.counterfactual.scope,
                },
                "retrieved_fact_scores": {
                    fact.fact_id: float(fact.score) for fact in output.retrieved_facts
                },
                "source_rate": self.source_rate,
                "victim_scope": "internal_care_without_external_expert",
            },
        )
        if evaluator is not None:
            context = dict(self.defense_context)
            context.setdefault("patient_id", self.patient_id or "")
            context.setdefault("history", None)
            released, defense_score, decision_metadata = self._apply_release_decision(
                released,
                evaluator,
                waveform,
                output,
                snapshot,
                self.sample_rate_hz,
                clean_snapshot,
                context,
            )
            snapshot.released = bool(released)
            snapshot.defense_score = float(defense_score)
            snapshot.metadata["defense_decision"] = decision_metadata
        return snapshot

    def counterfactual_guidance(
        self, clean_snapshot: VictimSnapshot
    ) -> tuple[int, int] | None:
        values = clean_snapshot.metadata.get("counterfactual_guidance")
        if (
            not isinstance(values, Mapping)
            or not bool(values.get("available"))
            or values.get("feature") is None
            or values.get("new_state") is None
        ):
            return None
        feature = str(values["feature"])
        try:
            feature_index = self.pipeline.causal_model.feature_names.index(feature)
        except ValueError:
            return None
        return feature_index, int(values["new_state"])

    @property
    def discretizer_thresholds(self) -> np.ndarray:
        thresholds = self.pipeline.discretizer.thresholds_
        if thresholds is None:
            raise RuntimeError("Victim discretizer is not fitted")
        return np.asarray(thresholds, dtype=np.float32)


class BudgetedVictim:
    """Counts complete-system submissions and enforces a hard query budget."""

    def __init__(self, victim: Any, budget: int):
        if budget < 1:
            raise ValueError("Query budget must be positive")
        self.victim = victim
        self.budget = int(budget)
        self.queries_used = 0
        self.queries_completed = 0
        self.queries_errored = 0

    @property
    def remaining(self) -> int:
        return self.budget - self.queries_used

    def submit(self, *args, **kwargs) -> VictimSnapshot:
        if self.remaining <= 0:
            raise QueryBudgetExceeded("Complete-system query budget is exhausted")
        # Charge at the call boundary: malformed inputs and downstream pipeline
        # errors cannot be retried for free after reaching the victim.
        self.queries_used += 1
        try:
            result = self.victim.evaluate(*args, **kwargs)
        except Exception:
            self.queries_errored += 1
            raise
        self.queries_completed += 1
        return result
