from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .causal import BayesianCausalModel
from .counterfactual import CounterfactualEngine
from .generation import TextGenerator
from .retrieval import CausalRetriever
from .types import (
    DiagnosisState,
    HistorySummary,
    ResponseResult,
    RetrievedFact,
)
from .verification import hallucination_risk, unsupported_claim_risk


@dataclass(slots=True)
class HistoryEntry:
    latent: np.ndarray
    record_id: str
    patient_id: str | None = None
    timestamp: float | None = None


class HistoryAgent:
    def __init__(self, feature_names: Sequence[str]):
        self.feature_names = list(feature_names)

    def analyze(
        self,
        current_latent: np.ndarray,
        history: Sequence[HistoryEntry] = (),
        patient_id: str | None = None,
    ) -> HistorySummary:
        current = np.asarray(current_latent, dtype=float)
        if current.shape != (len(self.feature_names),):
            raise ValueError("Current latent shape does not match feature names")
        if not history:
            return HistorySummary(
                "none", [], {}, "No longitudinal or surrogate history was available."
            )
        patient_entries = [item for item in history if patient_id and item.patient_id == patient_id]
        if patient_entries:
            reference = max(
                patient_entries,
                key=lambda item: item.timestamp if item.timestamp is not None else float("-inf"),
            )
            mode = "longitudinal"
        else:
            reference = min(
                history,
                key=lambda item: float(
                    np.linalg.norm(current - np.asarray(item.latent, dtype=float))
                ),
            )
            mode = "surrogate"
        delta_values = current - np.asarray(reference.latent, dtype=float)
        delta = {name: float(value) for name, value in zip(self.feature_names, delta_values)}
        strongest = sorted(delta.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        shifts = ", ".join(f"{name} {value:+.3f}" for name, value in strongest)
        return HistorySummary(
            mode=mode,
            reference_ids=[reference.record_id],
            delta=delta,
            summary=f"{mode.capitalize()} reference {reference.record_id}; largest latent shifts: {shifts}.",
        )


class DiagnosisAgent:
    def __init__(self, causal_model: BayesianCausalModel, driver_count: int = 5):
        self.causal_model = causal_model
        self.driver_count = driver_count
        self.counterfactual_engine = CounterfactualEngine(causal_model)

    def diagnose(
        self,
        discrete_latent: np.ndarray,
        counterfactual_target: str | int | None = None,
        neural_probabilities: np.ndarray | None = None,
    ) -> DiagnosisState:
        causal_probabilities = self.causal_model.posterior(discrete_latent)
        fused = self.causal_model.fused_posterior_array(discrete_latent, neural_probabilities)
        probabilities = {
            label: float(fused[index]) for index, label in enumerate(self.causal_model.label_names)
        }
        predicted_label = max(probabilities, key=probabilities.get)
        drivers = self.causal_model.rank_drivers(
            discrete_latent,
            self.driver_count,
            target_label=predicted_label,
            neural_probabilities=neural_probabilities,
        )
        attribution_scope = "reported_fused_posterior"
        if (
            not drivers
            and neural_probabilities is not None
            and self.causal_model.neural_weight >= 1.0 - 1e-12
        ):
            drivers = self.causal_model.rank_drivers(
                discrete_latent,
                self.driver_count,
                target_label=predicted_label,
                posterior_scope="bayesian_component",
            )
            attribution_scope = "bayesian_component" if drivers else "unavailable"
        counterfactual = None
        if counterfactual_target is not None:
            counterfactual_scope = (
                "bayesian_component"
                if self.causal_model.neural_weight >= 1.0 - 1e-12
                and neural_probabilities is not None
                else "reported_fused_posterior"
            )
            counterfactual = self.counterfactual_engine.search(
                discrete_latent,
                counterfactual_target,
                neural_probabilities=neural_probabilities,
                posterior_scope=counterfactual_scope,
            )
        return DiagnosisState(
            probabilities=probabilities,
            predicted_label=predicted_label,
            drivers=drivers,
            evidence_paths=self.causal_model.evidence_paths(drivers),
            counterfactual=counterfactual,
            causal_probabilities=causal_probabilities,
            neural_probabilities={
                label: float(neural_probabilities[index])
                for index, label in enumerate(self.causal_model.label_names)
            }
            if neural_probabilities is not None
            else {},
            neural_weight=self.causal_model.neural_weight
            if neural_probabilities is not None
            else 0.0,
            attribution_scope=attribution_scope,
        )


class ResponseAgent:
    def __init__(
        self,
        generator: TextGenerator,
        hallucination_threshold: float = 0.5,
        fuzzy_match_threshold: float = 0.32,
        fallback_to_rag_only: bool = True,
    ):
        self.generator = generator
        self.hallucination_threshold = hallucination_threshold
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.fallback_to_rag_only = fallback_to_rag_only

    @staticmethod
    def _prompt(
        query: str,
        diagnosis: DiagnosisState,
        facts: Sequence[RetrievedFact],
        history: HistorySummary,
        include_causal: bool,
    ) -> str:
        causal = (
            ", ".join(
                f"{driver.name}=state-{driver.state} (influence {driver.importance:.3f})"
                for driver in diagnosis.drivers
            )
            if include_causal and diagnosis.drivers
            else "None"
        )
        attribution_scope = diagnosis.attribution_scope.replace("_", " ")
        probabilities = ", ".join(
            f"{label}={probability:.3f}" for label, probability in diagnosis.probabilities.items()
        )
        evidence = (
            "\n".join(
                f"[Fact {index}] {fact.text} (source: {fact.source})"
                for index, fact in enumerate(facts, start=1)
            )
            or "No retrieved facts."
        )
        counterfactual = (
            diagnosis.counterfactual.message if diagnosis.counterfactual else "Not requested."
        )
        return (
            f"Patient Query: {query}\n"
            f"Temporal Context: {history.summary}\n"
            f"Predicted Diagnosis: {diagnosis.predicted_label}\n"
            f"Diagnosis Posterior: {probabilities}\n"
            f"Key Causal Factors (from VAE/Graph): {causal}\n"
            f"Attribution Scope: {attribution_scope}\n"
            f"Counterfactual Check: {counterfactual}\n"
            f"Retrieved Medical Facts (RAG):\n{evidence}\n"
            "State the model-predicted diagnosis exactly as supplied. Explain it clearly and "
            "medically grounded. Every clinical factual claim must cite its supporting fact tag "
            "(e.g., [Fact 1]). Do not infer unobserved waveform features."
        )

    @staticmethod
    def _with_diagnosis(explanation: str, diagnosis: DiagnosisState) -> str:
        header = f"Model-predicted diagnosis: {diagnosis.predicted_label}."
        return header + "\n" + explanation.strip()

    @staticmethod
    def _safe_evidence_only(diagnosis: DiagnosisState, facts: Sequence[RetrievedFact]) -> str:
        lines = [
            f"Model-predicted diagnosis: {diagnosis.predicted_label}.",
            "Evidence-only fallback:",
        ]
        if facts:
            lines.extend(
                f"- {fact.text} [Fact {index}]" for index, fact in enumerate(facts, start=1)
            )
        else:
            lines.append("Insufficient retrieved evidence is available for a grounded explanation.")
        lines.append(
            "The probabilistic output is a model finding and requires direct waveform and "
            "clinical review."
        )
        return "\n".join(lines)

    def respond(
        self,
        query: str,
        diagnosis: DiagnosisState,
        facts: Sequence[RetrievedFact],
        history: HistorySummary,
    ) -> ResponseResult:
        prompt = self._prompt(query, diagnosis, facts, history, include_causal=True)
        draft = self._with_diagnosis(self.generator.generate(prompt, facts), diagnosis)
        risk = hallucination_risk(draft, facts, self.fuzzy_match_threshold)
        claim_risk = unsupported_claim_risk(draft, facts, self.fuzzy_match_threshold)
        unsafe = risk > self.hallucination_threshold or claim_risk > 0
        if not unsafe or not self.fallback_to_rag_only:
            return ResponseResult(
                draft,
                risk,
                False,
                draft,
                unsupported_claim_score=claim_risk,
            )

        fallback_prompt = self._prompt(query, diagnosis, facts, history, include_causal=False)
        fallback = self._with_diagnosis(self.generator.generate(fallback_prompt, facts), diagnosis)
        fallback_risk = hallucination_risk(fallback, facts, self.fuzzy_match_threshold)
        fallback_claim_risk = unsupported_claim_risk(fallback, facts, self.fuzzy_match_threshold)
        abstained = fallback_risk > self.hallucination_threshold or fallback_claim_risk > 0
        explanation = self._safe_evidence_only(diagnosis, facts) if abstained else fallback
        final_risk = hallucination_risk(explanation, facts, self.fuzzy_match_threshold)
        final_claim_risk = unsupported_claim_risk(explanation, facts, self.fuzzy_match_threshold)
        explanation += "\n\n(Note: Fallback used due to high hallucination risk.)"
        return ResponseResult(
            explanation,
            final_risk,
            True,
            draft,
            fallback,
            unsupported_claim_score=final_claim_risk,
            abstained=abstained,
        )


class CausalRAGAgent:
    def __init__(self, retriever: CausalRetriever, top_k: int = 5, min_score: float = 1e-8):
        self.retriever = retriever
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, query: str, diagnosis: DiagnosisState) -> list[RetrievedFact]:
        return self.retriever.retrieve(
            query,
            diagnosis.drivers,
            diagnosis.probabilities,
            self.top_k,
            self.min_score,
            diagnosis.predicted_label,
        )
