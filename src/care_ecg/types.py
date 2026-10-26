from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ECGRecord:
    signal: Any
    label: int | str | None = None
    patient_id: str | None = None
    timestamp: float | None = None
    record_id: str | None = None
    report: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CausalDriver:
    name: str
    state: int
    importance: float
    effect: float
    alternative_state: int | None = None
    path: list[str] = field(default_factory=list)
    scope: str = "reported_fused_posterior"


@dataclass(slots=True)
class CounterfactualResult:
    found: bool
    target_label: str
    feature: str | None
    old_state: int | None
    new_state: int | None
    posterior_before: dict[str, float]
    posterior_after: dict[str, float]
    edit_distance: int
    message: str
    scope: str = "reported_fused_posterior"


@dataclass(slots=True)
class RetrievedFact:
    fact_id: str
    text: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HistorySummary:
    mode: str
    reference_ids: list[str]
    delta: dict[str, float]
    summary: str


@dataclass(slots=True)
class DiagnosisState:
    probabilities: dict[str, float]
    predicted_label: str
    drivers: list[CausalDriver]
    evidence_paths: list[list[str]]
    counterfactual: CounterfactualResult | None = None
    causal_probabilities: dict[str, float] = field(default_factory=dict)
    neural_probabilities: dict[str, float] = field(default_factory=dict)
    neural_weight: float = 0.0
    attribution_scope: str = "reported_fused_posterior"


@dataclass(slots=True)
class ResponseResult:
    explanation: str
    hallucination_score: float
    used_fallback: bool
    raw_with_causal: str
    raw_rag_only: str | None = None
    unsupported_claim_score: float = 0.0
    abstained: bool = False


@dataclass(slots=True)
class PipelineOutput:
    query: str
    probabilities: dict[str, float]
    diagnosis: str
    rationale: str
    continuous_biomarkers: dict[str, float]
    discrete_biomarkers: dict[str, int]
    causal_drivers: list[CausalDriver]
    causal_paths: list[list[str]]
    retrieved_facts: list[RetrievedFact]
    hallucination_score: float
    used_fallback: bool
    raw_explanation_with_causal: str
    raw_explanation_rag_only: str | None = None
    unsupported_claim_score: float = 0.0
    abstained: bool = False
    history: HistorySummary | None = None
    counterfactual: CounterfactualResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
