from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher

from .retrieval import tokenize
from .types import RetrievedFact

FACT_TAG_PATTERN = re.compile(r"\[Fact\s+(\d+)\]", re.IGNORECASE)
NONCLINICAL_PREFIXES = (
    "model-predicted diagnosis:",
    "model posterior:",
    "model-derived causal drivers:",
    "bayesian-component latent sensitivity factors:",
    "reported-posterior latent sensitivity factors:",
    "model confidence for the stated",
    "question addressed:",
    "counterfactual sensitivity:",
    "temporal context:",
    "evidence-grounded assessment:",
    "evidence-only fallback:",
    "insufficient retrieved evidence",
    "the probabilistic output",
    "note: fallback used",
    "(note: fallback used",
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def _sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+(?!\[Fact\s+\d+\])|\n+", text)
        if part.strip()
    ]


def fuzzy_fact_match(fact: str, explanation: str, threshold: float = 0.32) -> bool:
    """Return whether the explanation reflects a retrieved fact.

    This is intentionally the paper's lightweight fuzzy gate, not a claim-level
    natural-language-inference guarantee.
    """
    fact_tokens = set(tokenize(fact))
    if not fact_tokens:
        return False
    best = 0.0
    for sentence in _sentences(explanation):
        sentence_tokens = set(tokenize(sentence))
        if not sentence_tokens:
            continue
        containment = len(fact_tokens & sentence_tokens) / len(fact_tokens)
        sequence = SequenceMatcher(
            None, " ".join(sorted(fact_tokens)), " ".join(sorted(sentence_tokens))
        ).ratio()
        best = max(best, 0.7 * containment + 0.3 * sequence)
    return best >= threshold


def hallucination_risk(
    explanation: str,
    facts: Sequence[RetrievedFact],
    fuzzy_threshold: float = 0.32,
) -> float:
    """Compute HR = 1 - mean(1[retrieved fact is reflected in explanation])."""
    if not facts:
        return 1.0
    matches = [fuzzy_fact_match(fact.text, explanation, fuzzy_threshold) for fact in facts]
    return float(1.0 - sum(matches) / len(matches))


def _claim_supported(claim: str, fact: str, threshold: float) -> bool:
    claim_without_tags = FACT_TAG_PATTERN.sub("", claim)
    claim_tokens = {token for token in tokenize(claim_without_tags) if token not in STOPWORDS}
    fact_tokens = {token for token in tokenize(fact) if token not in STOPWORDS}
    if not claim_tokens or not fact_tokens:
        return False
    precision = len(claim_tokens & fact_tokens) / len(claim_tokens)
    sequence = SequenceMatcher(
        None, " ".join(sorted(claim_tokens)), " ".join(sorted(fact_tokens))
    ).ratio()
    return 0.7 * precision + 0.3 * sequence >= threshold


def unsupported_claim_risk(
    explanation: str,
    facts: Sequence[RetrievedFact],
    fuzzy_threshold: float = 0.32,
) -> float:
    """Estimate the fraction of generated claims unsupported by their cited facts.

    This complements the paper's fact-recall HR, which cannot detect an extra fabricated
    claim when all retrieved facts are also repeated.
    """
    claims: list[bool] = []
    for sentence in _sentences(explanation):
        normalized = sentence.lstrip("- ").lower()
        if normalized.startswith(NONCLINICAL_PREFIXES):
            continue
        citations = [int(value) for value in FACT_TAG_PATTERN.findall(sentence)]
        supported = any(
            1 <= citation <= len(facts)
            and _claim_supported(sentence, facts[citation - 1].text, fuzzy_threshold)
            for citation in citations
        )
        claims.append(supported)
    if not claims:
        return 0.0
    return float(1.0 - sum(claims) / len(claims))
