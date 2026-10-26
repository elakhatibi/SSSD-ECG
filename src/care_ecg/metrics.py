from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher

import numpy as np

from .retrieval import tokenize
from .types import CausalDriver, RetrievedFact
from .verification import hallucination_risk, unsupported_claim_risk

NEGATION_TOKENS = {"deny", "denied", "denies", "no", "not", "without"}


def _plausible_fuzzy_candidate(alias: Sequence[str], window: Sequence[str]) -> bool:
    """Cheap high-recall filter before the comparatively expensive edit matcher."""

    if len(alias) == 1:
        left, right = alias[0], window[0]
        return left[:1] == right[:1] and abs(len(left) - len(right)) <= max(1, len(left) // 4)
    overlap = len(set(alias) & set(window)) / len(set(alias))
    return overlap >= 0.4


def classification_metrics(
    expected: Sequence[str | int], predicted: Sequence[str | int]
) -> dict[str, float]:
    if len(expected) != len(predicted) or not expected:
        raise ValueError("Expected and predicted labels must be non-empty and equal length")
    labels = sorted(set(expected) | set(predicted), key=str)
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(expected, predicted))
        fp = sum(a != label and b == label for a, b in zip(expected, predicted))
        fn = sum(a == label and b != label for a, b in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float(np.mean(np.asarray(expected) == np.asarray(predicted))),
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
    }


def lexical_cosine(left: str, right: str) -> float:
    a, b = Counter(tokenize(left)), Counter(tokenize(right))
    vocabulary = set(a) | set(b)
    if not vocabulary:
        return 0.0
    numerator = sum(a[word] * b[word] for word in vocabulary)
    norm_a = sum(value * value for value in a.values()) ** 0.5
    norm_b = sum(value * value for value in b.values()) ** 0.5
    return float(numerator / (norm_a * norm_b)) if norm_a and norm_b else 0.0


def causal_retrieval_coverage(explanation: str, drivers: Sequence[CausalDriver]) -> float:
    if not drivers:
        return 0.0
    explanation_tokens = set(tokenize(explanation))
    covered = sum(
        bool(set(tokenize(driver.name))) and set(tokenize(driver.name)).issubset(explanation_tokens)
        for driver in drivers
    )
    return covered / len(drivers)


def grounding_metrics(
    query: str,
    explanation: str,
    facts: Sequence[RetrievedFact],
    drivers: Sequence[CausalDriver] = (),
) -> dict[str, float]:
    evidence = " ".join(fact.text for fact in facts)
    claim_risk = unsupported_claim_risk(explanation, facts)
    attribution_available = bool(drivers)
    fused_attribution_available = attribution_available and all(
        driver.scope == "reported_fused_posterior" for driver in drivers
    )
    return {
        "crc": causal_retrieval_coverage(explanation, drivers),
        "causal_attribution_availability": float(attribution_available),
        "fused_causal_attribution_availability": float(fused_attribution_available),
        "groundedness": 1.0 - claim_risk,
        "context_relevance": lexical_cosine(explanation, query),
        "hallucination_rate": hallucination_risk(explanation, facts),
        "unsupported_claim_rate": claim_risk,
        "semantic_relevance": lexical_cosine(explanation, evidence),
    }


class SCPTextMapper:
    """Boundary- and negation-aware text-to-SCP mapping with ordered mentions."""

    def __init__(self, aliases: Mapping[str, Sequence[str]], fuzzy_threshold: float = 0.84):
        self.aliases = {
            code: [tokens for value in [code, *values] if (tokens := tokenize(value))]
            for code, values in aliases.items()
        }
        self.fuzzy_threshold = fuzzy_threshold

    def map(self, text: str) -> list[str]:
        words = tokenize(text)
        mentions: list[tuple[int, float, int, str]] = []
        for code_order, (code, aliases) in enumerate(self.aliases.items()):
            best: tuple[int, float] | None = None
            # Prefer exact clinical aliases and avoid fuzzy scans when the report
            # already names a code or description explicitly.
            for alias in aliases:
                width = len(alias)
                for start in range(len(words) - width + 1):
                    if words[start : start + width] != alias:
                        continue
                    if any(token in NEGATION_TOKENS for token in words[max(0, start - 3) : start]):
                        continue
                    if best is None or start < best[0]:
                        best = (start, 1.0)
            if best is not None:
                mentions.append((best[0], -best[1], code_order, code))
                continue
            for alias in aliases:
                width = len(alias)
                for start in range(len(words) - width + 1):
                    window = words[start : start + width]
                    if any(token in NEGATION_TOKENS for token in words[max(0, start - 3) : start]):
                        continue
                    fuzzy = 0.0
                    if (width > 1 or len(alias[0]) >= 5) and _plausible_fuzzy_candidate(
                        alias, window
                    ):
                        fuzzy = SequenceMatcher(None, " ".join(alias), " ".join(window)).ratio()
                    if fuzzy >= self.fuzzy_threshold and (
                        best is None or start < best[0] or (start == best[0] and fuzzy > best[1])
                    ):
                        best = (start, fuzzy)
            if best is not None:
                mentions.append((best[0], -best[1], code_order, code))
        return [code for _, _, _, code in sorted(mentions)]


def scp_text_metrics(
    expected: Sequence[str], texts: Sequence[str], mapper: SCPTextMapper
) -> dict[str, float]:
    """Report both strict first-mention and permissive any-mention SCP scores."""
    if len(expected) != len(texts) or not expected:
        raise ValueError("Expected labels and texts must be non-empty and equal length")
    mapped = [mapper.map(text) for text in texts]
    first = [codes[0] if codes else "__NO_SCP_MATCH__" for codes in mapped]
    first_metrics = classification_metrics(expected, first)
    hits = [label in codes for label, codes in zip(expected, mapped)]
    per_record_precision = [
        (1.0 / len(codes) if hit else 0.0) if codes else 0.0 for hit, codes in zip(hits, mapped)
    ]
    per_record_recall = [float(hit) for hit in hits]
    per_record_f1 = [
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
        for precision, recall in zip(per_record_precision, per_record_recall)
    ]
    return {
        **{f"scp_first_{name}": value for name, value in first_metrics.items()},
        "scp_any_accuracy": float(np.mean(hits)),
        "scp_any_precision": float(np.mean(per_record_precision)),
        "scp_any_recall": float(np.mean(per_record_recall)),
        "scp_any_f1": float(np.mean(per_record_f1)),
        "scp_mapping_coverage": float(np.mean([bool(codes) for codes in mapped])),
        "scp_codes_per_report": float(np.mean([len(codes) for codes in mapped])),
    }


def verifier_diagnostics(
    expected: Sequence[str | int],
    predicted: Sequence[str | int],
    used_fallback: Sequence[bool],
) -> dict[str, float]:
    if not (len(expected) == len(predicted) == len(used_fallback)) or not expected:
        raise ValueError("Verifier inputs must be non-empty and equal length")
    passed = np.logical_not(np.asarray(used_fallback, dtype=bool))
    correct = np.asarray(expected) == np.asarray(predicted)
    wrong = np.logical_not(correct)
    return {
        "verifier_pass_rate": float(passed.mean()),
        "wrong_diagnosis_count": float(wrong.sum()),
        "wrong_diagnosis_pass_rate": float(passed[wrong].mean()) if wrong.any() else 0.0,
    }
