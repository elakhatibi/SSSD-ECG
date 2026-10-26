from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .types import CausalDriver, RetrievedFact

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


@dataclass(slots=True)
class KnowledgeDocument:
    doc_id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TfidfEncoder:
    """Dependency-light unigram/bigram TF-IDF encoder for offline use."""

    def __init__(self):
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    @staticmethod
    def _terms(text: str) -> list[str]:
        words = tokenize(text)
        return words + [f"{a}_{b}" for a, b in pairwise(words)]

    def fit(self, texts: Sequence[str]) -> TfidfEncoder:
        if not texts:
            raise ValueError("Cannot fit an empty knowledge index")
        terms = [set(self._terms(text)) for text in texts]
        vocabulary = sorted(set().union(*terms))
        self.vocabulary = {term: i for i, term in enumerate(vocabulary)}
        document_frequency = np.zeros(len(vocabulary), dtype=float)
        for row in terms:
            for term in row:
                document_frequency[self.vocabulary[term]] += 1
        self.idf = np.log((1 + len(texts)) / (1 + document_frequency)) + 1.0
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("TF-IDF encoder has not been fitted")
        matrix = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float32)
        for row_index, text in enumerate(texts):
            terms = self._terms(text)
            counts: dict[int, int] = {}
            for term in terms:
                index = self.vocabulary.get(term)
                if index is not None:
                    counts[index] = counts.get(index, 0) + 1
            if counts:
                denominator = len(terms)
                for index, count in counts.items():
                    matrix[row_index, index] = (
                        (1.0 + math.log(count)) * self.idf[index] / denominator
                    )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms > 0, norms, 1.0)


class CausalRetriever:
    """Offline evidence index queried with clinician text plus causal state."""

    def __init__(
        self,
        backend: str = "tfidf",
        sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        metadata_weight: float = 0.2,
    ):
        if backend not in {"tfidf", "sentence-transformer"}:
            raise ValueError("backend must be tfidf or sentence-transformer")
        if not 0 <= metadata_weight <= 1:
            raise ValueError("metadata_weight must be in [0, 1]")
        self.backend = backend
        self.sentence_model = sentence_model
        self.metadata_weight = float(metadata_weight)
        self.documents: list[KnowledgeDocument] = []
        self.embeddings: np.ndarray | None = None
        self._encoder: Any = None

    def _make_encoder(self):
        if self.backend == "tfidf":
            return TfidfEncoder()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Sentence-transformer retrieval requires `pip install 'care-ecg[rag]'`"
            ) from exc
        return SentenceTransformer(self.sentence_model)

    def fit(self, documents: Sequence[KnowledgeDocument]) -> CausalRetriever:
        if not documents:
            raise ValueError("At least one knowledge document is required")
        self.documents = list(documents)
        self._encoder = self._make_encoder()
        texts = [self._index_text(document) for document in self.documents]
        if self.backend == "tfidf":
            self._encoder.fit(texts)
            self.embeddings = self._encoder.transform(texts)
        else:
            self.embeddings = np.asarray(
                self._encoder.encode(texts, normalize_embeddings=True), dtype=np.float32
            )
        return self

    @staticmethod
    def _index_text(document: KnowledgeDocument) -> str:
        """Add auditable clinical metadata to the searchable representation."""

        values: list[str] = [document.text, document.source]
        for key in (
            "diagnostic_class",
            "diagnostic_superclass",
            "target_label",
            "source_label",
            "scp_code",
            "relation",
            "scope",
        ):
            value = document.metadata.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value)
        return " ".join(values)

    @staticmethod
    def _document_diagnosis_labels(
        document: KnowledgeDocument,
        diagnosis_labels: Sequence[str],
    ) -> set[str]:
        """Return CARE labels that metadata explicitly assigns to a document.

        ``source_label`` and ``target_label`` also describe non-diagnostic graph
        nodes (for example, the PTB-XL dataset or an SCP code).  Restricting
        those fields to the model's label vocabulary prevents them from being
        mistaken for diagnoses without trying to infer diagnoses from prose.
        """

        canonical = {str(label).strip().casefold(): str(label) for label in diagnosis_labels}
        values = [
            document.metadata.get(key)
            for key in (
                "diagnostic_class",
                "diagnostic_superclass",
                "target_label",
                "source_label",
            )
        ]
        return {
            canonical[value.strip().casefold()]
            for value in values
            if isinstance(value, str) and value.strip().casefold() in canonical
        }

    @classmethod
    def _document_diagnosis_score(
        cls, document: KnowledgeDocument, probabilities: dict[str, float]
    ) -> float:
        labels = cls._document_diagnosis_labels(document, list(probabilities))
        return max((probabilities.get(label, 0.0) for label in labels), default=0.0)

    def enrich_query(
        self,
        query: str,
        drivers: Sequence[CausalDriver],
        probabilities: dict[str, float],
    ) -> str:
        diagnoses = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
        driver_text = " ".join(f"{d.name} state {d.state}" for d in drivers)
        diagnosis_text = " ".join(name for name, _ in diagnoses)
        return f"{query} ECG causal factors {driver_text} diagnostic hypotheses {diagnosis_text}".strip()

    def retrieve(
        self,
        query: str,
        drivers: Sequence[CausalDriver] = (),
        probabilities: dict[str, float] | None = None,
        top_k: int = 5,
        min_score: float = 1e-8,
        diagnosed_label: str | None = None,
    ) -> list[RetrievedFact]:
        if self.embeddings is None or self._encoder is None:
            raise RuntimeError("Knowledge index has not been fitted")
        probabilities = dict(probabilities or {})
        if probabilities:
            values = np.asarray(list(probabilities.values()), dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
                raise ValueError("Diagnosis probabilities must be finite, non-negative, and nonzero")
            probabilities = {
                name: float(value / values.sum())
                for (name, _), value in zip(probabilities.items(), values)
            }
        enriched = self.enrich_query(query, drivers, probabilities)
        if self.backend == "tfidf":
            query_vector = self._encoder.transform([enriched])[0]
        else:
            query_vector = np.asarray(
                self._encoder.encode([enriched], normalize_embeddings=True)[0], dtype=np.float32
            )
        lexical_scores = self.embeddings @ query_vector
        structured_scores = np.asarray(
            [self._document_diagnosis_score(document, probabilities) for document in self.documents],
            dtype=np.float32,
        )
        scores = (
            (1.0 - self.metadata_weight) * lexical_scores
            + self.metadata_weight * structured_scores
        )
        selected_diagnosis: str | None = None
        if probabilities:
            if diagnosed_label is None:
                selected_diagnosis = max(probabilities, key=probabilities.get)
            else:
                label_lookup = {label.casefold(): label for label in probabilities}
                selected_diagnosis = label_lookup.get(str(diagnosed_label).strip().casefold())
                if selected_diagnosis is None:
                    raise ValueError("Diagnosed label must be represented in probabilities")
            eligible = np.asarray(
                [
                    not (
                        labels := self._document_diagnosis_labels(
                            document, list(probabilities)
                        )
                    )
                    or selected_diagnosis in labels
                    for document in self.documents
                ]
            )
            scores = np.where(eligible, scores, -np.inf)
        count = min(max(int(top_k), 0), len(self.documents))
        ranking = [
            int(index)
            for index in np.argsort(-scores, kind="stable")
            if float(scores[index]) >= min_score
        ][:count]
        return [
            RetrievedFact(
                fact_id=self.documents[i].doc_id,
                text=self.documents[i].text,
                source=self.documents[i].source,
                score=float(scores[i]),
                metadata={
                    **self.documents[i].metadata,
                    "retrieval_backend": self.backend,
                    "retrieval_lexical_score": float(lexical_scores[i]),
                    "retrieval_structured_score": float(structured_scores[i]),
                    "retrieval_metadata_weight": self.metadata_weight,
                    "retrieval_diagnosed_label": selected_diagnosis,
                    "retrieval_document_diagnosis_labels": sorted(
                        self._document_diagnosis_labels(
                            self.documents[i], list(probabilities)
                        )
                    ),
                },
            )
            for i in ranking
        ]

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "sentence_model": self.sentence_model,
            "metadata_weight": self.metadata_weight,
            "documents": [asdict(document) for document in self.documents],
        }

    @classmethod
    def from_dict(cls, values: dict) -> CausalRetriever:
        result = cls(
            values["backend"],
            values.get("sentence_model", "all-MiniLM-L6-v2"),
            values.get("metadata_weight", 0.2),
        )
        return result.fit([KnowledgeDocument(**value) for value in values["documents"]])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CausalRetriever:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_jsonl_documents(path: str | Path) -> list[KnowledgeDocument]:
    documents: list[KnowledgeDocument] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                documents.append(KnowledgeDocument(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid knowledge JSONL at line {line_number}") from exc
    return documents
