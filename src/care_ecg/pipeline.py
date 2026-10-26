from __future__ import annotations

import json
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

import numpy as np
import torch

from .agents import CausalRAGAgent, DiagnosisAgent, HistoryAgent, HistoryEntry, ResponseAgent
from .causal import BayesianCausalModel
from .config import CAREConfig
from .data import apply_global_standardizer, fit_global_standardizer, preprocess_ecg
from .discretization import QuantileDiscretizer
from .expert import blend_external_expert
from .generation import HuggingFaceGenerator, OpenAIResponsesGenerator, TemplateGenerator
from .models import CausalVAE
from .retrieval import CausalRetriever, KnowledgeDocument, load_jsonl_documents
from .training import (
    CausalVAETrainer,
    latent_diagnostics,
    resolve_device,
    select_class_biases,
    select_probability_fusion,
    set_seed,
    stratified_holdout_indices,
)
from .types import PipelineOutput


class CAREECGPipeline:
    """End-to-end implementation of the two-stream CARE-ECG framework."""

    def __init__(
        self,
        config: CAREConfig,
        model: CausalVAE,
        discretizer: QuantileDiscretizer,
        causal_model: BayesianCausalModel,
        retriever: CausalRetriever,
        normalization_stats: tuple[float, float] | None = None,
    ):
        set_seed(config.seed)
        self.config = config
        self.model = model
        self.discretizer = discretizer
        self.causal_model = causal_model
        self.retriever = retriever
        self.normalization_stats = normalization_stats
        if config.data.normalization == "global" and normalization_stats is None:
            raise ValueError("Global normalization requires fitted training statistics")
        self.device = resolve_device(config.device)
        self.model.to(self.device).eval()
        if config.generation.backend == "openai":
            generator = OpenAIResponsesGenerator(
                config.generation.model,
                config.generation.temperature,
                config.generation.max_tokens,
            )
        elif config.generation.backend == "huggingface":
            generator = HuggingFaceGenerator(
                config.generation.model,
                config.generation.temperature,
                config.generation.max_tokens,
            )
        else:
            generator = TemplateGenerator()
        self.history_agent = HistoryAgent(causal_model.feature_names)
        self.diagnosis_agent = DiagnosisAgent(causal_model)
        self.rag_agent = CausalRAGAgent(
            retriever, config.retrieval.top_k, config.retrieval.min_score
        )
        self.response_agent = ResponseAgent(
            generator,
            config.generation.hallucination_threshold,
            config.generation.fuzzy_match_threshold,
            config.generation.fallback_to_rag_only,
        )

    @classmethod
    def build_model(cls, config: CAREConfig, num_classes: int = 0) -> CausalVAE:
        set_seed(config.seed)
        return CausalVAE(
            leads=config.data.leads,
            latent_dim=config.model.latent_dim,
            hidden_dim=config.model.hidden_dim,
            backbone=config.model.backbone,
            lstm_layers=config.model.lstm_layers,
            dropout=config.model.dropout,
            num_classes=num_classes,
            classifier_hidden_dim=config.model.classifier_hidden_dim,
            inception_kernel_size=config.model.inception_kernel_size,
            classifier_input=config.model.classifier_input,
        )

    @classmethod
    def train(
        cls,
        signals: np.ndarray,
        labels: np.ndarray,
        label_names: Sequence[str],
        documents: Sequence[KnowledgeDocument],
        config: CAREConfig | None = None,
        clinical_prior_edges: Sequence[tuple[str, str]] = (),
        validation_fraction: float = 0.1,
        progress=None,
        validation_signals: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
    ) -> tuple[CAREECGPipeline, object]:
        config = config or CAREConfig()
        config.validate()
        x = np.asarray(signals, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        if x.ndim != 3 or x.shape[1] != config.data.leads or len(x) != len(y):
            raise ValueError("Expected signals [N, configured leads, T] and labels [N]")
        if len(set(label_names)) != len(label_names):
            raise ValueError("label_names must be unique")
        if not len(y) or y.min() < 0 or y.max() >= len(label_names):
            raise ValueError("Labels must be contiguous indices into label_names")
        if validation_signals is None:
            train_indices, validation_indices = stratified_holdout_indices(
                y, validation_fraction, config.seed
            )
            train_x = x[train_indices]
            train_y = y[train_indices]
            validation_x = x[validation_indices] if len(validation_indices) else None
            validation_y = y[validation_indices] if len(validation_indices) else None
        else:
            validation_x = np.asarray(validation_signals, dtype=np.float32)
            if validation_x.ndim != 3 or validation_x.shape[1:] != x.shape[1:]:
                raise ValueError("Validation signals must match training [leads, samples]")
            if not len(validation_x):
                raise ValueError("Explicit validation split is empty")
            if validation_labels is None:
                raise ValueError("Explicit supervised validation requires validation_labels")
            validation_y = np.asarray(validation_labels, dtype=np.int64)
            if len(validation_y) != len(validation_x):
                raise ValueError("Validation labels must match validation signals")
            if validation_y.min() < 0 or validation_y.max() >= len(label_names):
                raise ValueError("Validation labels must be indices into label_names")
            train_x, train_y = x, y
        normalization_stats = None
        if config.data.normalization == "global":
            normalization_stats = fit_global_standardizer(train_x)
            train_x = apply_global_standardizer(train_x, *normalization_stats)
            if validation_x is not None:
                validation_x = apply_global_standardizer(validation_x, *normalization_stats)
        model = cls.build_model(config, num_classes=len(label_names))
        trainer = CausalVAETrainer(model, config)
        history = trainer.fit(
            train_x,
            validation_fraction=0.0,
            progress=progress,
            validation_signals=validation_x,
            labels=train_y,
            validation_labels=validation_y,
        )
        latent = trainer.encode(train_x)
        history.diagnostics.update(latent_diagnostics(latent))
        names = [f"z{i}" for i in range(config.model.latent_dim)]
        discretizer = QuantileDiscretizer(config.discretizer.bins, names)
        discrete = discretizer.fit_transform(latent)
        causal_model = BayesianCausalModel(label_names, names, config.discretizer.bins).fit(
            discrete,
            train_y,
            max_parents=config.graph.max_parents,
            equivalent_sample_size=config.graph.equivalent_sample_size,
            alpha=config.graph.alpha,
            clinical_prior_edges=clinical_prior_edges,
        )
        if validation_x is not None and validation_y is not None:
            validation_latent = trainer.encode(validation_x)
            validation_discrete = discretizer.transform(validation_latent)
            causal_probabilities = np.stack(
                [causal_model.posterior_array(states) for states in validation_discrete]
            )
            neural_probabilities = trainer.predict_proba(validation_x)
            neural_weight, fused_probabilities, fusion_diagnostics = select_probability_fusion(
                causal_probabilities,
                neural_probabilities,
                validation_y,
            )
            causal_model.neural_weight = neural_weight
            class_biases, _, calibration_diagnostics = select_class_biases(
                fused_probabilities,
                validation_y,
            )
            causal_model.class_biases = class_biases
            history.diagnostics["probability_fusion"] = fusion_diagnostics
            history.diagnostics["class_bias_calibration"] = calibration_diagnostics
        retriever = CausalRetriever(
            config.retrieval.backend,
            config.retrieval.sentence_model,
            config.retrieval.metadata_weight,
        ).fit(documents)
        return cls(
            config,
            model,
            discretizer,
            causal_model,
            retriever,
            normalization_stats,
        ), history

    @torch.inference_mode()
    def encode_with_neural(
        self, signal: np.ndarray, source_rate: int | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        latent, probabilities = self.encode_batch_with_neural(
            np.asarray(signal, dtype=np.float32)[None],
            source_rate,
            batch_size=1,
        )
        return latent[0], None if probabilities is None else probabilities[0]

    @torch.inference_mode()
    def encode_batch_with_neural(
        self,
        signals: np.ndarray,
        source_rate: int | None = None,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Encode a waveform batch without changing single-record inference semantics."""
        x = np.asarray(signals, dtype=np.float32)
        if x.ndim != 3 or not len(x):
            raise ValueError("ECG batch must have shape [records, leads, samples]")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if source_rate is not None:
            x = np.stack(
                [
                    preprocess_ecg(
                        signal,
                        source_rate,
                        self.config.data.sample_rate,
                        self.config.data.seconds,
                        self.config.data.leads,
                        "none"
                        if self.config.data.normalization == "global"
                        else self.config.data.normalization,
                    )
                    for signal in x
                ]
            )
        elif x.shape[1:] != (self.config.data.leads, self.config.data.samples):
            raise ValueError(
                "Without source_rate, signals must already have configured "
                "[records, leads, samples] shape"
            )
        if self.config.data.normalization == "global":
            x = apply_global_standardizer(x, *self.normalization_stats)

        latent_batches: list[np.ndarray] = []
        probability_batches: list[np.ndarray] = []
        has_classifier: bool | None = None
        for start in range(0, len(x), batch_size):
            tensor = torch.as_tensor(x[start : start + batch_size], device=self.device)
            z, _, _, logits = self.model.encode_and_classify(tensor, sample=False)
            latent_batches.append(z.cpu().numpy())
            batch_has_classifier = logits is not None
            if has_classifier is None:
                has_classifier = batch_has_classifier
            elif has_classifier != batch_has_classifier:  # pragma: no cover - model invariant
                raise RuntimeError("Classifier availability changed between batches")
            if logits is not None:
                probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
        latents = np.concatenate(latent_batches, axis=0)
        probabilities = (
            np.concatenate(probability_batches, axis=0) if probability_batches else None
        )
        return latents, probabilities

    def encode(self, signal: np.ndarray, source_rate: int | None = None) -> np.ndarray:
        latent, _ = self.encode_with_neural(signal, source_rate)
        return latent

    def run(
        self,
        signal: np.ndarray,
        query: str,
        source_rate: int | None = None,
        history: Sequence[HistoryEntry] = (),
        patient_id: str | None = None,
        counterfactual_target: str | int | None = None,
        external_expert_probabilities: Sequence[float] | np.ndarray | None = None,
        external_expert_weight: float | None = None,
        external_expert_source: str | None = None,
    ) -> PipelineOutput:
        if not query.strip():
            raise ValueError("Clinician query must not be empty")
        latent, neural_probabilities = self.encode_with_neural(signal, source_rate)
        return self.run_encoded(
            latent,
            query,
            history=history,
            patient_id=patient_id,
            counterfactual_target=counterfactual_target,
            neural_probabilities=neural_probabilities,
            external_expert_probabilities=external_expert_probabilities,
            external_expert_weight=external_expert_weight,
            external_expert_source=external_expert_source,
        )

    def run_encoded(
        self,
        latent: Sequence[float] | np.ndarray,
        query: str,
        history: Sequence[HistoryEntry] = (),
        patient_id: str | None = None,
        counterfactual_target: str | int | None = None,
        neural_probabilities: Sequence[float] | np.ndarray | None = None,
        external_expert_probabilities: Sequence[float] | np.ndarray | None = None,
        external_expert_weight: float | None = None,
        external_expert_source: str | None = None,
    ) -> PipelineOutput:
        """Run causal and report agents from a precomputed deterministic encoding."""
        if not query.strip():
            raise ValueError("Clinician query must not be empty")
        latent = np.asarray(latent, dtype=np.float32).reshape(-1)
        if latent.shape != (len(self.causal_model.feature_names),):
            raise ValueError("Latent vector does not match the saved causal feature set")
        if neural_probabilities is not None:
            neural_probabilities = np.asarray(neural_probabilities, dtype=float).reshape(-1)
        neural_probabilities, expert_audit = blend_external_expert(
            neural_probabilities,
            external_expert_probabilities,
            external_expert_weight,
            external_expert_source,
            self.causal_model.label_names,
        )
        discrete = self.discretizer.transform(latent)
        history_state = self.history_agent.analyze(latent, history, patient_id)
        diagnosis = self.diagnosis_agent.diagnose(
            discrete,
            counterfactual_target,
            neural_probabilities,
        )
        facts = self.rag_agent.retrieve(query, diagnosis)
        response = self.response_agent.respond(query, diagnosis, facts, history_state)
        diagnosis_fusion = {
            "neural_weight": diagnosis.neural_weight,
            "class_biases": [float(value) for value in self.causal_model.class_biases],
            "causal_probabilities": diagnosis.causal_probabilities,
            "neural_probabilities": diagnosis.neural_probabilities,
            "selected_on": "validation",
        }
        if expert_audit is not None:
            diagnosis_fusion["external_expert"] = expert_audit
        return PipelineOutput(
            query=query,
            probabilities=diagnosis.probabilities,
            diagnosis=diagnosis.predicted_label,
            rationale=response.explanation,
            continuous_biomarkers={
                name: float(value) for name, value in zip(self.causal_model.feature_names, latent)
            },
            discrete_biomarkers={
                name: int(value) for name, value in zip(self.causal_model.feature_names, discrete)
            },
            causal_drivers=diagnosis.drivers,
            causal_paths=diagnosis.evidence_paths,
            retrieved_facts=facts,
            hallucination_score=response.hallucination_score,
            used_fallback=response.used_fallback,
            raw_explanation_with_causal=response.raw_with_causal,
            raw_explanation_rag_only=response.raw_rag_only,
            unsupported_claim_score=response.unsupported_claim_score,
            abstained=response.abstained,
            history=history_state,
            counterfactual=diagnosis.counterfactual,
            metadata={
                "framework": "CARE-ECG",
                "generator": self.config.generation.backend,
                "diagnosis_fusion": diagnosis_fusion,
                "causal_attribution": {
                    "scope": diagnosis.attribution_scope,
                    "available": bool(diagnosis.drivers),
                    "component_available": bool(diagnosis.drivers),
                    "fused_available": bool(diagnosis.drivers)
                    and diagnosis.attribution_scope == "reported_fused_posterior",
                    "causal_component_weight": 1.0 - diagnosis.neural_weight,
                    "epistemic_status": (
                        "observational latent-state sensitivity; not a physiological intervention"
                    ),
                },
                "warning": "Research software; not a medical device or clinical diagnosis.",
            },
        )

    def make_history_entries(
        self,
        signals: np.ndarray,
        record_ids: Sequence[str] | None = None,
        patient_ids: Sequence[str] | None = None,
        timestamps: Sequence[float] | None = None,
        source_rate: int | None = None,
    ) -> list[HistoryEntry]:
        x = np.asarray(signals, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError("Historical signals must have shape [records, leads, samples]")
        count = len(x)
        for name, values in (
            ("record_ids", record_ids),
            ("patient_ids", patient_ids),
            ("timestamps", timestamps),
        ):
            if values is not None and len(values) != count:
                raise ValueError(f"{name} length does not match historical signals")
        return [
            HistoryEntry(
                latent=self.encode(signal, source_rate),
                record_id=str(record_ids[index]) if record_ids is not None else str(index),
                patient_id=str(patient_ids[index]) if patient_ids is not None else None,
                timestamp=float(timestamps[index]) if timestamps is not None else None,
            )
            for index, signal in enumerate(x)
        ]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(directory / "config.yaml")
        torch.save(
            {"state_dict": self.model.state_dict(), "model_config": self.model.init_config},
            directory / "causal_vae.pt",
        )
        self.discretizer.save(directory / "discretizer.json")
        self.causal_model.save(directory / "causal_graph.json")
        self.retriever.save(directory / "knowledge_index.json")
        if self.normalization_stats is not None:
            mean, scale = self.normalization_stats
            (directory / "normalization.json").write_text(
                json.dumps({"policy": "global", "mean": mean, "scale": scale}, indent=2),
                encoding="utf-8",
            )
        (directory / "manifest.json").write_text(
            json.dumps({"format_version": 1, "framework": "CARE-ECG"}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls, directory: str | Path, config_override: CAREConfig | None = None
    ) -> CAREECGPipeline:
        directory = Path(directory)
        config = config_override or CAREConfig.from_yaml(directory / "config.yaml")
        checkpoint = torch.load(directory / "causal_vae.pt", map_location="cpu", weights_only=True)
        model = CausalVAE(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        normalization_stats = None
        normalization_path = directory / "normalization.json"
        if normalization_path.exists():
            values = json.loads(normalization_path.read_text(encoding="utf-8"))
            normalization_stats = (float(values["mean"]), float(values["scale"]))
        return cls(
            config,
            model,
            QuantileDiscretizer.load(directory / "discretizer.json"),
            BayesianCausalModel.load(directory / "causal_graph.json"),
            CausalRetriever.load(directory / "knowledge_index.json"),
            normalization_stats,
        )


def bundled_demo_documents() -> list[KnowledgeDocument]:
    resource = resources.files("care_ecg").joinpath("assets/demo_knowledge.jsonl")
    with resources.as_file(resource) as path:
        return load_jsonl_documents(path)
