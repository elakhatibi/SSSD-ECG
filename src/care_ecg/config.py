from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    leads: int = 12
    sample_rate: int = 500
    seconds: int = 10
    normalization: str = "robust"

    @property
    def samples(self) -> int:
        return self.sample_rate * self.seconds


@dataclass(slots=True)
class ModelConfig:
    latent_dim: int = 16
    hidden_dim: int = 128
    backbone: str = "resnet"
    lstm_layers: int = 2
    dropout: float = 0.1
    classifier_hidden_dim: int = 64
    inception_kernel_size: int = 15
    classifier_input: str = "latent"


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta: float = 1e-3
    beta_warmup_epochs: int = 10
    classification_weight: float = 1.0
    class_weighting: str = "balanced"
    label_smoothing: float = 0.05
    gradient_clip: float = 1.0
    augmentation_probability: float = 0.5
    noise_std: float = 0.01
    amplitude_scale: float = 0.05
    time_shift_fraction: float = 0.01
    lead_dropout_probability: float = 0.01
    baseline_wander_std: float = 0.02
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    restore_best_weights: bool = True
    lr_scheduler: str = "none"
    minimum_learning_rate: float = 1e-5


@dataclass(slots=True)
class DiscretizerConfig:
    bins: int = 3


@dataclass(slots=True)
class GraphConfig:
    max_parents: int = 3
    equivalent_sample_size: float = 5.0
    alpha: float = 1.0


@dataclass(slots=True)
class RetrievalConfig:
    backend: str = "tfidf"
    top_k: int = 5
    min_score: float = 1e-8
    sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    metadata_weight: float = 0.2


@dataclass(slots=True)
class GenerationConfig:
    backend: str = "template"
    model: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: int = 600
    hallucination_threshold: float = 0.5
    fuzzy_match_threshold: float = 0.32
    fallback_to_rag_only: bool = True


@dataclass(slots=True)
class CAREConfig:
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    discretizer: DiscretizerConfig = field(default_factory=DiscretizerConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def validate(self) -> None:
        if self.data.leads < 1 or self.data.sample_rate < 1 or self.data.seconds < 1:
            raise ValueError("ECG dimensions and sample rate must be positive")
        if self.data.normalization not in {"robust", "zscore", "global", "none"}:
            raise ValueError("data.normalization must be robust, zscore, global, or none")
        if (
            self.model.latent_dim < 1
            or self.model.hidden_dim < 2
            or self.model.classifier_hidden_dim < 1
        ):
            raise ValueError("Model dimensions must be positive")
        if self.model.backbone == "bilstm" and self.model.hidden_dim % 2:
            raise ValueError("BiLSTM hidden_dim must be even")
        if self.model.backbone not in {"bilstm", "resnet", "inceptiontime", "mamba"}:
            raise ValueError(
                "model.backbone must be 'bilstm', 'resnet', 'inceptiontime', or 'mamba'"
            )
        if self.model.lstm_layers < 1 or not 0 <= self.model.dropout < 1:
            raise ValueError("Model layers must be positive and dropout must be in [0, 1)")
        if self.model.backbone == "inceptiontime":
            if self.model.hidden_dim % 4:
                raise ValueError("InceptionTime hidden_dim must be divisible by four")
            if self.model.lstm_layers < 3:
                raise ValueError("InceptionTime requires at least three temporal layers")
        if self.model.inception_kernel_size < 7 or self.model.inception_kernel_size % 2 == 0:
            raise ValueError("model.inception_kernel_size must be an odd integer of at least 7")
        if self.model.classifier_input not in {"latent", "encoder", "hybrid"}:
            raise ValueError("model.classifier_input must be latent, encoder, or hybrid")
        if self.training.epochs < 1 or self.training.batch_size < 1:
            raise ValueError("Training epochs and batch size must be positive")
        if (
            self.training.learning_rate <= 0
            or self.training.weight_decay < 0
            or self.training.beta < 0
            or self.training.classification_weight < 0
        ):
            raise ValueError("Learning rate must be positive and beta must be non-negative")
        if self.training.beta_warmup_epochs < 0 or self.training.gradient_clip <= 0:
            raise ValueError("Invalid warm-up or gradient clipping configuration")
        if self.training.class_weighting not in {"none", "balanced"}:
            raise ValueError("training.class_weighting must be 'none' or 'balanced'")
        if not 0 <= self.training.label_smoothing < 1:
            raise ValueError("training.label_smoothing must be in [0, 1)")
        for name in (
            "augmentation_probability",
            "lead_dropout_probability",
        ):
            if not 0 <= getattr(self.training, name) <= 1:
                raise ValueError(f"training.{name} must be in [0, 1]")
        for name in (
            "noise_std",
            "amplitude_scale",
            "time_shift_fraction",
            "baseline_wander_std",
            "early_stopping_min_delta",
        ):
            if getattr(self.training, name) < 0:
                raise ValueError(f"training.{name} must be non-negative")
        if self.training.time_shift_fraction > 0.5:
            raise ValueError("training.time_shift_fraction must not exceed 0.5")
        if self.training.early_stopping_patience < 0:
            raise ValueError("training.early_stopping_patience must be non-negative")
        if self.training.lr_scheduler not in {"none", "cosine"}:
            raise ValueError("training.lr_scheduler must be 'none' or 'cosine'")
        if self.training.minimum_learning_rate < 0:
            raise ValueError("training.minimum_learning_rate must be non-negative")
        if (
            self.training.lr_scheduler == "cosine"
            and self.training.minimum_learning_rate > self.training.learning_rate
        ):
            raise ValueError(
                "Cosine minimum_learning_rate must not exceed training.learning_rate"
            )
        if self.discretizer.bins < 2:
            raise ValueError("At least two quantile bins are required")
        if self.graph.max_parents < 0 or self.graph.equivalent_sample_size <= 0:
            raise ValueError("Invalid graph structure-learning configuration")
        if self.graph.alpha <= 0:
            raise ValueError("Graph smoothing alpha must be positive")
        if self.retrieval.top_k < 1:
            raise ValueError("retrieval.top_k must be positive")
        if not -1 <= self.retrieval.min_score <= 1:
            raise ValueError("retrieval.min_score must be in [-1, 1]")
        if not 0 <= self.retrieval.metadata_weight <= 1:
            raise ValueError("retrieval.metadata_weight must be in [0, 1]")
        if not 0 <= self.generation.hallucination_threshold <= 1:
            raise ValueError("Hallucination threshold must be in [0, 1]")
        if self.generation.backend not in {"template", "openai", "huggingface"}:
            raise ValueError("generation.backend must be template, openai, or huggingface")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CAREConfig:
        known = {
            "data": DataConfig,
            "model": ModelConfig,
            "training": TrainingConfig,
            "discretizer": DiscretizerConfig,
            "graph": GraphConfig,
            "retrieval": RetrievalConfig,
            "generation": GenerationConfig,
        }
        root = {k: v for k, v in values.items() if k not in known}
        for key, kind in known.items():
            if key in values:
                root[key] = kind(**values[key])
        result = cls(**root)
        result.validate()
        return result

    @classmethod
    def from_yaml(cls, path: str | Path) -> CAREConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise TypeError("Configuration root must be a mapping")
        return cls.from_dict(values)

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
