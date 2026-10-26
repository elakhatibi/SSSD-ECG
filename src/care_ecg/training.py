from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .config import CAREConfig
from .data import TorchECGDataset
from .models import CausalVAE


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(slots=True)
class TrainingHistory:
    train: list[dict[str, float]]
    validation: list[dict[str, float]]
    diagnostics: dict[str, object] = field(default_factory=dict)


def latent_diagnostics(values: np.ndarray, variance_threshold: float = 1e-4) -> dict[str, object]:
    """Summarize posterior-mean activity so collapsed runs are visible."""
    latent = np.asarray(values, dtype=float)
    if latent.ndim != 2 or not len(latent):
        raise ValueError("Latent values must have shape [records, dimensions]")
    if variance_threshold < 0:
        raise ValueError("variance_threshold must be non-negative")
    variances = np.var(latent, axis=0)
    active = variances > variance_threshold
    return {
        "records": len(latent),
        "latent_dimensions": int(latent.shape[1]),
        "variance_threshold": float(variance_threshold),
        "active_units": int(active.sum()),
        "active_fraction": float(active.mean()),
        "feature_variances": [float(value) for value in variances],
    }


def stratified_holdout_indices(
    labels: np.ndarray, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic class-stratified holdout without emptying a class."""
    y = np.asarray(labels).reshape(-1)
    if not len(y):
        raise ValueError("Labels must not be empty")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction == 0:
        return np.arange(len(y), dtype=np.int64), np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    validation: list[int] = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        count = min(max(round(len(indices) * validation_fraction), 1), len(indices) - 1)
        if count > 0:
            validation.extend(int(index) for index in indices[:count])
    validation_indices = np.asarray(sorted(validation), dtype=np.int64)
    mask = np.ones(len(y), dtype=bool)
    mask[validation_indices] = False
    training_indices = np.flatnonzero(mask).astype(np.int64)
    if not len(training_indices):
        raise ValueError("Training split is empty")
    return training_indices, validation_indices


def balanced_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Return inverse-frequency weights whose mean sample weight is one."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if not len(y) or y.min() < 0 or y.max() >= num_classes:
        raise ValueError("Labels must be valid class indices")
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training labels are missing classes: {missing}")
    weights = len(y) / (num_classes * counts)
    return weights.astype(np.float32)


def _normalize_probabilities(values: np.ndarray, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or not len(probabilities) or probabilities.shape[1] < 2:
        raise ValueError(f"{name} must have shape [records, classes] with at least two classes")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError(f"{name} rows must have positive mass")
    return probabilities / totals


def _probability_classification_metrics(
    probabilities: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    return _classification_metrics(probabilities.argmax(axis=1), labels, probabilities.shape[1])


def _classification_metrics(
    predicted: np.ndarray, labels: np.ndarray, num_classes: int
) -> dict[str, float]:
    confusion = np.bincount(labels * num_classes + predicted, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )
    values = confusion.astype(np.float64)
    true_positive = np.diag(values)
    precision = true_positive / np.maximum(values.sum(axis=0), 1.0)
    recall = true_positive / np.maximum(values.sum(axis=1), 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(true_positive.sum() / values.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


def apply_class_biases(
    probabilities: np.ndarray, class_biases: Sequence[float]
) -> np.ndarray:
    """Apply additive log-probability biases and return normalized probabilities."""
    normalized = _normalize_probabilities(probabilities, "probabilities")
    biases = np.asarray(class_biases, dtype=np.float64).reshape(-1)
    if biases.shape != (normalized.shape[1],) or not np.isfinite(biases).all():
        raise ValueError("class_biases must be finite with one value per class")
    if not np.any(biases):
        return normalized
    logits = np.log(np.clip(normalized, np.finfo(np.float64).tiny, None)) + biases
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def select_class_biases(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    anchor_class: int = 0,
    max_abs_bias: float = 2.0,
    step_sizes: Sequence[float] = (0.2, 0.05, 0.01),
    max_passes_per_step: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Select anchored class biases using validation labels only.

    This is a bounded coordinate search rather than an exhaustive Cartesian
    grid. The anchor bias remains zero, removing the otherwise unidentified
    common additive offset. Selection maximizes macro-F1, then accuracy, and
    finally favors the lowest-L1 bias vector when predictions tie.
    """
    normalized = _normalize_probabilities(probabilities, "probabilities")
    expected = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(expected) != len(normalized):
        raise ValueError("Labels must match probability rows")
    if expected.min() < 0 or expected.max() >= normalized.shape[1]:
        raise ValueError("Labels are outside the probability class range")
    if not isinstance(anchor_class, (int, np.integer)) or not 0 <= anchor_class < normalized.shape[1]:
        raise ValueError("anchor_class must be a valid class index")
    if not np.isfinite(max_abs_bias) or max_abs_bias <= 0:
        raise ValueError("max_abs_bias must be finite and positive")
    steps = tuple(float(step) for step in step_sizes)
    if not steps or not np.isfinite(steps).all() or any(step <= 0 for step in steps):
        raise ValueError("step_sizes must contain finite positive values")
    if not isinstance(max_passes_per_step, (int, np.integer)) or max_passes_per_step < 1:
        raise ValueError("max_passes_per_step must be a positive integer")

    log_probabilities = np.log(np.clip(normalized, np.finfo(np.float64).tiny, None))
    biases = np.zeros(normalized.shape[1], dtype=np.float64)
    initial_metrics = _classification_metrics(
        log_probabilities.argmax(axis=1), expected, normalized.shape[1]
    )
    selected_metrics = initial_metrics
    evaluations = 0
    trace: list[dict[str, object]] = []

    def score(candidate: np.ndarray) -> tuple[dict[str, float], tuple[object, ...]]:
        predicted = (log_probabilities + candidate).argmax(axis=1)
        metrics = _classification_metrics(predicted, expected, normalized.shape[1])
        # Remaining deterministic tie breakers do not alter the stated metric objective.
        key: tuple[object, ...] = (
            metrics["macro_f1"],
            metrics["accuracy"],
            -float(np.abs(candidate).sum()),
            tuple(-np.abs(candidate)),
            tuple(-candidate),
        )
        return metrics, key

    selected_metrics, selected_key = score(biases)
    for step in steps:
        grid = np.arange(-max_abs_bias, max_abs_bias + step / 2.0, step)
        grid = np.unique(np.clip(np.append(grid, [0.0]), -max_abs_bias, max_abs_bias))
        for pass_index in range(max_passes_per_step):
            pass_changed = False
            for class_index in range(normalized.shape[1]):
                if class_index == anchor_class:
                    continue
                coordinate_biases = biases.copy()
                coordinate_metrics = selected_metrics
                coordinate_key = selected_key
                for value in grid:
                    candidate = biases.copy()
                    candidate[class_index] = float(value)
                    metrics, key = score(candidate)
                    evaluations += 1
                    if key > coordinate_key:
                        coordinate_biases = candidate
                        coordinate_metrics = metrics
                        coordinate_key = key
                if not np.array_equal(coordinate_biases, biases):
                    previous = float(biases[class_index])
                    biases = coordinate_biases
                    selected_metrics = coordinate_metrics
                    selected_key = coordinate_key
                    pass_changed = True
                    trace.append(
                        {
                            "step_size": step,
                            "pass": pass_index + 1,
                            "class_index": class_index,
                            "previous_bias": previous,
                            "selected_bias": float(biases[class_index]),
                            "macro_f1": selected_metrics["macro_f1"],
                            "accuracy": selected_metrics["accuracy"],
                        }
                    )
            if not pass_changed:
                break

    calibrated = apply_class_biases(normalized, biases)
    diagnostics: dict[str, object] = {
        "selection_split": "validation",
        "method": "anchored_additive_log_probability_coordinate_search",
        "objective": "macro_f1",
        "tie_breakers": ["accuracy", "lower_l1_bias"],
        "anchor_class": int(anchor_class),
        "max_abs_bias": float(max_abs_bias),
        "step_sizes": list(steps),
        "max_passes_per_step": int(max_passes_per_step),
        "candidate_evaluations": evaluations,
        "initial_metrics": initial_metrics,
        "selected_metrics": selected_metrics,
        "selected_class_biases": [float(value) for value in biases],
        "trace": trace,
    }
    return biases, calibrated, diagnostics


def select_probability_fusion(
    bn_probabilities: np.ndarray,
    neural_probabilities: np.ndarray,
    labels: np.ndarray,
    candidate_weights: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[float, np.ndarray, dict[str, object]]:
    """Select a BN/neural convex blend using validation labels only.

    The returned weight is the neural-model coefficient. Selection maximizes
    macro-F1, then accuracy, then favors the lower neural weight so an equally
    performing causal BN remains the simpler choice.
    """
    bn = _normalize_probabilities(bn_probabilities, "bn_probabilities")
    neural = _normalize_probabilities(neural_probabilities, "neural_probabilities")
    if bn.shape != neural.shape:
        raise ValueError("BN and neural probabilities must have identical shapes")
    expected = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(expected) != len(bn):
        raise ValueError("Labels must match probability rows")
    if expected.min() < 0 or expected.max() >= bn.shape[1]:
        raise ValueError("Labels are outside the probability class range")
    weights = sorted({float(weight) for weight in candidate_weights})
    if not weights or not np.isfinite(weights).all() or weights[0] < 0 or weights[-1] > 1:
        raise ValueError("candidate_weights must contain finite values in [0, 1]")

    candidates: list[dict[str, float]] = []
    best_key: tuple[float, float, float] | None = None
    selected_weight = 0.0
    selected_probabilities = bn
    selected_metrics: dict[str, float] = {}
    for weight in weights:
        fused = (1.0 - weight) * bn + weight * neural
        metrics = _probability_classification_metrics(fused, expected)
        candidate = {"neural_weight": weight, **metrics}
        candidates.append(candidate)
        key = (metrics["macro_f1"], metrics["accuracy"], -weight)
        if best_key is None or key > best_key:
            best_key = key
            selected_weight = weight
            selected_probabilities = fused
            selected_metrics = metrics
    diagnostics: dict[str, object] = {
        "selection_split": "validation",
        "objective": "macro_f1",
        "tie_breakers": ["accuracy", "lower_neural_weight"],
        "selected_neural_weight": selected_weight,
        "selected_metrics": selected_metrics,
        "candidates": candidates,
    }
    return selected_weight, selected_probabilities, diagnostics


class _LabeledECGDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = np.asarray(signals, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        if self.signals.ndim != 3 or len(self.signals) != len(self.labels):
            raise ValueError("Labeled ECG data must have matching [records, leads, samples]")

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return (
            torch.as_tensor(self.signals[index], dtype=torch.float32),
            torch.as_tensor(self.labels[index], dtype=torch.long),
        )


class ECGAugmenter:
    """Mild morphology-preserving waveform perturbations for denoising training."""

    def __init__(self, config: CAREConfig):
        training = config.training
        self.probability = training.augmentation_probability
        self.noise_std = training.noise_std
        self.amplitude_scale = training.amplitude_scale
        self.time_shift_fraction = training.time_shift_fraction
        self.lead_dropout_probability = training.lead_dropout_probability
        self.baseline_wander_std = training.baseline_wander_std

    def __call__(self, batch: Tensor) -> Tensor:
        if self.probability == 0 or not len(batch):
            return batch
        result = batch.clone()
        active = torch.rand((len(batch), 1, 1), device=batch.device) < self.probability
        lead_scale = batch.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)

        if self.amplitude_scale:
            factor = 1.0 + (2.0 * torch.rand_like(active, dtype=batch.dtype) - 1.0) * (
                self.amplitude_scale
            )
            result = torch.where(active, result * factor, result)

        if self.noise_std:
            noise = torch.randn_like(result) * lead_scale * self.noise_std
            result = torch.where(active, result + noise, result)

        if self.baseline_wander_std:
            time = torch.linspace(0.0, 1.0, batch.shape[-1], device=batch.device, dtype=batch.dtype)
            cycles = 0.5 + 1.5 * torch.rand(
                (len(batch), 1, 1), device=batch.device, dtype=batch.dtype
            )
            phase = (
                2.0
                * torch.pi
                * torch.rand(
                    (len(batch), batch.shape[1], 1), device=batch.device, dtype=batch.dtype
                )
            )
            wander = torch.sin(2.0 * torch.pi * cycles * time + phase)
            wander = wander * lead_scale * self.baseline_wander_std
            result = torch.where(active, result + wander, result)

        max_shift = round(batch.shape[-1] * self.time_shift_fraction)
        if max_shift:
            shifted = result.clone()
            shifts = torch.randint(
                -max_shift,
                max_shift + 1,
                (len(batch),),
                device=batch.device,
            )
            for index, shift_tensor in enumerate(shifts):
                if not bool(active[index]):
                    continue
                shift = int(shift_tensor)
                shifted[index].zero_()
                if shift > 0:
                    shifted[index, :, shift:] = result[index, :, :-shift]
                elif shift < 0:
                    shifted[index, :, :shift] = result[index, :, -shift:]
                else:
                    shifted[index] = result[index]
            result = shifted

        if self.lead_dropout_probability:
            dropped = (
                torch.rand((len(batch), batch.shape[1], 1), device=batch.device)
                < self.lead_dropout_probability
            ) & active
            all_dropped = dropped.squeeze(-1).all(dim=1)
            dropped[all_dropped, 0, 0] = False
            result = result.masked_fill(dropped, 0.0)
        return result


class CausalVAETrainer:
    def __init__(self, model: CausalVAE, config: CAREConfig):
        self.model = model
        self.config = config
        self.device = resolve_device(config.device)
        self.model.to(self.device)
        self.augmenter = ECGAugmenter(config)
        self.class_weights: Tensor | None = None

    def _epoch(
        self,
        loader: DataLoader,
        beta: float,
        optimizer: torch.optim.Optimizer | None,
    ) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)
        totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
        examples = 0
        confusion = (
            torch.zeros((self.model.num_classes, self.model.num_classes), dtype=torch.int64)
            if self.model.num_classes
            else None
        )
        for loaded in loader:
            if isinstance(loaded, (tuple, list)):
                clean_batch, labels = loaded
                labels = labels.to(self.device)
            else:
                clean_batch, labels = loaded, None
            clean_batch = clean_batch.to(self.device)
            model_input = self.augmenter(clean_batch) if training else clean_batch
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                output = self.model(model_input)
                losses = self.model.loss(
                    output,
                    clean_batch,
                    beta,
                    labels=labels,
                    classification_weight=self.config.training.classification_weight,
                    class_weights=self.class_weights,
                    label_smoothing=self.config.training.label_smoothing,
                )
                if training:
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip
                    )
                    optimizer.step()
            examples += len(clean_batch)
            for name, loss in losses.items():
                totals.setdefault(name, 0.0)
                totals[name] += float(loss.detach()) * len(clean_batch)
            if labels is not None and confusion is not None and output.logits is not None:
                predicted = output.logits.detach().argmax(dim=1)
                indices = labels.detach() * self.model.num_classes + predicted
                confusion += torch.bincount(
                    indices.cpu(), minlength=self.model.num_classes**2
                ).reshape(self.model.num_classes, self.model.num_classes)
        metrics = {name: value / max(examples, 1) for name, value in totals.items()}
        if confusion is not None and int(confusion.sum()):
            values = confusion.to(dtype=torch.float64)
            true_positive = values.diag()
            precision = true_positive / values.sum(dim=0).clamp_min(1.0)
            recall = true_positive / values.sum(dim=1).clamp_min(1.0)
            f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
            metrics.update(
                {
                    "accuracy": float(true_positive.sum() / values.sum()),
                    "macro_precision": float(precision.mean()),
                    "macro_recall": float(recall.mean()),
                    "macro_f1": float(f1.mean()),
                }
            )
        return metrics

    def fit(
        self,
        signals: np.ndarray,
        validation_fraction: float = 0.1,
        progress: Callable[[int, dict[str, float], dict[str, float]], None] | None = None,
        validation_signals: np.ndarray | None = None,
        *,
        labels: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
    ) -> TrainingHistory:
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        set_seed(self.config.seed)
        signal_values = np.asarray(signals, dtype=np.float32)
        label_values = None if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)
        if label_values is not None:
            if self.model.num_classes < 1:
                raise ValueError("Supervised training requires a model classification head")
            if len(label_values) != len(signal_values):
                raise ValueError("Training labels must match training signals")
            if (
                not len(label_values)
                or label_values.min() < 0
                or label_values.max() >= self.model.num_classes
            ):
                raise ValueError("Training labels are outside the model class range")
            dataset: Dataset = _LabeledECGDataset(signal_values, label_values)
        else:
            if validation_labels is not None:
                raise ValueError("Validation labels require training labels")
            dataset = TorchECGDataset(signal_values)
        if not len(dataset):
            raise ValueError("Training split is empty")
        generator = torch.Generator().manual_seed(self.config.seed)
        if validation_signals is not None:
            train_set = dataset
            validation_values = np.asarray(validation_signals, dtype=np.float32)
            if label_values is not None:
                if validation_labels is None:
                    raise ValueError("Supervised validation requires validation_labels")
                validation_label_values = np.asarray(validation_labels, dtype=np.int64).reshape(-1)
                if len(validation_label_values) != len(validation_values):
                    raise ValueError("Validation labels must match validation signals")
                if (
                    not len(validation_label_values)
                    or validation_label_values.min() < 0
                    or validation_label_values.max() >= self.model.num_classes
                ):
                    raise ValueError("Validation labels are outside the model class range")
                val_set: Dataset = _LabeledECGDataset(validation_values, validation_label_values)
            else:
                val_set = TorchECGDataset(validation_values)
            if not len(val_set):
                raise ValueError("Explicit validation split is empty")
            training_labels = label_values
        else:
            if validation_labels is not None:
                raise ValueError("validation_labels requires validation_signals")
            if label_values is not None:
                train_indices, validation_indices = stratified_holdout_indices(
                    label_values, validation_fraction, self.config.seed
                )
                train_set = Subset(dataset, train_indices.tolist())
                val_set = Subset(dataset, validation_indices.tolist())
                training_labels = label_values[train_indices]
            else:
                val_size = round(len(dataset) * validation_fraction)
                train_size = len(dataset) - val_size
                if train_size < 1:
                    raise ValueError("Training split is empty")
                train_set, val_set = random_split(
                    dataset, [train_size, val_size], generator=generator
                )
                training_labels = None
        if training_labels is not None and self.config.training.class_weighting == "balanced":
            weights = balanced_class_weights(training_labels, self.model.num_classes)
            self.class_weights = torch.as_tensor(weights, device=self.device)
        else:
            self.class_weights = None
        train_loader = DataLoader(
            train_set,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            generator=generator,
        )
        val_loader = (
            DataLoader(val_set, batch_size=self.config.training.batch_size)
            if len(val_set)
            else None
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.config.training.minimum_learning_rate,
            )
            if self.config.training.lr_scheduler == "cosine"
            else None
        )
        history = TrainingHistory([], [])
        learning_rates: list[float] = []
        warmup = max(self.config.training.beta_warmup_epochs, 1)
        best_score: float | None = None
        best_epoch: int | None = None
        best_metrics: dict[str, float] = {}
        best_state: dict[str, Tensor] | None = None
        epochs_without_improvement = 0
        early_stopped = False
        for epoch in range(1, self.config.training.epochs + 1):
            learning_rates.append(float(optimizer.param_groups[0]["lr"]))
            beta = self.config.training.beta * min(1.0, epoch / warmup)
            train_metrics = self._epoch(train_loader, beta, optimizer)
            validation_metrics = (
                self._epoch(val_loader, beta, None) if val_loader is not None else {}
            )
            history.train.append(train_metrics)
            history.validation.append(validation_metrics)
            if progress:
                progress(epoch, train_metrics, validation_metrics)
            if scheduler is not None:
                scheduler.step()
            if validation_metrics:
                monitor = "macro_f1" if "macro_f1" in validation_metrics else "loss"
                score = (
                    validation_metrics[monitor]
                    if monitor == "macro_f1"
                    else -validation_metrics[monitor]
                )
                if (
                    best_score is None
                    or score > best_score + self.config.training.early_stopping_min_delta
                ):
                    best_score = score
                    best_epoch = epoch
                    best_metrics = dict(validation_metrics)
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in self.model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    patience = self.config.training.early_stopping_patience
                    if patience and epochs_without_improvement >= patience:
                        early_stopped = True
                        break
        if self.config.training.restore_best_weights and best_state is not None:
            self.model.load_state_dict(best_state)
        history.diagnostics.update(
            {
                "epochs_completed": len(history.train),
                "best_epoch": best_epoch,
                "best_validation_metrics": best_metrics,
                "early_stopped": early_stopped,
                "class_weights": self.class_weights.detach().cpu().tolist()
                if self.class_weights is not None
                else None,
                "lr_scheduler": self.config.training.lr_scheduler,
                "learning_rates": learning_rates,
            }
        )
        return history

    @torch.inference_mode()
    def encode(self, signals: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        self.model.eval()
        loader = DataLoader(
            TorchECGDataset(signals),
            batch_size=batch_size or self.config.training.batch_size,
        )
        values: list[np.ndarray] = []
        for batch in loader:
            z, _, _ = self.model.encode(batch.to(self.device), sample=False)
            values.append(z.cpu().numpy())
        return np.concatenate(values, axis=0)

    @torch.inference_mode()
    def predict_proba(self, signals: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """Run the optional neural classification head in bounded batches."""
        if self.model.num_classes < 1:
            raise RuntimeError("This model was created without a classification head")
        dataset = TorchECGDataset(np.asarray(signals, dtype=np.float32))
        if not len(dataset):
            raise ValueError("Prediction signals must not be empty")
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size or self.config.training.batch_size)
        probabilities: list[np.ndarray] = []
        for batch in loader:
            logits = self.model.classification_logits(batch.to(self.device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probabilities, axis=0)

    def predict(self, signals: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        return self.predict_proba(signals, batch_size).argmax(axis=1)

    def save_history(self, history: TrainingHistory, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "train": history.train,
                    "validation": history.validation,
                    "diagnostics": history.diagnostics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
