from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
    raise ImportError("Neural components require PyTorch: `pip install torch`") from exc


def _group_count(channels: int, maximum: int = 8) -> int:
    upper = max(1, min(maximum, channels // 4))
    return next(group for group in range(upper, 0, -1) if channels % group == 0)


class BiLSTMBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim // 2,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.output_dim = hidden_dim

    def forward(self, sequence: Tensor) -> Tensor:
        encoded, _ = self.rnn(sequence)
        return encoded


class MambaBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError(
                "The Mamba backbone requires a compatible `mamba-ssm` installation; "
                "use backbone='bilstm' for the paper's fallback."
            ) from exc
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, sequence: Tensor) -> Tensor:
        value = self.projection(sequence)
        for layer, norm in zip(self.layers, self.norms):
            value = norm(value + self.dropout(layer(value)))
        return value


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        groups = _group_count(channels)
        padding = 2 * dilation
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(value + self.network(value))


class ResNetBackbone(nn.Module):
    """CPU-practical dilated residual temporal backbone for ECG waveforms."""

    def __init__(self, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        if layers < 1:
            raise ValueError("Residual backbone requires at least one layer")
        self.blocks = nn.Sequential(
            *[
                ResidualBlock1D(hidden_dim, dilation=2 ** (index % 4), dropout=dropout)
                for index in range(layers)
            ]
        )
        self.output_dim = hidden_dim

    def forward(self, sequence: Tensor) -> Tensor:
        return self.blocks(sequence.transpose(1, 2)).transpose(1, 2)


def _inception_kernel_sizes(kernel_size: int) -> tuple[int, int, int]:
    if kernel_size < 7 or kernel_size % 2 == 0:
        raise ValueError("InceptionTime kernel_size must be an odd integer of at least 7")

    def odd_floor(value: int) -> int:
        value = max(3, value)
        return value if value % 2 else value - 1

    return kernel_size, odd_floor(kernel_size // 2), odd_floor(kernel_size // 4)


class InceptionModule1D(nn.Module):
    """Multi-scale InceptionTime module adapted to fixed-width ECG features."""

    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        if channels % 4:
            raise ValueError("InceptionTime hidden_dim must be divisible by four")
        bottleneck_channels = max(4, min(32, channels // 4))
        branch_channels = channels // 4
        self.bottleneck = nn.Conv1d(channels, bottleneck_channels, kernel_size=1, bias=False)
        self.convolution_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_channels,
                    branch_channels,
                    kernel_size=size,
                    padding=size // 2,
                    bias=False,
                )
                for size in _inception_kernel_sizes(kernel_size)
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(channels, branch_channels, kernel_size=1, bias=False),
        )
        self.normalization = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, value: Tensor) -> Tensor:
        bottleneck = self.bottleneck(value)
        branches = [branch(bottleneck) for branch in self.convolution_branches]
        branches.append(self.pool_branch(value))
        return self.dropout(self.activation(self.normalization(torch.cat(branches, dim=1))))


class InceptionTimeBackbone(nn.Module):
    """InceptionTime-style multi-scale stack with shortcuts every three modules."""

    residual_interval = 3

    def __init__(self, channels: int, layers: int, dropout: float, kernel_size: int):
        super().__init__()
        if layers < self.residual_interval:
            raise ValueError("InceptionTime backbone requires at least three layers")
        _inception_kernel_sizes(kernel_size)
        self.blocks = nn.ModuleList(
            [InceptionModule1D(channels, kernel_size, dropout) for _ in range(layers)]
        )
        self.shortcuts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(channels),
                )
                for _ in range(layers // self.residual_interval)
            ]
        )
        self.activation = nn.ReLU()
        self.output_dim = channels

    def forward(self, sequence: Tensor) -> Tensor:
        value = sequence.transpose(1, 2)
        residual = value
        shortcut_index = 0
        for index, block in enumerate(self.blocks, start=1):
            value = block(value)
            if index % self.residual_interval == 0:
                value = self.activation(value + self.shortcuts[shortcut_index](residual))
                residual = value
                shortcut_index += 1
        return value.transpose(1, 2)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        leads: int,
        hidden_dim: int = 128,
        backbone: str = "bilstm",
        layers: int = 2,
        dropout: float = 0.1,
        inception_kernel_size: int = 15,
    ):
        super().__init__()
        widths = (32, 64, hidden_dim)
        if backbone == "inceptiontime":
            # Preserve more temporal detail than the 64x downsampling used by
            # the recurrent/residual paths before applying multi-scale filters.
            stem_width = max(8, hidden_dim // 2)
            self.stem = nn.Sequential(
                nn.Conv1d(leads, stem_width, kernel_size=15, stride=2, padding=7),
                nn.GroupNorm(_group_count(stem_width), stem_width),
                nn.SiLU(),
                nn.Conv1d(stem_width, hidden_dim, kernel_size=9, stride=2, padding=4),
                nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
                nn.SiLU(),
            )
            self.temporal = InceptionTimeBackbone(
                hidden_dim, layers, dropout, inception_kernel_size
            )
            self.attention = nn.Linear(hidden_dim, 1)
            return

        blocks: list[nn.Module] = []
        in_channels = leads
        for width in widths:
            blocks.extend(
                [
                    nn.Conv1d(in_channels, width, kernel_size=9, stride=4, padding=4),
                    nn.GroupNorm(_group_count(width), width),
                    nn.SiLU(),
                ]
            )
            in_channels = width
        self.stem = nn.Sequential(*blocks)
        if backbone == "bilstm":
            self.temporal = BiLSTMBackbone(hidden_dim, hidden_dim, layers, dropout)
        elif backbone == "resnet":
            self.temporal = ResNetBackbone(hidden_dim, layers, dropout)
        elif backbone == "mamba":
            self.temporal = MambaBackbone(hidden_dim, hidden_dim, layers, dropout)
        else:
            raise ValueError("backbone must be bilstm, resnet, inceptiontime, or mamba")
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        sequence = self.stem(x).transpose(1, 2)
        sequence = self.temporal(sequence)
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=-1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class WaveformDecoder(nn.Module):
    def __init__(self, latent_dim: int, leads: int, hidden_dim: int):
        super().__init__()
        seed_length = 20
        self.seed_length = seed_length
        self.hidden_dim = hidden_dim
        self.project = nn.Linear(latent_dim, hidden_dim * seed_length)
        self.network = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, 128, 8, stride=4, padding=2),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ConvTranspose1d(128, 64, 8, stride=4, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose1d(64, 32, 8, stride=4, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, leads, kernel_size=9, padding=4),
        )

    def forward(self, z: Tensor, target_length: int) -> Tensor:
        seed = self.project(z).reshape(len(z), self.hidden_dim, self.seed_length)
        reconstruction = self.network(seed)
        if reconstruction.shape[-1] != target_length:
            reconstruction = F.interpolate(
                reconstruction, size=target_length, mode="linear", align_corners=False
            )
        return reconstruction


@dataclass
class VAEOutput:
    reconstruction: Tensor
    z: Tensor
    mu: Tensor
    logvar: Tensor
    logits: Tensor | None = None


class CausalVAE(nn.Module):
    """Temporal VAE used to expose compact latent ECG biomarkers.

    The standard normal factorized prior implements the independence pressure
    specified by the paper. The class exposes posterior means for deterministic
    downstream graph fitting and inference.
    """

    def __init__(
        self,
        leads: int = 12,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        backbone: str = "bilstm",
        lstm_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 0,
        classifier_hidden_dim: int = 64,
        inception_kernel_size: int = 15,
        classifier_input: str = "latent",
    ):
        super().__init__()
        if num_classes < 0:
            raise ValueError("num_classes must be non-negative")
        if classifier_hidden_dim < 1:
            raise ValueError("classifier_hidden_dim must be positive")
        if classifier_input not in {"latent", "encoder", "hybrid"}:
            raise ValueError("classifier_input must be latent, encoder, or hybrid")
        self.leads = leads
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.backbone_name = backbone
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.classifier_hidden_dim = classifier_hidden_dim
        self.inception_kernel_size = inception_kernel_size
        self.classifier_input = classifier_input
        self.encoder = TemporalEncoder(
            leads,
            hidden_dim,
            backbone,
            lstm_layers,
            dropout,
            inception_kernel_size,
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = WaveformDecoder(latent_dim, leads, hidden_dim)
        classifier_input_dim = {
            "latent": latent_dim,
            "encoder": hidden_dim,
            "hybrid": hidden_dim + latent_dim,
        }[classifier_input]
        self.classifier = (
            nn.Sequential(
                nn.LayerNorm(classifier_input_dim),
                nn.Linear(classifier_input_dim, classifier_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, num_classes),
            )
            if num_classes
            else None
        )

    @property
    def init_config(self) -> dict[str, Any]:
        return {
            "leads": self.leads,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "backbone": self.backbone_name,
            "lstm_layers": self.lstm_layers,
            "dropout": self.dropout,
            "num_classes": self.num_classes,
            "classifier_hidden_dim": self.classifier_hidden_dim,
            "inception_kernel_size": self.inception_kernel_size,
            "classifier_input": self.classifier_input,
        }

    def _posterior(self, hidden: Tensor, sample: bool) -> tuple[Tensor, Tensor, Tensor]:
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden).clamp(-12.0, 12.0)
        z = self.reparameterize(mu, logvar) if sample else mu
        return z, mu, logvar

    def encode(self, x: Tensor, sample: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        return self._posterior(self.encoder(x), sample)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z: Tensor, target_length: int) -> Tensor:
        return self.decoder(z, target_length)

    def classify_latent(self, latent: Tensor) -> Tensor:
        if self.classifier is None:
            raise RuntimeError("This model was created without a classification head")
        if self.classifier_input != "latent":
            raise RuntimeError(
                "classify_latent is only available when classifier_input='latent'; "
                "use classification_logits for waveform classification"
            )
        return self.classifier(latent)

    def _classify_features(self, hidden: Tensor, mu: Tensor) -> Tensor | None:
        if self.classifier is None:
            return None
        if self.classifier_input == "latent":
            features = mu
        elif self.classifier_input == "encoder":
            features = hidden
        else:
            features = torch.cat((hidden, mu), dim=1)
        return self.classifier(features)

    def encode_and_classify(
        self, x: Tensor, sample: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Encode once and optionally classify without invoking the decoder."""
        hidden = self.encoder(x)
        z, mu, logvar = self._posterior(hidden, sample)
        return z, mu, logvar, self._classify_features(hidden, mu)

    def classification_logits(self, x: Tensor) -> Tensor:
        """Return deterministic logits without running the waveform decoder."""
        _, _, _, logits = self.encode_and_classify(x, sample=False)
        if logits is None:
            raise RuntimeError("This model was created without a classification head")
        return logits

    def forward(self, x: Tensor) -> VAEOutput:
        z, mu, logvar, logits = self.encode_and_classify(x, sample=self.training)
        return VAEOutput(self.decode(z, x.shape[-1]), z, mu, logvar, logits)

    @staticmethod
    def loss(
        output: VAEOutput,
        target: Tensor,
        beta: float = 1.0,
        labels: Tensor | None = None,
        classification_weight: float = 1.0,
        class_weights: Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> dict[str, Tensor]:
        reconstruction = F.mse_loss(output.reconstruction, target, reduction="mean")
        kl = -0.5 * torch.mean(1.0 + output.logvar - output.mu.square() - output.logvar.exp())
        total = reconstruction + beta * kl
        losses = {"loss": total, "reconstruction": reconstruction, "kl": kl}
        if labels is not None:
            if output.logits is None:
                raise ValueError("Labels require a model with a classification head")
            classification = F.cross_entropy(
                output.logits,
                labels,
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
            losses["classification"] = classification
            losses["loss"] = total + classification_weight * classification
        return losses
