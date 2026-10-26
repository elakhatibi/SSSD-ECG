from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class QuantileDiscretizer:
    """Map continuous latent biomarkers to ordinal, training-set quantile bins."""

    def __init__(self, n_bins: int = 3, feature_names: list[str] | None = None):
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        self.n_bins = n_bins
        self.feature_names = feature_names
        self.thresholds_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> QuantileDiscretizer:
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or len(x) < 2:
            raise ValueError("values must be a two-dimensional array with at least two rows")
        if not np.all(np.isfinite(x)):
            raise ValueError("Latent values must be finite")
        if self.feature_names is None:
            self.feature_names = [f"z{i}" for i in range(x.shape[1])]
        if len(self.feature_names) != x.shape[1]:
            raise ValueError("feature_names length does not match latent dimension")
        quantiles = np.arange(1, self.n_bins) / self.n_bins
        self.thresholds_ = np.quantile(x, quantiles, axis=0).T
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("Discretizer has not been fitted")
        x = np.asarray(values, dtype=float)
        one_dimensional = x.ndim == 1
        if one_dimensional:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.thresholds_.shape[0]:
            raise ValueError("Input latent dimensions do not match fitted thresholds")
        result = np.empty(x.shape, dtype=np.int64)
        for feature in range(x.shape[1]):
            result[:, feature] = np.searchsorted(
                self.thresholds_[feature], x[:, feature], side="right"
            )
        return result[0] if one_dimensional else result

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)

    def state_name(self, state: int) -> str:
        if state < 0 or state >= self.n_bins:
            raise ValueError("Invalid bin state")
        if self.n_bins == 3:
            return ("low", "mid", "high")[state]
        return f"quantile_{state + 1}_of_{self.n_bins}"

    def to_dict(self) -> dict:
        if self.thresholds_ is None:
            raise RuntimeError("Cannot serialize an unfitted discretizer")
        return {
            "n_bins": self.n_bins,
            "feature_names": self.feature_names,
            "thresholds": self.thresholds_.tolist(),
        }

    @classmethod
    def from_dict(cls, values: dict) -> QuantileDiscretizer:
        result = cls(int(values["n_bins"]), list(values["feature_names"]))
        result.thresholds_ = np.asarray(values["thresholds"], dtype=float)
        return result

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> QuantileDiscretizer:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
