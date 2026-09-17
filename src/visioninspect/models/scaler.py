"""Z-score feature standardisation.

Distance-based classifiers are scale sensitive: ``log_area`` spans ~10 units
while ``circularity`` spans ~1, so without standardisation a single feature
would dominate the metric. Statistics are fitted on the training split only,
which prevents test-set leakage.
"""

from __future__ import annotations

import numpy as np

from ..exceptions import ModelError


class StandardScaler:
    """Fit per-feature mean and standard deviation, then whiten."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "StandardScaler":
        """Estimate statistics from a ``(n_samples, n_features)`` matrix."""
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ModelError("StandardScaler.fit expects a non-empty 2-D array")
        self.mean_ = features.mean(axis=0)
        std = features.std(axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)   # constant features stay put
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply the fitted transform."""
        if self.mean_ is None or self.std_ is None:
            raise ModelError("StandardScaler must be fitted before transform()")
        features = np.asarray(features, dtype=np.float64)
        if features.shape[-1] != self.mean_.shape[0]:
            raise ModelError(
                f"Expected {self.mean_.shape[0]} features, got {features.shape[-1]}"
            )
        return (features - self.mean_) / self.std_

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Convenience wrapper around :meth:`fit` + :meth:`transform`."""
        return self.fit(features).transform(features)

    def state_dict(self) -> dict:
        """Serialisable parameters."""
        if self.mean_ is None or self.std_ is None:
            raise ModelError("Nothing to serialise: scaler is not fitted")
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> "StandardScaler":
        """Rebuild a scaler from :meth:`state_dict` output."""
        scaler = cls()
        scaler.mean_ = np.asarray(state["mean"], dtype=np.float64)
        scaler.std_ = np.asarray(state["std"], dtype=np.float64)
        return scaler
