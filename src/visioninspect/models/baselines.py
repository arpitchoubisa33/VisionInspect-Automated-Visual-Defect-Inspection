"""Baseline classifiers implemented from scratch, for model comparison.

A model-selection study is only meaningful against alternatives, so two
contrasting baselines sit alongside the k-NN:

* :class:`NearestCentroid` - the simplest possible prototype classifier. One
  mean vector per class; a query takes the label of the nearest mean. It has
  no hyper-parameters and assumes each class forms a single compact cluster.
* :class:`GaussianNaiveBayes` - a generative model. It estimates a per-class,
  per-feature Gaussian and multiplies the likelihoods, which assumes features
  are conditionally independent given the class. That assumption is known to
  be violated here (area, perimeter and elongation are correlated by
  construction), so its score quantifies how much that violation costs.

Both expose the same ``fit`` / ``predict`` / ``predict_one`` interface as
:class:`~visioninspect.models.knn.KNNClassifier`, so the cross-validation code
treats every candidate identically.
"""

from __future__ import annotations

import numpy as np

from ..exceptions import ModelError
from ..logger import get_logger
from .scaler import StandardScaler

LOGGER = get_logger("models.baselines")


class NearestCentroid:
    """Assign the label of the closest class mean in standardised space."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.classes_: list[str] = []
        self._centroids: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: list[str]) -> "NearestCentroid":
        """Compute one centroid per class."""
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or len(features) != len(labels):
            raise ModelError("features must be 2-D and match the label count")

        scaled = self.scaler.fit_transform(features)
        self.classes_ = sorted(set(labels))
        labels_array = np.asarray(labels)
        self._centroids = np.vstack([
            scaled[labels_array == name].mean(axis=0) for name in self.classes_
        ])
        LOGGER.info("Fitted NearestCentroid on %d classes", len(self.classes_))
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Softmax over negative distances, so outputs are comparable to k-NN."""
        if self._centroids is None:
            raise ModelError("Model must be fitted before predicting")
        scaled = self.scaler.transform(np.atleast_2d(features))
        distances = np.linalg.norm(
            scaled[:, np.newaxis, :] - self._centroids[np.newaxis, :, :], axis=2)
        scores = np.exp(-distances)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> list[str]:
        """Return the nearest-centroid label for each query."""
        return [self.classes_[i]
                for i in np.argmax(self.predict_proba(features), axis=1)]

    def predict_one(self, vector: np.ndarray) -> tuple[str, float]:
        """Classify a single descriptor as ``(label, confidence)``."""
        proba = self.predict_proba(np.atleast_2d(vector))[0]
        best = int(np.argmax(proba))
        return self.classes_[best], float(proba[best])


class GaussianNaiveBayes:
    """Per-class diagonal-covariance Gaussian model with a Laplace-style floor."""

    def __init__(self, var_floor: float = 1e-6) -> None:
        self.scaler = StandardScaler()
        self.classes_: list[str] = []
        self.var_floor = var_floor
        self._means: np.ndarray | None = None
        self._vars: np.ndarray | None = None
        self._log_priors: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: list[str]) -> "GaussianNaiveBayes":
        """Estimate class priors, means and variances."""
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or len(features) != len(labels):
            raise ModelError("features must be 2-D and match the label count")

        scaled = self.scaler.fit_transform(features)
        self.classes_ = sorted(set(labels))
        labels_array = np.asarray(labels)

        means, variances, priors = [], [], []
        for name in self.classes_:
            subset = scaled[labels_array == name]
            if len(subset) < 2:
                raise ModelError(f"Class '{name}' needs at least 2 samples")
            means.append(subset.mean(axis=0))
            # The floor keeps a near-constant feature from producing an
            # infinite log-likelihood.
            variances.append(np.maximum(subset.var(axis=0), self.var_floor))
            priors.append(len(subset) / len(scaled))

        self._means = np.vstack(means)
        self._vars = np.vstack(variances)
        self._log_priors = np.log(np.asarray(priors))
        LOGGER.info("Fitted GaussianNaiveBayes on %d classes", len(self.classes_))
        return self

    def _log_likelihood(self, scaled: np.ndarray) -> np.ndarray:
        """Log P(x | class) + log P(class) for every query and class."""
        assert self._means is not None and self._vars is not None
        diff = scaled[:, np.newaxis, :] - self._means[np.newaxis, :, :]
        exponent = -0.5 * np.sum(diff ** 2 / self._vars[np.newaxis, :, :], axis=2)
        normaliser = -0.5 * np.sum(np.log(2 * np.pi * self._vars), axis=1)
        return exponent + normaliser[np.newaxis, :] + self._log_priors

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Posterior class probabilities, computed stably in log space."""
        if self._means is None:
            raise ModelError("Model must be fitted before predicting")
        scaled = self.scaler.transform(np.atleast_2d(features))
        log_posterior = self._log_likelihood(scaled)
        log_posterior -= log_posterior.max(axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        return posterior / posterior.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> list[str]:
        """Return the maximum-a-posteriori label for each query."""
        return [self.classes_[i]
                for i in np.argmax(self.predict_proba(features), axis=1)]

    def predict_one(self, vector: np.ndarray) -> tuple[str, float]:
        """Classify a single descriptor as ``(label, confidence)``."""
        proba = self.predict_proba(np.atleast_2d(vector))[0]
        best = int(np.argmax(proba))
        return self.classes_[best], float(proba[best])
