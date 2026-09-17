"""k-Nearest-Neighbours classifier implemented from first principles.

Written with NumPy rather than pulled from a library so that the distance
metric, the tie-breaking rule and the distance weighting are all explicit and
inspectable - which is the point of a course project. Prediction is fully
vectorised: distances for the whole query batch are computed with the
expansion ``|a-b|^2 = |a|^2 - 2a.b + |b|^2``, giving one matrix product
instead of a Python double loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..exceptions import ModelError
from ..logger import get_logger
from .scaler import StandardScaler

LOGGER = get_logger("models.knn")


class KNNClassifier:
    """Distance-weighted k-NN over standardised feature vectors."""

    def __init__(self, k: int = 5, weighted: bool = True) -> None:
        if k < 1:
            raise ModelError("k must be >= 1")
        self.k = k
        self.weighted = weighted
        self.scaler = StandardScaler()
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self.classes_: list[str] = []

    # ---------------------------------------------------------------- fitting
    def fit(self, features: np.ndarray, labels: list[str]) -> "KNNClassifier":
        """Memorise the standardised training set."""
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or len(features) != len(labels):
            raise ModelError("features must be 2-D and match the label count")
        if len(features) < self.k:
            raise ModelError(
                f"k={self.k} exceeds the training set size ({len(features)})"
            )

        self.classes_ = sorted(set(labels))
        index = {name: i for i, name in enumerate(self.classes_)}
        self._X = self.scaler.fit_transform(features)
        self._y = np.array([index[label] for label in labels], dtype=np.int64)
        LOGGER.info("Fitted k-NN (k=%d, weighted=%s) on %d samples, %d classes",
                    self.k, self.weighted, len(features), len(self.classes_))
        return self

    # ------------------------------------------------------------- inference
    def _distances(self, query: np.ndarray) -> np.ndarray:
        """Squared Euclidean distances between every query and training row."""
        assert self._X is not None
        q_sq = np.sum(query ** 2, axis=1, keepdims=True)
        x_sq = np.sum(self._X ** 2, axis=1)[np.newaxis, :]
        cross = query @ self._X.T
        return np.maximum(q_sq - 2.0 * cross + x_sq, 0.0)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return a ``(n_queries, n_classes)`` matrix of vote shares."""
        if self._X is None or self._y is None:
            raise ModelError("Model must be fitted before predicting")

        query = self.scaler.transform(np.atleast_2d(features))
        distances = self._distances(query)
        neighbours = np.argsort(distances, axis=1, kind="stable")[:, : self.k]

        scores = np.zeros((query.shape[0], len(self.classes_)), dtype=np.float64)
        for row, idxs in enumerate(neighbours):
            # Inverse-distance weighting makes close neighbours count more and
            # removes the ambiguity of an even-numbered vote tie.
            weights = (1.0 / (np.sqrt(distances[row, idxs]) + 1e-6)
                       if self.weighted else np.ones(len(idxs)))
            for weight, neighbour in zip(weights, idxs):
                scores[row, self._y[neighbour]] += weight

        totals = scores.sum(axis=1, keepdims=True)
        return scores / np.where(totals < 1e-12, 1.0, totals)

    def predict(self, features: np.ndarray) -> list[str]:
        """Return the most probable class name per query."""
        proba = self.predict_proba(features)
        return [self.classes_[i] for i in np.argmax(proba, axis=1)]

    def predict_one(self, vector: np.ndarray) -> tuple[str, float]:
        """Classify a single descriptor, returning ``(label, confidence)``."""
        proba = self.predict_proba(np.atleast_2d(vector))[0]
        best = int(np.argmax(proba))
        return self.classes_[best], float(proba[best])

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        """Write the model to a human-readable JSON file."""
        if self._X is None or self._y is None:
            raise ModelError("Cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "k": self.k,
            "weighted": self.weighted,
            "classes": self.classes_,
            "scaler": self.scaler.state_dict(),
            "X": self._X.tolist(),
            "y": self._y.tolist(),
        }), encoding="utf-8")
        LOGGER.info("Saved model -> %s", path)

    @classmethod
    def load(cls, path: Path) -> "KNNClassifier":
        """Restore a model written by :meth:`save`."""
        path = Path(path)
        if not path.exists():
            raise ModelError(f"Model file not found: {path}")
        state = json.loads(path.read_text(encoding="utf-8"))
        model = cls(k=state["k"], weighted=state["weighted"])
        model.classes_ = state["classes"]
        model.scaler = StandardScaler.from_state(state["scaler"])
        model._X = np.asarray(state["X"], dtype=np.float64)
        model._y = np.asarray(state["y"], dtype=np.int64)
        return model
