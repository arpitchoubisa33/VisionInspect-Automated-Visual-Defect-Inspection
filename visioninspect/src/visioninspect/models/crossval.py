"""Stratified k-fold cross-validation and model comparison.

A single hold-out split on ~135 region descriptors gives a noisy estimate: one
or two samples moving across the boundary shifts accuracy by several points.
K-fold cross-validation reuses every sample as both training and validation
data and reports a mean with a spread, which is what the model-selection study
in ``scripts/model_selection.py`` compares.

Folds are **stratified** - each fold holds roughly the same class proportions
as the full set - so no fold accidentally omits a defect class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..exceptions import ModelError
from ..logger import get_logger

LOGGER = get_logger("models.crossval")


@dataclass
class CVResult:
    """Cross-validation outcome for one candidate model."""

    name: str
    fold_scores: list[float]

    @property
    def mean(self) -> float:
        """Mean accuracy across folds."""
        return float(np.mean(self.fold_scores))

    @property
    def std(self) -> float:
        """Standard deviation across folds - the stability of the estimate."""
        return float(np.std(self.fold_scores))

    def to_dict(self) -> dict:
        """JSON-friendly summary."""
        return {
            "model": self.name,
            "mean_accuracy": round(self.mean, 4),
            "std_accuracy": round(self.std, 4),
            "fold_scores": [round(score, 4) for score in self.fold_scores],
        }

    def __str__(self) -> str:
        return f"{self.name:<24} {self.mean:.4f} +/- {self.std:.4f}"


def stratified_folds(labels: list[str], n_splits: int = 5,
                     seed: int = 42) -> list[np.ndarray]:
    """Return ``n_splits`` index arrays with class proportions preserved."""
    if n_splits < 2:
        raise ModelError("n_splits must be at least 2")

    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels)
    folds: list[list[int]] = [[] for _ in range(n_splits)]

    for name in sorted(set(labels)):
        indices = np.flatnonzero(labels_array == name)
        if len(indices) < n_splits:
            raise ModelError(
                f"Class '{name}' has {len(indices)} samples, "
                f"fewer than n_splits={n_splits}"
            )
        # Deal the shuffled members of each class round-robin across folds.
        for position, index in enumerate(rng.permutation(indices)):
            folds[position % n_splits].append(int(index))

    return [np.asarray(sorted(fold)) for fold in folds]


def cross_validate(model_factory: Callable[[], object], features: np.ndarray,
                   labels: list[str], name: str, n_splits: int = 5,
                   seed: int = 42) -> CVResult:
    """Score one model across stratified folds.

    Args:
        model_factory: Zero-argument callable returning a *fresh* model, so no
            state leaks between folds.
        features: ``(n_samples, n_features)`` descriptor matrix.
        labels: Ground-truth label per row.
        name: Display name for the result.
        n_splits: Number of folds.
        seed: Seed controlling the fold assignment.
    """
    features = np.asarray(features, dtype=np.float64)
    folds = stratified_folds(labels, n_splits, seed)
    all_indices = np.arange(len(labels))
    scores: list[float] = []

    for fold_number, validation_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(all_indices, validation_idx)
        model = model_factory()
        model.fit(features[train_idx], [labels[i] for i in train_idx])
        predictions = model.predict(features[validation_idx])
        truth = [labels[i] for i in validation_idx]
        accuracy = float(np.mean([p == t for p, t in zip(predictions, truth)]))
        scores.append(accuracy)
        LOGGER.debug("%s fold %d/%d -> %.4f", name, fold_number,
                     n_splits, accuracy)

    result = CVResult(name=name, fold_scores=scores)
    LOGGER.info("%s", result)
    return result


def compare_models(candidates: dict[str, Callable[[], object]],
                   features: np.ndarray, labels: list[str],
                   n_splits: int = 5, seed: int = 42) -> list[CVResult]:
    """Cross-validate every candidate and return results, best mean first."""
    results = [cross_validate(factory, features, labels, name, n_splits, seed)
               for name, factory in candidates.items()]
    results.sort(key=lambda r: (r.mean, -r.std), reverse=True)
    return results
