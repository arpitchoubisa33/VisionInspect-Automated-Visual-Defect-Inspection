"""Model evaluation metrics implemented directly from their definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvaluationResult:
    """Aggregated classification metrics for one experiment."""

    labels: list[str]
    confusion: np.ndarray
    accuracy: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    macro_f1: float = 0.0

    def to_dict(self) -> dict:
        """JSON-friendly representation for the report writer."""
        return {
            "labels": self.labels,
            "confusion_matrix": self.confusion.tolist(),
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_class": {
                name: {k: round(v, 4) for k, v in metrics.items()}
                for name, metrics in self.per_class.items()
            },
        }

    def format_table(self) -> str:
        """Render a console-friendly per-class metric table."""
        head = f"{'class':<10}{'precision':>11}{'recall':>9}{'f1':>8}{'support':>9}"
        rows = [head, "-" * len(head)]
        for name in self.labels:
            m = self.per_class[name]
            rows.append(f"{name:<10}{m['precision']:>11.3f}{m['recall']:>9.3f}"
                        f"{m['f1']:>8.3f}{int(m['support']):>9d}")
        rows.append("-" * len(head))
        rows.append(f"{'accuracy':<10}{self.accuracy:>11.3f}"
                    f"{'macro-f1':>17}{self.macro_f1:>8.3f}"[:len(head)])
        return "\n".join(rows)


def confusion_matrix(y_true: list[str], y_pred: list[str],
                     labels: list[str]) -> np.ndarray:
    """Rows are ground truth, columns are predictions."""
    index = {name: i for i, name in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        if truth in index and pred in index:
            matrix[index[truth], index[pred]] += 1
    return matrix


def evaluate(y_true: list[str], y_pred: list[str],
             labels: list[str] | None = None) -> EvaluationResult:
    """Compute accuracy plus per-class precision, recall, F1 and support."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Cannot evaluate an empty prediction set")

    labels = labels or sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels)
    total = matrix.sum()
    accuracy = float(np.trace(matrix) / total) if total else 0.0

    per_class: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    for i, name in enumerate(labels):
        true_positive = float(matrix[i, i])
        predicted = float(matrix[:, i].sum())
        actual = float(matrix[i, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1": f1, "support": actual}
        f1_scores.append(f1)

    return EvaluationResult(
        labels=labels, confusion=matrix, accuracy=accuracy,
        per_class=per_class, macro_f1=float(np.mean(f1_scores)),
    )


def cross_validate_k(features: np.ndarray, labels: list[str],
                     k_values: list[int], seed: int = 42) -> dict[int, float]:
    """Hold-out sweep over candidate ``k`` values; returns accuracy per k."""
    from .knn import KNNClassifier  # local import avoids a circular dependency

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    cut = int(len(order) * 0.75)
    train_idx, val_idx = order[:cut], order[cut:]

    scores: dict[int, float] = {}
    y_train = [labels[i] for i in train_idx]
    y_val = [labels[i] for i in val_idx]
    for k in k_values:
        if k > len(train_idx):
            continue
        model = KNNClassifier(k=k).fit(features[train_idx], y_train)
        preds = model.predict(features[val_idx])
        scores[k] = float(np.mean([p == t for p, t in zip(preds, y_val)]))
    return scores
