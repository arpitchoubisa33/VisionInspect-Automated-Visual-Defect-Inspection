"""Visual artefacts: annotated overlays, pipeline strips and metric plots."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")                     # headless rendering for CI/servers
import matplotlib.pyplot as plt
import numpy as np

from ..detect.inspector import InspectionResult
from ..logger import get_logger
from ..models.evaluate import EvaluationResult

LOGGER = get_logger("report.visualize")

COLOURS = {
    "scratch": (60, 180, 255),
    "spot": (80, 220, 100),
    "crack": (70, 70, 240),
    "good": (180, 180, 180),
}


def annotate(result: InspectionResult) -> np.ndarray:
    """Draw bounding boxes, contours and labels onto the processed frame."""
    if result.processed is None:
        raise ValueError("InspectionResult carries no processed image")

    canvas = cv2.cvtColor(result.processed, cv2.COLOR_GRAY2BGR)
    for finding, region in zip(result.findings, result.regions):
        colour = COLOURS.get(finding.label, (0, 255, 255))
        x, y, w, h = finding.bbox
        cv2.drawContours(canvas, [region.contour], -1, colour, 1)
        cv2.rectangle(canvas, (x - 3, y - 3), (x + w + 3, y + h + 3), colour, 1)
        cv2.putText(canvas, f"{finding.label} {finding.confidence:.2f}",
                    (max(x - 3, 2), max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

    banner = (0, 160, 0) if not result.is_defective else (0, 0, 200)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), banner, -1)
    cv2.putText(canvas, f"VERDICT: {result.verdict.upper()}", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_overlay(result: InspectionResult, out_dir: Path) -> Path:
    """Write the annotated overlay for one inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(result.source).stem}_annotated.png"
    cv2.imwrite(str(path), annotate(result))
    return path


def save_pipeline_strip(result: InspectionResult, out_dir: Path) -> Path:
    """Save a 3-panel figure: preprocessed image, defect mask, overlay."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    panels = [
        (result.processed, "1. Preprocessed", "gray"),
        (result.mask, "2. Defect mask", "gray"),
        (cv2.cvtColor(annotate(result), cv2.COLOR_BGR2RGB), "3. Classified", None),
    ]
    for axis, (image, title, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title, fontsize=10)
        axis.axis("off")

    fig.suptitle(f"{Path(result.source).name} - verdict: {result.verdict}",
                 fontsize=11)
    fig.tight_layout()
    path = out_dir / f"{Path(result.source).stem}_pipeline.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def save_confusion_matrix(evaluation: EvaluationResult, out_dir: Path) -> Path:
    """Render the confusion matrix as an annotated heat map."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.2, 4.6))
    matrix = evaluation.confusion
    image = axis.imshow(matrix, cmap="Blues")

    axis.set_xticks(range(len(evaluation.labels)), evaluation.labels, rotation=30)
    axis.set_yticks(range(len(evaluation.labels)), evaluation.labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"Confusion matrix (accuracy {evaluation.accuracy:.1%})")

    peak = matrix.max() if matrix.size else 1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      color="white" if matrix[i, j] > peak / 2 else "black")

    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_k_sweep(scores: dict[int, float], out_dir: Path) -> Path:
    """Plot validation accuracy against the neighbourhood size k."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.2, 3.4))
    ks = sorted(scores)
    axis.plot(ks, [scores[k] for k in ks], marker="o")
    axis.set_xlabel("k (neighbours)")
    axis.set_ylabel("validation accuracy")
    axis.set_title("k-NN hyper-parameter sweep")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "k_sweep.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
