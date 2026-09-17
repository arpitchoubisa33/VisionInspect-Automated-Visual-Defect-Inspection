#!/usr/bin/env python3
"""Degradation study: how far can image quality drop before inspection fails?

A perfect score on pristine renders says little about a real production line,
where cameras defocus, lighting drifts and sensors get hot. This script
re-scores the held-out split under three controlled degradations and writes a
plot plus a JSON table to ``outputs/figures``.

Run:
    python scripts/robustness_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visioninspect.config import AppConfig                       # noqa: E402
from visioninspect.data.loader import discover_samples, stratified_split  # noqa: E402
from visioninspect.detect.inspector import DefectInspector       # noqa: E402
from visioninspect.logger import configure_logging               # noqa: E402
from visioninspect.models.knn import KNNClassifier               # noqa: E402


def add_noise(image: np.ndarray, sigma: float, rng: np.random.Generator):
    """Additive Gaussian sensor noise."""
    noisy = image.astype(np.float32) + rng.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_blur(image: np.ndarray, ksize: int, _rng=None):
    """Camera defocus, modelled as a Gaussian point-spread function."""
    if ksize <= 1:
        return image
    k = int(ksize) | 1
    return cv2.GaussianBlur(image, (k, k), 0)


def shift_brightness(image: np.ndarray, delta: float, _rng=None):
    """Global illumination drift."""
    return np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)


DEGRADATIONS = {
    "gaussian_noise_sigma": (add_noise, [0, 4, 8, 12, 16, 20, 25]),
    "defocus_blur_ksize": (add_blur, [1, 3, 5, 7, 9, 11]),
    "brightness_shift": (shift_brightness, [-60, -40, -20, 0, 20, 40, 60]),
}


def main() -> int:
    """Score the test split across every degradation level."""
    config = AppConfig()
    configure_logging("WARNING")

    model_file = config.output_dir / "model" / "knn_model.json"
    if not model_file.exists():
        print("No trained model found. Run: python run.py pipeline")
        return 1

    samples = discover_samples(config.data.dataset_dir)
    _, test = stratified_split(samples, config.model.test_ratio,
                               config.model.random_seed)
    inspector = DefectInspector(config, KNNClassifier.load(model_file))
    rng = np.random.default_rng(0)

    table: dict[str, dict[str, float]] = {}
    for name, (fn, levels) in DEGRADATIONS.items():
        scores: dict[str, float] = {}
        for level in levels:
            correct = 0
            for sample in test:
                image = fn(sample.load(), level, rng)
                result = inspector.inspect_array(image, source=sample.path.name)
                correct += int(result.verdict == sample.label)
            scores[str(level)] = round(correct / len(test), 4)
            print(f"{name:<24} {level:>5} -> accuracy {scores[str(level)]:.3f}")
        table[name] = scores

    out_dir = config.output_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "robustness.json").write_text(json.dumps(table, indent=2),
                                             encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, (name, scores) in zip(axes, table.items()):
        xs = [float(k) for k in scores]
        axis.plot(xs, list(scores.values()), marker="o")
        axis.set_title(name.replace("_", " "), fontsize=10)
        axis.set_ylabel("accuracy")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.3)
    fig.suptitle("VisionInspect robustness under image degradation")
    fig.tight_layout()
    fig.savefig(out_dir / "robustness.png", dpi=140)
    print(f"\nSaved -> {out_dir / 'robustness.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
