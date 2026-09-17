#!/usr/bin/env python3
"""Model-selection study: which classifier should the pipeline use?

Mines region descriptors from the training split, then compares candidate
classifiers under identical stratified 5-fold cross-validation:

* k-NN at several neighbourhood sizes (distance-weighted and unweighted)
* Nearest Centroid  - single prototype per class, no hyper-parameters
* Gaussian Naive Bayes - generative, assumes conditional independence

Also reports a feature-ablation: how much accuracy each descriptor family
contributes, which justifies keeping all thirteen features.

Writes ``outputs/figures/model_selection.{json,png}``.

Run:
    python scripts/model_selection.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visioninspect.config import AppConfig                          # noqa: E402
from visioninspect.data.loader import discover_samples, stratified_split  # noqa: E402
from visioninspect.detect.inspector import DefectInspector          # noqa: E402
from visioninspect.features.extractor import FEATURE_NAMES          # noqa: E402
from visioninspect.logger import configure_logging                  # noqa: E402
from visioninspect.models.baselines import (GaussianNaiveBayes,     # noqa: E402
                                            NearestCentroid)
from visioninspect.models.crossval import compare_models, cross_validate  # noqa: E402
from visioninspect.models.knn import KNNClassifier                  # noqa: E402
from visioninspect.pipeline.preprocess import preprocess            # noqa: E402
from visioninspect.pipeline.segment import segment                  # noqa: E402
from visioninspect.features.extractor import extract                # noqa: E402

# Descriptor families, used for the ablation study.
FAMILIES = {
    "shape": ("log_area", "elongation", "circularity", "extent",
              "solidity", "convexity", "thinness", "rect_fill"),
    "photometric": ("contrast", "grad_energy"),
    "hu_moments": ("hu1", "hu2", "hu3"),
}


def mine_descriptors(config, samples, sigma: float = 0.0):
    """Extract one descriptor per defect image, optionally under sensor noise.

    On pristine renders every candidate model scores 1.000, which tells us
    nothing about which to deploy. Adding Gaussian noise before the pipeline
    runs produces the harder, more realistic descriptor set that actually
    separates the candidates.
    """
    rng = np.random.default_rng(0)
    vectors, labels = [], []
    for sample in samples:
        if sample.label == "good":
            continue
        image = sample.load().astype(np.float32)
        if sigma > 0:
            image = image + rng.normal(0, sigma, image.shape)
        image = np.clip(image, 0, 255).astype(np.uint8)
        processed = preprocess(image, config.preprocess)
        regions, _ = segment(processed, config.segment)
        if not regions:
            continue
        vectors.append(extract(processed, regions[0]))
        labels.append(sample.label)
    return np.vstack(vectors), labels


def candidate_models() -> dict:
    """Every classifier entered into the comparison."""
    return {
        "k-NN (k=1)": lambda: KNNClassifier(k=1),
        "k-NN (k=3, weighted)": lambda: KNNClassifier(k=3, weighted=True),
        "k-NN (k=5, weighted)": lambda: KNNClassifier(k=5, weighted=True),
        "k-NN (k=5, uniform)": lambda: KNNClassifier(k=5, weighted=False),
        "k-NN (k=9, weighted)": lambda: KNNClassifier(k=9, weighted=True),
        "Nearest Centroid": NearestCentroid,
        "Gaussian Naive Bayes": GaussianNaiveBayes,
    }


def ablation(features: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Cross-validated accuracy for each descriptor family and for all of them."""
    scores: dict[str, float] = {}
    for family, names in FAMILIES.items():
        columns = [FEATURE_NAMES.index(name) for name in names]
        result = cross_validate(lambda: KNNClassifier(k=5),
                                features[:, columns], labels,
                                f"{family} only")
        scores[f"{family} only"] = round(result.mean, 4)

    for family, names in FAMILIES.items():
        columns = [i for i, name in enumerate(FEATURE_NAMES)
                   if name not in names]
        result = cross_validate(lambda: KNNClassifier(k=5),
                                features[:, columns], labels,
                                f"without {family}")
        scores[f"without {family}"] = round(result.mean, 4)

    full = cross_validate(lambda: KNNClassifier(k=5), features, labels,
                          "all 13 features")
    scores["all 13 features"] = round(full.mean, 4)
    return scores


def learning_curve(features: np.ndarray, labels: list[str],
                   sizes=(3, 5, 8, 12, 20, 30)) -> dict[str, dict[int, float]]:
    """How much labelled data does each model need?

    With plenty of data every candidate saturates at 1.000, so the informative
    question is not which model is best but which is usable when only a handful
    of labelled defects exist - the realistic situation on a new product line.
    """
    rng = np.random.default_rng(7)
    labels_array = np.asarray(labels)
    curves: dict[str, dict[int, float]] = {}

    models = {"k-NN (k=5)": lambda: KNNClassifier(k=5),
              "Nearest Centroid": NearestCentroid,
              "Gaussian Naive Bayes": GaussianNaiveBayes}

    for name, factory in models.items():
        curve: dict[int, float] = {}
        for size in sizes:
            scores = []
            for trial in range(5):          # average over 5 random subsets
                train_idx, test_idx = [], []
                for class_name in sorted(set(labels)):
                    members = rng.permutation(
                        np.flatnonzero(labels_array == class_name))
                    train_idx.extend(members[:size].tolist())
                    test_idx.extend(members[size:].tolist())
                if not test_idx or len(train_idx) < 5:
                    continue
                model = factory()
                model.fit(features[train_idx],
                          [labels[i] for i in train_idx])
                predictions = model.predict(features[test_idx])
                truth = [labels[i] for i in test_idx]
                scores.append(float(np.mean(
                    [p == t for p, t in zip(predictions, truth)])))
            if scores:
                curve[size] = round(float(np.mean(scores)), 4)
        curves[name] = curve
    return curves


def plot(results, ablation_scores: dict[str, float],
         curves: dict[str, dict[int, float]], out_dir: Path) -> Path:
    """Model comparison, feature ablation and learning curves."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (left, right, curve_ax) = plt.subplots(1, 3, figsize=(16, 4.6))

    for name, curve in curves.items():
        xs = sorted(curve)
        curve_ax.plot(xs, [curve[x] for x in xs], marker="o", label=name)
    curve_ax.set_xlabel("labelled regions per class")
    curve_ax.set_ylabel("held-out accuracy")
    curve_ax.set_title("Learning curves")
    curve_ax.set_ylim(0, 1.05)
    curve_ax.legend(fontsize=7)
    curve_ax.grid(alpha=0.3)

    names = [r.name for r in results][::-1]
    means = [r.mean for r in results][::-1]
    errors = [r.std for r in results][::-1]
    left.barh(names, means, xerr=errors, color="#60a5fa",
              edgecolor="#1f2430", capsize=3)
    left.set_xlim(0, 1.05)
    left.set_xlabel("5-fold CV accuracy")
    left.set_title("Model selection (sigma=18 noise)")
    left.grid(axis="x", alpha=0.3)

    keys = list(ablation_scores)[::-1]
    right.barh(keys, [ablation_scores[k] for k in keys], color="#34d399",
               edgecolor="#1f2430")
    right.set_xlim(0, 1.05)
    right.set_xlabel("5-fold CV accuracy (k-NN, k=5)")
    right.set_title("Feature ablation (sigma=18 noise)")
    right.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "model_selection.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    """Run the full model-selection study."""
    config = AppConfig()
    configure_logging("WARNING")

    samples = discover_samples(config.data.dataset_dir)
    train, _ = stratified_split(samples, config.model.test_ratio,
                                config.model.random_seed)
    features, labels = DefectInspector(config).build_training_set(train)
    print(f"Descriptor matrix: {features.shape[0]} regions "
          f"x {features.shape[1]} features")

    clean = compare_models(candidate_models(), features, labels, n_splits=5)
    print("\nStratified 5-fold CV - pristine images")
    print("-" * 52)
    for result in clean:
        print(f"  {result}")

    # Harder, more discriminating condition.
    noisy_features, noisy_labels = mine_descriptors(config, train, sigma=18.0)
    print(f"\nStratified 5-fold CV - sigma=18 sensor noise "
          f"({len(noisy_labels)} regions recovered)")
    print("-" * 52)
    results = compare_models(candidate_models(), noisy_features,
                             noisy_labels, n_splits=5)
    for result in results:
        print(f"  {result}")

    print("\nFeature ablation under noise (k-NN, k=5)")
    print("-" * 52)
    scores = ablation(noisy_features, noisy_labels)
    for name, score in scores.items():
        print(f"  {name:<24} {score:.4f}")

    print("\nLearning curves (mean of 5 random subsets)")
    print("-" * 52)
    curves = learning_curve(features, labels)
    for name, curve in curves.items():
        formatted = "  ".join(f"n={k}:{v:.3f}" for k, v in sorted(curve.items()))
        print(f"  {name:<22} {formatted}")

    out_dir = config.output_dir / "figures"
    payload = {
        "n_regions_clean": int(features.shape[0]),
        "n_regions_noisy": int(noisy_features.shape[0]),
        "n_features": int(features.shape[1]),
        "model_comparison_clean": [r.to_dict() for r in clean],
        "model_comparison_noisy": [r.to_dict() for r in results],
        "feature_ablation_noisy": scores,
        "learning_curves": {k: {str(n): v for n, v in c.items()}
                            for k, c in curves.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_selection.json").write_text(json.dumps(payload, indent=2),
                                                  encoding="utf-8")
    print(f"\nSaved -> {plot(results, scores, curves, out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
