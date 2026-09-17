"""Dataset discovery, validation and train/test splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..exceptions import DatasetError, ImageLoadError
from ..logger import get_logger

LOGGER = get_logger("data.loader")
VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    """One labelled image on disk."""

    path: Path
    label: str

    def load(self) -> np.ndarray:
        """Read the image as grayscale, raising a typed error on failure."""
        return load_image(self.path)


def load_image(path: Path) -> np.ndarray:
    """Load a single grayscale image with explicit validation.

    Raises:
        ImageLoadError: if the file is missing, unsupported or undecodable.
    """
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"Image does not exist: {path}")
    if path.suffix.lower() not in VALID_SUFFIXES:
        raise ImageLoadError(f"Unsupported image format '{path.suffix}': {path}")

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        raise ImageLoadError(f"Could not decode image: {path}")
    return img


def discover_samples(dataset_dir: Path) -> list[Sample]:
    """Scan a folder-per-class dataset directory.

    Raises:
        DatasetError: if the directory is absent or contains no usable images.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise DatasetError(
            f"Dataset directory not found: {dataset_dir}. "
            "Run 'python run.py generate' first."
        )

    samples: list[Sample] = []
    for class_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for file in sorted(class_dir.iterdir()):
            if file.suffix.lower() in VALID_SUFFIXES:
                samples.append(Sample(path=file, label=class_dir.name))

    if not samples:
        raise DatasetError(f"No images found under {dataset_dir}")

    LOGGER.info("Discovered %d samples across %d classes",
                len(samples), len({s.label for s in samples}))
    return samples


def class_distribution(samples: list[Sample]) -> dict[str, int]:
    """Count samples per class."""
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.label] = counts.get(sample.label, 0) + 1
    return counts


def stratified_split(samples: list[Sample], test_ratio: float, seed: int
                     ) -> tuple[list[Sample], list[Sample]]:
    """Split samples per class so both sides keep the original balance."""
    rng = np.random.default_rng(seed)
    train: list[Sample] = []
    test: list[Sample] = []

    by_label: dict[str, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)

    for label, group in sorted(by_label.items()):
        if len(group) < 2:
            raise DatasetError(f"Class '{label}' needs at least 2 samples")
        order = rng.permutation(len(group))
        n_test = max(1, int(round(len(group) * test_ratio)))
        for position, index in enumerate(order):
            (test if position < n_test else train).append(group[index])

    LOGGER.info("Split -> train=%d test=%d", len(train), len(test))
    return train, test
