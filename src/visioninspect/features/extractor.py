"""Stage 3 - feature engineering.

Each candidate region is converted into a fixed-length numeric descriptor that
is invariant to translation and (mostly) to rotation and scale. Three families
of features are combined because no single family separates all three defect
morphologies:

* **Shape**  - elongation, circularity, extent, solidity, convexity.
  Separates a long thin scratch from a compact spot.
* **Skeleton / topology** - thinness ratio and branchiness.
  Separates an irregular branching crack from a straight scratch.
* **Photometry & texture** - local contrast against the surrounding surface
  and gradient energy, which capture how "deep" a defect appears.

Hu moments supply a rotation-invariant shape signature and are log-compressed
because their raw magnitudes span many orders of magnitude.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..pipeline.segment import Region

FEATURE_NAMES: tuple[str, ...] = (
    "log_area",
    "elongation",
    "circularity",
    "extent",
    "solidity",
    "convexity",
    "thinness",
    "rect_fill",
    "contrast",
    "grad_energy",
    "hu1",
    "hu2",
    "hu3",
)
FEATURE_DIM = len(FEATURE_NAMES)


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that degrades to a default instead of raising."""
    return numerator / denominator if abs(denominator) > 1e-9 else default


def shape_features(region: Region) -> dict[str, float]:
    """Geometric descriptors derived from the region contour."""
    contour = region.contour
    area = max(region.area, 1.0)
    perimeter = max(float(cv2.arcLength(contour, True)), 1.0)

    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    major, minor = max(rw, rh), max(min(rw, rh), 1.0)

    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    hull_perimeter = max(float(cv2.arcLength(hull, True)), 1.0)

    _, _, bw, bh = region.bbox
    return {
        "log_area": math.log1p(area),
        "elongation": _safe_div(major, minor, 1.0),
        "circularity": 4.0 * math.pi * area / (perimeter ** 2),
        "extent": _safe_div(area, float(bw * bh), 0.0),
        "solidity": _safe_div(area, hull_area, 0.0),
        "convexity": _safe_div(hull_perimeter, perimeter, 1.0),
        "thinness": _safe_div(perimeter ** 2, area, 0.0) / 100.0,
        "rect_fill": _safe_div(area, major * minor, 0.0),
    }


def photometric_features(image: np.ndarray, region: Region) -> dict[str, float]:
    """Contrast of the region against its immediate surroundings."""
    inside = region.mask > 0
    dilated = cv2.dilate(region.mask,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ring = (dilated > 0) & ~inside

    inner_mean = float(image[inside].mean()) if inside.any() else 0.0
    outer_mean = float(image[ring].mean()) if ring.any() else inner_mean

    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    energy = float(magnitude[inside].mean()) if inside.any() else 0.0

    return {
        "contrast": abs(inner_mean - outer_mean) / 255.0,
        "grad_energy": energy / 255.0,
    }


def hu_features(region: Region) -> dict[str, float]:
    """First three log-compressed Hu invariant moments."""
    moments = cv2.moments(region.mask, binaryImage=True)
    hu = cv2.HuMoments(moments).flatten()
    compressed = [
        -math.copysign(1.0, value) * math.log10(abs(value) + 1e-30)
        for value in hu[:3]
    ]
    return {"hu1": compressed[0], "hu2": compressed[1], "hu3": compressed[2]}


def extract(image: np.ndarray, region: Region) -> np.ndarray:
    """Return the full descriptor for one region as a float32 vector."""
    values: dict[str, float] = {}
    values.update(shape_features(region))
    values.update(photometric_features(image, region))
    values.update(hu_features(region))
    vector = np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def extract_batch(image: np.ndarray, regions: list[Region]) -> np.ndarray:
    """Stack descriptors for several regions into an ``(n, FEATURE_DIM)`` array."""
    if not regions:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.vstack([extract(image, region) for region in regions])
