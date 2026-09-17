"""Stage 2 - background suppression and candidate-region extraction.

Defects are local deviations from an otherwise smooth surface, so the
background is *estimated* rather than assumed: a large-kernel morphological
closing/opening pair reconstructs the defect-free surface, and the residual
between the image and that estimate isolates anomalies regardless of the
global illumination level. Otsu's method then picks the threshold
automatically, morphology cleans the mask, and connected components become
candidate regions of interest.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import SegmentConfig
from ..logger import get_logger

LOGGER = get_logger("pipeline.segment")

# Detection threshold expressed in robust standard deviations above the
# residual noise floor, tuned on the validation split (see docs/DESIGN.md).
MAD_SIGMA = 4.5
ABSOLUTE_FLOOR = 22.0


@dataclass
class Region:
    """A candidate defect region produced by segmentation."""

    contour: np.ndarray
    mask: np.ndarray          # binary mask of this region only, full frame size
    bbox: tuple[int, int, int, int]   # x, y, w, h
    area: float

    @property
    def centroid(self) -> tuple[float, float]:
        """Area-weighted centre of the region."""
        moments = cv2.moments(self.contour)
        if abs(moments["m00"]) < 1e-6:
            x, y, w, h = self.bbox
            return (x + w / 2.0, y + h / 2.0)
        return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def estimate_background(image: np.ndarray, ksize: int) -> np.ndarray:
    """Reconstruct the defect-free surface with a large structuring element."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)


def residual_map(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Absolute deviation of the image from its estimated background."""
    return cv2.absdiff(image, background)


def binarise(residual: np.ndarray, sensitivity: float) -> np.ndarray:
    """Threshold the residual using robust (median/MAD) noise statistics.

    Otsu's method was tried first but fails here: on a defect-free part the
    residual contains only sensor noise, and Otsu - which always splits the
    histogram into two classes - happily invents a "defect" class out of that
    noise. Instead the noise floor is estimated with the median absolute
    deviation, which is unaffected by the small fraction of outlier pixels a
    real defect contributes, and anything beyond 4.5 sigma is flagged. The
    absolute floor stops the detector from firing on an unusually clean frame
    where the MAD collapses towards zero.
    """
    smoothed = cv2.GaussianBlur(residual.astype(np.float32), (3, 3), 0)
    median = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - median))) * 1.4826

    threshold = max(median + (MAD_SIGMA / sensitivity) * mad,
                    ABSOLUTE_FLOOR / sensitivity)
    LOGGER.debug("Residual median=%.2f mad=%.2f -> threshold=%.2f",
                 median, mad, threshold)
    return (smoothed > threshold).astype(np.uint8) * 255


def clean_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    """Remove speckle and bridge small gaps inside a defect."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)


def extract_regions(mask: np.ndarray, config: SegmentConfig) -> list[Region]:
    """Turn a binary mask into area-filtered, largest-first regions."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    regions: list[Region] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < config.min_area:
            continue
        single = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(single, [contour], -1, 255, thickness=cv2.FILLED)
        regions.append(
            Region(contour=contour, mask=single,
                   bbox=cv2.boundingRect(contour), area=area)
        )

    regions.sort(key=lambda r: r.area, reverse=True)
    LOGGER.debug("Extracted %d regions (kept %d)", len(contours), len(regions))
    return regions[: config.max_regions]


def segment(image: np.ndarray, config: SegmentConfig
            ) -> tuple[list[Region], np.ndarray]:
    """Full segmentation chain.

    Returns:
        ``(regions, mask)`` where ``mask`` is the cleaned binary defect mask.
    """
    background = estimate_background(image, config.background_ksize)
    residual = residual_map(image, background)
    raw_mask = binarise(residual, config.sensitivity)
    mask = clean_mask(raw_mask, config.morph_ksize)
    return extract_regions(mask, config), mask
