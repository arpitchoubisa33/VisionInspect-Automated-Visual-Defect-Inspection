"""Stage 1 - image conditioning.

Raw inspection frames vary in size, illumination and sensor noise. This stage
normalises them so that downstream thresholds are scene-independent:

1. resize to a fixed square working resolution (bilinear / area);
2. median filter to remove salt-and-pepper sensor noise while keeping edges;
3. CLAHE to equalise local contrast under uneven lighting.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import PreprocessConfig
from ..exceptions import ImageLoadError
from ..logger import get_logger

LOGGER = get_logger("pipeline.preprocess")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a single-channel view of a grayscale, BGR or BGRA image."""
    if image is None or image.size == 0:
        raise ImageLoadError("Empty image passed to preprocessing")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ImageLoadError(f"Unsupported image shape: {image.shape}")


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    """Resize to ``size x size``, using area interpolation when downscaling."""
    interp = cv2.INTER_AREA if image.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interp)


def denoise(image: np.ndarray, ksize: int) -> np.ndarray:
    """Edge-preserving median filter."""
    return cv2.medianBlur(image, ksize)


def equalise(image: np.ndarray, clip: float, grid: int) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalisation."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(image)


def preprocess(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Run the full conditioning chain and return a ``uint8`` grayscale image."""
    gray = to_grayscale(image)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resized = resize_square(gray, config.target_size)
    smoothed = denoise(resized, config.median_ksize)
    equalised = equalise(smoothed, config.clahe_clip, config.clahe_grid)
    LOGGER.debug("Preprocessed image -> shape=%s mean=%.1f",
                 equalised.shape, float(equalised.mean()))
    return equalised
