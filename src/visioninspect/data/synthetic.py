"""Synthetic defect-image generator.

Public industrial-inspection datasets are large and licence-restricted, so the
project ships a deterministic generator that renders brushed-metal surfaces
with three defect morphologies. This keeps the repository self-contained and
makes every experiment exactly reproducible from a seed.

Rendered classes
----------------
good     : clean textured surface
scratch  : long, thin, straight dark stroke
spot     : compact circular blemish (pit or stain)
crack    : irregular branching dark path
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .. import DEFECT_CLASSES
from ..logger import get_logger

LOGGER = get_logger("data.synthetic")


def _base_surface(size: int, rng: np.random.Generator) -> np.ndarray:
    """Render a brushed-metal background with uneven illumination."""
    surface = rng.normal(loc=150, scale=6, size=(size, size))

    # Horizontal brushing texture.
    brush = rng.normal(0, 9, size=(size, 1)) @ np.ones((1, size))
    surface += cv2.GaussianBlur(brush, (1, 9), 0)

    # Smooth illumination gradient (vignetting from an off-centre lamp).
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = rng.uniform(0.3, 0.7, size=2) * size
    radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / size
    surface += 28 * (1.0 - radial) - 10

    surface = cv2.GaussianBlur(surface, (3, 3), 0)
    return np.clip(surface, 0, 255).astype(np.uint8)


def _draw_scratch(img: np.ndarray, rng: np.random.Generator) -> None:
    """Draw a straight, elongated, low-intensity stroke."""
    size = img.shape[0]
    length = int(rng.uniform(0.35, 0.65) * size)
    angle = rng.uniform(0, math.pi)
    cx, cy = rng.uniform(0.3, 0.7, size=2) * size
    dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
    p1 = (int(cx - dx), int(cy - dy))
    p2 = (int(cx + dx), int(cy + dy))
    darkness = int(rng.uniform(55, 90))
    cv2.line(img, p1, p2, color=int(max(10, 150 - darkness)),
             thickness=int(rng.integers(2, 4)), lineType=cv2.LINE_AA)


def _draw_spot(img: np.ndarray, rng: np.random.Generator) -> None:
    """Draw a compact, near-circular blemish."""
    size = img.shape[0]
    radius = int(rng.uniform(0.032, 0.062) * size)
    cx, cy = (rng.uniform(0.25, 0.75, size=2) * size).astype(int)
    blob = np.zeros_like(img)
    cv2.circle(blob, (int(cx), int(cy)), radius, 255, -1, lineType=cv2.LINE_AA)
    blob = cv2.GaussianBlur(blob, (5, 5), 0).astype(np.float32) / 255.0
    delta = rng.uniform(62, 90) * (1 if rng.random() < 0.75 else -1)
    img[:] = np.clip(img.astype(np.float32) - delta * blob, 0, 255).astype(np.uint8)


def _draw_crack(img: np.ndarray, rng: np.random.Generator) -> None:
    """Draw an irregular random-walk path with one or two branches."""
    size = img.shape[0]
    x, y = (rng.uniform(0.25, 0.75, size=2) * size).astype(float)
    angle = rng.uniform(0, 2 * math.pi)
    points = [(int(x), int(y))]
    for _ in range(int(rng.integers(14, 22))):
        angle += rng.normal(0, 0.55)
        step = rng.uniform(4, 9)
        x = float(np.clip(x + math.cos(angle) * step, 4, size - 5))
        y = float(np.clip(y + math.sin(angle) * step, 4, size - 5))
        points.append((int(x), int(y)))

    value = int(rng.uniform(55, 85))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], value, 2, lineType=cv2.LINE_AA)

    # Branch off the main path, which is what distinguishes cracks from
    # scratches in the solidity / convexity descriptors.
    branch_at = int(rng.integers(3, max(4, len(points) - 4)))
    bx, by = points[branch_at]
    bangle = rng.uniform(0, 2 * math.pi)
    for _ in range(int(rng.integers(5, 9))):
        bangle += rng.normal(0, 0.6)
        nx = int(np.clip(bx + math.cos(bangle) * 6, 4, size - 5))
        ny = int(np.clip(by + math.sin(bangle) * 6, 4, size - 5))
        cv2.line(img, (bx, by), (nx, ny), value, 2, lineType=cv2.LINE_AA)
        bx, by = nx, ny


_DRAWERS = {"scratch": _draw_scratch, "spot": _draw_spot, "crack": _draw_crack}


def render_sample(label: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Render a single labelled grayscale sample.

    Args:
        label: One of ``good``, ``scratch``, ``spot`` or ``crack``.
        size: Side length of the square output image.
        rng: Seeded generator, so output is reproducible.

    Returns:
        ``uint8`` grayscale image of shape ``(size, size)``.
    """
    img = _base_surface(size, rng)
    if label in _DRAWERS:
        _DRAWERS[label](img, rng)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 3, img.shape), 0, 255)
    return img.astype(np.uint8)


def generate_dataset(dest: Path, samples_per_class: int, size: int,
                     seed: int = 7) -> dict[str, int]:
    """Materialise a class-balanced dataset on disk as ``<dest>/<label>/*.png``.

    Returns a mapping of label to number of images written.
    """
    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}
    labels = ("good",) + DEFECT_CLASSES

    for label in labels:
        folder = Path(dest) / label
        folder.mkdir(parents=True, exist_ok=True)
        for idx in range(samples_per_class):
            img = render_sample(label, size, rng)
            cv2.imwrite(str(folder / f"{label}_{idx:04d}.png"), img)
        counts[label] = samples_per_class
        LOGGER.info("Generated %d '%s' images -> %s", samples_per_class, label, folder)

    return counts
