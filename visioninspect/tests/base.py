"""Shared test helpers.

The suite uses the standard library's ``unittest`` so it runs on any Python
installation with no extra tooling:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from visioninspect.config import AppConfig                        # noqa: E402
from visioninspect.data.synthetic import render_sample            # noqa: E402
from visioninspect.pipeline.preprocess import preprocess          # noqa: E402
from visioninspect.pipeline.segment import segment                # noqa: E402


class VisionTestCase(unittest.TestCase):
    """Base case providing a default config and deterministic sample images."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = AppConfig()

    @staticmethod
    def sample(label: str, seed: int = 123, size: int = 256) -> np.ndarray:
        """Render a reproducible labelled image."""
        return render_sample(label, size, np.random.default_rng(seed))

    def first_region(self, label: str, seed: int = 123):
        """Preprocess a rendered sample and return ``(image, largest_region)``."""
        image = preprocess(self.sample(label, seed), self.config.preprocess)
        regions, _ = segment(image, self.config.segment)
        self.assertTrue(regions, f"expected at least one region for '{label}'")
        return image, regions[0]
