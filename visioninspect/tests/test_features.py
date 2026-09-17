"""Tests for descriptor extraction and the invariances it must provide."""

import unittest

import numpy as np

from base import VisionTestCase
from visioninspect.features.extractor import (FEATURE_DIM, FEATURE_NAMES,
                                              extract, extract_batch)


class TestFeatures(VisionTestCase):
    """Descriptors must be finite, fixed-length and class-discriminative."""

    def _mean_feature(self, label: str, name: str, seeds=(1, 2, 3, 4)) -> float:
        index = FEATURE_NAMES.index(name)
        values = []
        for seed in seeds:
            image, region = self.first_region(label, seed)
            values.append(float(extract(image, region)[index]))
        return float(np.mean(values))

    def test_vector_has_expected_shape(self):
        image, region = self.first_region("scratch")
        self.assertEqual(extract(image, region).shape, (FEATURE_DIM,))
        self.assertEqual(FEATURE_DIM, len(FEATURE_NAMES))

    def test_vector_is_finite(self):
        image, region = self.first_region("crack")
        self.assertTrue(np.all(np.isfinite(extract(image, region))))

    def test_scratch_is_more_elongated_than_spot(self):
        """The property that separates linear from compact defects."""
        self.assertGreater(self._mean_feature("scratch", "elongation"),
                           self._mean_feature("spot", "elongation"))

    def test_spot_is_more_circular_than_crack(self):
        self.assertGreater(self._mean_feature("spot", "circularity"),
                           self._mean_feature("crack", "circularity"))

    def test_crack_is_less_solid_than_spot(self):
        """Branching cracks fill their convex hull far less than round spots."""
        self.assertLess(self._mean_feature("crack", "solidity"),
                        self._mean_feature("spot", "solidity"))

    def test_defect_has_measurable_contrast(self):
        self.assertGreater(self._mean_feature("spot", "contrast"), 0.0)

    def test_extract_batch_on_empty_region_list(self):
        empty = extract_batch(np.zeros((32, 32), np.uint8), [])
        self.assertEqual(empty.shape, (0, FEATURE_DIM))


if __name__ == "__main__":
    unittest.main()
