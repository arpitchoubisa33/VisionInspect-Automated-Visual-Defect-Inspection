"""Tests for background suppression and candidate-region extraction."""

import unittest

import numpy as np

from base import VisionTestCase
from visioninspect.pipeline.preprocess import preprocess
from visioninspect.pipeline.segment import (binarise, clean_mask,
                                            estimate_background, segment)


class TestSegmentation(VisionTestCase):
    """Segmentation must find real defects without inventing them."""

    def test_clean_surface_yields_no_regions(self):
        """The critical false-positive guard: no defect on a good part."""
        for seed in (1, 2, 3, 4, 5):
            with self.subTest(seed=seed):
                image = preprocess(self.sample("good", seed),
                                   self.config.preprocess)
                regions, _ = segment(image, self.config.segment)
                self.assertEqual(regions, [])

    def test_every_defect_type_is_detected(self):
        for label in ("scratch", "spot", "crack"):
            for seed in (1, 2, 3):
                with self.subTest(label=label, seed=seed):
                    image = preprocess(self.sample(label, seed),
                                       self.config.preprocess)
                    regions, mask = segment(image, self.config.segment)
                    self.assertGreaterEqual(len(regions), 1)
                    self.assertEqual(mask.max(), 255)

    def test_region_area_respects_minimum(self):
        image, region = self.first_region("crack")
        self.assertGreaterEqual(region.area, self.config.segment.min_area)

    def test_background_estimate_is_smoother_than_input(self):
        image = preprocess(self.sample("scratch"), self.config.preprocess)
        background = estimate_background(image,
                                         self.config.segment.background_ksize)
        self.assertLessEqual(background.std(), image.std() + 1e-6)

    def test_binarise_on_flat_residual_flags_nothing(self):
        flat = np.zeros((128, 128), dtype=np.uint8)
        self.assertEqual(binarise(flat, 1.0).max(), 0)

    def test_clean_mask_removes_isolated_speckle(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10, 10] = 255                      # a single stray pixel
        self.assertEqual(clean_mask(mask, 3).sum(), 0)

    def test_regions_are_sorted_by_area(self):
        image = preprocess(self.sample("crack"), self.config.preprocess)
        regions, _ = segment(image, self.config.segment)
        areas = [r.area for r in regions]
        self.assertEqual(areas, sorted(areas, reverse=True))

    def test_region_centroid_is_inside_bbox(self):
        _, region = self.first_region("spot")
        cx, cy = region.centroid
        x, y, w, h = region.bbox
        self.assertTrue(x <= cx <= x + w and y <= cy <= y + h)


if __name__ == "__main__":
    unittest.main()
