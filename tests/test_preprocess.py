"""Tests for the image-conditioning stage."""

import unittest

import numpy as np

from base import VisionTestCase
from visioninspect.exceptions import ImageLoadError
from visioninspect.pipeline.preprocess import (preprocess, resize_square,
                                               to_grayscale)


class TestPreprocess(VisionTestCase):
    """Conditioning must normalise shape, type and contrast deterministically."""

    def test_colour_image_is_converted(self):
        colour = np.zeros((120, 90, 3), dtype=np.uint8)
        self.assertEqual(to_grayscale(colour).ndim, 2)

    def test_rgba_image_is_converted(self):
        rgba = np.zeros((40, 40, 4), dtype=np.uint8)
        self.assertEqual(to_grayscale(rgba).ndim, 2)

    def test_empty_image_raises(self):
        with self.assertRaises(ImageLoadError):
            to_grayscale(np.array([], dtype=np.uint8))

    def test_resize_produces_square(self):
        out = resize_square(np.zeros((300, 120), dtype=np.uint8), 256)
        self.assertEqual(out.shape, (256, 256))

    def test_preprocess_normalises_shape_and_type(self):
        out = preprocess(self.sample("good"), self.config.preprocess)
        size = self.config.preprocess.target_size
        self.assertEqual(out.shape, (size, size))
        self.assertEqual(out.dtype, np.uint8)

    def test_preprocess_is_deterministic(self):
        image = self.sample("scratch")
        first = preprocess(image, self.config.preprocess)
        second = preprocess(image, self.config.preprocess)
        self.assertTrue(np.array_equal(first, second))

    def test_preprocess_handles_non_square_input(self):
        rect = np.full((180, 320), 140, dtype=np.uint8)
        self.assertEqual(preprocess(rect, self.config.preprocess).shape,
                         (256, 256))


if __name__ == "__main__":
    unittest.main()
