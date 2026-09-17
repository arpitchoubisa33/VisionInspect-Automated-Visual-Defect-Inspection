"""Validation tests for the configuration layer."""

import json
import tempfile
import unittest
from pathlib import Path

from base import VisionTestCase
from visioninspect.config import (AppConfig, ModelConfig, PreprocessConfig,
                                  SegmentConfig)
from visioninspect.exceptions import ConfigError


class TestConfig(VisionTestCase):
    """Configuration objects must reject invalid values at construction time."""

    def test_defaults_are_valid(self):
        config = AppConfig()
        self.assertGreaterEqual(config.preprocess.target_size, 64)
        self.assertGreaterEqual(config.model.k, 1)

    def test_preprocess_rejects_bad_values(self):
        for kwargs in ({"target_size": 16}, {"median_ksize": 4},
                       {"clahe_clip": 0.0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ConfigError):
                    PreprocessConfig(**kwargs)

    def test_segment_rejects_even_kernel(self):
        with self.assertRaises(ConfigError):
            SegmentConfig(background_ksize=40)

    def test_segment_rejects_out_of_range_sensitivity(self):
        with self.assertRaises(ConfigError):
            SegmentConfig(sensitivity=9.0)

    def test_model_rejects_bad_k(self):
        with self.assertRaises(ConfigError):
            ModelConfig(k=0)

    def test_model_rejects_bad_test_ratio(self):
        with self.assertRaises(ConfigError):
            ModelConfig(test_ratio=0.9)

    def test_from_json_applies_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps({"model": {"k": 7}, "log_level": "DEBUG"}))
            config = AppConfig.from_json(path)
        self.assertEqual(config.model.k, 7)
        self.assertEqual(config.log_level, "DEBUG")

    def test_from_json_missing_file_raises(self):
        with self.assertRaises(ConfigError):
            AppConfig.from_json(Path("/nonexistent/cfg.json"))

    def test_to_dict_is_json_serialisable(self):
        json.dumps(AppConfig().to_dict())


if __name__ == "__main__":
    unittest.main()
