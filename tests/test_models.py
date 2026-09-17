"""Tests for the from-scratch scaler, k-NN classifier and metrics."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from base import VisionTestCase
from visioninspect.exceptions import ModelError
from visioninspect.models.evaluate import (confusion_matrix, cross_validate_k,
                                           evaluate)
from visioninspect.models.knn import KNNClassifier
from visioninspect.models.scaler import StandardScaler


def toy_data():
    """Two well-separated Gaussian blobs in 3-D."""
    rng = np.random.default_rng(0)
    features = np.vstack([rng.normal(0, 0.4, size=(30, 3)),
                          rng.normal(5, 0.4, size=(30, 3))])
    return features, ["a"] * 30 + ["b"] * 30


class TestScaler(VisionTestCase):
    """Standardisation must whiten features and survive degenerate columns."""

    def test_scaler_whitens_features(self):
        data = np.array([[1.0, 100.0], [3.0, 300.0], [5.0, 500.0]])
        scaled = StandardScaler().fit_transform(data)
        self.assertTrue(np.allclose(scaled.mean(axis=0), 0, atol=1e-9))
        self.assertTrue(np.allclose(scaled.std(axis=0), 1, atol=1e-9))

    def test_scaler_handles_constant_column(self):
        data = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
        self.assertTrue(np.all(np.isfinite(StandardScaler().fit_transform(data))))

    def test_scaler_requires_fit(self):
        with self.assertRaises(ModelError):
            StandardScaler().transform(np.zeros((2, 2)))

    def test_scaler_rejects_wrong_width(self):
        scaler = StandardScaler().fit(np.zeros((4, 3)))
        with self.assertRaises(ModelError):
            scaler.transform(np.zeros((2, 5)))


class TestKNN(VisionTestCase):
    """The classifier must separate clean data and fail loudly when misused."""

    def test_knn_separates_two_blobs(self):
        features, labels = toy_data()
        model = KNNClassifier(k=3).fit(features, labels)
        self.assertEqual(model.predict([[0, 0, 0]]), ["a"])
        self.assertEqual(model.predict([[5, 5, 5]]), ["b"])

    def test_probabilities_sum_to_one(self):
        features, labels = toy_data()
        proba = KNNClassifier(k=5).fit(features, labels).predict_proba(features[:5])
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_predict_one_returns_confidence(self):
        features, labels = toy_data()
        label, confidence = KNNClassifier(k=3).fit(features, labels).predict_one(
            np.array([0.0, 0.0, 0.0]))
        self.assertEqual(label, "a")
        self.assertGreater(confidence, 0.5)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(ModelError):
            KNNClassifier().predict(np.zeros((1, 3)))

    def test_k_larger_than_dataset_raises(self):
        with self.assertRaises(ModelError):
            KNNClassifier(k=50).fit(np.zeros((4, 2)), ["a", "a", "b", "b"])

    def test_invalid_k_raises(self):
        with self.assertRaises(ModelError):
            KNNClassifier(k=0)

    def test_save_and_load_roundtrip(self):
        features, labels = toy_data()
        model = KNNClassifier(k=3).fit(features, labels)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            restored = KNNClassifier.load(path)
        self.assertEqual(restored.predict(features), model.predict(features))

    def test_load_missing_model_raises(self):
        with self.assertRaises(ModelError):
            KNNClassifier.load(Path("/nonexistent/model.json"))


class TestMetrics(VisionTestCase):
    """Metric code is implemented from definitions, so it is tested directly."""

    def test_confusion_matrix_counts(self):
        matrix = confusion_matrix(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
        self.assertEqual(matrix.tolist(), [[1, 1], [0, 1]])

    def test_evaluate_perfect_prediction(self):
        result = evaluate(["a", "b", "a"], ["a", "b", "a"])
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.macro_f1, 1.0)
        self.assertIn("precision", result.format_table())

    def test_evaluate_known_precision_recall(self):
        result = evaluate(["a", "a", "b", "b"], ["a", "b", "b", "b"])
        self.assertAlmostEqual(result.per_class["a"]["recall"], 0.5)
        self.assertAlmostEqual(result.per_class["b"]["precision"], 2 / 3)

    def test_evaluate_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            evaluate([], [])

    def test_evaluate_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            evaluate(["a"], ["a", "b"])

    def test_cross_validate_returns_score_per_k(self):
        features, labels = toy_data()
        scores = cross_validate_k(features, labels, [1, 3, 5])
        self.assertEqual(sorted(scores), [1, 3, 5])
        self.assertTrue(all(0.0 <= v <= 1.0 for v in scores.values()))

    def test_result_is_serialisable(self):
        payload = evaluate(["a", "b"], ["a", "b"]).to_dict()
        self.assertIn("confusion_matrix", payload)


if __name__ == "__main__":
    unittest.main()
