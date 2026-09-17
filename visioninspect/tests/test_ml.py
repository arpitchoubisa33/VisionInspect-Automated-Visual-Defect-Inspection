"""Tests for the baseline classifiers and cross-validation machinery."""

import unittest

import numpy as np

from base import VisionTestCase
from visioninspect.exceptions import ModelError
from visioninspect.models.baselines import GaussianNaiveBayes, NearestCentroid
from visioninspect.models.crossval import (compare_models, cross_validate,
                                           stratified_folds)
from visioninspect.models.knn import KNNClassifier


def blobs(n_per_class: int = 30):
    """Three separated Gaussian blobs in 4-D."""
    rng = np.random.default_rng(1)
    centres = {"a": 0.0, "b": 5.0, "c": 10.0}
    features, labels = [], []
    for name, centre in centres.items():
        features.append(rng.normal(centre, 0.5, size=(n_per_class, 4)))
        labels += [name] * n_per_class
    return np.vstack(features), labels


class TestBaselines(VisionTestCase):
    """Both baselines must satisfy the same interface contract as k-NN."""

    def test_nearest_centroid_separates_blobs(self):
        features, labels = blobs()
        model = NearestCentroid().fit(features, labels)
        self.assertEqual(model.predict([[0, 0, 0, 0]]), ["a"])
        self.assertEqual(model.predict([[10, 10, 10, 10]]), ["c"])

    def test_naive_bayes_separates_blobs(self):
        features, labels = blobs()
        model = GaussianNaiveBayes().fit(features, labels)
        self.assertEqual(model.predict([[5, 5, 5, 5]]), ["b"])

    def test_probabilities_sum_to_one(self):
        features, labels = blobs()
        for model in (NearestCentroid().fit(features, labels),
                      GaussianNaiveBayes().fit(features, labels)):
            with self.subTest(model=type(model).__name__):
                proba = model.predict_proba(features[:6])
                self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_predict_one_returns_label_and_confidence(self):
        features, labels = blobs()
        label, confidence = NearestCentroid().fit(
            features, labels).predict_one(np.zeros(4))
        self.assertEqual(label, "a")
        self.assertTrue(0.0 < confidence <= 1.0)

    def test_predict_before_fit_raises(self):
        for model in (NearestCentroid(), GaussianNaiveBayes()):
            with self.subTest(model=type(model).__name__):
                with self.assertRaises(ModelError):
                    model.predict(np.zeros((1, 4)))

    def test_naive_bayes_rejects_singleton_class(self):
        with self.assertRaises(ModelError):
            GaussianNaiveBayes().fit(np.zeros((3, 2)), ["a", "a", "b"])

    def test_naive_bayes_survives_constant_feature(self):
        """The variance floor must prevent a division by zero."""
        features, labels = blobs()
        features[:, 2] = 1.0                      # a perfectly constant column
        model = GaussianNaiveBayes().fit(features, labels)
        self.assertTrue(np.all(np.isfinite(model.predict_proba(features[:4]))))


class TestCrossValidation(VisionTestCase):
    """Folds must be disjoint, exhaustive and class-balanced."""

    def test_folds_are_disjoint_and_exhaustive(self):
        _, labels = blobs(20)
        folds = stratified_folds(labels, n_splits=5)
        merged = np.concatenate(folds)
        self.assertEqual(len(merged), len(labels))
        self.assertEqual(len(set(merged.tolist())), len(labels))

    def test_folds_preserve_class_balance(self):
        _, labels = blobs(20)
        labels_array = np.asarray(labels)
        for fold in stratified_folds(labels, n_splits=5):
            counts = {name: int((labels_array[fold] == name).sum())
                      for name in set(labels)}
            self.assertEqual(len(set(counts.values())), 1)

    def test_too_few_splits_raises(self):
        _, labels = blobs(10)
        with self.assertRaises(ModelError):
            stratified_folds(labels, n_splits=1)

    def test_class_smaller_than_n_splits_raises(self):
        with self.assertRaises(ModelError):
            stratified_folds(["a"] * 10 + ["b"] * 2, n_splits=5)

    def test_cross_validate_scores_every_fold(self):
        features, labels = blobs()
        result = cross_validate(lambda: KNNClassifier(k=3), features, labels,
                                "k-NN", n_splits=5)
        self.assertEqual(len(result.fold_scores), 5)
        self.assertGreater(result.mean, 0.9)
        self.assertGreaterEqual(result.std, 0.0)

    def test_compare_models_sorts_best_first(self):
        features, labels = blobs()
        results = compare_models(
            {"k-NN": lambda: KNNClassifier(k=3),
             "Nearest Centroid": NearestCentroid}, features, labels, n_splits=4)
        means = [r.mean for r in results]
        self.assertEqual(means, sorted(means, reverse=True))

    def test_result_is_serialisable(self):
        features, labels = blobs()
        payload = cross_validate(NearestCentroid, features, labels,
                                 "NC", n_splits=3).to_dict()
        self.assertEqual(payload["model"], "NC")
        self.assertIn("mean_accuracy", payload)


if __name__ == "__main__":
    unittest.main()
