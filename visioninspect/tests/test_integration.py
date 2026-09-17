"""End-to-end integration tests over a small generated dataset."""

import shutil
import tempfile
import unittest
from pathlib import Path

from base import VisionTestCase
from visioninspect.config import AppConfig, DataConfig, ModelConfig
from visioninspect.data.loader import (class_distribution, discover_samples,
                                       load_image, stratified_split)
from visioninspect.data.synthetic import generate_dataset
from visioninspect.detect.inspector import DefectInspector
from visioninspect.exceptions import DatasetError, ImageLoadError
from visioninspect.models.evaluate import evaluate
from visioninspect.report import reporter, visualize


class TestEndToEnd(VisionTestCase):
    """Generate a small dataset once, then exercise the whole pipeline on it."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.tmp = Path(tempfile.mkdtemp(prefix="visioninspect_test_"))
        dataset = cls.tmp / "dataset"
        generate_dataset(dataset, samples_per_class=12, size=256, seed=3)

        cls.config = AppConfig(
            data=DataConfig(dataset_dir=dataset, samples_per_class=12),
            model=ModelConfig(k=3, test_ratio=0.25),
            output_dir=cls.tmp / "outputs",
        )
        samples = discover_samples(dataset)
        cls.train_samples, cls.test_samples = stratified_split(samples, 0.25, 42)
        cls.inspector = DefectInspector(cls.config)
        cls.inspector.train(cls.train_samples)
        cls.results = cls.inspector.inspect_batch(cls.test_samples)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # -------------------------------------------------------------- dataset
    def test_dataset_is_balanced(self):
        counts = class_distribution(discover_samples(self.config.data.dataset_dir))
        self.assertEqual(set(counts), {"good", "scratch", "spot", "crack"})
        self.assertEqual(len(set(counts.values())), 1)

    def test_missing_dataset_raises(self):
        with self.assertRaises(DatasetError):
            discover_samples(self.tmp / "absent")

    def test_unsupported_file_raises(self):
        bad = self.tmp / "notes.txt"
        bad.write_text("hello")
        with self.assertRaises(ImageLoadError):
            load_image(bad)

    def test_split_is_stratified(self):
        self.assertEqual(set(class_distribution(self.train_samples)),
                         set(class_distribution(self.test_samples)))

    # ------------------------------------------------------------ inference
    def test_good_images_are_passed(self):
        """No false alarms on defect-free parts."""
        goods = [s for s in self.test_samples if s.label == "good"]
        verdicts = [self.inspector.inspect_file(s.path).verdict for s in goods]
        self.assertEqual(verdicts, ["good"] * len(goods))

    def test_defects_are_caught(self):
        """Recall on the defective class: nothing bad may ship."""
        defects = [s for s in self.test_samples if s.label != "good"]
        flagged = [self.inspector.inspect_file(s.path).is_defective
                   for s in defects]
        self.assertTrue(all(flagged))

    def test_end_to_end_accuracy_is_high(self):
        truth = {s.path.name: s.label for s in self.test_samples}
        y_true = [truth[Path(r.source).name] for r in self.results]
        y_pred = [r.verdict for r in self.results]
        self.assertGreaterEqual(evaluate(y_true, y_pred).accuracy, 0.85)

    def test_latency_budget(self):
        """Non-functional requirement: under 250 ms per frame."""
        self.assertLess(max(r.elapsed_ms for r in self.results), 250.0)

    def test_result_is_serialisable(self):
        payload = self.results[0].to_dict()
        self.assertLessEqual({"source", "verdict", "findings"}, payload.keys())

    def test_corrupt_file_is_skipped_not_fatal(self):
        """A bad frame must not abort a production batch."""
        from visioninspect.data.loader import Sample
        broken = self.tmp / "broken.png"
        broken.write_bytes(b"not really a png")
        batch = list(self.test_samples[:2]) + [Sample(path=broken, label="good")]
        self.assertEqual(len(self.inspector.inspect_batch(batch)), 2)

    # -------------------------------------------------------------- reports
    def test_reports_are_written(self):
        out = self.tmp / "reports"
        csv_path = reporter.write_csv(self.results, out / "r.csv")
        json_path = reporter.write_json(self.results, out / "r.json")
        md_path = reporter.write_markdown(self.results, out / "r.md")
        self.assertTrue(csv_path.exists() and json_path.exists())
        self.assertIn("Batch Inspection Summary", md_path.read_text())

    def test_summary_statistics(self):
        stats = reporter.summarise(self.results)
        self.assertEqual(stats["images_inspected"], len(self.results))
        self.assertTrue(0.0 <= stats["pass_rate"] <= 1.0)

    def test_overlay_is_rendered(self):
        defective = next(r for r in self.results if r.is_defective)
        path = visualize.save_overlay(defective, self.tmp / "overlays")
        self.assertTrue(path.exists() and path.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
