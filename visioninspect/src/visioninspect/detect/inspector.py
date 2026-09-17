"""Stage 4 - the inspection facade.

:class:`DefectInspector` wires preprocessing, segmentation, feature extraction
and classification into one object with a small surface area
(``train`` / ``inspect``). Keeping the orchestration here means the CLI, the
tests and any future GUI all drive the same code path.

Verdict rule
------------
An image is ``good`` when segmentation finds no region above the minimum area.
Otherwise the verdict is the class of the highest-severity region, where
severity is region area weighted by classifier confidence - a large confident
crack outranks a small uncertain speck.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import AppConfig
from ..data.loader import Sample
from ..exceptions import ModelError
from ..features.extractor import extract, FEATURE_DIM
from ..logger import get_logger
from ..models.knn import KNNClassifier
from ..pipeline.preprocess import preprocess
from ..pipeline.segment import Region, segment

LOGGER = get_logger("detect.inspector")


@dataclass
class DefectFinding:
    """One classified defect inside an inspected image."""

    label: str
    confidence: float
    area: float
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]

    @property
    def severity(self) -> float:
        """Area scaled by confidence; used to pick the dominant defect."""
        return self.area * self.confidence

    def to_dict(self) -> dict:
        """JSON-friendly view."""
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "area_px": round(self.area, 1),
            "bbox": list(self.bbox),
            "centroid": [round(self.centroid[0], 1), round(self.centroid[1], 1)],
        }


@dataclass
class InspectionResult:
    """Everything produced for a single inspected image."""

    source: str
    verdict: str
    findings: list[DefectFinding] = field(default_factory=list)
    elapsed_ms: float = 0.0
    processed: np.ndarray | None = None
    mask: np.ndarray | None = None
    regions: list[Region] = field(default_factory=list)

    @property
    def is_defective(self) -> bool:
        """True when at least one defect was accepted."""
        return self.verdict != "good"

    def to_dict(self) -> dict:
        """Serialisable summary (image buffers are intentionally excluded)."""
        return {
            "source": self.source,
            "verdict": self.verdict,
            "defect_count": len(self.findings),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "findings": [f.to_dict() for f in self.findings],
        }


class DefectInspector:
    """Facade over the whole computer-vision pipeline."""

    def __init__(self, config: AppConfig, model: KNNClassifier | None = None):
        self.config = config
        self.model = model

    # ------------------------------------------------------------- training
    def build_training_set(self, samples: list[Sample]
                           ) -> tuple[np.ndarray, list[str]]:
        """Segment every training image and label each region with its class.

        Regions found in ``good`` images are discarded rather than used as a
        fourth class: they are residual texture, not a defect morphology, and
        including them would blur the decision boundary. Clean images are
        instead handled by the "no region -> good" rule at inference time.
        """
        vectors: list[np.ndarray] = []
        labels: list[str] = []

        for sample in samples:
            if sample.label == "good":
                continue
            image = preprocess(sample.load(), self.config.preprocess)
            regions, _ = segment(image, self.config.segment)
            if not regions:
                LOGGER.debug("No region found in %s", sample.path.name)
                continue
            # The dominant region carries the annotated defect.
            vectors.append(extract(image, regions[0]))
            labels.append(sample.label)

        if not vectors:
            raise ModelError("No defect regions could be extracted for training")
        LOGGER.info("Built training matrix: %d x %d", len(vectors), FEATURE_DIM)
        return np.vstack(vectors), labels

    def train(self, samples: list[Sample]) -> KNNClassifier:
        """Fit the classifier on region descriptors mined from ``samples``."""
        features, labels = self.build_training_set(samples)
        model = KNNClassifier(k=self.config.model.k,
                              weighted=self.config.model.weighted)
        self.model = model.fit(features, labels)
        return self.model

    # ------------------------------------------------------------ inference
    def inspect_array(self, image: np.ndarray, source: str = "<array>"
                      ) -> InspectionResult:
        """Run the pipeline on an in-memory image."""
        if self.model is None:
            raise ModelError("Inspector has no model; call train() or load one")

        started = time.perf_counter()
        processed = preprocess(image, self.config.preprocess)
        regions, mask = segment(processed, self.config.segment)

        findings: list[DefectFinding] = []
        for region in regions:
            label, confidence = self.model.predict_one(extract(processed, region))
            findings.append(DefectFinding(
                label=label, confidence=confidence, area=region.area,
                bbox=region.bbox, centroid=region.centroid,
            ))

        verdict = (max(findings, key=lambda f: f.severity).label
                   if findings else "good")
        elapsed = (time.perf_counter() - started) * 1000.0

        return InspectionResult(
            source=source, verdict=verdict, findings=findings,
            elapsed_ms=elapsed, processed=processed, mask=mask, regions=regions,
        )

    def inspect_file(self, path: Path) -> InspectionResult:
        """Run the pipeline on an image file."""
        from ..data.loader import load_image
        path = Path(path)
        result = self.inspect_array(load_image(path), source=str(path))
        LOGGER.info("%-28s -> %-8s (%d region(s), %.1f ms)",
                    path.name, result.verdict, len(result.findings),
                    result.elapsed_ms)
        return result

    def inspect_batch(self, samples: list[Sample]) -> list[InspectionResult]:
        """Inspect many samples, isolating per-image failures."""
        results: list[InspectionResult] = []
        for sample in samples:
            try:
                results.append(self.inspect_file(sample.path))
            except Exception as exc:                     # noqa: BLE001
                # One corrupt frame must not abort a production batch.
                LOGGER.error("Skipping %s: %s", sample.path.name, exc)
        return results
