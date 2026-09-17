"""Stage 5 - analytics and report generation.

Produces three artefacts from a batch of inspections:

* ``inspection_report.csv``  - one row per image, for spreadsheets / MES import
* ``inspection_report.json`` - full nested detail, for downstream services
* ``summary.md``             - human-readable quality summary
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from ..detect.inspector import InspectionResult
from ..logger import get_logger
from ..models.evaluate import EvaluationResult

LOGGER = get_logger("report.reporter")


def summarise(results: list[InspectionResult]) -> dict:
    """Aggregate batch-level quality statistics."""
    total = len(results)
    defective = sum(1 for r in results if r.is_defective)
    by_verdict: dict[str, int] = {}
    for result in results:
        by_verdict[result.verdict] = by_verdict.get(result.verdict, 0) + 1

    latencies = [r.elapsed_ms for r in results] or [0.0]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "images_inspected": total,
        "defective": defective,
        "pass_rate": round((total - defective) / total, 4) if total else 0.0,
        "defect_rate": round(defective / total, 4) if total else 0.0,
        "verdict_counts": dict(sorted(by_verdict.items())),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "throughput_fps": round(1000.0 / (sum(latencies) / len(latencies)), 2)
        if sum(latencies) else 0.0,
    }


def write_csv(results: list[InspectionResult], path: Path) -> Path:
    """One row per inspected image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "verdict", "defect_count", "top_confidence",
                         "total_defect_area_px", "latency_ms"])
        for result in results:
            top = max((f.confidence for f in result.findings), default=0.0)
            area = sum(f.area for f in result.findings)
            writer.writerow([Path(result.source).name, result.verdict,
                             len(result.findings), f"{top:.4f}",
                             f"{area:.1f}", f"{result.elapsed_ms:.2f}"])
    LOGGER.info("Wrote CSV report -> %s", path)
    return path


def write_json(results: list[InspectionResult], path: Path,
               evaluation: EvaluationResult | None = None,
               extra: dict | None = None) -> Path:
    """Full machine-readable report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarise(results),
        "results": [r.to_dict() for r in results],
    }
    if evaluation is not None:
        payload["evaluation"] = evaluation.to_dict()
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote JSON report -> %s", path)
    return path


def write_markdown(results: list[InspectionResult], path: Path,
                   evaluation: EvaluationResult | None = None) -> Path:
    """Short human-readable quality summary."""
    stats = summarise(results)
    lines = [
        "# VisionInspect - Batch Inspection Summary", "",
        f"*Generated {stats['generated_at']}*", "",
        "## Throughput & yield", "",
        "| Metric | Value |", "| --- | --- |",
        f"| Images inspected | {stats['images_inspected']} |",
        f"| Defective | {stats['defective']} |",
        f"| Pass rate | {stats['pass_rate']:.1%} |",
        f"| Average latency | {stats['avg_latency_ms']} ms |",
        f"| Throughput | {stats['throughput_fps']} images/s |", "",
        "## Verdict distribution", "",
        "| Verdict | Count |", "| --- | --- |",
    ]
    lines += [f"| {name} | {count} |"
              for name, count in stats["verdict_counts"].items()]

    if evaluation is not None:
        lines += ["", "## Classification performance", "",
                  f"Accuracy **{evaluation.accuracy:.2%}**, "
                  f"macro-F1 **{evaluation.macro_f1:.3f}**", "",
                  "| Class | Precision | Recall | F1 | Support |",
                  "| --- | --- | --- | --- | --- |"]
        for name in evaluation.labels:
            m = evaluation.per_class[name]
            lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} "
                         f"| {m['f1']:.3f} | {int(m['support'])} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote Markdown summary -> %s", path)
    return path
