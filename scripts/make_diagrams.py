#!/usr/bin/env python3
"""Render the design diagrams as PNGs for the PDF report.

The Markdown docs use Mermaid, which GitHub renders natively but a PDF cannot.
Rather than depend on a Node toolchain, the same diagrams are drawn here with
Matplotlib primitives so the report builds anywhere the project runs.

Run:
    python scripts/make_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse

OUT = Path(__file__).resolve().parents[1] / "docs" / "diagrams"

INK = "#1f2430"
MUTED = "#6b7280"
PALETTE = {
    "pres": "#dbeafe", "app": "#fde68a", "dom": "#bbf7d0",
    "infra": "#e9d5ff", "store": "#fecaca", "plain": "#f1f5f9",
}


def _canvas(width: float, height: float, title: str):
    """Create a titled, axis-free canvas in data units 0..100."""
    fig, axis = plt.subplots(figsize=(width, height))
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.axis("off")
    axis.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=10)
    return fig, axis


def box(axis, x, y, w, h, text, fill="plain", fontsize=8, bold=False):
    """Draw a rounded box centred at ``(x, y)`` with wrapped label text."""
    axis.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.6,rounding_size=1.6",
        facecolor=PALETTE.get(fill, fill), edgecolor=INK, linewidth=1.0))
    axis.text(x, y, text, ha="center", va="center", fontsize=fontsize,
              color=INK, fontweight="bold" if bold else "normal", linespacing=1.35)


def group(axis, x, y, w, h, label, colour):
    """Draw a dashed container with a caption in its top-left corner."""
    axis.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.8,rounding_size=1.6",
        facecolor=colour, edgecolor=MUTED, linewidth=1.0,
        linestyle="--", alpha=0.35))
    axis.text(x - w / 2 + 1.0, y + h / 2 + 1.4, label, ha="left", va="bottom",
              fontsize=8.5, color=MUTED, fontweight="bold")


def arrow(axis, start, end, style="-|>", dashed=False, label=None,
          colour=INK, rad=0.0):
    """Draw a connector between two points."""
    axis.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=11,
        linewidth=1.0, color=colour,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 1.4,
                  label, ha="center", fontsize=7, color=MUTED, style="italic")


def save(fig, name: str) -> Path:
    """Write a figure into the diagrams folder."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(OUT.parents[1])}")
    return path


# ------------------------------------------------------------ 1. architecture
def architecture() -> None:
    """Layered system-architecture diagram."""
    fig, ax = _canvas(10, 8.2, "System Architecture — layered view")

    group(ax, 50, 90, 94, 11, "PRESENTATION", "#dbeafe")
    box(ax, 28, 90, 30, 7.5, "run.py\nentry point", "pres")
    box(ax, 68, 90, 34, 7.5, "cli.py\nargparse sub-commands", "pres")

    group(ax, 50, 73, 94, 11, "APPLICATION", "#fde68a")
    box(ax, 50, 73, 46, 7.5, "detect/inspector.py\nDefectInspector facade · verdict logic", "app")

    group(ax, 50, 48, 94, 24, "DOMAIN — computer vision", "#bbf7d0")
    box(ax, 18, 54, 28, 8.5, "preprocess.py\nresize · median · CLAHE", "dom", 7.2)
    box(ax, 50, 54, 28, 8.5, "segment.py\nbackground · MAD · regions", "dom", 7.2)
    box(ax, 82, 54, 28, 8.5, "extractor.py\n13-D descriptors", "dom", 7.2)
    box(ax, 34, 41, 30, 8.5, "knn.py · scaler.py\nweighted k-NN", "dom", 7.2)
    box(ax, 70, 41, 30, 8.5, "evaluate.py\nconfusion · P/R/F1 · k sweep", "dom", 7.2)

    group(ax, 50, 21, 94, 20, "INFRASTRUCTURE", "#e9d5ff")
    box(ax, 18, 25, 28, 7.5, "data/synthetic.py\ndataset generator", "infra", 7.2)
    box(ax, 50, 25, 28, 7.5, "data/loader.py\nvalidation · split", "infra", 7.2)
    box(ax, 82, 25, 28, 7.5, "report/reporter.py\nCSV · JSON · MD", "infra", 7.2)
    box(ax, 34, 14, 30, 7.5, "report/visualize.py\noverlays · plots", "infra", 7.2)
    box(ax, 70, 14, 30, 7.5, "config · logger · exceptions", "infra", 7.2)

    box(ax, 50, 3, 44, 6, "File system — dataset · model.json · outputs", "store", 8, True)

    arrow(ax, (28, 86.2), (44, 77))
    arrow(ax, (68, 86.2), (56, 77))
    arrow(ax, (44, 69.2), (20, 58.5))
    arrow(ax, (32, 54), (36, 54))
    arrow(ax, (64, 54), (68, 54))
    arrow(ax, (50, 49.7), (38, 45.5))
    arrow(ax, (82, 49.7), (46, 43), rad=0.12)
    arrow(ax, (56, 69.2), (70, 45.5), dashed=True)
    arrow(ax, (34, 36.7), (34, 29), dashed=True)
    arrow(ax, (70, 36.7), (78, 29), dashed=True)
    arrow(ax, (50, 9), (50, 6.5), style="<|-|>")
    save(fig, "01_architecture.png")


# ---------------------------------------------------------------- 2. workflow
def workflow() -> None:
    """Process-flow diagram for a single inspection."""
    fig, ax = _canvas(8.6, 10.5, "Process Flow — single image inspection")

    steps = [
        (50, 95, "Load image"),
        (50, 86, "Grayscale · resize 256×256"),
        (50, 78, "Median blur 3×3"),
        (50, 70, "CLAHE  clip 1.5 · 4×4 tiles"),
        (50, 62, "Estimate background\nclose ∘ open · 41×41 ellipse"),
        (50, 53, "Residual = | image − background |"),
        (50, 45, "Threshold at median + 4.5·MAD"),
        (50, 37, "Morphological open, then close"),
        (50, 29, "Find contours · filter by min area"),
    ]
    for x, y, text in steps:
        box(ax, x, y, 56, 6.4, text, "dom", 8)
    for i in range(len(steps) - 1):
        arrow(ax, (50, steps[i][1] - 3.6), (50, steps[i + 1][1] + 3.6))

    # Decision diamond.
    ax.plot([50, 66, 50, 34, 50], [24, 18, 12, 18, 24], color=INK, lw=1.0)
    ax.fill([50, 66, 50, 34], [24, 18, 12, 18], color="#fde68a", alpha=0.85)
    ax.text(50, 18, "regions\nfound?", ha="center", va="center", fontsize=8)
    arrow(ax, (50, 25.8), (50, 24.2))

    box(ax, 15, 18, 24, 6.4, "Verdict = good", "#bbf7d0", 8, True)
    arrow(ax, (34, 18), (27, 18), label="no")

    box(ax, 82, 18, 28, 7.6, "For each region:\n13 features → k-NN", "dom", 8)
    arrow(ax, (66, 18), (68, 18), label="yes")
    box(ax, 82, 8, 28, 7.6, "Verdict = class of\nmax(area × confidence)", "#bbf7d0", 8, True)
    arrow(ax, (82, 14.2), (82, 11.8))
    box(ax, 40, 3, 40, 5.6, "Annotate overlay · append to report", "infra", 8)
    arrow(ax, (68, 8), (60, 3.5), rad=0.15)
    arrow(ax, (15, 14.8), (25, 4.5), rad=-0.15)
    save(fig, "02_workflow.png")


# ---------------------------------------------------------------- 3. use case
def use_case() -> None:
    """Actors and the use cases they drive."""
    fig, ax = _canvas(10, 7.6, "Use Case Diagram")

    ax.add_patch(FancyBboxPatch((28, 4), 44, 92,
                                boxstyle="round,pad=0.8,rounding_size=2",
                                facecolor="#f8fafc", edgecolor=MUTED, lw=1.2))
    ax.text(50, 93, "VisionInspect", ha="center", fontsize=10,
            fontweight="bold", color=INK)

    cases = [
        (50, 86, "Inspect a single part"),
        (50, 77, "View annotated overlay"),
        (50, 68, "Run a batch inspection"),
        (50, 59, "Evaluate accuracy"),
        (50, 50, "Tune detection sensitivity"),
        (50, 41, "Export CSV / JSON reports"),
        (50, 32, "Generate dataset"),
        (50, 23, "Train classifier"),
        (50, 14, "Run robustness study"),
        (50, 7, "Run test suite"),
    ]
    for x, y, text in cases:
        ax.add_patch(Ellipse((x, y), 38, 7.2, facecolor="#bbf7d0",
                             edgecolor=INK, lw=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=8, color=INK)

    actors = [(11, 80, "Line\nOperator", [86, 77]),
              (11, 50, "Quality\nEngineer", [68, 59, 50, 41]),
              (89, 24, "Developer /\nReviewer", [32, 23, 14, 7])]
    for ax_x, ax_y, name, targets in actors:
        ax.plot([ax_x], [ax_y + 4], marker="o", ms=8, color=INK)
        ax.plot([ax_x, ax_x], [ax_y + 2.4, ax_y - 2], color=INK, lw=1.2)
        ax.plot([ax_x - 3, ax_x + 3], [ax_y + 0.6, ax_y + 0.6], color=INK, lw=1.2)
        ax.plot([ax_x, ax_x - 2.6], [ax_y - 2, ax_y - 6], color=INK, lw=1.2)
        ax.plot([ax_x, ax_x + 2.6], [ax_y - 2, ax_y - 6], color=INK, lw=1.2)
        ax.text(ax_x, ax_y - 9.5, name, ha="center", fontsize=8.5,
                fontweight="bold", color=INK)
        for target in targets:
            start = (ax_x + 5, ax_y) if ax_x < 50 else (ax_x - 5, ax_y)
            end = (31, target) if ax_x < 50 else (69, target)
            arrow(ax, start, end, style="-", colour=MUTED, rad=0.06)

    arrow(ax, (69, 86), (69, 78), dashed=True, colour="#2563eb", rad=-0.5)
    ax.text(76, 82, "«include»", fontsize=6.5, color="#2563eb", style="italic")
    arrow(ax, (31, 68), (31, 87), dashed=True, colour="#2563eb", rad=0.5)
    ax.text(21, 76, "«include»", fontsize=6.5, color="#2563eb", style="italic")
    save(fig, "03_use_case.png")


# ------------------------------------------------------------------- 4. class
def class_diagram() -> None:
    """Core domain classes and their relationships."""
    fig, ax = _canvas(11, 8.6, "Class Diagram — core domain")

    def uml(x, top, w, name, attrs, ops, fill="dom"):
        """Draw a UML class box whose height is derived from its content."""
        line = 2.6
        height = 10.0 + line * (len(attrs) + len(ops))
        y = top - height / 2
        box(ax, x, y, w, height, "", fill)
        cursor = top - 3.0
        ax.text(x, cursor, name, ha="center", va="center", fontsize=8.2,
                fontweight="bold", color=INK)
        cursor -= 2.6
        ax.plot([x - w / 2 + 1, x + w / 2 - 1], [cursor] * 2, color=INK, lw=0.8)
        for attr in attrs:
            cursor -= line
            ax.text(x - w / 2 + 2.2, cursor, attr, ha="left", va="center",
                    fontsize=6.5, color=INK)
        cursor -= line * 0.6
        ax.plot([x - w / 2 + 1, x + w / 2 - 1], [cursor] * 2, color=INK, lw=0.8)
        for op in ops:
            cursor -= line
            ax.text(x - w / 2 + 2.2, cursor, op, ha="left", va="center",
                    fontsize=6.5, color=INK)
        return top - height       # bottom edge

    uml(18, 98, 32, "AppConfig",
        ["+ data: DataConfig", "+ preprocess: PreprocessConfig",
         "+ segment: SegmentConfig", "+ model: ModelConfig"],
        ["+ from_json(path)", "+ to_dict()"], "infra")

    uml(63, 98, 38, "DefectInspector",
        ["- config: AppConfig", "- model: KNNClassifier"],
        ["+ build_training_set(samples)", "+ train(samples)",
         "+ inspect_array(image)", "+ inspect_file(path)",
         "+ inspect_batch(samples)"], "app")

    uml(16, 58, 28, "Region",
        ["+ contour: ndarray", "+ mask: ndarray", "+ bbox: tuple",
         "+ area: float"], ["+ centroid()"])

    uml(52, 58, 30, "InspectionResult",
        ["+ source: str", "+ verdict: str", "+ findings: list",
         "+ elapsed_ms: float"], ["+ is_defective()", "+ to_dict()"])

    uml(85, 58, 26, "DefectFinding",
        ["+ label: str", "+ confidence: float", "+ area: float",
         "+ bbox: tuple"], ["+ severity()", "+ to_dict()"])

    uml(27, 24, 32, "KNNClassifier",
        ["- k: int", "- scaler: StandardScaler", "- _X, _y: ndarray"],
        ["+ fit()   + predict()", "+ predict_one()   + save()"])

    uml(71, 24, 30, "StandardScaler",
        ["- mean_: ndarray", "- std_: ndarray"],
        ["+ fit()   + transform()", "+ state_dict()"])

    arrow(ax, (34.5, 90), (43.5, 90), label="configured by")
    arrow(ax, (57, 73), (54, 62), style="-|>")
    ax.text(60, 67, "produces", fontsize=6.5, color=MUTED, style="italic")
    arrow(ax, (67.5, 45), (71.5, 45), style="-|>")
    ax.text(62, 47.5, "1..*", fontsize=6.5, color=MUTED)
    arrow(ax, (30.5, 45), (36.5, 45), dashed=True)
    ax.text(30, 47.5, "derived from", fontsize=6.5, color=MUTED, style="italic")
    # Elbow route down the gap between Region and InspectionResult.
    ax.plot([46, 33.5], [72, 72], color=INK, lw=1.0, linestyle="--")
    ax.plot([33.5, 33.5], [72, 30], color=INK, lw=1.0, linestyle="--")
    arrow(ax, (33.5, 30), (33.5, 24.5), dashed=True)
    ax.text(35, 36, "uses", fontsize=6.5, color=MUTED, style="italic")
    arrow(ax, (43.5, 15), (55.5, 15), style="-|>")
    ax.text(48.5, 17.5, "composed of", fontsize=6.5, color=MUTED, style="italic")
    save(fig, "04_class.png")


# ---------------------------------------------------------------- 5. sequence
def sequence() -> None:
    """Sequence diagram for the inspect command."""
    fig, ax = _canvas(10.5, 7.8, "Sequence Diagram — inspect a single image")

    lanes = [(6, "User"), (20, "cli.py"), (36, "DefectInspector"),
             (52, "segment"), (66, "extractor"), (81, "KNNClassifier"),
             (95, "visualize")]
    for x, name in lanes:
        box(ax, x, 94, 12.5, 6, name, "pres", 7.0, True)
        ax.plot([x, x], [90, 6], color=MUTED, lw=0.8, linestyle="--")

    # (source lane, destination lane, y, label, is_return_message)
    msgs = [
        (6, 20, 86, "run.py inspect part.png", False),
        (20, 81, 80, "load(model.json)", False),
        (20, 36, 73, "inspect_file(path)", False),
        (36, 36, 67, "preprocess(image, config)", False),
        (36, 52, 61, "segment(image, config)", False),
        (52, 36, 55, "regions[], mask", True),
        (36, 66, 47, "extract(image, region)", False),
        (66, 36, 41, "13-D descriptor", True),
        (36, 81, 35, "predict_one(descriptor)", False),
        (81, 36, 29, "(label, confidence)", True),
        (36, 20, 20, "InspectionResult", True),
        (20, 95, 14, "save_overlay(result)", False),
        (20, 6, 8, "verdict + file paths", True),
    ]
    for src, dst, y, text, returning in msgs:
        if src == dst:
            ax.add_patch(FancyArrowPatch((src, y + 1.5), (src, y - 1.5),
                         arrowstyle="-|>", mutation_scale=9, lw=1.0,
                         color=INK, connectionstyle="arc3,rad=-2.2"))
            ax.text(src + 5, y, text, fontsize=6.6, color=INK, va="center")
        else:
            arrow(ax, (src, y), (dst, y), dashed=returning,
                  colour=MUTED if returning else INK)
            ax.text((src + dst) / 2, y + 1.3, text, ha="center",
                    fontsize=6.6, color=INK)

    # Loop frame over the per-region messages.
    ax.add_patch(FancyBboxPatch((29, 26), 58, 25, boxstyle="square,pad=0",
                                facecolor="none", edgecolor="#2563eb",
                                lw=0.9, linestyle="--"))
    ax.text(30, 49.5, "loop  [for each region]", fontsize=6.6,
            color="#2563eb", fontweight="bold")
    ax.text(44, 23, "alt  no region → verdict = good (loop skipped)",
            fontsize=6.6, color="#b45309", style="italic")
    save(fig, "05_sequence.png")


# ---------------------------------------------------------------------- 6. ER
def er_diagram() -> None:
    """Logical entity-relationship model of the inspection store."""
    fig, ax = _canvas(10.5, 8.2, "ER Diagram — inspection data model")

    def entity(x, top, w, name, fields, fill="infra"):
        """Draw an entity box sized to its field list."""
        line = 2.9
        height = 8.5 + line * len(fields)
        y = top - height / 2
        box(ax, x, y, w, height, "", fill)
        cursor = top - 3.2
        ax.text(x, cursor, name, ha="center", va="center", fontsize=8.2,
                fontweight="bold", color=INK)
        cursor -= 2.8
        ax.plot([x - w / 2 + 1, x + w / 2 - 1], [cursor] * 2, color=INK, lw=0.8)
        for field in fields:
            cursor -= line
            ax.text(x - w / 2 + 2.2, cursor, field, ha="left", va="center",
                    fontsize=6.6, color=INK)

    entity(17, 98, 30, "DATASET",
           ["dataset_dir  PK", "samples_per_class", "image_size", "random_seed"])
    entity(17, 70, 30, "IMAGE",
           ["path  PK", "dataset_dir  FK", "true_label", "split"], "dom")
    entity(17, 40, 30, "INSPECTION",
           ["inspection_id  PK", "image_path  FK", "verdict", "elapsed_ms"], "dom")
    entity(53, 40, 30, "FINDING",
           ["finding_id  PK", "inspection_id  FK", "label  FK",
            "confidence", "area_px", "bbox / centroid"], "dom")
    entity(53, 82, 30, "DEFECT_CLASS",
           ["label  PK", "description", "disposition"], "store")
    entity(81, 98, 26, "MODEL",
           ["model_path  PK", "k", "class_names", "scaler_mean / std"])
    entity(81, 66, 26, "TRAINING_REGION",
           ["region_id  PK", "model_path  FK", "label  FK", "feature_vector"])

    def relate(p1, p2, left, right):
        arrow(ax, p1, p2, style="-", colour=INK)
        ax.text(p1[0] + 1.4, p1[1] - 2.4, left, fontsize=6.4, color="#b45309")
        ax.text(p2[0] + 1.4, p2[1] + 1.4, right, fontsize=6.4, color="#b45309")

    relate((17, 78), (17, 70), "1", "N")
    relate((17, 48), (17, 40), "1", "0..1")
    relate((32, 28), (38, 28), "1", "N")
    relate((53, 40), (53, 65), "N", "1")
    relate((81, 78), (81, 66), "1", "N")
    relate((68, 55), (60, 65), "N", "1")

    # MODEL is used by every INSPECTION: routed along the bottom margin.
    ax.plot([81, 81, 17, 17], [37, 6, 6, 18], color=MUTED, lw=1.0,
            linestyle="--")
    arrow(ax, (17, 12), (17, 18), dashed=True, colour=MUTED)
    ax.text(52, 7.5, "model used by inspection", fontsize=6.6,
            color=MUTED, style="italic", ha="center")
    save(fig, "06_er.png")


def main() -> int:
    """Render every diagram."""
    print("Rendering design diagrams...")
    architecture()
    workflow()
    use_case()
    class_diagram()
    sequence()
    er_diagram()
    print(f"Done -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
