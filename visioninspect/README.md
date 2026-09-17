# VisionInspect — Automated Visual Defect Inspection

A complete classical **computer-vision** pipeline that inspects surface images,
localises defects, classifies their morphology (**scratch / spot / crack**) and
produces inspection reports — with no deep learning, no GPU and no reference
("golden") image.

**Headline results on the held-out test split (60 images):** 100% accuracy,
0 false alarms on defect-free parts, ~6 ms per image (≈166 images/s on CPU),
and 93% accuracy retained under σ = 16 additive sensor noise.

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [How it works](#how-it-works)
- [Technologies used](#technologies-used)
- [Installation](#installation)
- [Running the project](#running-the-project)
- [Testing](#testing)
- [Results](#results)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Documentation](#documentation)

---

## Overview

Manual surface inspection is slow, inconsistent and typically catches only
60–80% of defects. VisionInspect replaces the "look at the part under a lamp"
step with a deterministic pipeline:

```
image → preprocess → estimate background → residual → threshold
      → segment regions → extract 13 descriptors → k-NN → verdict + report
```

The central design idea is that **a defect is whatever the surface is not**.
Instead of comparing each part against a stored golden image (which breaks the
moment the product or the lighting changes), the defect-free surface is
*reconstructed from the image itself* by morphological opening and closing with
a large structuring element. Subtracting that estimate leaves only local
anomalies. A robust median/MAD noise model then decides which of those
anomalies are real, so the detector does not invent defects on a clean part.

Each surviving region is described by thirteen interpretable features —
elongation separates a scratch from a spot, solidity separates a branching
crack from a straight scratch, circularity separates a spot from both — and a
from-scratch distance-weighted k-NN classifier assigns the defect type.

## Features

**Three major functional modules** (see [`docs/DESIGN.md`](docs/DESIGN.md)):

| # | Module | Responsibility |
| --- | --- | --- |
| **M1** | **Data & Preprocessing** | Synthetic dataset generation, image discovery and validation, stratified splitting, resize / denoise / CLAHE conditioning. |
| **M2** | **Detection & Classification** | Background estimation, residual thresholding, morphological cleanup, region extraction, 13-D feature engineering, k-NN training and inference, verdict logic. |
| **M3** | **Analytics & Reporting** | Confusion matrix and per-class precision / recall / F1, hyper-parameter sweep, annotated overlays and pipeline figures, CSV / JSON / Markdown reports. |

Additional capabilities:

- Reproducible **synthetic dataset generator** — the project runs offline from a clean checkout.
- **Reference-free detection** — no golden image, no manual ROI setup.
- **From-scratch k-NN, Nearest Centroid, Gaussian Naive Bayes and all metrics** — no scikit-learn; the distance metric, weighting, variance floor and tie-breaking are all explicit.
- **Model-selection study** — stratified 5-fold cross-validation, feature ablation and learning curves comparing all three classifiers.
- **JSON model persistence** — train once, inspect anywhere.
- **Robustness study** — accuracy re-measured under noise, defocus and illumination drift.
- **77 unit and integration tests** using only the standard library.

## How it works

| Stage | Technique | Why this choice |
| --- | --- | --- |
| 1. Conditioning | Resize → median blur (3×3) → CLAHE (clip 1.5, 4×4 tiles) | Median preserves thin scratch edges that a Gaussian would erase; CLAHE removes the illumination gradient so a single threshold works across the frame. |
| 2. Background estimation | Morphological close → open, 41×41 ellipse | Reconstructs the surface *without* the defect, removing the need for a golden reference. |
| 3. Residual | Absolute difference | Isolates local anomalies of either polarity (dark scratch or bright stain). |
| 4. Thresholding | median + 4.5 × MAD, with an absolute floor | Otsu was tried first and **fails**: on a clean part it splits sensor noise into a "defect" class. MAD is unaffected by the defect's outlier pixels. |
| 5. Cleanup | Open then close (3×3), area filter | Removes speckle, bridges gaps inside one defect. |
| 6. Features | 8 shape + 2 photometric + 3 Hu moments | Interpretable and rotation-invariant; separable with only ~100 training regions. |
| 7. Classification | Distance-weighted k-NN (k=5) on z-scored features | Non-parametric, trains instantly, and its errors are explainable by looking at the neighbours. |
| 8. Verdict | No region → `good`; otherwise the class of the highest area × confidence region | A large confident crack outranks a small uncertain speck. |

## Technologies used

| Tool | Version | Role |
| --- | --- | --- |
| Python | 3.10+ | Implementation language |
| OpenCV (`opencv-python`) | ≥ 4.8 | Filtering, morphology, contours, moments |
| NumPy | ≥ 1.24 | Vectorised maths, distance computation |
| Matplotlib | ≥ 3.7 | Confusion matrix, sweep and robustness plots |
| ReportLab | ≥ 4.0 | Optional PDF report build |
| `unittest`, `argparse`, `dataclasses`, `logging`, `csv`, `json` | stdlib | Testing, CLI, typed config, observability, reporting |
| Git | — | Version control |

## Installation

```bash
# 1. Clone
git clone <your-repository-url>
cd visioninspect

# 2. Create an isolated environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

No dataset download is required — the dataset is generated locally.

## Running the project

### Everything at once

```bash
python run.py pipeline --samples 60
```

This generates the dataset, trains the classifier, evaluates the held-out
split, and writes reports and figures to `outputs/`.

### Step by step

```bash
# 1. Build a balanced synthetic dataset (240 images, 4 classes)
python run.py generate --samples 60

# 2. Train the k-NN classifier and persist it
python run.py train

# 3. Inspect a single image (writes an annotated overlay)
python run.py inspect data/dataset/crack/crack_0003.png

# 4. Score the test split and write CSV / JSON / Markdown reports
python run.py evaluate --overlays 8

# 5. Optional: robustness study under noise, blur and brightness drift
python scripts/robustness_study.py

# 6. Optional: model selection — CV, feature ablation, learning curves
python scripts/model_selection.py

# 7. Optional: rebuild the PDF project report
python scripts/build_report.py
```

### Useful flags

```bash
python run.py --log-level DEBUG evaluate      # verbose pipeline tracing
python run.py --config config.example.json train   # JSON config override
python run.py --version
python run.py inspect --help
```

### Example output

```
$ python run.py inspect data/dataset/crack/crack_0003.png

Verdict: CRACK  (1 region(s), 6.1 ms)
  [1] crack    conf=1.00 area=486px bbox=(88, 103, 71, 64)
Overlay : outputs/overlays/crack_0003_annotated.png
Pipeline: outputs/overlays/crack_0003_pipeline.png
```

### Generated artefacts

```
outputs/
├── model/knn_model.json          # persisted classifier
├── inspection_report.csv         # one row per image
├── inspection_report.json        # full detail + evaluation + config
├── summary.md                    # human-readable batch summary
├── figures/confusion_matrix.png
├── figures/k_sweep.png
├── figures/robustness.png
├── overlays/*_annotated.png      # boxes, labels, confidences
├── overlays/*_pipeline.png       # 3-panel: preprocessed | mask | classified
└── logs/visioninspect.log
```

## Testing

The suite uses only the standard library, so no test runner needs installing:

```bash
# Run all 77 tests
python -m unittest discover -s tests -v

# Run one module
python -m unittest tests.test_segment -v

# Run one test
python -m unittest tests.test_segment.TestSegmentation.test_clean_surface_yields_no_regions
```

| Test module | Covers | Tests |
| --- | --- | --- |
| `test_config.py` | Config validation, JSON overrides, error paths | 8 |
| `test_preprocess.py` | Grayscale/RGBA conversion, resizing, determinism | 7 |
| `test_segment.py` | **Zero false positives on clean parts**, detection of all three defect types, mask cleanup, region ordering | 8 |
| `test_features.py` | Descriptor shape and finiteness, and the class-separability properties the classifier depends on | 7 |
| `test_models.py` | Scaler whitening, k-NN correctness, persistence round-trip, metric maths against hand-computed values | 20 |
| `test_ml.py` | Baseline classifiers, stratified fold construction, cross-validation and model comparison | 14 |
| `test_integration.py` | End-to-end accuracy, latency budget, corrupt-file resilience, report writing | 13 |

Expected result: `Ran 77 tests ... OK` in roughly one second.

## Results

Held-out test split, 60 images (15 per class), k = 5:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| crack | 1.000 | 1.000 | 1.000 | 15 |
| good | 1.000 | 1.000 | 1.000 | 15 |
| scratch | 1.000 | 1.000 | 1.000 | 15 |
| spot | 1.000 | 1.000 | 1.000 | 15 |
| **Overall** | — | — | **accuracy 1.000 / macro-F1 1.000** | 60 |

Average latency **6.0 ms** per image, peak 6.9 ms (≈166 images/s, single CPU
thread). The `k` sweep is flat at 1.00 for k ∈ {1…11}, confirming the classes
are well separated in feature space rather than the result of a lucky `k`.

Stratified 5-fold cross-validation over seven candidate classifiers found six
tied at 1.000 (every k-NN variant and Nearest Centroid), with Gaussian Naive
Bayes at 0.993 — so model choice is not the bottleneck here. Learning curves
do separate them: k-NN and Nearest Centroid reach ceiling from five labelled
regions per class, while Naive Bayes plateaus near 0.98 because its
conditional-independence assumption is violated by construction. Full analysis
in [`docs/ML.md`](docs/ML.md).

A perfect score on pristine renders would be a weak claim on its own, so
`scripts/robustness_study.py` re-scores the same split under controlled
degradation:

| Degradation | Level where accuracy first drops | Accuracy at the worst level tested |
| --- | --- | --- |
| Gaussian sensor noise | σ = 8 (0.950) | σ = 25 → 0.733 |
| Defocus blur | k = 11 (0.983) | k = 11 → 0.983 |
| Brightness shift | +40 (0.950) | +60 → 0.933 |

The system is essentially immune to illumination drift and defocus — CLAHE and
the morphological background estimate absorb both — and degrades gracefully
under sensor noise, which is the failure mode to watch on a real line.

> **Honest limitation:** these numbers are measured on synthetic imagery whose
> defect morphologies are drawn from known generators. Real castings carry
> overlapping defects, oblique lighting and machining marks, so accuracy on
> production data would be lower. The robustness study is included precisely
> because a headline "100%" on clean synthetic data should not be read as a
> production claim. See `docs/DESIGN.md` for the full discussion.

## Project structure

```
visioninspect/
├── README.md                     # this file
├── statement.md                  # problem statement, scope, users, features
├── requirements.txt
├── config.example.json
├── run.py                        # entry point
├── src/visioninspect/
│   ├── __init__.py
│   ├── config.py                 # typed, validated configuration
│   ├── exceptions.py             # typed error hierarchy
│   ├── logger.py                 # centralised logging
│   ├── cli.py                    # argparse sub-commands
│   ├── data/
│   │   ├── synthetic.py          # dataset generator
│   │   └── loader.py             # discovery, validation, stratified split
│   ├── pipeline/
│   │   ├── preprocess.py         # resize, denoise, CLAHE
│   │   └── segment.py            # background estimate, MAD threshold, regions
│   ├── features/
│   │   └── extractor.py          # 13-D shape + photometric + Hu descriptors
│   ├── models/
│   │   ├── scaler.py             # z-score standardisation
│   │   ├── knn.py                # from-scratch distance-weighted k-NN
│   │   ├── baselines.py          # Nearest Centroid, Gaussian Naive Bayes
│   │   ├── crossval.py           # stratified k-fold CV, model comparison
│   │   └── evaluate.py           # confusion matrix, P/R/F1, k sweep
│   ├── detect/
│   │   └── inspector.py          # pipeline facade + verdict logic
│   └── report/
│       ├── visualize.py          # overlays and plots
│       └── reporter.py           # CSV / JSON / Markdown reports
├── tests/                        # 63 unittest tests
├── scripts/
│   ├── robustness_study.py       # degradation experiments
│   ├── model_selection.py        # CV comparison, ablation, learning curves
│   ├── make_diagrams.py          # renders the design diagrams
│   └── build_report.py           # rebuilds the PDF report
└── docs/
    ├── DESIGN.md                 # architecture, diagrams, design rationale
    ├── ML.md                     # dataset, model selection, evaluation methodology
    └── diagrams/                 # rendered diagram images
```

## Configuration

Every tunable lives in a validated dataclass in `src/visioninspect/config.py`
and can be overridden from JSON:

```bash
python run.py --config config.example.json evaluate
```

The knob most worth tuning in the field is `segment.sensitivity`: values above
1.0 lower the detection threshold (catch fainter defects, risk false alarms),
values below 1.0 raise it. Invalid values are rejected at construction with a
`ConfigError` rather than failing silently deep in the pipeline.

## Documentation

- [`statement.md`](statement.md) — problem statement, scope, target users
- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture, UML diagrams, design rationale, requirements traceability
- [`docs/ML.md`](docs/ML.md) — dataset description, model selection rationale, evaluation methodology
- `Project_Report.pdf` — the full submission report

## License

MIT — see [LICENSE](LICENSE).
