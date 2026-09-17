# Machine Learning Methodology

This document covers the three artefacts the VITyarthi brief requires for
computation-heavy or ML-related courses: **dataset description**, **model
selection rationale**, and **evaluation methodology**. It complements
[`DESIGN.md`](DESIGN.md), which covers architecture and the computer-vision
pipeline.

Reproduce everything here with:

```bash
python run.py pipeline --samples 60      # dataset, training, evaluation
python scripts/model_selection.py        # model comparison, ablation, learning curves
python scripts/robustness_study.py       # degradation study
```

---

## 1. Dataset description

### 1.1 Why a synthetic dataset

Public surface-defect datasets (NEU-DET, DAGM 2007, MVTec-AD) are large,
licence-restricted and require a download. A generator was written instead so
that the project is self-contained, reproducible from a seed, and free of
licensing constraints. The honest cost of that choice is realism, discussed in
§4.4.

### 1.2 Generation process

Each image is rendered in three layers by `src/visioninspect/data/synthetic.py`:

1. **Base surface** — Gaussian intensity field (μ = 150, σ = 6) plus a
   horizontally smoothed noise field that imitates brushed-metal texture.
2. **Illumination** — a radial gradient from a lamp placed at a random position
   in the central 40% of the frame, producing up to ±28 grey levels of
   vignetting. This is what forces the pipeline to use adaptive rather than
   global thresholding.
3. **Defect** — zero or one defect drawn with randomised geometry, followed by
   a final σ = 3 sensor-noise pass.

### 1.3 Class definitions

| Class | Rendering | Distinguishing property | Real-world analogue |
| --- | --- | --- | --- |
| `good` | no defect layer | — | acceptable part |
| `scratch` | straight anti-aliased line, length 35–65% of frame, thickness 2–3 px, Δintensity 55–90 | high elongation, high solidity | tooling or handling mark |
| `spot` | Gaussian-blurred filled circle, radius 3.2–6.2% of frame, Δintensity 62–90, 75% dark / 25% bright | high circularity, compact | pit, stain or corrosion |
| `crack` | 14–22 step random walk (heading jitter σ = 0.55 rad) plus a 5–9 step branch | elongated **but** low solidity from branching | structural fracture |

### 1.4 Dataset statistics

| Property | Value |
| --- | --- |
| Images | 240 (60 per class, perfectly balanced) |
| Resolution | 256 × 256, 8-bit grayscale |
| Working resolution after preprocessing | 256 × 256 |
| Train / test split | 180 / 60, stratified, seed 42 |
| Defect regions mined for training | 135 (45 per defect class) |
| Descriptor dimensionality | 13 |
| Generation seed | 7 (fully deterministic) |

### 1.5 From images to a learning problem

The classifier does **not** see raw images. The CV pipeline first segments
candidate regions, and each region becomes one 13-dimensional training example.
This is deliberate: it converts a 65,536-pixel input into 13 interpretable
numbers, which is why the model trains on 135 examples instead of the thousands
a CNN would need.

Two labelling decisions matter:

- **Only the largest region per training image is labelled.** Its class is
  taken from the folder label. Smaller regions in the same image are unlabelled
  residual texture and would add noise.
- **Regions found in `good` images are discarded, not used as a fourth class.**
  Residual texture is not a defect *morphology*; making it a class blurs the
  decision boundary. Defect-free parts are instead handled structurally — no
  region above the minimum area means `good` — which measured **0 false
  positives** on the test split.

### 1.6 Feature space

| Family | Features | Rationale |
| --- | --- | --- |
| **Shape** (8) | `log_area`, `elongation`, `circularity`, `extent`, `solidity`, `convexity`, `thinness`, `rect_fill` | Separates linear from compact defects, and branching from straight |
| **Photometric** (2) | `contrast` (region vs surrounding ring), `grad_energy` (mean Sobel magnitude) | Captures how pronounced a defect is |
| **Hu moments** (3) | `hu1`, `hu2`, `hu3`, sign-preserving log-compressed | Rotation- and scale-invariant shape signature |

All features are z-score standardised with statistics fitted **on the training
fold only**, which prevents test-set leakage. This is not optional: `log_area`
spans roughly 10 units while `circularity` spans 1, so without standardisation
a single feature would dominate the Euclidean metric.

---

## 2. Model selection rationale

### 2.1 Candidates and why each was considered

| Model | Type | Why considered |
| --- | --- | --- |
| **k-NN** (k ∈ {1,3,5,9}, distance-weighted and uniform) | Non-parametric, instance-based | No training cost, no distributional assumption, errors explainable by inspecting the neighbours |
| **Nearest Centroid** | Prototype-based | Simplest possible baseline; no hyper-parameters. If it matches k-NN, the classes form compact single clusters |
| **Gaussian Naive Bayes** | Generative, parametric | Tests how much the conditional-independence assumption costs, given that area, perimeter and elongation are correlated by construction |
| CNN | Deep, learned features | **Rejected before implementation**: needs thousands of labelled defects and a GPU, and its failures are opaque. With 135 examples it would overfit immediately |
| SVM with RBF kernel | Margin-based | **Deferred to future work**: likely competitive, but adds a kernel and two hyper-parameters without addressing the actual bottleneck (see §2.4) |

All three implemented candidates are written from scratch in NumPy
(`models/knn.py`, `models/baselines.py`) — no scikit-learn — so the distance
metric, weighting scheme, variance floor and tie-breaking rules are explicit.

### 2.2 Stratified 5-fold cross-validation

A single hold-out split on 135 examples is noisy: two samples crossing the
boundary move accuracy by 1.5 points. Every candidate is therefore scored under
identical **stratified 5-fold** cross-validation (`models/crossval.py`), where
each fold preserves the class proportions of the full set.

| Model | Mean accuracy | Std |
| --- | --- | --- |
| k-NN (k = 1) | 1.0000 | 0.0000 |
| k-NN (k = 3, weighted) | 1.0000 | 0.0000 |
| k-NN (k = 5, weighted) | 1.0000 | 0.0000 |
| k-NN (k = 5, uniform) | 1.0000 | 0.0000 |
| k-NN (k = 9, weighted) | 1.0000 | 0.0000 |
| Nearest Centroid | 1.0000 | 0.0000 |
| Gaussian Naive Bayes | 0.9926 | 0.0148 |

**Interpretation.** Six of seven candidates are tied at ceiling. The honest
conclusion is not "k-NN is the best model" but "**the region-classification
sub-problem is close to linearly separable in this feature space, so model
choice is not the bottleneck**". Repeating the comparison on descriptors mined
from images degraded with σ = 18 sensor noise produced the same tie, confirming
this is a property of the feature space rather than of pristine input.

### 2.3 Learning curves — the experiment that does separate the models

Since accuracy saturates with plenty of data, the discriminating question is
how much labelled data each model needs. That is the realistic constraint on a
new product line, where only a handful of defect examples exist.

| Labelled regions per class | k-NN (k=5) | Nearest Centroid | Gaussian NB |
| --- | --- | --- | --- |
| 3 | 0.986 | **0.994** | 0.965 |
| 5 | **1.000** | 0.998 | 0.980 |
| 8 | 0.998 | 0.998 | 0.986 |
| 12 | **1.000** | **1.000** | 0.980 |
| 20 | **1.000** | **1.000** | 0.989 |
| 30 | **1.000** | **1.000** | 0.987 |

Each point averages five random training subsets. Two findings:

- **Gaussian Naive Bayes plateaus near 0.98** and never closes the gap. More
  data does not help, because the ceiling comes from its false
  conditional-independence assumption, not from sample scarcity. This is a
  textbook bias-limited curve.
- **Nearest Centroid is strongest in the extreme low-data regime** (n = 3),
  which makes sense: estimating one mean per class is far more stable than
  trusting individual neighbours when neighbours are scarce.

### 2.4 Feature ablation

Cross-validated accuracy (k-NN, k = 5) with descriptor families removed:

| Feature set | Accuracy |
| --- | --- |
| Shape only (8 features) | 1.000 |
| Hu moments only (3) | 0.968 |
| Photometric only (2) | 0.813 |
| Without shape | 0.984 |
| Without photometric | 1.000 |
| Without Hu moments | 1.000 |
| **All 13 features** | **1.000** |

Shape features carry the discriminative load — unsurprising, since the three
classes are defined by morphology. Photometric features alone reach only 0.813,
confirming that *how dark* a defect is says much less than *what shape* it is.
The redundancy is intentional: photometric and Hu features cost microseconds
and provide fallback signal when shape estimates degrade under blur, which the
robustness study confirms (accuracy holds at 0.983 with an 11×11 defocus
kernel).

### 2.5 Decision

**Distance-weighted k-NN with k = 5** is deployed, because:

1. It is tied for best on every accuracy measure.
2. It reaches ceiling from just 5 labelled examples per class.
3. Distance weighting removes tie ambiguity at even k and degrades gracefully.
4. Its errors are explainable — you can print the five neighbours that voted.
5. It has no training cost, so retraining on new product lines is instant.

k = 5 rather than k = 1 because, although both score 1.000 here, k = 1 has no
averaging and would be the first to break on noisier production data.

**The real bottleneck is detection, not classification.** Every end-to-end
error in the project's history came from segmentation — Otsu hallucinating
defects on clean parts (35% accuracy) and CLAHE erasing spots (93%) — never
from the classifier. That is why the engineering effort went into the
background estimator and threshold model rather than into a fancier classifier.

---

## 3. Evaluation methodology

### 3.1 Protocol

| Aspect | Choice | Reason |
| --- | --- | --- |
| Split | 75 / 25 stratified, seed 42 | Class balance preserved on both sides |
| Leakage control | Scaler statistics fitted on training data only; test images never touched during training | Standard practice; enforced in `KNNClassifier.fit` |
| Model selection | Stratified 5-fold CV **on the training split only** | The test split is used once, for the final number |
| Reproducibility | Every RNG explicitly seeded; config serialised into the JSON report | Identical results across runs and machines |
| End-to-end scoring | Metrics computed on the **full pipeline verdict**, not on ground-truth regions | Measures what the system actually outputs, including detection failures |

That last point matters: it would be easy — and misleading — to score the
classifier on hand-picked regions. Scoring the end-to-end verdict means a
missed detection counts as an error, which is why the earlier 93% run showed
`spot` recall at 0.733 rather than hiding the failure inside the detector.

### 3.2 Metrics

Implemented from their definitions in `models/evaluate.py`:

- **Accuracy** — trace of the confusion matrix over its sum.
- **Per-class precision, recall, F1** — precision guards against false alarms
  (a rejected good part costs money); recall guards against escapes (a shipped
  defect costs more).
- **Macro-F1** — unweighted class mean, so no class is masked by the others.
- **Confusion matrix** — shows *which* classes are confused, not just how often.
- **Latency** — mean and max milliseconds per image, since an inspection system
  that is accurate but slow cannot be deployed on a line.

### 3.3 Final results (held-out test split, 60 images, never used for tuning)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| crack | 1.000 | 1.000 | 1.000 | 15 |
| good | 1.000 | 1.000 | 1.000 | 15 |
| scratch | 1.000 | 1.000 | 1.000 | 15 |
| spot | 1.000 | 1.000 | 1.000 | 15 |

Accuracy **1.000**, macro-F1 **1.000**, mean latency **6.0 ms**
(≈166 images/s, single CPU thread).

### 3.4 Stress testing — the number that actually matters

A perfect score on pristine synthetic renders is a weak claim, so
`scripts/robustness_study.py` re-scores the same split under three controlled
degradations:

| Degradation | Levels tested | Result |
| --- | --- | --- |
| Gaussian sensor noise | σ = 0 … 25 | 1.000 at σ ≤ 4, 0.933 at σ = 16, **0.733 at σ = 25** |
| Defocus blur | k = 1 … 11 | 1.000 up to k = 9, 0.983 at k = 11 |
| Brightness shift | −60 … +60 grey levels | 1.000 from −60 to +20, 0.933 at +60 |

The system is effectively immune to illumination drift (the morphological
background estimate and CLAHE absorb it) and to defocus, and degrades
gracefully under sensor noise. **Sensor noise is the documented failure mode**:
heavy noise raises the MAD-estimated noise floor above faint defects, so they
stop being detected. On a real line this argues for controlling camera gain and
exposure rather than for a different classifier.

### 3.5 Threats to validity

1. **Synthetic data.** Defects come from known generators, so classes are
   cleanly separable. Real castings carry overlapping defects, machining marks
   and oblique lighting; accuracy on production data would be lower.
2. **Ceiling effect.** With every candidate at 1.000, the comparison cannot
   rank models on accuracy alone — which is why learning curves and ablation
   were added rather than presenting the tie as a result.
3. **Single dominant defect per image.** Multi-defect parts are handled at
   inference but not represented in training.
4. **Small sample.** 135 training regions; cross-validation mitigates but does
   not eliminate variance.
5. **Test split reused across development.** The final split was scored several
   times while the detector was being fixed. The robustness study partly
   compensates by testing conditions never used for tuning.

### 3.6 What would strengthen the evaluation

- Benchmark against NEU-DET to quantify the synthetic-to-real gap.
- Add an SVM and a small CNN once a real dataset makes the comparison meaningful.
- Report a precision-recall curve by sweeping `segment.sensitivity`, giving
  operators an explicit escape-rate / false-alarm trade-off curve.
- Nested cross-validation, so hyper-parameter choice and performance estimation
  are fully separated.
