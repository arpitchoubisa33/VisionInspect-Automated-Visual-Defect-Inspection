# Problem Statement

## 1. Problem

Surface-defect inspection on manufacturing lines is still largely performed by
human operators. A person examining parts under a lamp for an eight-hour shift
suffers measurable attention decay: inspection literature consistently reports
that manual visual inspection catches only 60–80% of defects, that two
inspectors frequently disagree on the same part, and that throughput is capped
by how fast a human can look. The cost of a miss is asymmetric — a scratched or
cracked component that reaches assembly is far more expensive to recall than one
rejected at the station.

The defects themselves are also not interchangeable. A **scratch** is usually
cosmetic and the part can be reworked; a **crack** is a structural failure and
the part must be scrapped; a **spot** (pit or stain) may indicate a drifting
process upstream. A system that only answers "good or bad" throws away the
information a process engineer actually needs.

**The problem this project solves:** automatically inspect a grayscale surface
image, decide whether it is defective, localise each defect, and classify its
morphology — without a labelled pixel-level dataset, without a GPU, and fast
enough to keep up with a line.

## 2. Scope

### In scope

- Single-camera, grayscale inspection of approximately planar surfaces.
- Automatic localisation of defect regions with no manual region-of-interest
  setup and no golden reference image.
- Classification of each region into **scratch**, **spot** or **crack**.
- A per-image verdict, per-batch analytics, and CSV/JSON/Markdown reports that
  a quality system could consume.
- A reproducible synthetic dataset generator, so the whole project runs from a
  clean checkout with no external downloads.
- A quantified robustness study under noise, defocus and illumination drift.

### Out of scope

- Colour, 3-D, thermal or X-ray inspection.
- Deep learning. The project deliberately uses classical computer vision and a
  from-scratch classifier so that every decision is inspectable and the system
  trains on tens rather than thousands of images.
- Real-time camera capture, PLC integration and reject-actuator control.
  The system exposes a batch/file interface that such a layer would call.
- Sub-pixel dimensional metrology.

## 3. Target Users

| User | What they need from the system |
| --- | --- |
| **Line operator** | An instant, unambiguous pass/fail verdict with the defect highlighted on screen, so they can act without interpreting numbers. |
| **Quality engineer** | Per-defect-type counts and trends, and the ability to tune detection sensitivity when the product or lighting changes. |
| **Process engineer** | Evidence of which defect morphology is rising, to trace it back to a machine or a tool upstream. |
| **Student / reviewer** | A readable, testable reference implementation of a complete classical CV pipeline. |

## 4. High-Level Features

1. **Reproducible dataset module** — renders brushed-metal surfaces with three
   parameterised defect morphologies from a fixed seed; no download required.
2. **Adaptive preprocessing** — resize, edge-preserving median denoise and
   contrast-limited adaptive histogram equalisation, so thresholds downstream
   do not depend on the lighting of the scene.
3. **Reference-free defect segmentation** — the defect-free surface is
   *estimated* by morphological reconstruction and subtracted; the residual is
   thresholded against a robust median/MAD noise model.
4. **Hand-crafted descriptors** — thirteen shape, topology, photometric and Hu
   moment features per region, chosen so each defect class is separable by a
   property a human can name.
5. **From-scratch distance-weighted k-NN classifier** — with z-score
   standardisation, JSON persistence and a hyper-parameter sweep.
6. **Verdict logic and analytics** — severity-ranked verdicts, batch pass rate,
   latency statistics and confusion-matrix evaluation.
7. **Reporting and visualisation** — annotated overlays, three-panel pipeline
   figures, confusion-matrix and k-sweep plots, plus CSV, JSON and Markdown
   reports.
8. **Robustness study** — accuracy re-measured under Gaussian noise, defocus
   blur and brightness shift, quantifying where the system degrades.

## 5. Success Criteria

| Criterion | Target | Achieved |
| --- | --- | --- |
| Overall classification accuracy on the held-out split | ≥ 90% | **100%** |
| False alarms on defect-free parts | ≤ 5% | **0%** |
| Defect recall (nothing defective passes) | ≥ 95% | **100%** |
| Inference latency per image | < 250 ms | **≈ 11 ms** |
| Accuracy retained at σ = 16 sensor noise | ≥ 80% | **93.3%** |
