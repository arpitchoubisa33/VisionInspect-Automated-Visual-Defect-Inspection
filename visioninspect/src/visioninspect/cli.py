"""Command-line interface - the user-facing workflow layer.

Sub-commands
------------
``generate``  build the synthetic dataset
``train``     mine region descriptors, fit and persist the k-NN model
``inspect``   run a single image through the pipeline and save an overlay
``evaluate``  score the held-out test split and write the full report
``pipeline``  generate -> train -> evaluate in one call (demo path)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import AppConfig
from .data.loader import (class_distribution, discover_samples,
                          stratified_split)
from .data.synthetic import generate_dataset
from .detect.inspector import DefectInspector
from .exceptions import VisionInspectError
from .logger import configure_logging, get_logger
from .models.evaluate import cross_validate_k, evaluate
from .models.knn import KNNClassifier
from .report import reporter, visualize

LOGGER = get_logger("cli")


def _load_config(args: argparse.Namespace) -> AppConfig:
    """Build the config, honouring an optional JSON override file."""
    config = AppConfig.from_json(Path(args.config)) if args.config else AppConfig()
    configure_logging(args.log_level or config.log_level,
                      config.output_dir / "logs" / "visioninspect.log")
    return config


def _model_path(config: AppConfig) -> Path:
    return config.output_dir / "model" / "knn_model.json"


# --------------------------------------------------------------- commands
def cmd_generate(args: argparse.Namespace) -> int:
    """Create the synthetic dataset on disk."""
    config = _load_config(args)
    counts = generate_dataset(
        dest=config.data.dataset_dir,
        samples_per_class=args.samples or config.data.samples_per_class,
        size=config.data.image_size,
        seed=config.data.random_seed,
    )
    print(f"Dataset ready at {config.data.dataset_dir}: {counts}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Fit the classifier and persist it."""
    config = _load_config(args)
    samples = discover_samples(config.data.dataset_dir)
    print(f"Class distribution: {class_distribution(samples)}")

    train_samples, _ = stratified_split(samples, config.model.test_ratio,
                                        config.model.random_seed)
    inspector = DefectInspector(config)
    model = inspector.train(train_samples)
    model.save(_model_path(config))
    print(f"Trained k-NN on {len(model.classes_)} classes -> "
          f"{_model_path(config)}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect a single image and write an annotated overlay."""
    config = _load_config(args)
    inspector = DefectInspector(config, KNNClassifier.load(_model_path(config)))
    result = inspector.inspect_file(Path(args.image))

    overlay = visualize.save_overlay(result, config.output_dir / "overlays")
    strip = visualize.save_pipeline_strip(result, config.output_dir / "overlays")

    print(f"\nVerdict: {result.verdict.upper()}  "
          f"({len(result.findings)} region(s), {result.elapsed_ms:.1f} ms)")
    for i, finding in enumerate(result.findings, 1):
        print(f"  [{i}] {finding.label:<8} conf={finding.confidence:.2f} "
              f"area={finding.area:.0f}px bbox={finding.bbox}")
    print(f"Overlay : {overlay}\nPipeline: {strip}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Score the held-out split and write CSV/JSON/Markdown reports."""
    config = _load_config(args)
    samples = discover_samples(config.data.dataset_dir)
    train_samples, test_samples = stratified_split(
        samples, config.model.test_ratio, config.model.random_seed)

    inspector = DefectInspector(config)
    model_file = _model_path(config)
    if model_file.exists() and not args.retrain:
        inspector.model = KNNClassifier.load(model_file)
        LOGGER.info("Loaded existing model from %s", model_file)
    else:
        inspector.train(train_samples)
        inspector.model.save(model_file)

    results = inspector.inspect_batch(test_samples)
    truth_by_name = {s.path.name: s.label for s in test_samples}
    y_true = [truth_by_name[Path(r.source).name] for r in results]
    y_pred = [r.verdict for r in results]
    evaluation = evaluate(y_true, y_pred,
                          labels=sorted(set(y_true) | set(y_pred)))

    out = config.output_dir
    reporter.write_csv(results, out / "inspection_report.csv")
    reporter.write_markdown(results, out / "summary.md", evaluation)
    features, labels = inspector.build_training_set(train_samples)
    k_scores = cross_validate_k(features, labels, [1, 3, 5, 7, 9, 11],
                                config.model.random_seed)
    reporter.write_json(results, out / "inspection_report.json", evaluation,
                        extra={"k_sweep": k_scores, "config": config.to_dict()})

    visualize.save_confusion_matrix(evaluation, out / "figures")
    visualize.save_k_sweep(k_scores, out / "figures")
    for result in results[: args.overlays]:
        visualize.save_pipeline_strip(result, out / "overlays")

    print("\n" + evaluation.format_table())
    print(f"\nk sweep: {k_scores}")
    print(f"Reports written to {out}")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Run generate -> train -> evaluate end to end."""
    cmd_generate(args)
    cmd_train(args)
    args.retrain = False
    return cmd_evaluate(args)


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser for every sub-command."""
    parser = argparse.ArgumentParser(
        prog="visioninspect",
        description="Automated visual defect inspection (computer vision).")
    parser.add_argument("--version", action="version",
                        version=f"VisionInspect {__version__}")
    parser.add_argument("--config", help="path to a JSON config override")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="build the dataset")
    generate.add_argument("--samples", type=int, help="images per class")
    generate.set_defaults(func=cmd_generate)

    train = subparsers.add_parser("train", help="fit and save the classifier")
    train.set_defaults(func=cmd_train)

    inspect = subparsers.add_parser("inspect", help="inspect one image")
    inspect.add_argument("image", help="path to the image to inspect")
    inspect.set_defaults(func=cmd_inspect)

    evaluate_cmd = subparsers.add_parser("evaluate", help="score the test split")
    evaluate_cmd.add_argument("--retrain", action="store_true")
    evaluate_cmd.add_argument("--overlays", type=int, default=6,
                              help="how many pipeline strips to save")
    evaluate_cmd.set_defaults(func=cmd_evaluate)

    full = subparsers.add_parser("pipeline", help="generate + train + evaluate")
    full.add_argument("--samples", type=int, help="images per class")
    full.add_argument("--overlays", type=int, default=6)
    full.set_defaults(func=cmd_pipeline, retrain=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; converts expected errors into clean exit codes."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except VisionInspectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
