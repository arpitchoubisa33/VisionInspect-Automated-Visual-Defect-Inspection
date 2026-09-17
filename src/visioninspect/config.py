"""Typed configuration objects with validation.

Configuration is expressed as frozen dataclasses instead of loose dicts so
that invalid values fail fast (maintainability + reliability requirements).
Values can be overridden from a JSON file via :meth:`AppConfig.from_json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .exceptions import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PreprocessConfig:
    """Parameters for the image-conditioning stage."""

    target_size: int = 256
    median_ksize: int = 3
    clahe_clip: float = 1.5
    clahe_grid: int = 4

    def __post_init__(self) -> None:
        if self.target_size < 64:
            raise ConfigError("target_size must be at least 64 pixels")
        if self.median_ksize % 2 == 0 or self.median_ksize < 1:
            raise ConfigError("median_ksize must be a positive odd integer")
        if self.clahe_clip <= 0:
            raise ConfigError("clahe_clip must be positive")


@dataclass(frozen=True)
class SegmentConfig:
    """Parameters for background suppression and region extraction."""

    background_ksize: int = 41
    morph_ksize: int = 3
    min_area: int = 40
    max_regions: int = 8
    sensitivity: float = 1.0

    def __post_init__(self) -> None:
        if self.background_ksize % 2 == 0:
            raise ConfigError("background_ksize must be odd")
        if self.min_area <= 0:
            raise ConfigError("min_area must be positive")
        if not 0.2 <= self.sensitivity <= 3.0:
            raise ConfigError("sensitivity must be between 0.2 and 3.0")


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters for the k-NN classifier."""

    k: int = 5
    weighted: bool = True
    test_ratio: float = 0.25
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ConfigError("k must be >= 1")
        if not 0.05 <= self.test_ratio <= 0.5:
            raise ConfigError("test_ratio must be between 0.05 and 0.5")


@dataclass(frozen=True)
class DataConfig:
    """Dataset location and synthetic-generation settings."""

    dataset_dir: Path = PROJECT_ROOT / "data" / "dataset"
    samples_per_class: int = 60
    image_size: int = 256
    random_seed: int = 7


@dataclass(frozen=True)
class AppConfig:
    """Root configuration aggregating every stage of the pipeline."""

    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: Path = PROJECT_ROOT / "outputs"
    log_level: str = "INFO"

    @classmethod
    def from_json(cls, path: Path) -> "AppConfig":
        """Build a config from a JSON file containing partial overrides."""
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"Config file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Config file is not valid JSON: {exc}") from exc

        data = DataConfig(**{**asdict(DataConfig()), **raw.get("data", {})})
        pre = PreprocessConfig(**{**asdict(PreprocessConfig()), **raw.get("preprocess", {})})
        seg = SegmentConfig(**{**asdict(SegmentConfig()), **raw.get("segment", {})})
        mod = ModelConfig(**{**asdict(ModelConfig()), **raw.get("model", {})})
        return cls(
            data=data,
            preprocess=pre,
            segment=seg,
            model=mod,
            output_dir=Path(raw.get("output_dir", PROJECT_ROOT / "outputs")),
            log_level=raw.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable view of the configuration."""
        out = asdict(self)
        out["data"]["dataset_dir"] = str(self.data.dataset_dir)
        out["output_dir"] = str(self.output_dir)
        return out
