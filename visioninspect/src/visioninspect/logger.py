"""Centralised logging setup (non-functional requirement: observability).

Every module obtains its logger through :func:`get_logger` so that log
formatting, level and destination are controlled from exactly one place.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger once per process.

    Args:
        level: Logging level name, e.g. ``"DEBUG"``.
        log_file: Optional path; when given, logs are mirrored to this file.
    """
    global _CONFIGURED
    root = logging.getLogger("visioninspect")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if _CONFIGURED:
        return

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(file_handler)

    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger."""
    return logging.getLogger(f"visioninspect.{name}")
