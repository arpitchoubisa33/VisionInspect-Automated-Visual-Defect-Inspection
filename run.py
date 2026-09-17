#!/usr/bin/env python3
"""Repository entry point.

Adds ``src/`` to the import path so the project runs from a clean checkout
without an editable install, then delegates to the CLI.

Usage:
    python run.py pipeline --samples 60
    python run.py inspect data/dataset/crack/crack_0000.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from visioninspect.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
