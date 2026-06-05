#!/usr/bin/env python3
"""Entry point: run the 3-agent Multi-Agent AutoML pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import RAW_DATA_PATH  # noqa: E402
from src.orchestrator import Orchestrator  # noqa: E402


def main() -> None:
    if not RAW_DATA_PATH.exists():
        from scripts.prepare_sample_data import main as prepare

        print("Preparing sample dataset...")
        prepare()

    Orchestrator().run()


if __name__ == "__main__":
    main()
