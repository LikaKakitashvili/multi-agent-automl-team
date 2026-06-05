#!/usr/bin/env python3
"""Download Titanic dataset as raw_data.csv for the AutoML pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, RAW_DATA_PATH, ensure_dirs  # noqa: E402


def main() -> None:
    ensure_dirs()
    try:
        df = pd.read_csv(
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        )
    except Exception:
        # Offline fallback via seaborn-style synthetic subset
        from sklearn.datasets import fetch_openml

        data = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
        df = data.frame

    if "survived" in df.columns and "Survived" not in df.columns:
        df = df.rename(columns={"survived": "Survived"})
    if "Survived" in df.columns:
        df["Survived"] = df["Survived"].astype(int)

    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved {len(df)} rows to {RAW_DATA_PATH}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
