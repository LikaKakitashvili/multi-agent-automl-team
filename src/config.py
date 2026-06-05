from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

RAW_DATA_PATH = DATA_DIR / "raw_data.csv"
CLEAN_DATA_PATH = OUTPUT_DIR / "clean_data.csv"
ENGINEERED_DATA_PATH = OUTPUT_DIR / "engineered_data.csv"
AGENT_LOG_PATH = OUTPUT_DIR / "agent_logs.txt"
FINAL_REPORT_PATH = OUTPUT_DIR / "FINAL_REPORT.md"
CLEANER_REPORT_PATH = OUTPUT_DIR / "agent1_structured_report.json"
ENGINEER_REPORT_PATH = OUTPUT_DIR / "agent2_structured_report.json"
TRAINER_REPORT_PATH = OUTPUT_DIR / "agent3_structured_report.json"

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "25"))
MAX_TRAINING_ITERATIONS = int(os.getenv("MAX_TRAINING_ITERATIONS", "5"))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
