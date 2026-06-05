from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import CLEAN_DATA_PATH, ENGINEERED_DATA_PATH, RAW_DATA_PATH
from src.structured_report import StructuredReport


@dataclass
class PipelineContext:
    """Shared mutable state passed between agents."""

    raw_path: str = str(RAW_DATA_PATH)
    clean_path: str = str(CLEAN_DATA_PATH)
    engineered_path: str = str(ENGINEERED_DATA_PATH)
    cleaner_report: StructuredReport | None = None
    engineer_report: StructuredReport | None = None
    trainer_report: StructuredReport | None = None
    target_column: str | None = None
    training_metrics: dict = field(default_factory=dict)
    training_code_history: list[str] = field(default_factory=list)

    @property
    def cleaner_summary(self) -> str:
        return self.cleaner_report.summary if self.cleaner_report else ""

    @property
    def engineer_summary(self) -> str:
        return self.engineer_report.summary if self.engineer_report else ""

    @property
    def trainer_summary(self) -> str:
        return self.trainer_report.summary if self.trainer_report else ""

    def load_raw(self) -> pd.DataFrame:
        return pd.read_csv(self.raw_path)

    def load_clean(self) -> pd.DataFrame:
        return pd.read_csv(self.clean_path)

    def load_engineered(self) -> pd.DataFrame:
        return pd.read_csv(self.engineered_path)

    def save_clean(self, df: pd.DataFrame) -> None:
        df.to_csv(self.clean_path, index=False)

    def save_engineered(self, df: pd.DataFrame) -> None:
        df.to_csv(self.engineered_path, index=False)
