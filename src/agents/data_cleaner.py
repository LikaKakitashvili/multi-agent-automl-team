from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.context import PipelineContext
from src.tools.cleaner import CleanerToolkit


class DataCleanerAgent(BaseAgent):
    name = "Agent 1 — The Data Cleaner (The Auditor)"

    def __init__(self, ctx: PipelineContext, logger) -> None:
        super().__init__(logger)
        self.ctx = ctx
        self.toolkit = CleanerToolkit(ctx)

    def system_prompt(self) -> str:
        return """You are The Data Cleaner ("The Auditor") on an autonomous ML team.

Your job: inspect the raw CSV, reason about data quality, and prepare a technically sound dataset.

Rules:
- Use tools to inspect the data before making decisions. Do NOT guess.
- Decide imputation strategy (mean/median/mode) per column based on distribution and semantics.
- Use detect_outliers / handle_outliers when numeric columns have extreme values (cap or drop — justify your choice).
- Drop columns only when justified (e.g., unique IDs, >90% missing, constant column).
- You may cast dtypes when values are stored as strings but represent numbers or categories.
- Do NOT apply one-size-fits-all rules; justify each action from what you observe.
- When finished, call persist_clean_data, then complete_handoff with a structured JSON report for the Feature Engineer.
"""

    def initial_user_message(self) -> str:
        return (
            f"Audit and clean the dataset at: {self.ctx.raw_path}\n"
            "Identify the target/label column if obvious, but do not drop it.\n"
            "Save cleaned data before handoff."
        )

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "inspect_metadata",
                    "description": "Returns shape, dtypes, null counts, and cardinality per column.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_column_stats",
                    "description": "Distribution stats for numeric columns or top values for categorical.",
                    "parameters": {
                        "type": "object",
                        "properties": {"col": {"type": "string"}},
                        "required": ["col"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "impute_missing",
                    "description": "Fill NaNs using mean, median, or mode.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "col": {"type": "string"},
                            "strategy": {
                                "type": "string",
                                "enum": ["mean", "median", "mode"],
                            },
                        },
                        "required": ["col", "strategy"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "drop_column",
                    "description": "Remove an unusable column.",
                    "parameters": {
                        "type": "object",
                        "properties": {"col": {"type": "string"}},
                        "required": ["col"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "detect_outliers",
                    "description": "Detect outliers in a numeric column using IQR or z-score.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "col": {"type": "string"},
                            "method": {"type": "string", "enum": ["iqr", "zscore"]},
                        },
                        "required": ["col"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "handle_outliers",
                    "description": "Cap outliers to bounds or drop outlier rows.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "col": {"type": "string"},
                            "action": {"type": "string", "enum": ["cap", "drop"]},
                            "detection": {"type": "string", "enum": ["iqr", "zscore"]},
                        },
                        "required": ["col", "action"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "cast_column",
                    "description": "Cast column to int, float, str, or category.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "col": {"type": "string"},
                            "dtype": {
                                "type": "string",
                                "enum": ["int", "float", "str", "category"],
                            },
                        },
                        "required": ["col", "dtype"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "persist_clean_data",
                    "description": "Write the current dataframe to clean_data.csv.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    def run_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return self.toolkit.dispatch(name, arguments)
