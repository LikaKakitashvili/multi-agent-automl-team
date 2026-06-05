from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.context import PipelineContext
from src.structured_report import StructuredReport
from src.tools.engineer import EngineerToolkit


class FeatureEngineerAgent(BaseAgent):
    name = "Agent 2 — The Feature Engineer (The Architect)"

    def __init__(self, ctx: PipelineContext, logger) -> None:
        super().__init__(logger)
        self.ctx = ctx
        self.toolkit = EngineerToolkit(ctx)

    def system_prompt(self) -> str:
        return """You are The Feature Engineer ("The Architect") on an autonomous ML team.

Your job: increase information density using domain logic, encode categoricals, and select predictive features.

Rules:
- Read the prior agent's summary and inspect clean data first.
- Create at least one new meaningful feature (ratio, interaction, binning via expression, etc.).
- Use correlation_analysis and select_top_features to reduce redundancy — choose k based on dataset size.
- Encode categoricals with onehot or label encoding as appropriate.
- Integer columns with low cardinality (e.g. Pclass) can be encoded with encode_categorical.
- Do NOT use fixed recipes; explain strategy in your structured handoff report.
- When finished, call persist_engineered_data, then complete_handoff with structured JSON.
"""

    def initial_user_message(self) -> str:
        prior = StructuredReport.format_for_agent(self.ctx.cleaner_report)
        return (
            f"Engineer features from: {self.ctx.clean_path}\n\n"
            f"Structured report from Agent 1:\n{prior}\n\n"
            "Preserve the target column. Save engineered_data.csv before handoff."
        )

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "inspect_metadata",
                    "description": "Overview of clean dataset columns and nulls.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_interaction",
                    "description": (
                        "Python/pandas expression using variable df, e.g. "
                        "df['fare_per_person'] = df['Fare'] / df['FamilySize']"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "encode_categorical",
                    "description": (
                        "One-hot or label encode a categorical column "
                        "(including low-cardinality integers like Pclass)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "col": {"type": "string"},
                            "method": {"type": "string", "enum": ["onehot", "label"]},
                        },
                        "required": ["col"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "correlation_analysis",
                    "description": "Pearson correlation of numeric features with target.",
                    "parameters": {
                        "type": "object",
                        "properties": {"target": {"type": "string"}},
                        "required": ["target"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "select_top_features",
                    "description": "Keep top k features by mutual information with target.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string"},
                            "k": {"type": "integer"},
                        },
                        "required": ["target", "k"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "persist_engineered_data",
                    "description": "Write engineered_data.csv.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    def run_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return self.toolkit.dispatch(name, arguments)
