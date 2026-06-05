from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.config import MAX_TRAINING_ITERATIONS, OPENAI_MODEL
from src.context import PipelineContext
from src.structured_report import StructuredReport
from src.tools.trainer import TrainerToolkit


class ModelTrainerAgent(BaseAgent):
    name = "Agent 3 — The Model Trainer (The Coder)"

    def __init__(self, ctx: PipelineContext, logger) -> None:
        super().__init__(logger)
        self.ctx = ctx
        self.toolkit = TrainerToolkit(ctx)
        self.iteration = 0

    def system_prompt(self) -> str:
        return """You are The Model Trainer ("The Coder") on an autonomous ML team.

Your job: write and execute Python code to train an XGBoost classifier (or regressor if target is continuous).

Critical feedback loop:
1. Generate complete training code and run it via execute_python_code.
2. Read stdout metrics (accuracy, recall, f1). Code should print lines like:
   accuracy=0.82
   recall=0.75
   f1=0.78
   Or print: METRICS_JSON: {"accuracy": 0.82, "recall": 0.75, "f1": 0.78}
3. Decide if performance is good enough for this dataset. If not, change hyperparameters
   (learning_rate, max_depth, n_estimators, subsample, etc.) and retrain.
4. Do NOT use fixed hyperparameters across runs — adapt based on results.
5. When satisfied, call complete_handoff with a structured report including final metrics.

Code requirements:
- Read CSV from DATA_PATH (already defined in the execution environment).
- Use train/test split and sklearn metrics.
- Handle non-numeric columns appropriately.
- Use xgboost.XGBClassifier or XGBRegressor.
"""

    def initial_user_message(self) -> str:
        target_hint = self.ctx.target_column or "infer from prior structured reports"
        cleaner = StructuredReport.format_for_agent(self.ctx.cleaner_report)
        engineer = StructuredReport.format_for_agent(self.ctx.engineer_report)
        return (
            f"Train XGBoost on: {self.ctx.engineered_path}\n\n"
            f"Structured report from Agent 1:\n{cleaner}\n\n"
            f"Structured report from Agent 2:\n{engineer}\n\n"
            f"Likely target column: {target_hint}\n"
            f"You may run up to {MAX_TRAINING_ITERATIONS} training iterations."
        )

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_python_code",
                    "description": "Execute generated training script; returns stdout/stderr and parsed metrics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code_string": {
                                "type": "string",
                                "description": "Python code body (DATA_PATH is predefined).",
                            }
                        },
                        "required": ["code_string"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_last_metrics",
                    "description": "Return metrics parsed from the last successful run.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    def run_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "execute_python_code":
            self.iteration += 1
            self.logger.log(self.name, f"Training iteration {self.iteration}/{MAX_TRAINING_ITERATIONS}")
        return self.toolkit.dispatch(name, arguments)

    def run(self) -> StructuredReport:
        """Override: enforce feedback loop with explicit iteration budget."""
        self.logger.section(f"Starting {self.name}")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.initial_user_message()},
        ]

        client = OpenAI(api_key=self.client.api_key)
        tools = self.completion_tools()

        for turn in range(1, MAX_TRAINING_ITERATIONS * 8 + 1):
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if msg.content:
                self.logger.log(self.name, msg.content.strip())

            if not msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content or ""})
                if self.iteration >= 1 and self.ctx.training_metrics:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "If metrics are acceptable, call complete_handoff with a "
                                "structured report. Otherwise execute_python_code with "
                                "improved hyperparameters."
                            ),
                        }
                    )
                    continue
                messages.append(
                    {
                        "role": "user",
                        "content": "Generate training code and call execute_python_code.",
                    }
                )
                continue

            messages.append(msg.model_dump())
            for call in msg.tool_calls:
                fn = call.function
                args = json.loads(fn.arguments or "{}")
                self.logger.log(self.name, f"Tool call: {fn.name}")

                if fn.name == "complete_handoff":
                    if self.iteration < 1:
                        result = json.dumps(
                            {
                                "error": "You must execute at least one training script before handoff."
                            }
                        )
                        messages.append(
                            {"role": "tool", "tool_call_id": call.id, "content": result}
                        )
                        continue
                    if not args.get("metrics") and self.ctx.training_metrics:
                        args["metrics"] = self.ctx.training_metrics
                    return self._finalize_handoff(args)

                if fn.name == "execute_python_code" and self.iteration >= MAX_TRAINING_ITERATIONS:
                    result = json.dumps(
                        {
                            "error": f"Max training iterations ({MAX_TRAINING_ITERATIONS}) reached. "
                            "Call complete_handoff with best results."
                        }
                    )
                else:
                    result = self.run_tool(fn.name, args)

                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        raise RuntimeError(f"{self.name} did not complete within turn limit.")
