from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI

from src.config import MAX_AGENT_TURNS, OPENAI_API_KEY, OPENAI_MODEL
from src.logging_utils import AgentLogger
from src.structured_report import StructuredReport


class BaseAgent(ABC):
    name: str = "BaseAgent"

    def __init__(self, logger: AgentLogger) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.logger = logger
        self.handoff_report: StructuredReport | None = None

    @property
    def handoff_summary(self) -> str:
        return self.handoff_report.summary if self.handoff_report else ""

    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @abstractmethod
    def initial_user_message(self) -> str:
        ...

    @abstractmethod
    def tool_schemas(self) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def run_tool(self, name: str, arguments: dict[str, Any]) -> str:
        ...

    def completion_tools(self) -> list[dict[str, Any]]:
        return self.tool_schemas() + [self._complete_handoff_schema()]

    @staticmethod
    def _complete_handoff_schema() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "complete_handoff",
                "description": (
                    "Call when your work is finished. Submit a structured report "
                    "for the next agent and the final pipeline report."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Narrative summary of what you did and why.",
                        },
                        "actions": {
                            "type": "array",
                            "description": "List of key decisions/actions taken.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string"},
                                    "target": {"type": "string"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["action", "target", "reason"],
                            },
                        },
                        "artifacts": {
                            "type": "object",
                            "description": "Produced files and dataset metadata.",
                        },
                        "notes_for_next_agent": {
                            "type": "string",
                            "description": "Actionable guidance for the next agent.",
                        },
                        "metrics": {
                            "type": "object",
                            "description": "Final model metrics (Agent 3 only).",
                        },
                    },
                    "required": ["summary", "actions", "artifacts", "notes_for_next_agent"],
                },
            },
        }

    def _finalize_handoff(self, args: dict[str, Any]) -> StructuredReport:
        report = StructuredReport.from_handoff(self.name, args)
        self.handoff_report = report
        self.logger.log(self.name, f"Structured handoff:\n{report.to_json()}")
        return report

    def run(self) -> StructuredReport:
        self.logger.section(f"Starting {self.name}")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.initial_user_message()},
        ]

        for turn in range(1, MAX_AGENT_TURNS + 1):
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=self.completion_tools(),
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if msg.content:
                self.logger.log(self.name, msg.content.strip())

            if not msg.tool_calls:
                if turn == MAX_AGENT_TURNS:
                    raise RuntimeError(f"{self.name} exceeded max turns without handoff.")
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue using tools until the task is done, then call "
                            "complete_handoff with a structured report."
                        ),
                    }
                )
                continue

            messages.append(msg.model_dump())
            for call in msg.tool_calls:
                fn = call.function
                args = json.loads(fn.arguments or "{}")
                self.logger.log(self.name, f"Tool call: {fn.name}({json.dumps(args)[:500]})")

                if fn.name == "complete_handoff":
                    return self._finalize_handoff(args)

                result = self.run_tool(fn.name, args)
                preview = result[:800] + ("..." if len(result) > 800 else "")
                self.logger.log(self.name, f"Tool result ({fn.name}): {preview}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

        raise RuntimeError(f"{self.name} exceeded {MAX_AGENT_TURNS} turns.")
