from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StructuredReport:
    """Structured handoff document passed between agents."""

    agent: str
    summary: str
    actions: list[dict[str, str]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    notes_for_next_agent: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_handoff(cls, agent: str, args: dict[str, Any]) -> StructuredReport:
        actions = args.get("actions") or []
        if isinstance(actions, list):
            normalized = [
                {
                    "action": str(item.get("action", "")),
                    "target": str(item.get("target", "")),
                    "reason": str(item.get("reason", "")),
                }
                for item in actions
                if isinstance(item, dict)
            ]
        else:
            normalized = []

        metrics_raw = args.get("metrics") or {}
        metrics = {
            str(k): float(v)
            for k, v in metrics_raw.items()
            if isinstance(v, (int, float))
        }

        return cls(
            agent=agent,
            summary=str(args.get("summary", "")),
            actions=normalized,
            artifacts=dict(args.get("artifacts") or {}),
            notes_for_next_agent=str(args.get("notes_for_next_agent", "")),
            metrics=metrics,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def narrative(self) -> str:
        lines = [self.summary]
        if self.actions:
            lines.append("\n**Actions taken:**")
            for item in self.actions:
                lines.append(
                    f"- {item.get('action', 'action')} on `{item.get('target', '')}`: "
                    f"{item.get('reason', '')}"
                )
        if self.notes_for_next_agent:
            lines.append(f"\n**Notes for next agent:** {self.notes_for_next_agent}")
        if self.metrics:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
            lines.append(f"\n**Metrics:** {metric_str}")
        return "\n".join(lines)

    @staticmethod
    def format_for_agent(report: StructuredReport | None) -> str:
        if report is None:
            return "(no prior structured report)"
        return report.to_json()


def save_report(report: StructuredReport, path) -> None:
    path.write_text(report.to_json(), encoding="utf-8")
