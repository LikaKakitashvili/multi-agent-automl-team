from __future__ import annotations

from datetime import datetime, timezone

from src.config import AGENT_LOG_PATH, ensure_dirs


class AgentLogger:
    def __init__(self) -> None:
        ensure_dirs()
        self._path = AGENT_LOG_PATH
        if self._path.exists():
            self._path.write_text("", encoding="utf-8")

    def log(self, agent_name: str, message: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {agent_name}: {message}\n"
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)
        print(line, end="")

    def section(self, title: str) -> None:
        self.log("SYSTEM", f"{'=' * 60}")
        self.log("SYSTEM", title)
        self.log("SYSTEM", f"{'=' * 60}")
