from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.config import ENGINEERED_DATA_PATH, OUTPUT_DIR
from src.context import PipelineContext


class TrainerToolkit:
    """Tools for Agent 3: The Model Trainer."""

    def __init__(self, ctx: PipelineContext) -> None:
        self.ctx = ctx
        self._last_metrics: dict[str, Any] = {}

    def execute_python_code(self, code_string: str) -> dict[str, Any]:
        self.ctx.training_code_history.append(code_string)
        data_path = str(ENGINEERED_DATA_PATH.resolve())
        work_dir = OUTPUT_DIR / "training_runs"
        work_dir.mkdir(parents=True, exist_ok=True)

        preamble = (
            "import json\n"
            "import sys\n"
            f"DATA_PATH = r'''{data_path}'''\n"
        )
        full_code = preamble + code_string

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=work_dir, delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(full_code)
            script_path = Path(tmp.name)

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(work_dir),
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            metrics = self._parse_metrics(stdout)
            if metrics:
                self._last_metrics = metrics
                self.ctx.training_metrics = metrics
            return {
                "returncode": proc.returncode,
                "stdout": stdout[-8000:],
                "stderr": stderr[-4000:],
                "parsed_metrics": metrics,
                "success": proc.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {"error": "Execution timed out after 300s.", "success": False}
        finally:
            script_path.unlink(missing_ok=True)

    @staticmethod
    def _parse_metrics(stdout: str) -> dict[str, float]:
        patterns = {
            "accuracy": r"accuracy[:\s=]+([0-9.]+)",
            "recall": r"recall[:\s=]+([0-9.]+)",
            "f1": r"f1[_\- ]?score?[:\s=]+([0-9.]+)|\bf1[:\s=]+([0-9.]+)",
            "precision": r"precision[:\s=]+([0-9.]+)",
        }
        found: dict[str, float] = {}
        lower = stdout.lower()
        for key, pattern in patterns.items():
            match = re.search(pattern, lower)
            if match:
                value = next(g for g in match.groups() if g is not None)
                found[key] = float(value)
        json_match = re.search(r"METRICS_JSON:\s*(\{.*?\})", stdout, re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(1))
                for k, v in payload.items():
                    if isinstance(v, (int, float)):
                        found[k.lower()] = float(v)
            except json.JSONDecodeError:
                pass
        return found

    def get_last_metrics(self) -> dict[str, Any]:
        return self._last_metrics

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        handlers = {
            "execute_python_code": lambda a: self.execute_python_code(a["code_string"]),
            "get_last_metrics": lambda _: self.get_last_metrics(),
        }
        if name not in handlers:
            return json.dumps({"error": f"Unknown tool '{name}'."})
        result = handlers[name](arguments)
        return json.dumps(result, default=str)
