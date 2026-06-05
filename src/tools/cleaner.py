from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.context import PipelineContext


class CleanerToolkit:
    """Tools for Agent 1: The Data Cleaner."""

    def __init__(self, ctx: PipelineContext) -> None:
        self.ctx = ctx
        self._df = ctx.load_raw()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def inspect_metadata(self) -> dict[str, Any]:
        nulls = self._df.isnull().sum().to_dict()
        dtypes = {c: str(t) for c, t in self._df.dtypes.items()}
        nunique = {c: int(self._df[c].nunique(dropna=False)) for c in self._df.columns}
        return {
            "shape": list(self._df.shape),
            "columns": list(self._df.columns),
            "dtypes": dtypes,
            "null_counts": nulls,
            "unique_counts": nunique,
        }

    def _outlier_bounds(self, series: pd.Series, method: str) -> tuple[float, float, int]:
        clean = series.dropna()
        if clean.empty:
            return 0.0, 0.0, 0
        method = method.lower().strip()
        if method == "zscore":
            mean = float(clean.mean())
            std = float(clean.std())
            if std == 0:
                return mean, mean, 0
            z = (clean - mean).abs() / std
            mask = z > 3
            lower = float(clean.min()) if not mask.any() else float(clean[~mask].min())
            upper = float(clean.max()) if not mask.any() else float(clean[~mask].max())
            return lower, upper, int(mask.sum())
        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((clean < lower) | (clean > upper)).sum())
        return lower, upper, count

    def get_column_stats(self, col: str) -> dict[str, Any]:
        if col not in self._df.columns:
            return {"error": f"Column '{col}' not found."}
        series = self._df[col]
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe().to_dict()
            lower, upper, outlier_count = self._outlier_bounds(series, "iqr")
            return {
                "column": col,
                "type": "numeric",
                "stats": {k: (None if pd.isna(v) else float(v)) for k, v in desc.items()},
                "null_count": int(series.isnull().sum()),
                "outlier_count_iqr": outlier_count,
                "suggested_cap_bounds": {"lower": lower, "upper": upper},
            }
        values = series.astype(str).value_counts(dropna=False).head(20)
        return {
            "column": col,
            "type": "categorical",
            "top_values": {str(k): int(v) for k, v in values.items()},
            "unique_count": int(series.nunique(dropna=False)),
            "null_count": int(series.isnull().sum()),
        }

    def impute_missing(self, col: str, strategy: str) -> dict[str, Any]:
        if col not in self._df.columns:
            return {"error": f"Column '{col}' not found."}
        strategy = strategy.lower().strip()
        before = int(self._df[col].isnull().sum())
        if before == 0:
            return {"column": col, "message": "No missing values.", "filled": 0}

        if strategy == "mean":
            value = self._df[col].mean()
        elif strategy == "median":
            value = self._df[col].median()
        elif strategy == "mode":
            mode = self._df[col].mode(dropna=True)
            if mode.empty:
                return {"error": f"Cannot compute mode for '{col}'."}
            value = mode.iloc[0]
        else:
            return {"error": f"Unknown strategy '{strategy}'. Use mean, median, or mode."}

        self._df[col] = self._df[col].fillna(value)
        after = int(self._df[col].isnull().sum())
        return {
            "column": col,
            "strategy": strategy,
            "fill_value": None if pd.isna(value) else value,
            "filled": before - after,
            "remaining_nulls": after,
        }

    def drop_column(self, col: str) -> dict[str, Any]:
        if col not in self._df.columns:
            return {"error": f"Column '{col}' not found."}
        self._df = self._df.drop(columns=[col])
        return {"dropped": col, "new_shape": list(self._df.shape)}

    def cast_column(self, col: str, dtype: str) -> dict[str, Any]:
        """Optional helper: convert column dtype (int, float, str, category)."""
        if col not in self._df.columns:
            return {"error": f"Column '{col}' not found."}
        try:
            if dtype == "int":
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                self._df[col] = pd.to_numeric(self._df[col], errors="coerce")
            elif dtype in ("str", "string"):
                self._df[col] = self._df[col].astype(str)
            elif dtype == "category":
                self._df[col] = self._df[col].astype("category")
            else:
                return {"error": f"Unsupported dtype '{dtype}'."}
            return {"column": col, "new_dtype": str(self._df[col].dtype)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def detect_outliers(self, col: str, method: str = "iqr") -> dict[str, Any]:
        if col not in self._df.columns:
            return {"error": f"Column '{col}' not found."}
        series = self._df[col]
        if not pd.api.types.is_numeric_dtype(series):
            return {"error": f"Column '{col}' must be numeric for outlier detection."}
        lower, upper, count = self._outlier_bounds(series, method)
        return {
            "column": col,
            "method": method,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": count,
            "outlier_pct": round(100 * count / max(len(series.dropna()), 1), 2),
        }

    def handle_outliers(self, col: str, action: str, detection: str = "iqr") -> dict[str, Any]:
        if col not in self._df.columns:
            return {"error": f"Column '{col}' not found."}
        series = self._df[col]
        if not pd.api.types.is_numeric_dtype(series):
            return {"error": f"Column '{col}' must be numeric for outlier handling."}
        action = action.lower().strip()
        detection = detection.lower().strip()
        lower, upper, before = self._outlier_bounds(series, detection)
        if action == "cap":
            self._df[col] = series.clip(lower=lower, upper=upper)
            _, _, after = self._outlier_bounds(self._df[col], detection)
            return {
                "column": col,
                "action": "cap",
                "detection": detection,
                "bounds": {"lower": lower, "upper": upper},
                "outliers_before": before,
                "outliers_after": after,
            }
        if action == "drop":
            mask = (series < lower) | (series > upper)
            rows_before = len(self._df)
            self._df = self._df.loc[~mask].copy()
            return {
                "column": col,
                "action": "drop",
                "detection": detection,
                "bounds": {"lower": lower, "upper": upper},
                "rows_removed": int(mask.sum()),
                "new_shape": list(self._df.shape),
                "rows_before": rows_before,
            }
        return {"error": f"Unknown action '{action}'. Use cap or drop."}

    def persist_clean_data(self) -> dict[str, Any]:
        self.ctx.save_clean(self._df)
        return {"saved_to": self.ctx.clean_path, "shape": list(self._df.shape)}

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        handlers = {
            "inspect_metadata": lambda _: self.inspect_metadata(),
            "get_column_stats": lambda a: self.get_column_stats(a["col"]),
            "impute_missing": lambda a: self.impute_missing(a["col"], a["strategy"]),
            "drop_column": lambda a: self.drop_column(a["col"]),
            "cast_column": lambda a: self.cast_column(a["col"], a["dtype"]),
            "detect_outliers": lambda a: self.detect_outliers(
                a["col"], a.get("method", "iqr")
            ),
            "handle_outliers": lambda a: self.handle_outliers(
                a["col"], a["action"], a.get("detection", "iqr")
            ),
            "persist_clean_data": lambda _: self.persist_clean_data(),
        }
        if name not in handlers:
            return json.dumps({"error": f"Unknown tool '{name}'."})
        result = handlers[name](arguments)
        return json.dumps(result, default=str)
