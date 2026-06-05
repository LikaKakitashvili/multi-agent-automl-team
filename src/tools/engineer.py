from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from src.context import PipelineContext


class EngineerToolkit:
    """Tools for Agent 2: The Feature Engineer."""

    def __init__(self, ctx: PipelineContext) -> None:
        self.ctx = ctx
        self._df = ctx.load_clean()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def inspect_metadata(self) -> dict[str, Any]:
        nulls = self._df.isnull().sum().to_dict()
        dtypes = {c: str(t) for c, t in self._df.dtypes.items()}
        return {
            "shape": list(self._df.shape),
            "columns": list(self._df.columns),
            "dtypes": dtypes,
            "null_counts": nulls,
        }

    def create_interaction(self, expression: str) -> dict[str, Any]:
        """Evaluate a pandas expression that assigns a new column, e.g. \"df['a'] = df['x']/df['y']\"."""
        local_df = self._df
        try:
            exec(expression, {"df": local_df, "pd": pd, "np": np}, {})  # noqa: S102
            self._df = local_df
            return {"expression": expression, "columns": list(self._df.columns), "shape": list(self._df.shape)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "expression": expression}

    def _is_categorical_column(self, col: str) -> bool:
        series = self._df[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            return True
        if pd.api.types.is_numeric_dtype(series):
            return int(series.nunique(dropna=True)) <= 20
        return False

    def encode_categorical(self, col: str, method: str = "onehot") -> dict[str, Any]:
        if col not in self._df.columns:
            return {"error": f"Column '{col}' not found."}
        method = method.lower().strip()
        if not self._is_categorical_column(col):
            return {
                "error": (
                    f"Column '{col}' is not categorical "
                    f"(dtype={self._df[col].dtype}, "
                    f"unique={self._df[col].nunique(dropna=True)})."
                )
            }

        encoded = self._df[col].astype(str)

        if method == "label":
            self._df[col] = encoded.astype("category").cat.codes
            return {"column": col, "method": "label", "new_dtype": str(self._df[col].dtype)}

        if method in ("onehot", "one-hot"):
            dummies = pd.get_dummies(encoded, prefix=col, drop_first=True)
            self._df = self._df.drop(columns=[col]).join(dummies)
            return {
                "column": col,
                "method": "onehot",
                "new_columns": list(dummies.columns),
                "shape": list(self._df.shape),
            }

        return {"error": f"Unknown method '{method}'. Use onehot or label."}

    def correlation_analysis(self, target: str) -> dict[str, Any]:
        if target not in self._df.columns:
            return {"error": f"Target '{target}' not found."}
        self.ctx.target_column = target
        numeric = self._df.select_dtypes(include=[np.number])
        if target not in numeric.columns:
            return {"error": f"Target '{target}' must be numeric for correlation."}
        corrs = numeric.corr()[target].drop(target, errors="ignore").sort_values(
            key=abs, ascending=False
        )
        return {
            "target": target,
            "correlations": {k: float(v) for k, v in corrs.items() if pd.notna(v)},
        }

    def select_top_features(self, target: str, k: int) -> dict[str, Any]:
        if target not in self._df.columns:
            return {"error": f"Target '{target}' not found."}
        self.ctx.target_column = target
        feature_cols = [c for c in self._df.columns if c != target]
        X = self._df[feature_cols].copy()
        y = self._df[target]

        for col in X.columns:
            if X[col].dtype == object or str(X[col].dtype) == "category":
                X[col] = X[col].astype("category").cat.codes

        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(0)

        is_classification = y.nunique() <= 20 and not pd.api.types.is_float_dtype(y)
        try:
            if is_classification:
                scores = mutual_info_classif(X, y, random_state=42)
            else:
                scores = mutual_info_regression(X, y, random_state=42)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

        ranking = sorted(zip(feature_cols, scores), key=lambda x: x[1], reverse=True)
        top = [name for name, _ in ranking[:k]]
        keep = top + [target]
        self._df = self._df[keep]
        return {
            "target": target,
            "kept_features": top,
            "scores": {name: float(score) for name, score in ranking[:k]},
            "shape": list(self._df.shape),
        }

    def persist_engineered_data(self) -> dict[str, Any]:
        self.ctx.save_engineered(self._df)
        return {"saved_to": self.ctx.engineered_path, "shape": list(self._df.shape)}

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        handlers = {
            "inspect_metadata": lambda _: self.inspect_metadata(),
            "create_interaction": lambda a: self.create_interaction(a["expression"]),
            "encode_categorical": lambda a: self.encode_categorical(
                a["col"], a.get("method", "onehot")
            ),
            "correlation_analysis": lambda a: self.correlation_analysis(a["target"]),
            "select_top_features": lambda a: self.select_top_features(a["target"], int(a["k"])),
            "persist_engineered_data": lambda _: self.persist_engineered_data(),
        }
        if name not in handlers:
            return json.dumps({"error": f"Unknown tool '{name}'."})
        result = handlers[name](arguments)
        return json.dumps(result, default=str)
