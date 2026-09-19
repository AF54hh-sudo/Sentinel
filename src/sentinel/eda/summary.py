"""Reusable, serializable dataset profiling utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return shape, columns, data types, memory, and duplicate-row counts."""
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "column_names": frame.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "memory_bytes": int(frame.memory_usage(deep=True).sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def missing_value_summary(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.isna().sum()
    result = pd.DataFrame(
        {
            "missing_count": counts,
            "missing_percent": counts.div(max(len(frame), 1)).mul(100),
        }
    )
    return result.sort_values(["missing_count", "missing_percent"], ascending=False)


def numerical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    columns = ["count", "mean", "median", "std", "min", "p25", "p50", "p75", "max"]
    if numeric.empty:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        {
            "count": numeric.count(),
            "mean": numeric.mean(),
            "median": numeric.median(),
            "std": numeric.std(),
            "min": numeric.min(),
            "p25": numeric.quantile(0.25),
            "p50": numeric.quantile(0.50),
            "p75": numeric.quantile(0.75),
            "max": numeric.max(),
        }
    )


def categorical_summary(frame: pd.DataFrame, *, top_n: int = 10) -> dict[str, dict[str, Any]]:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    categorical = frame.select_dtypes(include=["object", "string", "category", "bool"])
    result: dict[str, dict[str, Any]] = {}
    for column in categorical:
        counts = categorical[column].value_counts(dropna=False).head(top_n)
        result[column] = {
            "unique_count": int(categorical[column].nunique(dropna=True)),
            "top_values": {str(key): int(value) for key, value in counts.items()},
        }
    return result


def iqr_outlier_summary(frame: pd.DataFrame, *, multiplier: float = 1.5) -> pd.DataFrame:
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    rows: list[dict[str, Any]] = []
    for column in frame.select_dtypes(include="number"):
        values = frame[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask = (values < lower) | (values > upper)
        rows.append(
            {
                "column": column,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": int(mask.sum()),
                "outlier_percent": float(mask.mean() * 100),
            }
        )
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def correlation_summary(frame: pd.DataFrame, *, method: str = "spearman") -> pd.DataFrame:
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    numeric = frame.select_dtypes(include="number")
    return numeric.corr(method=method) if not numeric.empty else pd.DataFrame()


def profile_dataset(tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Profile every table without mutating source data."""
    return {
        name: {
            "dataset": dataset_summary(frame),
            "missing": missing_value_summary(frame).reset_index(names="column").to_dict("records"),
            "numeric": numerical_summary(frame).reset_index(names="column").replace({np.nan: None}).to_dict("records"),
            "categorical": categorical_summary(frame),
            "iqr_outliers": iqr_outlier_summary(frame).reset_index().replace({np.nan: None}).to_dict("records"),
        }
        for name, frame in tables.items()
    }
