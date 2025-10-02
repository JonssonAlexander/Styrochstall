"""Datetime helpers for resampling and validation."""
from __future__ import annotations

import pandas as pd


def ensure_datetime_index(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    """Return a DataFrame indexed by the specified timestamp column."""

    if df.index.name == column and pd.api.types.is_datetime64_any_dtype(df.index):
        return df
    if column not in df.columns:
        raise KeyError(f"Timestamp column '{column}' is missing.")
    result = df.copy()
    result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    if result[column].isna().any():
        raise ValueError("Timestamp parsing failed for some rows.")
    return result.set_index(column).sort_index()
