"""Filesystem utilities for snapshot management."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist and return the path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet, ensuring parent directories exist."""

    ensure_directory(path.parent)
    df.to_parquet(path, index=True)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""

    return pd.read_parquet(path)


def list_snapshots(root: Path) -> List[Path]:
    """List snapshot directories sorted by timestamp."""

    if not root.exists():
        return []
    snapshots: Iterable[Path] = [p for p in root.iterdir() if p.is_dir()]
    return sorted(snapshots)
