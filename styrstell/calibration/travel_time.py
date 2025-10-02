"""Travel-time distribution estimation."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from styrstell.config import TravelTimeCalibrationConfig


def estimate_travel_time_distribution(trips: pd.DataFrame, config: TravelTimeCalibrationConfig) -> pd.DataFrame:
    """Estimate travel-time histograms by time-of-day bin.

    The input dataframe must contain either a `duration_minutes` column or
    `start_timestamp` and `end_timestamp` columns from which durations can be
    derived.
    """

    if "duration_minutes" in trips.columns:
        durations = trips["duration_minutes"].astype(float)
    elif {"start_timestamp", "end_timestamp"}.issubset(trips.columns):
        start = pd.to_datetime(trips["start_timestamp"], utc=True, errors="coerce")
        end = pd.to_datetime(trips["end_timestamp"], utc=True, errors="coerce")
        durations = (end - start).dt.total_seconds() / 60.0
    else:
        raise KeyError("Trips dataframe must include duration_minutes or start/end timestamps.")
    durations = durations.replace([np.inf, -np.inf], np.nan).dropna()
    durations = durations.clip(lower=1, upper=config.max_minutes)
    if durations.empty:
        raise ValueError("No valid trip durations available for calibration.")

    if "start_timestamp" in trips.columns:
        start_ts = pd.to_datetime(trips["start_timestamp"], utc=True, errors="coerce")
    else:
        start_ts = pd.Series(pd.date_range("2000-01-01", periods=len(durations), freq="1H"), index=durations.index)
    time_bin = start_ts.dt.floor(f"{config.bin_minutes}min").dt.time.astype(str)

    edges = np.arange(0, config.max_minutes + config.bin_minutes, config.bin_minutes, dtype=float)
    results: List[Dict[str, object]] = []
    smoothing = max(config.smoothing, 0.0)

    groups = time_bin.groupby(time_bin)
    for label, indices in groups.groups.items():
        sample = durations.loc[indices]
        if sample.empty:
            continue
        hist, _ = np.histogram(sample, bins=edges)
        weights = hist.astype(float) + smoothing
        weights /= weights.sum()
        for start, prob in zip(edges[:-1], weights):
            results.append(
                {
                    "time_bin": label,
                    "duration_bin_start": start,
                    "duration_bin_end": start + config.bin_minutes,
                    "probability": prob,
                    "count": int(hist[int(start / config.bin_minutes)]),
                }
            )
    return pd.DataFrame(results)
