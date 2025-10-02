"""Demand intensity calibration."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from styrstell.config import DemandCalibrationConfig


def estimate_demand_intensity(flows: pd.DataFrame, config: DemandCalibrationConfig) -> pd.DataFrame:
    """Estimate inhomogeneous Poisson intensities per station and interval."""

    if not {"departures_raw", "arrivals_raw"}.issubset(flows.columns):
        raise KeyError("Flows must contain departures_raw and arrivals_raw columns.")
    interval_minutes = _infer_interval_minutes(flows)
    grouped = flows.groupby(level="station_id")
    window = max(config.window_intervals, 1)
    if config.method == "kernel":
        bandwidth = config.bandwidth or window
        lambda_dep = grouped["departures_raw"].transform(lambda s: _gaussian_smooth(s, bandwidth))
        lambda_arr = grouped["arrivals_raw"].transform(lambda s: _gaussian_smooth(s, bandwidth))
    else:
        lambda_dep = grouped["departures_raw"].transform(lambda s: s.rolling(window, min_periods=1).mean())
        lambda_arr = grouped["arrivals_raw"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    scale = max(interval_minutes, 1e-6)
    result = flows.copy()
    result["lambda_departures"] = lambda_dep / scale
    result["lambda_arrivals"] = lambda_arr / scale
    result["interval_minutes"] = interval_minutes
    return result


def _infer_interval_minutes(flows: pd.DataFrame) -> float:
    timestamps = flows.index.get_level_values("timestamp").unique().sort_values()
    if len(timestamps) < 2:
        return 5.0
    diffs = timestamps.to_series().diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 5.0
    return float(diffs.median() / 60.0)


def _gaussian_smooth(series: pd.Series, bandwidth: int) -> pd.Series:
    if len(series) < 2:
        return series
    try:
        return series.rolling(window=bandwidth, win_type="gaussian", min_periods=1).mean(std=bandwidth / 2)
    except ValueError:
        # Win_type requires SciPy; fall back to moving average.
        return series.rolling(window=bandwidth, min_periods=1).mean()
