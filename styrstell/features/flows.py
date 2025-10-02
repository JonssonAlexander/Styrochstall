"""Flow inference heuristics from station availability deltas."""
from __future__ import annotations

import numpy as np
import pandas as pd

from styrstell.config import FeatureConfig


def infer_station_flows(panel: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Estimate departures, arrivals, and rebalancing flags for each station interval."""

    if not {"num_bikes_available", "num_docks_available"}.issubset(panel.columns):
        missing = {"num_bikes_available", "num_docks_available"} - set(panel.columns)
        raise KeyError(f"Panel missing required columns: {missing}")
    result = panel.copy()
    result["delta_bikes"] = (
        result.groupby(level="station_id")["num_bikes_available"].diff().fillna(0)
    )
    result["delta_docks"] = (
        result.groupby(level="station_id")["num_docks_available"].diff().fillna(0)
    )
    result["departures_raw"] = (-result["delta_bikes"]).clip(lower=0)
    result["arrivals_raw"] = result["delta_bikes"].clip(lower=0)
    window = max(config.smoothing_window, 1)
    result["departures"] = (
        result.groupby(level="station_id")["departures_raw"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    result["arrivals"] = (
        result.groupby(level="station_id")["arrivals_raw"].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    threshold = max(config.rebalancing_threshold, 1)
    result["is_rebalancing"] = (result["delta_bikes"].abs() >= threshold) & (
        (result["departures_raw"] == 0) | (result["arrivals_raw"] == 0)
    )
    result["net_flow"] = result["arrivals"] - result["departures"]
    result["utilization"] = result["num_bikes_available"] / (
        result["num_bikes_available"] + result["num_docks_available"].replace(0, np.nan)
    )
    result["utilization"] = result["utilization"].fillna(0.0).clip(lower=0.0, upper=1.0)
    return result
