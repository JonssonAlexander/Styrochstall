"""Panel construction helpers for station status data."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from styrstell.config import FeatureConfig


def build_station_panel(frames: Iterable[pd.DataFrame], config: FeatureConfig) -> pd.DataFrame:
    """Construct a regularly sampled station panel from raw status frames."""

    records = []
    for frame in frames:
        if frame.empty:
            continue
        local = frame.copy()
        ts_column = "snapshot_ts" if "snapshot_ts" in local.columns else "last_reported"
        local["timestamp"] = pd.to_datetime(local[ts_column], utc=True, errors="coerce")
        local = local.dropna(subset=["timestamp", "station_id"])
        for column in ["num_bikes_available", "num_docks_available", "is_renting", "is_returning"]:
            if column not in local.columns:
                local[column] = None
        records.append(
            local[
                [
                    "timestamp",
                    "station_id",
                    "num_bikes_available",
                    "num_docks_available",
                    "is_renting",
                    "is_returning",
                ]
            ]
        )
    if not records:
        raise ValueError("No station status frames provided.")
    combined = pd.concat(records, ignore_index=True)
    combined = combined.sort_values(["station_id", "timestamp"])
    combined = combined.drop_duplicates(subset=["station_id", "timestamp"], keep="last")

    groups = []
    for station_id, group in combined.groupby("station_id"):
        group = group.set_index("timestamp").sort_index()
        resampled = _resample_station(group, config)
        resampled["station_id"] = station_id
        groups.append(resampled)
    panel = pd.concat(groups)
    panel = panel.reset_index().set_index(["timestamp", "station_id"]).sort_index()
    if config.resample.drop_unobserved_stations:
        panel = panel.groupby(level="station_id").filter(lambda df: df.notna().any().any())
    return panel


def _resample_station(group: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    frequency = config.resample.frequency
    method = config.resample.fill_method
    if method == "none":
        return group.resample(frequency).asfreq()
    if method == "nearest":
        return group.resample(frequency).nearest(limit=1)
    return group.resample(frequency).ffill()
