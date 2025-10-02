"""Origin-Destination calibration via IPFP."""
from __future__ import annotations

from datetime import timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from styrstell.config import ODCalibrationConfig


def calibrate_od_matrix(flows: pd.DataFrame, stations: pd.DataFrame, config: ODCalibrationConfig) -> pd.DataFrame:
    """Calibrate time-varying OD matrices matching observed departures/arrivals."""

    required_station_cols = {"station_id", "lat", "lon"}
    if not required_station_cols.issubset(stations.columns):
        missing = required_station_cols - set(stations.columns)
        raise KeyError(f"Stations dataframe missing columns: {missing}")
    if not {"departures", "arrivals"}.issubset(flows.columns):
        raise KeyError("Flows dataframe must include departures and arrivals columns.")

    station_ids = stations["station_id"].astype(str).tolist()
    distances = _haversine_matrix(stations)
    prior = np.exp(-config.gravity_beta * distances)
    prior[prior <= 0] = 1e-12

    flows = flows.copy()
    flows = flows.reset_index()
    flows["timestamp"] = pd.to_datetime(flows["timestamp"], utc=True)
    slot_duration = timedelta(minutes=config.time_slot_minutes)
    flows["slot_start"] = flows["timestamp"].dt.floor(f"{config.time_slot_minutes}min")

    matrices = []
    for slot, slot_df in flows.groupby("slot_start"):
        origin_totals = slot_df.groupby("station_id")[["departures"]].sum().reindex(station_ids).fillna(0.0)
        dest_totals = slot_df.groupby("station_id")[["arrivals"]].sum().reindex(station_ids).fillna(0.0)
        if origin_totals.sum().iloc[0] == 0 or dest_totals.sum().iloc[0] == 0:
            continue
        matrix = _ipfp(prior, origin_totals["departures"].to_numpy(), dest_totals["arrivals"].to_numpy(), config)
        probability = matrix / matrix.sum(axis=1, keepdims=True)
        probability = np.nan_to_num(probability, nan=0.0)
        origin = station_ids
        dest = station_ids
        records = pd.DataFrame(
            {
                "slot_start": slot,
                "origin": np.repeat(origin, len(dest)),
                "destination": np.tile(dest, len(origin)),
                "flow": matrix.flatten(),
                "probability": probability.flatten(),
            }
        )
        matrices.append(records)
    if not matrices:
        return pd.DataFrame(columns=["slot_start", "origin", "destination", "flow", "probability"])
    result = pd.concat(matrices, ignore_index=True)
    result["slot_end"] = result["slot_start"] + slot_duration
    return result


def _haversine_matrix(stations: pd.DataFrame) -> np.ndarray:
    lat = np.radians(stations["lat"].to_numpy(dtype=float))
    lon = np.radians(stations["lon"].to_numpy(dtype=float))
    sin_lat = np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
    sin_lon = np.sin((lon[:, None] - lon[None, :]) / 2) ** 2
    a = sin_lat + np.cos(lat)[:, None] * np.cos(lat)[None, :] * sin_lon
    earth_radius_km = 6371.0
    return 2 * earth_radius_km * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _ipfp(prior: np.ndarray, row_targets: np.ndarray, col_targets: np.ndarray, config: ODCalibrationConfig) -> np.ndarray:
    matrix = prior.copy()
    matrix = np.maximum(matrix, 1e-12)
    row_targets = row_targets.astype(float)
    col_targets = col_targets.astype(float)
    for _ in range(config.max_iterations):
        row_sums = matrix.sum(axis=1)
        row_scale = np.divide(row_targets, row_sums, out=np.ones_like(row_targets), where=row_sums > 0)
        matrix *= row_scale[:, None]
        col_sums = matrix.sum(axis=0)
        col_scale = np.divide(col_targets, col_sums, out=np.ones_like(col_targets), where=col_sums > 0)
        matrix *= col_scale[None, :]
        if _converged(matrix, row_targets, col_targets, config.tolerance):
            break
    return matrix


def _converged(matrix: np.ndarray, row_targets: np.ndarray, col_targets: np.ndarray, tol: float) -> bool:
    row_error = np.abs(matrix.sum(axis=1) - row_targets).sum()
    col_error = np.abs(matrix.sum(axis=0) - col_targets).sum()
    total = np.sum(row_targets) + np.sum(col_targets)
    if total <= 0:
        return True
    return (row_error + col_error) / total < tol
