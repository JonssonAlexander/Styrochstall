"""Standalone Origin-Destination estimation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


@dataclass
class ODEstimationResult:
    """Container storing OD flows/probabilities with sampling utilities."""

    flows: pd.DataFrame
    probabilities: pd.DataFrame

    def sample_trips(self, n: int, random_state: Optional[np.random.Generator] = None) -> pd.DataFrame:
        """Generate ``n`` synthetic trips consistent with the OD matrix."""

        if n <= 0:
            return pd.DataFrame(columns=["origin", "destination"])
        rng = random_state or np.random.default_rng()
        matrix = self.probabilities.to_numpy(copy=False)
        origins = self.probabilities.index.to_numpy()
        destinations = self.probabilities.columns.to_numpy()
        flat_probs = matrix.flatten()
        if flat_probs.sum() == 0:
            raise ValueError("OD matrix probabilities sum to zero; cannot sample trips.")
        flat_probs = flat_probs / flat_probs.sum()
        choices = rng.choice(matrix.size, size=n, p=flat_probs)
        origin_indices = choices // matrix.shape[1]
        dest_indices = choices % matrix.shape[1]
        trips = pd.DataFrame(
            {
                "origin": origins[origin_indices],
                "destination": destinations[dest_indices],
            }
        )
        return trips


def load_delta_dataframe(path: Path) -> pd.DataFrame:
    """Load a processed dataset containing timestamps, station identifiers, and bike deltas."""

    if not path.exists():
        raise FileNotFoundError(f"Delta dataset not found: {path}")
    df = pd.read_parquet(path)
    expected = {"timestamp", "station_id", "lat", "lon", "delta_bikes"}
    missing = expected - set(df.columns)
    if missing:
        raise KeyError(f"Delta dataframe missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["station_id"] = df["station_id"].astype(str)
    return df[["timestamp", "station_id", "lat", "lon", "delta_bikes"]].copy()


def aggregate_flows(
    data: pd.DataFrame,
    window: str = "1H",
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Aggregate outflows and inflows per station over the provided window."""

    if data.empty:
        raise ValueError("Input dataframe is empty.")
    windowed = (
        data.set_index("timestamp")
        .groupby("station_id")["delta_bikes"]
        .resample(window)
        .sum()
        .reset_index()
    )
    outflows = (-windowed.loc[windowed["delta_bikes"] < 0, "delta_bikes"]).groupby(windowed.loc[windowed["delta_bikes"] < 0, "station_id"]).sum()
    inflows = windowed.loc[windowed["delta_bikes"] > 0, ["station_id", "delta_bikes"]].groupby("station_id").sum()["delta_bikes"]
    stations = data.drop_duplicates(subset="station_id").set_index("station_id")[["lat", "lon"]]
    outflows = outflows.reindex(stations.index, fill_value=0.0)
    inflows = inflows.reindex(stations.index, fill_value=0.0)
    return outflows.astype(float), inflows.astype(float), stations


def estimate_od_matrix(
    data: pd.DataFrame,
    beta: float = 0.6,
    window: str = "1H",
    tol: float = 1e-6,
    max_iterations: int = 200,
) -> ODEstimationResult:
    """Estimate an OD probability matrix using a gravity prior and IPF matching."""

    outflows, inflows, stations = aggregate_flows(data, window=window)
    totals = np.minimum(outflows.sum(), inflows.sum())
    if totals <= 0:
        raise ValueError("Outflows and inflows are zero; cannot estimate OD matrix.")
    distance_matrix = _haversine_matrix(stations[["lat", "lon"]].to_numpy())
    prior = np.exp(-beta * distance_matrix)
    np.fill_diagonal(prior, 0.0)
    prior[prior <= 0] = 1e-12

    matrix = _ipf(prior, outflows.to_numpy(), inflows.to_numpy(), max_iterations=max_iterations, tol=tol)
    # Convert to probability matrix by normalising each row
    with np.errstate(invalid="ignore", divide="ignore"):
        row_sums = matrix.sum(axis=1, keepdims=True)
        probabilities = np.divide(matrix, row_sums, where=row_sums > 0)
    probabilities = np.nan_to_num(probabilities, nan=0.0)
    flow_df = pd.DataFrame(matrix, index=stations.index, columns=stations.index)
    od_df = pd.DataFrame(probabilities, index=stations.index, columns=stations.index)
    return ODEstimationResult(flows=flow_df, probabilities=od_df)


def _haversine_matrix(coords: np.ndarray) -> np.ndarray:
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return EARTH_RADIUS_KM * c


def _ipf(
    prior: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    *,
    max_iterations: int,
    tol: float,
) -> np.ndarray:
    matrix = prior.copy().astype(float)
    matrix[matrix <= 0] = 1e-12
    row_targets = row_targets.astype(float)
    col_targets = col_targets.astype(float)
    total_flow = min(row_targets.sum(), col_targets.sum())
    if total_flow <= 0:
        raise ValueError("Target flows sum to zero; cannot run IPF.")
    for _ in range(max_iterations):
        row_sums = matrix.sum(axis=1)
        row_scale = np.divide(row_targets, row_sums, out=np.ones_like(row_targets), where=row_sums > 0)
        matrix *= row_scale[:, None]
        col_sums = matrix.sum(axis=0)
        col_scale = np.divide(col_targets, col_sums, out=np.ones_like(col_targets), where=col_sums > 0)
        matrix *= col_scale[None, :]
        if _has_converged(matrix, row_targets, col_targets, tol):
            break
    return matrix


def _has_converged(matrix: np.ndarray, row_targets: np.ndarray, col_targets: np.ndarray, tol: float) -> bool:
    row_error = np.abs(matrix.sum(axis=1) - row_targets).sum()
    col_error = np.abs(matrix.sum(axis=0) - col_targets).sum()
    denom = row_targets.sum() + col_targets.sum()
    return (row_error + col_error) / max(denom, 1e-12) < tol


__all__ = [
    "ODEstimationResult",
    "load_delta_dataframe",
    "aggregate_flows",
    "estimate_od_matrix",
]
