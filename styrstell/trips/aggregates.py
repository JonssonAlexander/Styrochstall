"""Aggregations for observed trip products."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

EDGE_EXPORT_COLUMNS = ["origin", "destination", "weight", "distance_km", "mean_duration_minutes"]


def build_edge_list(trips: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trips into an edge list with summary statistics."""

    if trips.empty:
        return pd.DataFrame(columns=[
            "origin_station_id",
            "destination_station_id",
            "trip_count",
            "mean_duration_minutes",
            "median_duration_minutes",
            "mean_distance_km",
        ])
    grouped = trips.groupby(["origin_station_id", "destination_station_id"])

    aggregated = grouped.agg(
        trip_count=("bike_id", "count"),
        mean_duration_minutes=("duration_seconds", lambda s: float(s.mean() / 60.0)),
        median_duration_minutes=("duration_seconds", lambda s: float(s.median() / 60.0)),
        mean_distance_km=("distance_km", "mean"),
    ).reset_index()
    aggregated["mean_distance_km"] = pd.to_numeric(aggregated["mean_distance_km"], errors="coerce")
    return aggregated


def build_od_matrix(trips: pd.DataFrame) -> pd.DataFrame:
    """Create a station x station OD matrix of trip counts."""

    if trips.empty:
        return pd.DataFrame()
    matrix = trips.pivot_table(
        index="origin_station_id",
        columns="destination_station_id",
        values="bike_id",
        aggfunc="count",
        fill_value=0,
    )
    matrix = matrix.sort_index().sort_index(axis=1)
    return matrix


def build_travel_time_distribution(trips: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly travel-time quantiles and averages."""

    if trips.empty:
        return pd.DataFrame(columns=["hour", "mean_minutes", "p50", "p80", "p95"])
    durations_minutes = trips["duration_seconds"] / 60.0
    departure_times = pd.to_datetime(trips["estimated_departure_time"], utc=True)
    hours = departure_times.dt.hour
    df = pd.DataFrame({"hour": hours, "duration": durations_minutes})

    def _quantile(series: pd.Series, q: float) -> float:
        clean = series.dropna()
        if clean.empty:
            return float("nan")
        return float(np.percentile(clean, q))

    distribution = df.groupby("hour").agg(
        mean_minutes=("duration", "mean"),
        p50=("duration", lambda s: _quantile(s, 50)),
        p80=("duration", lambda s: _quantile(s, 80)),
        p95=("duration", lambda s: _quantile(s, 95)),
    ).reset_index()
    return distribution.sort_values("hour").reset_index(drop=True)


def build_maps_lambda(trips: pd.DataFrame, interval_minutes: int = 60) -> pd.DataFrame:
    """Estimate departures per station per interval from MAPS-inferred trips."""

    if trips.empty:
        return pd.DataFrame(columns=["timestamp", "station_id", "lambda_departures"])
    data = trips.copy()
    data["estimated_departure_time"] = pd.to_datetime(data["estimated_departure_time"], utc=True, errors="coerce")
    data = data.dropna(subset=["estimated_departure_time", "origin_station_id"])
    data["slot_start"] = data["estimated_departure_time"].dt.floor(f"{interval_minutes}min")
    agg = (
        data.groupby(["slot_start", "origin_station_id"]).size().reset_index(name="departures")
    )
    agg["lambda_departures"] = agg["departures"] / max(interval_minutes, 1)
    agg = agg.rename(columns={"slot_start": "timestamp", "origin_station_id": "station_id"})
    agg["timestamp"] = pd.to_datetime(agg["timestamp"], utc=True)
    return agg[["timestamp", "station_id", "lambda_departures"]]


def export_edge_list_for_visualization(edge_list: pd.DataFrame) -> pd.DataFrame:
    """Format the edge list for existing OD visualization code."""

    if edge_list.empty:
        return pd.DataFrame(columns=EDGE_EXPORT_COLUMNS)
    formatted = edge_list.rename(
        columns={
            "origin_station_id": "origin",
            "destination_station_id": "destination",
            "trip_count": "weight",
            "mean_distance_km": "distance_km",
            "mean_duration_minutes": "mean_duration_minutes",
        }
    )[EDGE_EXPORT_COLUMNS]
    return formatted
