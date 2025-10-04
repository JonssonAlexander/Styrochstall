"""Aggregate observed trip durations into a travel-time cache for simulations."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from styrstell.utils import ensure_directory

app = typer.Typer(add_completion=False)


@app.command()
def run(
    trips_path: Path = typer.Option(
        Path("data/processed/maps_trips.parquet"),
        "--trips",
        help="Parquet file containing inferred trips with duration_seconds and estimated_departure_time.",
    ),
    output_path: Path = typer.Option(
        Path("data/processed/travel_time_cache.parquet"),
        "--output",
        help="Destination parquet for the travel time cache.",
    ),
    time_granularity: str = typer.Option(
        "15min",
        "--time-granularity",
        help="Floor departures to this pandas offset when grouping (e.g. 15min, 30min, 1H).",
    ),
    min_samples: int = typer.Option(
        1,
        "--min-samples",
        help="Minimum number of trips required for a time-bucketed OD pair to be retained.",
    ),
    include_global: bool = typer.Option(
        True,
        "--include-global/--no-include-global",
        help="If enabled, append overall mean minutes per OD pair with time_key='__global__'.",
    ),
) -> None:
    """Compute mean travel minutes for each (origin, destination, time key)."""

    if not trips_path.exists():
        raise FileNotFoundError(f"Trips file not found: {trips_path}")

    trips = pd.read_parquet(trips_path)
    required_cols = {
        "origin_station_id",
        "destination_station_id",
        "duration_seconds",
        "estimated_departure_time",
    }
    missing = required_cols - set(trips.columns)
    if missing:
        raise KeyError(f"Trips parquet missing columns: {missing}")
    if trips.empty:
        raise ValueError("Trips dataframe is empty; nothing to aggregate.")

    frame = trips.copy()
    frame["origin_station_id"] = frame["origin_station_id"].astype(str)
    frame["destination_station_id"] = frame["destination_station_id"].astype(str)
    frame["estimated_departure_time"] = pd.to_datetime(
        frame["estimated_departure_time"], utc=True
    )
    frame = frame.loc[frame["duration_seconds"] > 0].copy()
    if frame.empty:
        raise ValueError("All trips have non-positive duration; cannot build cache.")

    frame["time_key"] = (
        frame["estimated_departure_time"].dt.floor(time_granularity).dt.strftime("%H:%M")
    )
    frame["mean_minutes"] = frame["duration_seconds"] / 60.0

    grouped = (
        frame.groupby(["origin_station_id", "destination_station_id", "time_key"], as_index=False)
        .agg(mean_minutes=("mean_minutes", "mean"), trip_count=("mean_minutes", "size"))
    )
    grouped = grouped.loc[grouped["trip_count"] >= min_samples].copy()

    records = grouped[[
        "origin_station_id",
        "destination_station_id",
        "time_key",
        "mean_minutes",
        "trip_count",
    ]]
    records = records.rename(
        columns={
            "origin_station_id": "origin",
            "destination_station_id": "destination",
        }
    )

    if include_global:
        global_stats = (
            frame.groupby(["origin_station_id", "destination_station_id"], as_index=False)
            .agg(mean_minutes=("mean_minutes", "mean"), trip_count=("mean_minutes", "size"))
            .rename(
                columns={
                    "origin_station_id": "origin",
                    "destination_station_id": "destination",
                }
            )
        )
        global_stats["time_key"] = "__global__"
        records = pd.concat([records, global_stats], ignore_index=True, copy=False)

    ensure_directory(output_path.parent)
    records.to_parquet(output_path, index=False)
    typer.echo(
        f"Travel time cache saved to {output_path} with {len(records)} rows"
    )


if __name__ == "__main__":
    app()
