"""Prepare demand and inventory inputs for the Monte Carlo simulator."""
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
        help="MAPS inferred trips parquet.",
    ),
    station_counts_path: Path = typer.Option(
        Path("data/processed/gbfs_maps_station_counts.parquet"),
        "--station-counts",
        help="Parquet with time series of station bike counts.",
    ),
    rebalancing_events_path: Path = typer.Option(
        Path("data/processed/rebalancing_events.parquet"),
        "--rebalance-events",
        help="Parquet with rebalancing dropoff/pickup events.",
    ),
    station_info_path: Path = typer.Option(
        Path("data/processed/station_information.parquet"),
        "--station-info",
        help="Station metadata used for station capacities.",
    ),
    time_granularity: str = typer.Option(
        "1H",
        "--time-granularity",
        help="Time binning offset matching the Markov chain (e.g. 15min, 1H).",
    ),
    snapshot_source: str = typer.Option(
        "GBFS",
        "--snapshot-source",
        help="Which source to use for initial inventory snapshot (GBFS, MAPS, or BOTH).",
    ),
    inventory_strategy: str = typer.Option(
        "first",
        "--inventory-strategy",
        help="How to derive initial bikes per station: first, last, mean, or median.",
    ),
    departures_output: Path = typer.Option(
        Path("data/processed/simulation_departures.parquet"),
        "--departures-output",
        help="Output parquet with per-bin departure intensities.",
    ),
    inventory_output: Path = typer.Option(
        Path("data/processed/simulation_initial_inventory.parquet"),
        "--inventory-output",
        help="Output parquet with initial station inventory and capacities.",
    ),
    rebalancing_output: Path = typer.Option(
        Path("data/processed/simulation_rebalancing_events.parquet"),
        "--rebalance-output",
        help="Output parquet with rebalancing adjustments.",
    ),
) -> None:
    granularity = time_granularity.lower()

    # Demand departures
    if not trips_path.exists():
        raise FileNotFoundError(f"Trips file not found: {trips_path}")
    trips = pd.read_parquet(trips_path)
    required_trip_cols = {
        "origin_station_id",
        "estimated_departure_time",
    }
    if not required_trip_cols.issubset(trips.columns):
        missing = required_trip_cols - set(trips.columns)
        raise KeyError(f"Trips parquet missing columns: {missing}")
    trips = trips.copy()
    trips["origin_station_id"] = trips["origin_station_id"].astype(str)
    trips["estimated_departure_time"] = pd.to_datetime(
        trips["estimated_departure_time"], utc=True
    )
    trips["time_bin"] = trips["estimated_departure_time"].dt.floor(granularity)
    departures = (
        trips.groupby(["time_bin", "origin_station_id"], as_index=False)
        .size()
        .rename(columns={"origin_station_id": "station_id", "size": "departures"})
    )
    ensure_directory(departures_output.parent)
    departures.to_parquet(departures_output, index=False)

    # Initial inventory using earliest timestamp per station
    if not station_counts_path.exists():
        raise FileNotFoundError(f"Station counts not found: {station_counts_path}")
    counts = pd.read_parquet(station_counts_path)
    required_count_cols = {"timestamp", "station_id", "source", "count"}
    if not required_count_cols.issubset(counts.columns):
        missing = required_count_cols - set(counts.columns)
        raise KeyError(f"Station counts missing columns: {missing}")
    snapshot_source = snapshot_source.strip().upper()
    if snapshot_source not in {"GBFS", "MAPS", "BOTH"}:
        raise ValueError("snapshot_source must be one of GBFS, MAPS, BOTH")
    if snapshot_source != "BOTH":
        counts = counts.loc[counts["source"].str.upper() == snapshot_source].copy()
    else:
        counts = counts.copy()
    counts["timestamp"] = pd.to_datetime(counts["timestamp"], utc=True)
    counts["station_id"] = counts["station_id"].astype(str)
    if counts.empty:
        raise ValueError("No station counts available after filtering; cannot set initial inventory.")

    strategy = inventory_strategy.strip().lower()
    if strategy == "first":
        base_snapshot = (
            counts.sort_values("timestamp").groupby("station_id", as_index=False).first()
        )
        snapshot_time_col = base_snapshot["timestamp"].copy()
    elif strategy == "last":
        base_snapshot = (
            counts.sort_values("timestamp", ascending=False).groupby("station_id", as_index=False).first()
        )
        snapshot_time_col = base_snapshot["timestamp"].copy()
    elif strategy in {"mean", "average", "avg"}:
        base_snapshot = (
            counts.groupby("station_id", as_index=False)["count"].mean()
            .rename(columns={"count": "count"})
        )
        base_snapshot["timestamp"] = pd.NaT
        snapshot_time_col = base_snapshot["timestamp"].copy()
    elif strategy == "median":
        base_snapshot = (
            counts.groupby("station_id", as_index=False)["count"].median()
            .rename(columns={"count": "count"})
        )
        base_snapshot["timestamp"] = pd.NaT
        snapshot_time_col = base_snapshot["timestamp"].copy()
    else:
        raise ValueError("inventory_strategy must be one of first, last, mean, median")

    if "count" not in base_snapshot.columns:
        base_snapshot = base_snapshot.rename(columns={"value": "count"})
    base_snapshot = base_snapshot.rename(columns={"count": "bikes"})

    capacities = pd.DataFrame()
    if station_info_path.exists():
        station_meta = pd.read_parquet(station_info_path)
        if {"station_id", "capacity"}.issubset(station_meta.columns):
            capacities = station_meta[["station_id", "capacity"]].copy()
            capacities["station_id"] = capacities["station_id"].astype(str)

    initial_inventory = base_snapshot.merge(capacities, on="station_id", how="left")
    initial_inventory = initial_inventory.rename(
        columns={
            "count": "bikes",
        }
    )
    if "snapshot_time" not in initial_inventory.columns:
        initial_inventory["snapshot_time"] = snapshot_time_col
    else:
        initial_inventory["snapshot_time"] = snapshot_time_col
    initial_inventory["inventory_strategy"] = strategy
    initial_inventory["snapshot_source"] = snapshot_source
    ensure_directory(inventory_output.parent)
    initial_inventory.to_parquet(inventory_output, index=False)

    # Rebalancing adjustments
    if not rebalancing_events_path.exists():
        raise FileNotFoundError(f"Rebalancing events not found: {rebalancing_events_path}")
    reb = pd.read_parquet(rebalancing_events_path)
    required_reb_cols = {"timestamp", "station_id", "change"}
    if not required_reb_cols.issubset(reb.columns):
        missing = required_reb_cols - set(reb.columns)
        raise KeyError(f"Rebalancing events missing columns: {missing}")
    reb = reb.copy()
    reb["timestamp"] = pd.to_datetime(reb["timestamp"], utc=True)
    reb["station_id"] = reb["station_id"].astype(str)
    reb = reb.rename(columns={"change": "delta"})
    ensure_directory(rebalancing_output.parent)
    reb.to_parquet(rebalancing_output, index=False)

    typer.echo(
        f"Prepared departures ({len(departures)} rows), initial inventory ({len(initial_inventory)} stations),"
        f" and rebalancing events ({len(reb)} rows)."
    )


if __name__ == "__main__":
    app()
