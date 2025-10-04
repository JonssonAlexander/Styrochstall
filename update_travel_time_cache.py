"""Populate travel-time cache using haversine distance and optional cached data."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from styrstell.simulation.markov import MarkovChain
from styrstell.simulation.travel_time import CachedTravelTimeProvider
from styrstell.utils import ensure_directory

app = typer.Typer(add_completion=False)


@app.command()
def run(
    transitions_path: Path = typer.Option(
        Path("data/processed/markov_transitions.parquet"),
        "--transitions",
        help="Markov transitions parquet to source OD pairs and time bins.",
    ),
    station_info_path: Path = typer.Option(
        Path("data/processed/station_information.parquet"),
        "--station-info",
        help="Station metadata with station_id, lat, lon.",
    ),
    cache_path: Path = typer.Option(
        Path("data/processed/travel_time_cache.parquet"),
        "--cache",
        help="Parquet cache to read/update.",
    ),
    probability_threshold: float = typer.Option(
        0.0,
        "--probability-threshold",
        help="Minimum transition probability to request; helps limit API calls.",
    ),
    mean_speed_kmph: float = typer.Option(
        17.0,
        "--mean-speed-kmph",
        help="Cycling speed used to convert distance to travel minutes.",
    ),
) -> None:
    if not transitions_path.exists():
        raise FileNotFoundError(f"Transitions file not found: {transitions_path}")
    if not station_info_path.exists():
        raise FileNotFoundError(f"Station info not found: {station_info_path}")

    chain = MarkovChain.from_parquet(transitions_path)
    transitions = chain.transitions
    if probability_threshold > 0:
        transitions = transitions.loc[
            transitions["probability"] >= probability_threshold
        ].copy()
    transitions = transitions.sort_values("time_bin")

    station_meta = pd.read_parquet(station_info_path)
    required = {"station_id", "lat", "lon"}
    missing = required - set(station_meta.columns)
    if missing:
        raise KeyError(f"Station metadata missing columns: {missing}")
    coordinates = {
        str(row.station_id): (float(row.lat), float(row.lon))
        for row in station_meta.itertuples()
    }

    cache_frame = None
    if cache_path.exists():
        cache_frame = pd.read_parquet(cache_path)

    provider = CachedTravelTimeProvider(
        cache_frame,
        time_granularity=chain.metadata.get("time_granularity", "15min") or "15min",
        fallback_minutes=15.0,
        coordinates=coordinates,
        speed_kmph=mean_speed_kmph,
    )

    unique_pairs = transitions[["origin", "destination"]].drop_duplicates()
    total_targets = len(unique_pairs)
    typer.echo(f"Found {total_targets} unique OD pairs after filtering.")

    for time_bin, group in transitions.groupby("time_bin", sort=True):
        timestamp = pd.to_datetime(time_bin, utc=True)
        pairs = group[["origin", "destination"]].drop_duplicates()
        for row in pairs.itertuples(index=False):
            provider.get_minutes(str(row.origin), str(row.destination), timestamp)

    typer.echo("Travel-time cache populated using haversine estimates.")
    ensure_directory(cache_path.parent)
    provider.save(cache_path)


if __name__ == "__main__":
    app()
