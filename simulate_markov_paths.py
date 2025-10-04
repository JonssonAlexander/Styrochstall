"""Run Monte-Carlo path simulations using the Markov transition model."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import typer

from styrstell.simulation.markov import MarkovChain, MarkovSimulator
from styrstell.simulation.travel_time import (
    CachedTravelTimeProvider,
    GoogleMapsTravelTimeProvider,
)
from styrstell.utils import ensure_directory

app = typer.Typer(add_completion=False)


@app.command()
def run(
    transitions_path: Path = typer.Option(
        Path("data/processed/markov_transitions.parquet"),
        "--transitions",
        help="Parquet file produced by build_markov_chain.py",
    ),
    output_path: Path = typer.Option(
        Path("reports/markov_simulated_trips.parquet"),
        "--output",
        help="Parquet file to write simulated trip records.",
    ),
    start_time: datetime = typer.Option(
        datetime.utcnow(),
        "--start",
        help="Simulation start timestamp (UTC).",
    ),
    steps: int = typer.Option(12, "--steps", help="Number of transitions per path."),
    paths: int = typer.Option(100, "--paths", help="Number of independent paths to simulate."),
    origin: List[str] = typer.Option(
        None,
        "--origin",
        help="Optional list of starting stations. If omitted, sample uniformly from station ids.",
    ),
    travel_cache: Optional[Path] = typer.Option(
        Path("data/processed/travel_time_cache.parquet"),
        "--travel-cache",
        help="Optional parquet cache with columns [origin, destination, time_key, mean_minutes].",
    ),
    travel_granularity: str = typer.Option(
        "15min",
        "--travel-granularity",
        help="Time bucket for travel time caching (e.g. 15min, 30min).",
    ),
    fallback_minutes: float = typer.Option(
        12.0,
        "--fallback-minutes",
        help="Fallback travel time when cache or Maps has no data.",
    ),
    maps_api_key: Optional[str] = typer.Option(
        None,
        "--maps-api-key",
        help="If provided, use Google Maps Distance Matrix to augment travel times.",
    ),
    mean_speed_kmph: float = typer.Option(
        17.0,
        "--mean-speed-kmph",
        help="Assumed cycling speed when estimating travel time from distance (km/h).",
    ),
    station_info_path: Path = typer.Option(
        Path("data/processed/station_information.parquet"),
        "--station-info",
        help="Station metadata parquet providing lat/lon for Google Maps queries.",
    ),
    random_seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Seed for reproducible sampling.",
    ),
) -> None:
    """Generate simulated trips using the Markov transition model."""

    if not transitions_path.exists():
        raise FileNotFoundError(f"Transition file not found: {transitions_path}")

    chain = MarkovChain.from_parquet(transitions_path)

    cache_frame = None
    if travel_cache is not None and travel_cache.exists():
        cache_frame = pd.read_parquet(travel_cache)

    coordinates = {}
    if station_info_path.exists():
        station_meta = pd.read_parquet(station_info_path)
        if {"station_id", "lat", "lon"}.issubset(station_meta.columns):
            coordinates = {
                str(row.station_id): (float(row.lat), float(row.lon))
                for row in station_meta.itertuples()
            }

    if maps_api_key:
        provider = GoogleMapsTravelTimeProvider(
            api_key=maps_api_key,
            cache_path=travel_cache,
            time_granularity=travel_granularity,
            fallback_minutes=fallback_minutes,
            coordinates=coordinates,
            speed_kmph=mean_speed_kmph,
        )
    else:
        provider = CachedTravelTimeProvider(
            cache_frame,
            time_granularity=travel_granularity,
            fallback_minutes=fallback_minutes,
            coordinates=coordinates,
            speed_kmph=mean_speed_kmph,
        )

    rng = np.random.default_rng(random_seed)
    if origin:
        origins = [str(o) for o in origin]
    else:
        origins = rng.choice(chain.station_ids, size=paths, replace=True).tolist()

    simulator = MarkovSimulator(chain, provider, random_seed=random_seed)
    trips = simulator.simulate_paths(origins, steps, pd.to_datetime(start_time, utc=True))

    ensure_directory(output_path.parent)
    trips.to_parquet(output_path, index=False)
    typer.echo(
        f"Simulated {len(trips)} trips across {paths} paths; results saved to {output_path}"
    )


if __name__ == "__main__":
    app()
