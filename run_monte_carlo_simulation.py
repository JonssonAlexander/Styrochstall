"""CLI entry point for Monte Carlo simulation of station stockout risk."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import typer

from styrstell.simulation.markov import MarkovChain
from styrstell.simulation.monte_carlo import (
    MonteCarloSimulator,
    aggregate_stockout_probabilities,
)
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
        help="Markov transitions parquet produced from MAPS trips.",
    ),
    departures_path: Path = typer.Option(
        Path("data/processed/simulation_departures.parquet"),
        "--departures",
        help="Prepared departures parquet (time_bin, station_id, departures).",
    ),
    inventory_path: Path = typer.Option(
        Path("data/processed/simulation_initial_inventory.parquet"),
        "--inventory",
        help="Initial inventory parquet (station_id, bikes, capacity).",
    ),
    rebalancing_path: Path = typer.Option(
        Path("data/processed/simulation_rebalancing_events.parquet"),
        "--rebalance",
        help="Rebalancing events parquet (timestamp, station_id, delta).",
    ),
    station_info_path: Path = typer.Option(
        Path("data/processed/station_information.parquet"),
        "--station-info",
        help="Station metadata with station_id, lat, lon.",
    ),
    travel_cache: Path = typer.Option(
        Path("data/processed/travel_time_cache.parquet"),
        "--travel-cache",
        help="Travel time cache parquet to read and update.",
    ),
    maps_api_key: Optional[str] = typer.Option(
        None,
        "--maps-api-key",
        help="Google Maps API key for missing travel times.",
    ),
    num_replications: int = typer.Option(
        200,
        "--replications",
        help="Number of Monte Carlo replications per scenario.",
    ),
    low_stock_threshold: int = typer.Option(
        1,
        "--low-stock-threshold",
        help="Stockout definition threshold (<= value triggers risk).",
    ),
    scenarios: List[str] = typer.Option(
        ["demand_only", "demand_with_refill"],
        "--scenario",
        help="Simulation scenarios to run (demand_only, demand_with_refill).",
    ),
    mean_speed_kmph: float = typer.Option(
        17.0,
        "--mean-speed-kmph",
        help="Assumed cycling speed for haversine travel times (km/h).",
    ),
    demand_mode: str = typer.Option(
        "fixed",
        "--demand-mode",
        help="How to sample departures per station/hour (fixed or poisson).",
    ),
    outcomes_output: Path = typer.Option(
        Path("data/processed/monte_carlo_outcomes.parquet"),
        "--outcomes-output",
        help="Destination parquet for per-replication outcomes.",
    ),
    probability_output: Path = typer.Option(
        Path("data/processed/monte_carlo_stockout_probabilities.parquet"),
        "--probability-output",
        help="Destination parquet for aggregated stockout probabilities.",
    ),
    probability_mode: str = typer.Option(
        "cumulative",
        "--probability-mode",
        help="Stockout probability aggregation mode (cumulative or instantaneous).",
    ),
    inventory_log_output: Optional[Path] = typer.Option(
        Path("data/processed/monte_carlo_inventory_log.parquet"),
        "--inventory-log-output",
        help="Optional parquet to store per-bin inventory across replications.",
    ),
    downtime_output: Optional[Path] = typer.Option(
        Path("data/processed/monte_carlo_downtime.parquet"),
        "--downtime-output",
        help="Optional parquet to store station-level downtime statistics.",
    ),
) -> None:
    chain = MarkovChain.from_parquet(transitions_path)

    departures = pd.read_parquet(departures_path)
    inventory = pd.read_parquet(inventory_path)
    rebalancing = pd.read_parquet(rebalancing_path)

    coordinates = {}
    if station_info_path.exists():
        station_meta = pd.read_parquet(station_info_path)
        if {"station_id", "lat", "lon"}.issubset(station_meta.columns):
            coordinates = {
                str(row.station_id): (float(row.lat), float(row.lon))
                for row in station_meta.itertuples()
            }

    cache_frame = None
    if travel_cache.exists():
        cache_frame = pd.read_parquet(travel_cache)

    if maps_api_key:
        provider = GoogleMapsTravelTimeProvider(
            api_key=maps_api_key,
            cache_path=travel_cache,
            time_granularity=chain.metadata.get("time_granularity", "15min") or "15min",
            fallback_minutes=15.0,
            coordinates=coordinates,
            speed_kmph=mean_speed_kmph,
        )
    else:
        provider = CachedTravelTimeProvider(
            cache_frame,
            time_granularity=chain.metadata.get("time_granularity", "15min") or "15min",
            fallback_minutes=15.0,
            coordinates=coordinates,
            speed_kmph=mean_speed_kmph,
        )

    simulator = MonteCarloSimulator(
        chain=chain,
        departures=departures,
        initial_inventory=inventory,
        travel_provider=provider,
        rebalancing_events=rebalancing,
        time_granularity=chain.metadata.get("time_granularity", "1H") or "1H",
        low_stock_threshold=low_stock_threshold,
        demand_mode=demand_mode,
    )

    all_outcomes: List[pd.DataFrame] = []
    inventory_logs: List[pd.DataFrame] = []
    for scenario in scenarios:
        scenario_normalized = scenario if isinstance(scenario, str) else str(scenario)
        if scenario_normalized not in {"demand_only", "demand_with_refill"}:
            raise ValueError(f"Unknown scenario: {scenario}")
        typer.echo(f"Running scenario '{scenario_normalized}' with {num_replications} replications...")
        outcomes, inventory = simulator.simulate(  # type: ignore[arg-type]
            num_replications,
            scenario_normalized,
            log_inventory=inventory_log_output is not None,
        )
        all_outcomes.append(outcomes)
        if inventory_log_output is not None and inventory is not None and not inventory.empty:
            inventory_logs.append(inventory)

    outcomes_df = pd.concat(all_outcomes, ignore_index=True)
    ensure_directory(outcomes_output.parent)
    outcomes_df.to_parquet(outcomes_output, index=False)

    inventory_df = pd.concat(inventory_logs, ignore_index=True) if inventory_logs else None

    prob_df = aggregate_stockout_probabilities(
        outcomes_df,
        chain=chain,
        time_granularity=chain.metadata.get("time_granularity", "1H") or "1H",
        cumulative=probability_mode.strip().lower() != "instantaneous",
        inventory_log=inventory_df,
    )
    ensure_directory(probability_output.parent)
    prob_df.to_parquet(probability_output, index=False)

    if inventory_log_output is not None and inventory_df is not None:
        ensure_directory(inventory_log_output.parent)
        inventory_df.to_parquet(inventory_log_output, index=False)
        typer.echo(
            f"Inventory log saved to {inventory_log_output} with {len(inventory_df)} rows."
        )

    if downtime_output is not None and inventory_df is not None:
        downtime_time, downtime_station = _compute_downtime_statistics(
            inventory_df,
            chain.metadata.get("time_granularity", "1H") or "1H",
        )
        ensure_directory(downtime_output.parent)
        path_time = downtime_output.with_name(downtime_output.stem + "_time.parquet")
        path_station = downtime_output
        downtime_time.to_parquet(path_time, index=False)
        downtime_station.to_parquet(path_station, index=False)
        typer.echo(
            f"Downtime stats saved to {path_station} (station) and {path_time} (time)."
        )

    typer.echo(
        f"Saved outcomes ({len(outcomes_df)} rows) to {outcomes_output} and probabilities"
        f" ({len(prob_df)} rows) to {probability_output}."
    )


def _compute_downtime_statistics(
    inventory_df: pd.DataFrame,
    time_granularity: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = inventory_df.copy()
    df["time_bin"] = pd.to_datetime(df["time_bin"], utc=True)
    df["is_stockout"] = df["bikes"] <= 1
    offset = pd.to_timedelta(time_granularity.lower())
    bin_minutes = max(offset.total_seconds() / 60.0, 1.0)

    time_summary = (
        df.groupby(["scenario", "time_bin"], as_index=False)["is_stockout"]
        .mean()
        .rename(columns={"is_stockout": "instantaneous_stockout_probability"})
    )

    grouped_station = df.groupby(["scenario", "station_id"], as_index=False)["is_stockout"]
    station_mean = grouped_station.mean().rename(columns={"is_stockout": "stockout_fraction"})
    station_sum = grouped_station.sum().rename(columns={"is_stockout": "stockout_bins"})
    station_count = grouped_station.count().rename(columns={"is_stockout": "total_bins"})
    station_summary = (
        station_mean.merge(station_sum, on=["scenario", "station_id"])
        .merge(station_count, on=["scenario", "station_id"])
    )
    station_summary["downtime_minutes"] = station_summary["stockout_bins"] * bin_minutes
    station_summary["total_minutes"] = station_summary["total_bins"] * bin_minutes

    return time_summary, station_summary


if __name__ == "__main__":
    app()
