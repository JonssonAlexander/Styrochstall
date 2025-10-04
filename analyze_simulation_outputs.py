"""Diagnostics for Monte Carlo bike availability simulations."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from styrstell.utils import ensure_directory

app = typer.Typer(add_completion=False)


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


@app.command()
def run(
    trips_path: Path = typer.Option(
        Path("data/processed/maps_trips.parquet"),
        "--trips",
        help="MAPS inferred trips parquet.",
    ),
    inventory_path: Path = typer.Option(
        Path("data/processed/simulation_initial_inventory.parquet"),
        "--inventory",
        help="Initial inventory parquet produced by prepare_simulation_data.py",
    ),
    departures_path: Path = typer.Option(
        Path("data/processed/simulation_departures.parquet"),
        "--departures",
        help="Per-bin departure intensities parquet.",
    ),
    probability_path: Path = typer.Option(
        Path("data/processed/monte_carlo_stockout_probabilities.parquet"),
        "--probabilities",
        help="Aggregated stockout probability parquet.",
    ),
    inventory_log_path: Optional[Path] = typer.Option(
        Path("data/processed/monte_carlo_inventory_log.parquet"),
        "--inventory-log",
        help="Optional per-bin inventory log parquet (if produced).",
    ),
    downtime_station_path: Optional[Path] = typer.Option(
        Path("data/processed/monte_carlo_downtime.parquet"),
        "--downtime-station",
        help="Optional station-level downtime parquet (output of run_monte_carlo_simulation).",
    ),
    downtime_time_path: Optional[Path] = typer.Option(
        Path("data/processed/monte_carlo_downtime_time.parquet"),
        "--downtime-time",
        help="Optional time-level downtime parquet (output of run_monte_carlo_simulation).",
    ),
    output_path: Path = typer.Option(
        Path("reports/monte_carlo_diagnostics.md"),
        "--output",
        help="Markdown report summarising diagnostics.",
    ),
) -> None:
    trips = _read_parquet(trips_path)
    trips = trips.copy()
    trips["estimated_departure_time"] = pd.to_datetime(trips["estimated_departure_time"], utc=True)
    trips["hour"] = trips["estimated_departure_time"].dt.floor("1H")
    trips["station_id"] = trips["origin_station_id"].astype(str)

    trips_per_hour = trips.groupby("hour").size().rename("observed_departures")
    trips_per_station = trips.groupby("station_id").size().rename("observed_departures")
    trips_per_station_hour = trips.groupby(["station_id", "hour"]).size().rename("observed_departures")

    inventory = _read_parquet(inventory_path)
    total_initial_bikes = inventory["bikes"].sum()
    initial_stats = inventory["bikes"].describe()
    low_initial = (inventory["bikes"] <= 1).sum()

    departures = _read_parquet(departures_path)
    departures = departures.copy()
    departures["time_bin"] = pd.to_datetime(departures["time_bin"], utc=True)
    departures_per_hour = departures.groupby("time_bin")["departures"].sum().rename("lambda_sum")
    departures_per_station = departures.groupby("station_id")["departures"].sum().rename("lambda_sum")
    departures_per_station_hour = departures.groupby(["station_id", "time_bin"])["departures"].sum().rename("lambda_sum")

    # Align observed vs lambda on time and station
    hourly_comparison = trips_per_hour.to_frame().join(departures_per_hour, how="outer").fillna(0)
    station_comparison = trips_per_station.to_frame().join(departures_per_station, how="outer").fillna(0)
    station_hour_comparison = trips_per_station_hour.to_frame().join(
        departures_per_station_hour, how="outer"
    ).fillna(0)

    probability = _read_parquet(probability_path)
    probability["time_bin"] = pd.to_datetime(probability["time_bin"], utc=True)
    probability_summary = probability.groupby(["scenario", "time_bin"])[
        "stockout_probability"
    ].mean().rename("mean_probability")

    mean_bikes_summary = None
    inventory_log_summary = None
    downtime_summary = None
    downtime_station_summary = None
    if inventory_log_path is not None and inventory_log_path.exists():
        inventory_log = _read_parquet(inventory_log_path)
        inventory_log = inventory_log.copy()
        inventory_log["time_bin"] = pd.to_datetime(inventory_log["time_bin"], utc=True)
        inventory_log_summary = (
            inventory_log.groupby(["scenario", "time_bin"], as_index=False)["bikes"].mean()
        )
        inventory_log_summary = inventory_log_summary.rename(columns={"bikes": "mean_bikes"})
        mean_bikes_summary = (
            inventory_log.groupby(["scenario", "station_id"])["bikes"].mean().rename("mean_bikes")
        )
        # Downtime (number of bins with bikes <=1 and total minutes)
        inventory_log["is_stockout"] = inventory_log["bikes"] <= 1
        if downtime_time_path is None or not downtime_time_path.exists():
            downtime_summary = (
                inventory_log.groupby(["scenario", "time_bin"], as_index=False)["is_stockout"]
                .mean()
                .rename(columns={"is_stockout": "instantaneous_stockout_probability"})
            )
        if downtime_station_path is None or not downtime_station_path.exists():
            downtime_station_summary = (
                inventory_log.groupby(["scenario", "station_id"], as_index=False)["is_stockout"]
                .mean()
                .rename(columns={"is_stockout": "stockout_fraction"})
            )
        # Join mean bikes back into probability table (for quick inspection)
        probability = probability.merge(
            inventory_log_summary,
            on=["scenario", "time_bin"],
            how="left",
        )

    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("# Monte Carlo Simulation Diagnostics\n\n")

        fh.write("## Initial Inventory\n")
        fh.write(f"Total bikes: {total_initial_bikes:.1f}\n\n")
        fh.write(initial_stats.to_frame(name="bikes").to_markdown())
        fh.write("\n\n")
        fh.write(f"Stations with ≤1 bike: {low_initial} / {len(inventory)}\n\n")

        fh.write("## Demand Comparison (per hour)\n")
        fh.write(hourly_comparison.head(24).to_markdown())
        fh.write("\n\n")

        fh.write("## Demand Comparison (per station)\n")
        fh.write(station_comparison.sort_values("observed_departures", ascending=False).head(20).to_markdown())
        fh.write("\n\n")

        fh.write("## Demand Comparison (per station & hour)\n")
        fh.write(
            station_hour_comparison
            .sort_values("observed_departures", ascending=False)
            .head(30)
            .to_markdown()
        )
        fh.write("\n\n")

        fh.write("## Stockout Probability (mean across stations)\n")
        fh.write(probability_summary.head(24).to_markdown())
        fh.write("\n\n")

        if inventory_log_summary is not None:
            fh.write("## Mean Bikes Over Time\n")
            fh.write(inventory_log_summary.head(24).to_markdown(index=False))
            fh.write("\n\n")
        if mean_bikes_summary is not None:
            fh.write("## Mean Bikes Per Station\n")
            fh.write(mean_bikes_summary.sort_values(ascending=False).head(20).to_markdown())
            fh.write("\n\n")
        if downtime_summary is not None:
            fh.write("## Instantaneous Stockout Probability Over Time\n")
            fh.write(downtime_summary.head(24).to_markdown(index=False))
            fh.write("\n\n")
        if downtime_station_summary is not None:
            cols = downtime_station_summary.columns
            if "downtime_minutes" in cols:
                downtime_station_summary = downtime_station_summary.sort_values("downtime_minutes", ascending=False)
            elif "stockout_fraction" in cols:
                downtime_station_summary = downtime_station_summary.sort_values("stockout_fraction", ascending=False)
            fh.write("## Downtime Per Station\n")
            fh.write(downtime_station_summary.head(20).to_markdown(index=False))
            fh.write("\n\n")

        fh.write("## Notes\n")
        fh.write(
            "- Ensure `--inventory-log-output` was provided when running the Monte Carlo simulation to populate mean bikes.\n"
        )
        fh.write(
            "- Compare observed departures vs lambda; large discrepancies indicate demand scaling issues.\n"
        )

    # Also save probability augmented with mean bikes if available
    probability.to_parquet(probability_path, index=False)

    typer.echo(f"Diagnostics written to {output_path}")


if __name__ == "__main__":
    app()
    if downtime_summary is None and downtime_time_path is not None and downtime_time_path.exists():
        downtime_summary = _read_parquet(downtime_time_path)
    if downtime_station_summary is None and downtime_station_path is not None and downtime_station_path.exists():
        downtime_station_summary = _read_parquet(downtime_station_path)
