"""Build a time-dependent Markov chain of station transitions from OD data."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from styrstell.simulation.markov import MarkovChainBuilder
from styrstell.utils import ensure_directory

app = typer.Typer(add_completion=False)


def _aggregate_trips_to_od(trips_path: Path, time_granularity: str) -> pd.DataFrame:
    trips = pd.read_parquet(trips_path)
    required = {
        "origin_station_id",
        "destination_station_id",
        "estimated_departure_time",
        "duration_seconds",
    }
    missing = required - set(trips.columns)
    if missing:
        raise KeyError(f"Trips parquet missing columns: {missing}")
    frame = trips.copy()
    frame = frame.loc[frame["duration_seconds"] > 0].copy()
    if frame.empty:
        raise ValueError("Trips parquet has no positive-duration records.")

    granularity = time_granularity.lower()
    frame["origin"] = frame["origin_station_id"].astype(str)
    frame["destination"] = frame["destination_station_id"].astype(str)
    frame["slot_start"] = (
        pd.to_datetime(frame["estimated_departure_time"], utc=True).dt.floor(granularity)
    )
    grouped = (
        frame.groupby(["slot_start", "origin", "destination"], as_index=False)
        .size()
        .rename(columns={"size": "flow"})
    )
    grouped["flow"] = grouped["flow"].astype(float)
    grouped["probability"] = grouped.groupby(["slot_start", "origin"], group_keys=False)[
        "flow"
    ].apply(lambda x: x / x.sum())
    grouped["probability"] = grouped["probability"].astype(float)
    try:
        offset = pd.to_timedelta(granularity)
    except ValueError:
        offset = pd.Timedelta(0)
    grouped["slot_end"] = grouped["slot_start"] + offset
    return grouped


@app.command()
def run(
    tidy_od_path: Optional[Path] = typer.Option(
        Path("data/models/observed_od_tidy.parquet"),
        "--tidy-od",
        help="Parquet with pre-aggregated OD probabilities (slot_start, origin, destination, flow).",
    ),
    maps_trips_path: Optional[Path] = typer.Option(
        Path("data/processed/maps_trips.parquet"),
        "--maps-trips",
        help="Parquet with inferred trips from MAPS (origin_station_id, destination_station_id, estimated_departure_time, duration_seconds).",
    ),
    stations_path: Optional[Path] = typer.Option(
        Path("data/processed/station_information.parquet"),
        "--stations",
        help="Parquet file with station metadata (station_id, optional capacity).",
    ),
    output_path: Path = typer.Option(
        Path("data/processed/markov_transitions.parquet"),
        "--output",
        help="Where to write the transition table (parquet).",
    ),
    time_granularity: str = typer.Option(
        "1H",
        "--time-granularity",
        help="Pandas offset alias controlling the time bin for transitions (e.g. 30min, 1H).",
    ),
    smoothing: float = typer.Option(
        0.05,
        "--smoothing",
        help="Additive smoothing applied to each destination before normalisation.",
    ),
    ensure_self_loops: bool = typer.Option(
        True,
        "--ensure-self-loops/--no-ensure-self-loops",
        help="Guarantee a non-zero probability of staying at the same station in each bin.",
    ),
) -> None:
    """Compute the Markov transition table and persist it for simulation."""

    od: Optional[pd.DataFrame] = None

    if maps_trips_path is not None and maps_trips_path.exists():
        od = _aggregate_trips_to_od(maps_trips_path, time_granularity)
    elif tidy_od_path is not None and tidy_od_path.exists():
        od = pd.read_parquet(tidy_od_path)
        if od.empty:
            raise ValueError("OD table is empty; aborting.")
    else:
        raise FileNotFoundError(
            "Neither tidy OD parquet nor maps trips parquet could be loaded."
        )

    station_ids = None
    if stations_path is not None and stations_path.exists():
        stations = pd.read_parquet(stations_path)
        station_column = "station_id" if "station_id" in stations.columns else stations.index.name
        if station_column is None:
            raise ValueError("Station file missing station identifiers")
        station_ids = stations[station_column].astype(str).tolist()

    builder = MarkovChainBuilder(
        time_granularity=time_granularity,
        smoothing=smoothing,
        ensure_self_loops=ensure_self_loops,
    )
    chain = builder.fit(od, station_ids)
    ensure_directory(output_path.parent)
    chain.to_parquet(output_path)
    typer.echo(
        f"Markov transitions written to {output_path} with {len(chain.station_ids)} stations"
    )


if __name__ == "__main__":
    app()
