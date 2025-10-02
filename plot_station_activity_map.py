"""Render a Folium map with station-level trip activity."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import folium
import numpy as np
import pandas as pd
import typer

from styrstell.utils import ensure_directory

app = typer.Typer(add_completion=False)


@app.command()
def run(
    trips_path: Path = typer.Option(
        Path("data/processed/maps_trips.parquet"),
        "--trips",
        help="Parquet file containing inferred trips (maps_trips.parquet or trips.parquet).",
    ),
    station_info_path: Path = typer.Option(
        Path("data/processed/station_information.parquet"),
        "--station-info",
        help="Parquet file with station metadata (must include station_id, name, lat, lon).",
    ),
    metric: Literal["departures", "arrivals", "both"] = typer.Option(
        "departures",
        "--metric",
        help="Which trip metric to visualise: departures, arrivals, or both (net).",
    ),
    output_path: Path = typer.Option(
        Path("reports/station_activity_map.html"),
        "--output",
        help="Output HTML file for the Folium map.",
    ),
) -> None:
    """Generate an interactive station activity map highlighting trip volumes."""

    if not trips_path.exists():
        raise FileNotFoundError(f"Trips file not found: {trips_path}")
    if not station_info_path.exists():
        raise FileNotFoundError(f"Station information file not found: {station_info_path}")

    trips = pd.read_parquet(trips_path)
    if trips.empty:
        raise ValueError("Trips dataframe is empty; nothing to plot.")

    stations = pd.read_parquet(station_info_path)
    required_cols = {"station_id", "lat", "lon"}
    if not required_cols.issubset(stations.columns):
        missing = required_cols - set(stations.columns)
        raise KeyError(f"station_information missing columns: {missing}")
    stations = stations.copy()
    stations["station_id"] = stations["station_id"].astype(str)
    stations["station_name"] = (
        stations.get("name")
        .fillna(stations.get("short_name"))
        .fillna(stations["station_id"])
    )
    stations = stations.set_index("station_id")

    trips["origin_station_id"] = trips["origin_station_id"].astype(str)
    trips["destination_station_id"] = trips["destination_station_id"].astype(str)

    departures = trips.groupby("origin_station_id").size().rename("departures")
    arrivals = trips.groupby("destination_station_id").size().rename("arrivals")
    activity = pd.concat([departures, arrivals], axis=1).fillna(0)
    activity.index.name = "station_id"

    if metric == "departures":
        activity["value"] = activity["departures"]
    elif metric == "arrivals":
        activity["value"] = activity["arrivals"]
    else:
        activity["value"] = activity["departures"] - activity["arrivals"]

    if activity["value"].abs().sum() == 0:
        raise ValueError("Computed activity metric is zero for all stations.")

    activity = activity.join(stations, how="inner")
    if activity.empty:
        raise ValueError("No station metadata matched the trip stations.")

    center_lat = activity["lat"].mean()
    center_lon = activity["lon"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

    values = activity["value"].to_numpy()
    max_value = np.abs(values).max()
    if max_value <= 0:
        max_value = 1.0
    scale = 20 / max_value  # smaller scaling factor

    for station_id, row in activity.iterrows():
        value = row["value"]
        departures_count = row.get("departures", 0)
        arrivals_count = row.get("arrivals", 0)
        radius = max(4, abs(value) * scale)
        color = "green"
        if metric == "both":
            color = "red" if value < 0 else "blue"
        elif metric == "arrivals":
            color = "blue"
        else:
            color = "orange"
        popup_lines = [f"<b>{row['station_name']}</b>"]
        popup_lines.append(f"Departures: {int(departures_count)}")
        popup_lines.append(f"Arrivals: {int(arrivals_count)}")
        if metric == "both":
            popup_lines.append(f"Net: {int(value)}")
        else:
            popup_lines.append(f"{metric.title()}: {int(value)}")
        folium.CircleMarker(
            location=(row["lat"], row["lon"]),
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.6,
            weight=1,
            popup="<br>".join(popup_lines),
        ).add_to(fmap)

    ensure_directory(output_path.parent)
    fmap.save(output_path)
    typer.echo(f"Station activity map saved to {output_path}")


if __name__ == "__main__":
    app()
