"""Render a time-aware Folium map tracking station bike availability."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson
from branca.element import Element
import typer

from styrstell.utils import ensure_directory

app = typer.Typer(add_completion=False)


def _resolve_station_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Station metadata not found: {path}")
    stations = pd.read_parquet(path)
    required = {"station_id", "lat", "lon"}
    missing = required - set(stations.columns)
    if missing:
        raise KeyError(f"Station metadata missing columns: {missing}")
    stations = stations.copy()
    stations["station_id"] = stations["station_id"].astype(str)
    stations["station_name"] = (
        stations.get("name")
        .fillna(stations.get("short_name"))
        .fillna(stations["station_id"])
    )
    return stations.set_index("station_id")


def _color_for_count(count: float) -> str:
    if count <= 0:
        return "#d7191c"  # red
    if count <= 3:
        return "#fdae61"  # amber/yellow
    return "#1a9641"  # green


def _period_iso(delta: str) -> str:
    seconds = int(pd.Timedelta(delta).total_seconds())
    seconds = max(seconds, 1)
    return f"PT{seconds}S"


@app.command()
def run(
    counts_path: Path = typer.Option(
        Path("data/processed/gbfs_maps_station_counts.parquet"),
        "--counts",
        help="Parquet with station counts over time (expects timestamp, station_id, source, count).",
    ),
    station_info_path: Path = typer.Option(
        Path("data/processed/station_information.parquet"),
        "--station-info",
        help="Parquet with station metadata providing lat/lon.",
    ),
    output_path: Path = typer.Option(
        Path("reports/station_status_timemap.html"),
        "--output",
        help="Destination HTML file for the interactive map.",
    ),
    source: str = typer.Option(
        "GBFS",
        "--source",
        help="Filter counts to this data source (e.g. GBFS or MAPS).",
    ),
    time_granularity: str = typer.Option(
        "15min",
        "--time-granularity",
        help="Floor timestamps to this pandas offset before visualising.",
    ),
    max_frames: Optional[int] = typer.Option(
        200,
        "--max-frames",
        help="Maximum number of time bins to plot (downsamples evenly if exceeded).",
    ),
) -> None:
    """Create a TimestampedGeoJson map showing station inventory over time."""

    if not counts_path.exists():
        raise FileNotFoundError(f"Counts file not found: {counts_path}")

    counts = pd.read_parquet(counts_path)
    required_cols = {"timestamp", "station_id", "source", "count"}
    missing = required_cols - set(counts.columns)
    if missing:
        raise KeyError(f"Counts parquet missing columns: {missing}")
    if counts.empty:
        raise ValueError("Counts dataframe is empty; nothing to plot.")

    stations = _resolve_station_metadata(station_info_path)

    frame = counts.copy()
    frame = frame.loc[frame["source"] == source.upper()].copy()
    if frame.empty:
        raise ValueError(f"No rows for source '{source}'.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["station_id"] = frame["station_id"].astype(str)
    frame["time_bin"] = frame["timestamp"].dt.floor(time_granularity)
    frame = (
        frame.groupby(["time_bin", "station_id"], as_index=False)["count"].mean()
    )

    frame = frame.merge(
        stations[["lat", "lon", "station_name"]],
        left_on="station_id",
        right_index=True,
        how="inner",
    )
    if frame.empty:
        raise ValueError("No station metadata matched the counts stations.")

    frame = frame.sort_values(["time_bin", "station_id"]).reset_index(drop=True)
    unique_bins = frame["time_bin"].drop_duplicates().sort_values().to_list()
    if max_frames is not None and len(unique_bins) > max_frames:
        indices = np.linspace(0, len(unique_bins) - 1, max_frames)
        selected = {unique_bins[int(round(idx))] for idx in indices}
        frame = frame.loc[frame["time_bin"].isin(selected)].copy()
        frame = frame.sort_values(["time_bin", "station_id"]).reset_index(drop=True)
        unique_bins = frame["time_bin"].drop_duplicates().sort_values().to_list()

    if not unique_bins:
        raise ValueError("No time bins available after filtering.")

    center_lat = frame["lat"].mean()
    center_lon = frame["lon"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

    features = []
    for time_bin, group in frame.groupby("time_bin"):
        timestamp_iso = pd.Timestamp(time_bin).isoformat()
        for row in group.itertuples():
            count = float(row.count)
            color = _color_for_count(count)
            size = max(4.0, min(count, 25.0) + 4.0)
            popup = (
                f"<strong>{row.station_name}</strong><br>"
                f"Bikes available: {int(round(count))}<br>"
                f"Timestamp: {timestamp_iso}"
            )
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row.lon, row.lat],
                },
                "properties": {
                    "time": timestamp_iso,
                    "popup": popup,
                    "icon": "circle",
                    "iconstyle": {
                        "fillColor": color,
                        "fillOpacity": 0.35,
                        "stroke": True,
                        "color": color,
                        "opacity": 0.6,
                        "radius": size,
                    },
                },
            }
            features.append(feature)

    data = {"type": "FeatureCollection", "features": features}
    TimestampedGeoJson(
        data,
        period=_period_iso(time_granularity),
        duration=_period_iso(time_granularity),
        add_last_point=False,
        transition_time=int(1000 / 15),
        loop=False,
        auto_play=False,
        max_speed=15,
        loop_button=True,
        time_slider_drag_update=True,
    ).add_to(fmap)

    legend_html = """
    <div style="position: fixed; bottom: 20px; left: 20px; width: 220px; z-index: 9999; background: rgba(255, 255, 255, 0.9); padding: 10px 12px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.4);">
      <h4 style="margin: 0 0 6px; font-size: 14px;">Bike Availability</h4>
      <ul style="margin: 0; padding-left: 16px; font-size: 12px; line-height: 1.4;">
        <li><span style="color:#d7191c;">●</span> 0 bikes</li>
        <li><span style="color:#fdae61;">●</span> 1-3 bikes</li>
        <li><span style="color:#1a9641;">●</span> 4+ bikes</li>
      </ul>
    </div>
    """
    fmap.get_root().html.add_child(Element(legend_html))

    ensure_directory(output_path.parent)
    fmap.save(output_path)
    typer.echo(
        f"Station status time map saved to {output_path} with {len(unique_bins)} frames"
    )


if __name__ == "__main__":
    app()
