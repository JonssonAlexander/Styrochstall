"""Visualise stockout probabilities over time on a Folium map."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson
from branca.element import Element
import typer

from styrstell.utils import ensure_directory

app = typer.Typer(add_completion=False)


def _color_for_probability(prob: float) -> str:
    if prob >= 0.5:
        return "#a50026"
    if prob >= 0.3:
        return "#d73027"
    if prob >= 0.1:
        return "#fdae61"
    if prob >= 0.05:
        return "#fee08b"
    return "#1a9641"


def _period_iso(delta: str) -> str:
    seconds = int(pd.to_timedelta(delta.lower()).total_seconds())
    seconds = max(seconds, 1)
    return f"PT{seconds}S"


@app.command()
def run(
    probability_path: Path = typer.Option(
        Path("data/processed/monte_carlo_stockout_probabilities.parquet"),
        "--probabilities",
        help="Aggregated stockout probability parquet.",
    ),
    station_info_path: Path = typer.Option(
        Path("data/processed/station_information.parquet"),
        "--station-info",
        help="Station metadata with station_id, lat, lon.",
    ),
    scenario: str = typer.Option(
        "demand_only",
        "--scenario",
        help="Scenario to visualise (must match aggregated data).",
    ),
    output_path: Path = typer.Option(
        Path("reports/stockout_probability_map.html"),
        "--output",
        help="Destination HTML file for the interactive map.",
    ),
    time_granularity: str = typer.Option(
        "1H",
        "--time-granularity",
        help="Time granularity used in probabilities (e.g. 15min, 1H).",
    ),
) -> None:
    if not probability_path.exists():
        raise FileNotFoundError(f"Probability parquet not found: {probability_path}")
    prob_df = pd.read_parquet(probability_path)
    prob_df = prob_df.loc[prob_df["scenario"] == scenario].copy()
    if prob_df.empty:
        raise ValueError(f"No rows for scenario '{scenario}'.")

    if not station_info_path.exists():
        raise FileNotFoundError(f"Station info not found: {station_info_path}")
    station_meta = pd.read_parquet(station_info_path)
    required = {"station_id", "lat", "lon"}
    missing = required - set(station_meta.columns)
    if missing:
        raise KeyError(f"Station metadata missing columns: {missing}")
    station_meta = station_meta.copy()
    station_meta["station_id"] = station_meta["station_id"].astype(str)
    station_meta["station_name"] = (
        station_meta.get("name")
        .fillna(station_meta.get("short_name"))
        .fillna(station_meta["station_id"])
    )
    station_meta = station_meta.set_index("station_id")

    prob_df["time_bin"] = pd.to_datetime(prob_df["time_bin"], utc=True)
    prob_df["station_id"] = prob_df["station_id"].astype(str)
    prob_df = prob_df.merge(
        station_meta[["lat", "lon", "station_name"]],
        left_on="station_id",
        right_index=True,
        how="inner",
    )
    if prob_df.empty:
        raise ValueError("No station metadata matched probability rows.")

    # Aggregate in case multiple entries exist per station/time (e.g., after merges)
    agg_cols = {
        "stockout_probability": "mean",
        "lat": "first",
        "lon": "first",
        "station_name": "first",
    }
    if "mean_bikes" in prob_df.columns:
        agg_cols["mean_bikes"] = "mean"
    prob_df = (
        prob_df.groupby(["time_bin", "station_id"], as_index=False)
        .agg(agg_cols)
        .sort_values(["time_bin", "station_id"])
        .reset_index(drop=True)
    )
    time_bins = prob_df["time_bin"].drop_duplicates().to_list()

    center_lat = prob_df["lat"].mean()
    center_lon = prob_df["lon"].mean()
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

    # Build features with metrics (probability and bikes if available)
    features = []
    inventory_available = "mean_bikes" in prob_df.columns

    for _, row in prob_df.iterrows():
        timestamp_iso = pd.Timestamp(row["time_bin"]).isoformat()
        prob = float(row["stockout_probability"])
        color = _color_for_probability(prob)
        radius = 6 + prob * 20
        popup_lines = [f"<strong>{row['station_name']}</strong>"]
        popup_lines.append(f"Stockout probability: {prob:.2%}")
        if inventory_available and pd.notna(row.get("mean_bikes")):
            popup_lines.append(f"Mean bikes: {row['mean_bikes']:.1f}")
        popup_lines.append(f"Time bin: {timestamp_iso}")

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["lon"], row["lat"]],
            },
            "properties": {
                "time": timestamp_iso,
                "popup": "<br>".join(popup_lines),
                "icon": "circle",
                "iconstyle": {
                    "fillColor": color,
                    "fillOpacity": 0.65,
                    "stroke": True,
                    "color": color,
                    "opacity": 0.65,
                    "radius": radius,
                },
            },
        }
        features.append(feature)

    layer = TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period=_period_iso(time_granularity),
        duration="PT1S",
        add_last_point=False,
        transition_time=0,
        loop=False,
        auto_play=False,
        max_speed=10,
        loop_button=True,
        time_slider_drag_update=True,
    )
    layer.add_to(fmap)

    legend_html = """
    <div style=\"position: fixed; bottom: 20px; left: 20px; width: 240px; z-index: 9999; background: rgba(255, 255, 255, 0.9); padding: 10px 12px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.4);\">
      <h4 style=\"margin: 0 0 6px; font-size: 14px;\">Stockout Probability</h4>
      <ul style=\"margin: 0; padding-left: 16px; font-size: 12px; line-height: 1.4;\">
        <li><span style=\"color:#a50026;\">●</span> ≥ 50%</li>
        <li><span style=\"color:#d73027;\">●</span> 30–50%</li>
        <li><span style=\"color:#fdae61;\">●</span> 10–30%</li>
        <li><span style=\"color:#fee08b;\">●</span> 5–10%</li>
        <li><span style=\"color:#1a9641;\">●</span> &lt; 5%</li>
      </ul>
    </div>
    """
    fmap.get_root().html.add_child(Element(legend_html))

    ensure_directory(output_path.parent)
    fmap.save(output_path)
    typer.echo(
        f"Stockout probability map saved to {output_path} containing {len(time_bins)} time bins"
    )


if __name__ == "__main__":
    app()
