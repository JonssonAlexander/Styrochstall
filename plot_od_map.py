"""Render OD flows on an interactive Leaflet map (HTML output)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import folium
import numpy as np
import pandas as pd
import typer

from styrstell.utils import ensure_directory, read_parquet

app = typer.Typer(add_completion=False)


@app.command()
def run(
    station_path: Path = typer.Option(Path("data/processed/station_information.parquet"), help="Station metadata parquet with lat/lon."),
    flow_path: Path = typer.Option(Path("data/models/od_flow_matrix.parquet"), help="OD flow matrix parquet."),
    probability_path: Path = typer.Option(Path("data/models/od_probability_matrix.parquet"), help="OD probability matrix parquet."),
    top_k: int = typer.Option(75, help="Number of strongest OD links to display (0 = all)."),
    min_flow: float = typer.Option(1.0, help="Minimum flow weight to include when plotting."),
    annotate: bool = typer.Option(False, help="Whether to label stations on the markers."),
    label_top: int = typer.Option(10, help="Number of highest-capacity stations to label when --annotate is set (0 = none)."),
    boundary_path: Optional[Path] = typer.Option(
        None,
        "--boundary",
        help="Optional GeoJSON boundary to overlay beneath the network.",
    ),
    output_path: Path = typer.Option(Path("reports/od_graph.html"), help="HTML file to write."),
    zoom_start: int = typer.Option(12, help="Initial zoom level for the map."),
) -> None:
    """Build an interactive OD flow map using Folium (Leaflet)."""

    stations = _load_station_metadata(station_path)
    flows = read_parquet(flow_path)
    probabilities = read_parquet(probability_path)

    flows.index = flows.index.astype(str)
    flows.columns = flows.columns.astype(str)
    probabilities.index = probabilities.index.astype(str)
    probabilities.columns = probabilities.columns.astype(str)

    missing = set(flows.index) - set(stations.index)
    if missing:
        typer.echo(f"Warning: {len(missing)} stations missing coordinates; dropping them from map.")
        flows = flows.drop(index=missing, errors="ignore").drop(columns=missing, errors="ignore")
        probabilities = probabilities.drop(index=missing, errors="ignore").drop(columns=missing, errors="ignore")

    edges = _prepare_edge_table(flows, top_k=top_k, min_flow=min_flow)
    if edges.empty:
        raise typer.BadParameter("No edges left to plot; adjust 'top_k' or 'min_flow'.")

    # Base map centered on station centroid
    map_center = [stations["lat"].mean(), stations["lon"].mean()]
    fmap = folium.Map(location=map_center, zoom_start=zoom_start, tiles="cartodbpositron")

    if boundary_path is not None:
        _add_boundary(fmap, boundary_path)

    _add_edges(fmap, edges, stations)
    _add_nodes(fmap, stations, flows, annotate=annotate, label_top=label_top)

    ensure_directory(output_path.parent)
    fmap.save(str(output_path))
    typer.echo(f"Interactive OD map saved to {output_path}")


def _load_station_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Station metadata not found: {path}")
    df = read_parquet(path)
    if "station_id" in df.columns:
        df = df.set_index(df["station_id"].astype(str))
    df.index = df.index.astype(str)
    required = {"lat", "lon"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise KeyError(f"Station metadata missing columns: {missing}")
    return df


def _prepare_edge_table(flows: pd.DataFrame, top_k: int, min_flow: float) -> pd.DataFrame:
    df = flows.copy()
    matrix = df.to_numpy(copy=True)
    np.fill_diagonal(matrix, 0.0)
    df.loc[:, :] = matrix
    df = df.rename_axis(index="origin", columns="destination")
    stacked = df.stack().reset_index(name="flow")
    filtered = stacked[stacked["flow"] >= min_flow]
    if top_k > 0:
        filtered = filtered.sort_values("flow", ascending=False).head(top_k)
    return filtered


def _add_boundary(fmap: folium.Map, boundary_path: Path) -> None:
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file not found: {boundary_path}")
    with boundary_path.open("r", encoding="utf-8") as handle:
        geojson = json.load(handle)
    folium.GeoJson(geojson, name="Boundary", style_function=lambda _: {"fillOpacity": 0, "color": "#555", "weight": 1}).add_to(fmap)


def _add_nodes(
    fmap: folium.Map,
    stations: pd.DataFrame,
    flows: pd.DataFrame,
    annotate: bool,
    label_top: int,
) -> None:
    totals = flows.sum(axis=1) + flows.sum(axis=0)
    totals = totals.reindex(stations.index).fillna(0)
    min_total, max_total = totals.min(), totals.max() if totals.max() > 0 else 1.0

    capacities = stations.get("capacity")
    if capacities is not None and not capacities.isna().all() and label_top != 0:
        top_labels = capacities.sort_values(ascending=False).head(label_top).index
    elif label_top == 0:
        top_labels = pd.Index([])
    else:
        top_labels = totals.sort_values(ascending=False).head(label_top).index if label_top > 0 else pd.Index([])

    for station_id, row in stations.iterrows():
        total_flow = totals.loc[station_id]
        radius = np.interp(total_flow, (min_total, max_total), (4, 14))
        display_name = row.get("name") or row.get("short_name") or station_id
        popup_lines = [f"<b>{display_name}</b>"]
        short_name = row.get("short_name")
        if short_name and short_name != display_name:
            popup_lines.append(f"Station {short_name}")
        popup_lines.append(f"Total flow: {total_flow:.2f}")
        if "capacity" in row and pd.notna(row["capacity"]):
            popup_lines.append(f"Capacity: {int(row['capacity'])}")
        popup_text = "<br>".join(popup_lines)
        marker = folium.CircleMarker(
            location=(row["lat"], row["lon"]),
            radius=radius,
            color="#1f77b4",
            fill=True,
            fill_opacity=0.8,
            weight=1,
            popup=popup_text,
        )
        marker.add_to(fmap)
        if annotate and station_id in top_labels:
            label_text = display_name
            folium.map.Marker(
                location=(row["lat"], row["lon"]),
                icon=folium.DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(0, 0),
                    html=f"<div style='font-size:9px;color:#2c3e50;font-weight:bold;'>{label_text}</div>",
                ),
            ).add_to(fmap)


def _add_edges(fmap: folium.Map, edges: pd.DataFrame, stations: pd.DataFrame) -> None:
    max_flow = edges["flow"].max() if not edges.empty else 1.0
    for _, row in edges.iterrows():
        origin = row["origin"]
        dest = row["destination"]
        if origin not in stations.index or dest not in stations.index:
            continue
        start = stations.loc[origin, ["lat", "lon"]]
        end = stations.loc[dest, ["lat", "lon"]]
        weight = row["flow"]
        width = float(np.interp(weight, (0, max_flow), (1, 6)))
        folium.PolyLine(
            locations=[(start["lat"], start["lon"]), (end["lat"], end["lon"])],
            weight=width,
            color="#ff5733",
            opacity=0.6,
            popup=f"{origin} → {dest}: {weight:.2f}",
        ).add_to(fmap)


if __name__ == "__main__":
    app()
