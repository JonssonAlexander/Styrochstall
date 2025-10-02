"""Visualise absorbing stations and strongly connected components on a Folium map."""
from __future__ import annotations

from colorsys import hsv_to_rgb
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import folium
import pandas as pd
import typer

from styrstell.utils import ensure_directory

app = typer.Typer(add_completion=False)


@app.command()
def run(
    station_path: Path = typer.Option(
        Path("data/processed/station_information.parquet"),
        "--station-info",
        help="Parquet file with station metadata (station_id, name, lat, lon).",
    ),
    absorbing_path: Optional[Path] = typer.Option(
        Path("data/processed/absorbing_states.parquet"),
        "--absorbing",
        help="Parquet file with absorbing-state summary (optional).",
    ),
    components_path: Path = typer.Option(
        Path("data/processed/strongly_connected_components.parquet"),
        "--components",
        help="Parquet file containing strongly connected components (component_id, stations).",
    ),
    output_path: Path = typer.Option(
        Path("reports/graph_components_map.html"),
        "--output",
        help="Output HTML file for the map.",
    ),
) -> None:
    """Render absorbing stations and strongly connected components on a Folium map."""

    stations = _load_station_metadata(station_path)
    fmap = folium.Map(
        location=[stations["lat"].mean(), stations["lon"].mean()],
        zoom_start=12,
        tiles="cartodbpositron",
    )

    if absorbing_path is not None and absorbing_path.exists():
        absorbing = pd.read_parquet(absorbing_path)
        _add_absorbing_markers(fmap, absorbing, stations)
    else:
        typer.echo("Absorbing-state file missing; skipping absorbing markers.")

    if components_path.exists():
        components = pd.read_parquet(components_path)
        if not components.empty:
            _add_component_layers(fmap, components, stations)
        else:
            typer.echo("No strongly connected components to plot.")
    else:
        raise FileNotFoundError(f"Components file not found: {components_path}")

    folium.LayerControl(collapsed=False).add_to(fmap)
    ensure_directory(output_path.parent)
    fmap.save(output_path)
    typer.echo(f"Graph diagnostics map saved to {output_path}")


def _load_station_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Station metadata not found: {path}")
    df = pd.read_parquet(path)
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].astype(str)
        df = df.set_index("station_id")
    else:
        df.index = df.index.astype(str)
    required = {"lat", "lon"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise KeyError(f"Station metadata missing columns: {missing}")
    name_series = df.get("name")
    if name_series is None:
        name_series = pd.Series(index=df.index, dtype="object")

    short_series = df.get("short_name")
    if short_series is None:
        short_series = pd.Series(index=df.index, dtype="object")

    station_name = name_series.fillna(short_series)
    df["station_name"] = station_name.where(
        station_name.notna(), df.index.astype(str)
    )
    return df


def _add_absorbing_markers(fmap: folium.Map, absorbing: pd.DataFrame, stations: pd.DataFrame) -> None:
    layer = folium.FeatureGroup(name="Absorbing stations", show=True)
    for _, row in absorbing.iterrows():
        station_id = str(row["station_id"])
        if station_id not in stations.index:
            continue
        meta = stations.loc[station_id]
        tooltip = (
            f"{meta['station_name']}<br>"
            f"Self probability: {row['self_probability']:.2f}<br>"
            f"Residual: {row.get('residual_probability', float('nan')):.2f}"
        )
        folium.CircleMarker(
            location=(meta["lat"], meta["lon"]),
            radius=7,
            color="#d73027",
            fill=True,
            fill_opacity=0.85,
            weight=2,
            popup=tooltip,
        ).add_to(layer)
    layer.add_to(fmap)


def _generate_palette(count: int) -> List[str]:
    """Return ``count`` visually distinct hex colors using evenly spaced hues."""
    if count <= 0:
        return []

    palette: List[str] = []
    for idx in range(count):
        hue = (idx / count + 0.17) % 1.0  # offset to avoid overlapping ends
        r, g, b = hsv_to_rgb(hue, 0.65, 0.95)
        palette.append(
            f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"
        )
    return palette


def _add_component_layers(
    fmap: folium.Map,
    components: pd.DataFrame,
    stations: pd.DataFrame,
) -> None:
    colors = _generate_palette(len(components))

    for idx, (_, row) in enumerate(components.iterrows()):
        comp_id = row["component_id"]
        stations_list: Iterable[str] = row["stations"]
        color = colors[idx] if idx < len(colors) else "#444444"
        layer = folium.FeatureGroup(name=f"SCC {comp_id}", show=False)

        coords: List[Tuple[float, float]] = []
        names: List[str] = []
        for station_id in stations_list:
            sid = str(station_id)
            if sid not in stations.index:
                continue
            meta = stations.loc[sid]
            coords.append((meta["lat"], meta["lon"]))
            names.append(meta["station_name"])
            folium.CircleMarker(
                location=(meta["lat"], meta["lon"]),
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{meta['station_name']} (SCC {comp_id})",
            ).add_to(layer)

        if len(coords) >= 2:
            folium.PolyLine(
                locations=coords + [coords[0]],
                color=color,
                weight=2,
                opacity=0.6,
                tooltip=f"Cycle {comp_id}: {' → '.join(names)}",
            ).add_to(layer)

        layer.add_to(fmap)


if __name__ == "__main__":
    app()
