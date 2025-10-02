"""Visualise OD matrices as a geographic chord/graph plot."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from styrstell.utils import ensure_directory, read_parquet

try:
    import contextily as ctx
except ImportError:  # pragma: no cover - optional dependency
    ctx = None

app = typer.Typer(add_completion=False)


@app.command()
def run(
    station_path: Path = typer.Option(Path("data/processed/station_information.parquet"), help="Station metadata parquet with lat/lon."),
    flow_path: Path = typer.Option(Path("data/models/od_flow_matrix.parquet"), help="OD flow matrix parquet."),
    probability_path: Path = typer.Option(Path("data/models/od_probability_matrix.parquet"), help="OD probability matrix parquet."),
    top_k: int = typer.Option(50, help="Number of strongest OD links to display (0 = show all)."),
    min_flow: float = typer.Option(1.0, help="Minimum flow weight to include when plotting."),
    output_path: Optional[Path] = typer.Option(Path("reports/od_graph.png"), help="Optional path to save the figure."),
    annotate: bool = typer.Option(False, help="Whether to annotate station IDs on the plot."),
    basemap: bool = typer.Option(False, "--basemap/--no-basemap", help="Overlay an OpenStreetMap basemap (requires contextily)."),
    boundary_path: Optional[Path] = typer.Option(
        None,
        "--boundary",
        help="Optional GeoJSON boundary to draw behind the network (processed with the stdlib).",
    ),
) -> None:
    """Plot the OD flow graph with nodes positioned by geographic coordinates."""

    stations = _load_station_metadata(station_path)
    flows = read_parquet(flow_path)
    probabilities = read_parquet(probability_path)

    # Ensure consistent ordering and types
    flows.index = flows.index.astype(str)
    flows.columns = flows.columns.astype(str)
    probabilities.index = probabilities.index.astype(str)
    probabilities.columns = probabilities.columns.astype(str)

    missing = set(flows.index) - set(stations.index)
    if missing:
        typer.echo(f"Warning: {len(missing)} stations missing coordinates; dropping them from graph.")
        flows = flows.drop(index=missing, errors="ignore").drop(columns=missing, errors="ignore")
        probabilities = probabilities.drop(index=missing, errors="ignore").drop(columns=missing, errors="ignore")

    edges = _prepare_edge_table(flows, top_k=top_k, min_flow=min_flow)
    if edges.empty:
        raise typer.BadParameter("No edges left to plot; adjust 'top_k' or 'min_flow'.")

    fig, ax = plt.subplots(figsize=(10, 8))
    coords = _prepare_coordinates(stations, use_basemap=basemap)
    if boundary_path is not None:
        _plot_boundary(ax, boundary_path, use_basemap=basemap)
    _plot_nodes(ax, stations, flows, coords=coords, annotate=annotate, use_basemap=basemap)
    _plot_edges(ax, edges, coords=coords)

    if basemap:
        if ctx is None:
            raise RuntimeError(
                "contextily is required for basemap plotting. Install it or rerun with --no-basemap."
            )
        ctx.add_basemap(ax, crs="EPSG:3857")
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    ax.set_title("Origin-Destination Flow Graph")

    plt.tight_layout()
    if output_path:
        ensure_directory(output_path.parent)
        fig.savefig(output_path, dpi=200)
        typer.echo(f"Saved OD graph to {output_path}")
    plt.show()


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


def _prepare_coordinates(stations: pd.DataFrame, use_basemap: bool) -> pd.DataFrame:
    coords = stations[["lon", "lat"]].copy()
    coords.columns = ["lon", "lat"]
    if use_basemap:
        try:
            from pyproj import Transformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pyproj is required for basemap plotting. Install it or rerun with --no-basemap."
            ) from exc
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = transformer.transform(coords["lon"].to_numpy(), coords["lat"].to_numpy())
        coords["x"] = x
        coords["y"] = y
    else:
        coords["x"] = coords["lon"]
        coords["y"] = coords["lat"]
    return coords


def _plot_boundary(ax, boundary_path: Path, use_basemap: bool) -> None:
    import json

    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file not found: {boundary_path}")
    with boundary_path.open("r", encoding="utf-8") as handle:
        geojson = json.load(handle)

    geometries = []
    gtype = geojson.get("type")
    if gtype == "FeatureCollection":
        geometries = [feat.get("geometry") for feat in geojson.get("features", [])]
    elif gtype in {"Feature"}:
        geometries = [geojson.get("geometry")]
    else:
        geometries = [geojson]

    transformer = None
    if use_basemap:
        try:
            from pyproj import Transformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pyproj is required for basemap plotting. Install it or omit --basemap when using --boundary."
            ) from exc
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def project(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if transformer is None:
            return coords[:, 0], coords[:, 1]
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.asarray(x), np.asarray(y)

    for geometry in geometries:
        if not geometry:
            continue
        gtype = geometry.get("type")
        if gtype == "Polygon":
            for ring in geometry.get("coordinates", []):
                coords = np.array(ring)
                if coords.size == 0:
                    continue
                xs, ys = project(coords)
                ax.plot(xs, ys, color="gray", linewidth=0.8, alpha=0.6, zorder=1)
        elif gtype == "MultiPolygon":
            for polygon in geometry.get("coordinates", []):
                for ring in polygon:
                    coords = np.array(ring)
                    if coords.size == 0:
                        continue
                    xs, ys = project(coords)
                    ax.plot(xs, ys, color="gray", linewidth=0.8, alpha=0.6, zorder=1)


def _plot_nodes(
    ax,
    stations: pd.DataFrame,
    flows: pd.DataFrame,
    coords: pd.DataFrame,
    annotate: bool,
    use_basemap: bool,
) -> None:
    totals = flows.sum(axis=1) + flows.sum(axis=0)
    totals = totals.reindex(stations.index).fillna(0)
    sizes = np.interp(totals, (totals.min(), totals.max() if totals.max() > 0 else 1.0), (50, 250))
    ax.scatter(
        coords["x"],
        coords["y"],
        s=sizes,
        c="royalblue",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    if annotate:
        for station_id in stations.index:
            ax.text(
                coords.loc[station_id, "x"],
                coords.loc[station_id, "y"],
                station_id,
                fontsize=6,
                ha="center",
                va="center",
                color="black" if use_basemap else "dimgray",
            )


def _plot_edges(ax, edges: pd.DataFrame, coords: pd.DataFrame) -> None:
    max_flow = edges["flow"].max() if not edges.empty else 1.0
    for _, row in edges.iterrows():
        origin = row["origin"]
        dest = row["destination"]
        if origin not in coords.index or dest not in coords.index:
            continue
        start = coords.loc[origin, ["x", "y"]]
        end = coords.loc[dest, ["x", "y"]]
        weight = row["flow"]
        lw = np.interp(weight, (0, max_flow), (0.5, 3.0))
        ax.plot(
            [start["x"], end["x"]],
            [start["y"], end["y"]],
            color="orangered",
            alpha=0.5,
            linewidth=lw,
            zorder=2,
        )


if __name__ == "__main__":
    app()
