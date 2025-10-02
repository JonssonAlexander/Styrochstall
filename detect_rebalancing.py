"""Detect potential rebalancing activity based on station-level bike spikes."""
from __future__ import annotations

import heapq
import math
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import folium
import pandas as pd
import typer

from scripts.compare_gbfs_maps import (
    _aligned_timestamps,
    _load_gbfs_counts,
    _load_maps_counts,
    _needs_localize,
)
from styrstell import config as cfg
from styrstell.utils import ensure_directory

app = typer.Typer(add_completion=False)


@app.command()
def run(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Calibration config to reuse data paths.",
    ),
    source: Literal["MAPS", "GBFS"] = typer.Option(
        "MAPS", "--source",
        help="Which feed to analyse for rebalancing spikes (MAPS or GBFS).",
    ),
    threshold: int = typer.Option(
        5,
        "--threshold",
        help="Minimum absolute change in bike count to flag a spike.",
    ),
    events_output: Path = typer.Option(
        Path("data/processed/rebalancing_events.parquet"),
        "--events-output",
        help="Output path for individual spike events.",
    ),
    routes_output: Path = typer.Option(
        Path("data/processed/rebalancing_routes.parquet"),
        "--routes-output",
        help="Output path for inferred vehicle routes between spikes.",
    ),
    output_map: Optional[Path] = typer.Option(
        Path("reports/rebalancing_routes.html"),
        "--output-map",
        help="Optional Folium map visualising spikes and inferred routes.",
    ),
    neighbors: int = typer.Option(
        5,
        "--neighbors",
        help="Number of nearest neighbors per station when building the routing graph.",
    ),
    max_speed_kmph: float = typer.Option(
        35.0,
        "--max-speed-kmph",
        help="Maximum plausible speed for a rebalancing vehicle.",
    ),
    min_speed_kmph: float = typer.Option(
        5.0,
        "--min-speed-kmph",
        help="If inferred speed is below this, route is assumed to involve the depot.",
    ),
    depot_lat: Optional[float] = typer.Option(
        None,
        "--depot-lat",
        help="Latitude of the depot (used for slow-speed fallback).",
    ),
    depot_lon: Optional[float] = typer.Option(
        None,
        "--depot-lon",
        help="Longitude of the depot (used for slow-speed fallback).",
    ),
) -> None:
    """Flag large station spikes and infer shortest routes using Dijkstra on a station graph."""

    calibration_cfg = cfg.load_calibration_config(config_path)
    data_paths = calibration_cfg.data

    gbfs_root = Path(data_paths.raw) / "GBFS"
    maps_root = Path(data_paths.raw) / "MAPS"
    station_info_path = data_paths.processed / "station_information.parquet"

    if not station_info_path.exists():
        raise FileNotFoundError("station_information.parquet not found. Run calibration first.")

    station_info = pd.read_parquet(station_info_path)
    station_info["station_id"] = station_info["station_id"].astype(str)
    station_info["station_name"] = (
        station_info.get("name")
        .fillna(station_info.get("short_name"))
        .fillna(station_info["station_id"])
    )
    station_info = station_info.set_index("station_id")

    if not gbfs_root.exists() or not maps_root.exists():
        raise FileNotFoundError("Both data/raw/GBFS and data/raw/MAPS folders are required.")

    labels = _aligned_timestamps(gbfs_root, maps_root)
    if not labels:
        raise RuntimeError("No aligned snapshots found between GBFS and MAPS feeds.")

    # Build time series of station counts for the selected source feed
    records: List[dict] = []
    for label in labels:
        ts = pd.to_datetime(label).tz_localize("UTC") if _needs_localize(label) else pd.to_datetime(label)
        counts = _load_maps_counts(maps_root / label) if source == "MAPS" else _load_gbfs_counts(gbfs_root / label)
        for station_id, value in counts.items():
            records.append(
                {
                    "timestamp": ts,
                    "station_id": station_id,
                    "count": value,
                }
            )
    if not records:
        raise RuntimeError("No station counts available for spike detection.")

    counts_df = pd.DataFrame(records)
    pivot = counts_df.pivot_table(
        index="timestamp",
        columns="station_id",
        values="count",
        aggfunc="mean",
    ).sort_index()
    deltas = pivot.diff().dropna(how="all")

    events = []
    for timestamp, row in deltas.iterrows():
        for station_id, change in row.items():
            if pd.isna(change) or abs(change) < threshold:
                continue
            meta = station_info.loc[station_id] if station_id in station_info.index else None
            label = _station_label(meta, station_id)
            event_type = "pickup" if change < 0 else "dropoff"
            events.append(
                {
                    "timestamp": timestamp,
                    "station_id": station_id,
                    "station_name": label,
                    "change": int(change),
                    "movement_type": event_type,
                }
            )

    if not events:
        typer.echo("No station spikes exceeded the threshold; nothing to report.")
        return

    events_df = pd.DataFrame(events).sort_values("timestamp")
    ensure_directory(events_output.parent)
    events_df.to_parquet(events_output, index=False)
    typer.echo(f"Logged {len(events_df)} spike events to {events_output}")

    # Build routing graph (k-nearest neighbors) and infer routes from pickups to dropoffs
    graph = _build_graph(station_info, k=neighbors)
    routes = _infer_routes(
        events_df,
        station_info,
        graph,
        max_speed_kmph=max_speed_kmph,
        min_speed_kmph=min_speed_kmph,
        depot_coords=(depot_lat, depot_lon) if depot_lat is not None and depot_lon is not None else None,
    )
    routes_df = pd.DataFrame(routes)
    if not routes_df.empty:
        ensure_directory(routes_output.parent)
        routes_df.to_parquet(routes_output, index=False)
        typer.echo(f"Saved {len(routes_df)} inferred routes to {routes_output}")
    else:
        typer.echo("No pickup/dropoff pairs available to infer routes.")

    if output_map is None:
        return

    ensure_directory(output_map.parent)
    fmap = folium.Map(location=[station_info["lat"].mean(), station_info["lon"].mean()], zoom_start=12)

    # Add spike markers
    for _, row in events_df.iterrows():
        if row["station_id"] not in station_info.index:
            continue
        meta = station_info.loc[row["station_id"]]
        coords = [meta["lat"], meta["lon"]]
        color = "red" if row["movement_type"] == "pickup" else "blue"
        tooltip = (
            f"{row['timestamp'].tz_convert('UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}<br>"
            f"{row['station_name']}<br>"
            f"Change: {row['change']}"
        )
        folium.CircleMarker(
            location=coords,
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=tooltip,
        ).add_to(fmap)

    if not routes_df.empty:
        depot_coords = (depot_lat, depot_lon) if depot_lat is not None and depot_lon is not None else None
        if depot_coords:
            folium.Marker(
                location=list(depot_coords),
                icon=folium.Icon(color="black", icon="home", prefix="fa"),
                popup="Depot",
            ).add_to(fmap)
        for _, route in routes_df.iterrows():
            path_ids = route["path_station_ids"]
            path_coords = []
            for sid in path_ids:
                if sid == "DEPOT" and depot_coords:
                    path_coords.append(list(depot_coords))
                elif sid in station_info.index:
                    path_coords.append([station_info.loc[sid]["lat"], station_info.loc[sid]["lon"]])
            if len(path_coords) < 2:
                continue
            tooltip = (
                f"{route['timestamp'].tz_convert('UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}<br>"
                f"{route['origin_station_name']} → {route['destination_station_name']}<br>"
                f"Bikes moved: {route['moved_bikes']}<br>"
                f"Direct distance: {route['direct_distance_km']:.2f} km<br>"
                f"Route distance: {route['path_distance_km']:.2f} km<br>"
                f"Speed: {route['speed_kmph']:.2f} km/h"
            )
            folium.PolyLine(
                locations=path_coords,
                weight=2 + route["moved_bikes"],
                color="purple",
                opacity=0.7,
                tooltip=tooltip,
            ).add_to(fmap)

    fmap.save(output_map)
    typer.echo(f"Rebalancing map saved to {output_map}")


def _infer_routes(
    events_df: pd.DataFrame,
    station_info: pd.DataFrame,
    graph: Dict[str, List[Tuple[str, float]]],
    *,
    max_speed_kmph: float,
    min_speed_kmph: float,
    depot_coords: Optional[Tuple[float, float]],
) -> List[dict]:
    routes: List[dict] = []
    if events_df.empty:
        return routes

    pickups = events_df[events_df["movement_type"] == "pickup"]
    dropoffs = events_df[events_df["movement_type"] == "dropoff"]
    if pickups.empty or dropoffs.empty:
        return routes

    dropoff_pool: Dict[Tuple[pd.Timestamp, str], int] = {
        (row["timestamp"], row["station_id"]): int(row["change"])
        for _, row in dropoffs.iterrows()
    }

    for _, pickup in pickups.iterrows():
        timestamp = pickup["timestamp"]
        origin_id = pickup["station_id"]
        moved_bikes = abs(int(pickup["change"]))
        origin_meta = station_info.loc[origin_id] if origin_id in station_info.index else None
        if origin_meta is None:
            continue
        origin_name = _station_label(origin_meta, origin_id)

        best_dest = None
        best_distance = math.inf
        for (drop_ts, dest_id), drop_value in dropoff_pool.items():
            if drop_ts != timestamp or drop_value <= 0:
                continue
            dest_meta = station_info.loc[dest_id] if dest_id in station_info.index else None
            if dest_meta is None:
                continue
            distance = _haversine_km(
                origin_meta["lat"], origin_meta["lon"], dest_meta["lat"], dest_meta["lon"]
            )
            if distance < best_distance:
                best_distance = distance
                best_dest = (dest_id, min(moved_bikes, drop_value), dest_meta)
        if best_dest is None:
            continue

        dest_id, matched, dest_meta = best_dest
        dropoff_pool[(timestamp, dest_id)] -= matched
        if dropoff_pool[(timestamp, dest_id)] <= 0:
            dropoff_pool.pop((timestamp, dest_id))

        dest_name = _station_label(dest_meta, dest_id)
        path_ids, path_distance = _dijkstra_path(origin_id, dest_id, graph, station_info)
        time_delta_hours = _time_delta_hours(events_df, timestamp)
        if time_delta_hours <= 0:
            continue
        speed = path_distance / time_delta_hours if path_distance > 0 else 0.0
        if speed > max_speed_kmph:
            continue
        route_record = {
            "timestamp": timestamp,
            "origin_station_id": origin_id,
            "origin_station_name": origin_name,
            "destination_station_id": dest_id,
            "destination_station_name": dest_name,
            "moved_bikes": matched,
            "direct_distance_km": best_distance,
            "path_distance_km": path_distance,
            "path_station_ids": path_ids,
            "speed_kmph": speed,
            "duration_hours": time_delta_hours,
            "route_type": "standard",
        }

        if speed < min_speed_kmph and depot_coords is not None:
            depot_distance = _haversine_km(
                station_info.loc[origin_id]["lat"],
                station_info.loc[origin_id]["lon"],
                depot_coords[0],
                depot_coords[1],
            )
            route_record.update(
                {
                    "destination_station_id": "DEPOT",
                    "destination_station_name": "Depot",
                    "path_station_ids": [origin_id, "DEPOT"],
                    "path_distance_km": depot_distance,
                    "speed_kmph": depot_distance / time_delta_hours if time_delta_hours > 0 else 0.0,
                    "route_type": "depot_return",
                }
            )
            dropoff_pool[(timestamp, dest_id)] = dropoff_pool.get((timestamp, dest_id), 0) + matched

        routes.append(route_record)

    return routes


def _station_label(meta: Optional[pd.Series], station_id: str) -> str:
    if meta is None:
        return station_id
    return str(meta.get("station_name") or meta.get("name") or meta.get("short_name") or station_id)


def _build_graph(stations: pd.DataFrame, k: int) -> Dict[str, List[Tuple[str, float]]]:
    station_ids = stations.index.tolist()
    coords = stations[["lat", "lon"]].to_numpy()
    neighbors: Dict[str, List[Tuple[str, float]]] = {station: [] for station in station_ids}
    for i, origin in enumerate(station_ids):
        distances = []
        for j, dest in enumerate(station_ids):
            if origin == dest:
                continue
            dist = _haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            distances.append((dest, dist))
        distances.sort(key=lambda item: item[1])
        neighbors[origin] = distances[:k]
    return neighbors


def _dijkstra_path(
    origin: str,
    dest: str,
    graph: Dict[str, List[Tuple[str, float]]],
    station_info: pd.DataFrame,
) -> Tuple[List[str], float]:
    if origin == dest:
        return [origin], 0.0

    distances = {node: math.inf for node in graph}
    previous: Dict[str, Optional[str]] = {node: None for node in graph}
    distances[origin] = 0.0
    heap: List[Tuple[float, str]] = [(0.0, origin)]

    while heap:
        current_dist, node = heapq.heappop(heap)
        if node == dest:
            break
        if current_dist > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = current_dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = node
                heapq.heappush(heap, (new_dist, neighbor))

    if distances[dest] == math.inf:
        if origin in station_info.index and dest in station_info.index:
            direct = _haversine_km(
                station_info.loc[origin]["lat"],
                station_info.loc[origin]["lon"],
                station_info.loc[dest]["lat"],
                station_info.loc[dest]["lon"],
            )
        else:
            direct = math.inf
        return [origin, dest], direct

    path = []
    node = dest
    while node is not None:
        path.append(node)
        node = previous[node]
    path.reverse()
    return path, distances[dest]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def _time_delta_hours(events_df: pd.DataFrame, timestamp: pd.Timestamp) -> float:
    timestamps = events_df["timestamp"].sort_values().unique()
    idx = list(timestamps).index(timestamp)
    if idx == 0:
        return 0.0
    prev_ts = timestamps[idx - 1]
    delta_hours = (timestamp - prev_ts).total_seconds() / 3600.0
    return delta_hours


if __name__ == "__main__":
    app()
