"""Detect potential rebalancing activity based on station-level bike spikes."""
from __future__ import annotations

import heapq
import math
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import folium
import pandas as pd
import typer

from compare_gbfs_maps import (
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
    """Flag large station spikes and infer shortest routes using an assignment model."""

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
        station_info.get("name").fillna(station_info.get("short_name")).fillna(station_info["station_id"])
    )
    station_info = station_info.set_index("station_id")

    if not gbfs_root.exists() or not maps_root.exists():
        raise FileNotFoundError("Both data/raw/GBFS and data/raw/MAPS folders are required.")

    labels = _aligned_timestamps(gbfs_root, maps_root)
    if not labels:
        raise RuntimeError("No aligned snapshots found between GBFS and MAPS feeds.")

    events_df = _derive_events(labels, source, maps_root, gbfs_root, station_info, threshold)
    if events_df.empty:
        typer.echo("No station spikes exceeded the threshold; nothing to report.")
        return

    ensure_directory(events_output.parent)
    events_df.to_parquet(events_output, index=False)
    typer.echo(f"Logged {len(events_df)} spike events to {events_output}")

    graph = _build_graph(station_info, k=neighbors)
    depot_coords = (depot_lat, depot_lon) if depot_lat is not None and depot_lon is not None else None
    routes = _match_routes(
        events_df,
        station_info,
        graph,
        max_speed_kmph=max_speed_kmph,
        min_speed_kmph=min_speed_kmph,
        depot_coords=depot_coords,
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

    # Spike markers
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

    if depot_coords is not None:
        folium.Marker(
            location=list(depot_coords),
            icon=folium.Icon(color="black", icon="home", prefix="fa"),
            popup="Depot",
        ).add_to(fmap)

    for _, route in routes_df.iterrows():
        path_coords = []
        for station_id in route["path_station_ids"]:
            if station_id == "DEPOT" and depot_coords is not None:
                path_coords.append(list(depot_coords))
            elif station_id in station_info.index:
                meta = station_info.loc[station_id]
                path_coords.append([meta["lat"], meta["lon"]])
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


def _derive_events(
    labels: List[str],
    source: str,
    maps_root: Path,
    gbfs_root: Path,
    station_info: pd.DataFrame,
    threshold: int,
) -> pd.DataFrame:
    records: List[dict] = []
    for label in labels:
        ts = pd.to_datetime(label).tz_localize("UTC") if _needs_localize(label) else pd.to_datetime(label)
        counts = _load_maps_counts(maps_root / label) if source == "MAPS" else _load_gbfs_counts(gbfs_root / label)
        for station_id, value in counts.items():
            records.append({"timestamp": ts, "station_id": station_id, "count": value})

    counts_df = pd.DataFrame(records)
    pivot = (
        counts_df.pivot_table(index="timestamp", columns="station_id", values="count", aggfunc="mean")
        .fillna(0)
        .sort_index()
    )
    deltas = pivot.diff().dropna(how="all")

    events: List[dict] = []
    for timestamp, row in deltas.iterrows():
        for station_id, change in row.items():
            if pd.isna(change) or abs(change) < threshold:
                continue
            if station_id not in station_info.index:
                continue
            meta = station_info.loc[station_id]
            events.append(
                {
                    "timestamp": timestamp,
                    "station_id": station_id,
                    "station_name": _station_label(meta, station_id),
                    "change": int(change),
                    "movement_type": "pickup" if change < 0 else "dropoff",
                }
            )

    return pd.DataFrame(events).sort_values("timestamp") if events else pd.DataFrame()


def _match_routes(
    events_df: pd.DataFrame,
    station_info: pd.DataFrame,
    graph: Dict[str, List[Tuple[str, float]]],
    *,
    max_speed_kmph: float,
    min_speed_kmph: float,
    depot_coords: Optional[Tuple[float, float]],
) -> List[dict]:
    routes: List[dict] = []
    open_pickups: List[dict] = []
    depot = depot_coords
    timestamps = sorted(events_df["timestamp"].unique())

    for current_ts in timestamps:
        # register new pickups available from this timestamp forward
        for _, row in events_df[(events_df["movement_type"] == "pickup") & (events_df["timestamp"] == current_ts)].iterrows():
            if row["station_id"] not in station_info.index:
                continue
            open_pickups.append(
                {
                    "timestamp": row["timestamp"],
                    "station_id": row["station_id"],
                    "station_name": row["station_name"],
                    "remaining": abs(int(row["change"])),
                }
            )

        drop_events = events_df[(events_df["movement_type"] == "dropoff") & (events_df["timestamp"] == current_ts)]
        if drop_events.empty:
            continue

        for _, drop_event in drop_events.iterrows():
            if drop_event["station_id"] not in station_info.index:
                continue
            demand = abs(int(drop_event["change"]))
            if demand <= 0:
                continue

            eligible = [p for p in open_pickups if p["timestamp"] <= current_ts and p["remaining"] > 0]
            if not eligible:
                if depot is not None:
                    for _ in range(demand):
                        routes.append(
                            _build_depot_route(
                                {
                                    "timestamp": drop_event["timestamp"],
                                    "station_id": drop_event["station_id"],
                                    "station_name": drop_event["station_name"],
                                },
                                depot,
                                station_info,
                                max_speed_kmph,
                            )
                        )
                continue

            # choose pickups greedily by distance then timestamp
            selected_pickups: List[dict] = []
            remaining_demand = demand
            while remaining_demand > 0:
                eligible = [p for p in open_pickups if p["timestamp"] <= current_ts and p["remaining"] > 0]
                if not eligible:
                    break
                best = min(
                    eligible,
                    key=lambda p: (
                        _haversine_km(
                            station_info.loc[p["station_id"]]["lat"],
                            station_info.loc[p["station_id"]]["lon"],
                            station_info.loc[drop_event["station_id"]]["lat"],
                            station_info.loc[drop_event["station_id"]]["lon"],
                        ),
                        p["timestamp"],
                    ),
                )
                take = min(remaining_demand, best["remaining"])
                selected_pickups.append(
                    {
                        "timestamp": best["timestamp"],
                        "station_id": best["station_id"],
                        "station_name": best["station_name"],
                        "amount": take,
                    }
                )
                best["remaining"] -= take
                remaining_demand -= take
                if best["remaining"] == 0:
                    open_pickups.remove(best)

            if remaining_demand > 0:
                if depot is not None:
                    for _ in range(remaining_demand):
                        routes.append(
                            _build_depot_route(
                                {
                                    "timestamp": drop_event["timestamp"],
                                    "station_id": drop_event["station_id"],
                                    "station_name": drop_event["station_name"],
                                },
                                depot,
                                station_info,
                                max_speed_kmph,
                            )
                        )
                continue

            if not selected_pickups:
                continue

            selected_pickups.sort(key=lambda item: item["timestamp"])
            path_nodes = selected_pickups + [
                {
                    "timestamp": drop_event["timestamp"],
                    "station_id": drop_event["station_id"],
                    "station_name": drop_event["station_name"],
                    "amount": 0,
                }
            ]

            travel_start = selected_pickups[0]["timestamp"]
            travel_hours = max((drop_event["timestamp"] - travel_start).total_seconds() / 3600.0, 1 / 12)
            path_station_ids: List[str] = []
            total_distance = 0.0

            for idx in range(len(path_nodes) - 1):
                current_node = path_nodes[idx]
                next_node = path_nodes[idx + 1]
                start_id = current_node["station_id"]
                end_id = next_node["station_id"]
                if start_id == end_id:
                    continue
                if (
                    start_id not in station_info.index
                    or end_id not in station_info.index
                ):
                    if depot_coords is None:
                        continue
                    if start_id == "DEPOT":
                        start_lat, start_lon = depot_coords
                    else:
                        start_lat, start_lon = station_info.loc[start_id]["lat"], station_info.loc[start_id]["lon"]
                    if end_id == "DEPOT":
                        end_lat, end_lon = depot_coords
                    else:
                        end_lat, end_lon = station_info.loc[end_id]["lat"], station_info.loc[end_id]["lon"]
                    leg_path = [start_id, end_id]
                    leg_distance = _haversine_km(start_lat, start_lon, end_lat, end_lon)
                else:
                    leg_path, leg_distance = _dijkstra_path(start_id, end_id, graph, station_info)
                    if leg_distance == math.inf:
                        leg_distance = _haversine_km(
                            station_info.loc[start_id]["lat"],
                            station_info.loc[start_id]["lon"],
                            station_info.loc[end_id]["lat"],
                            station_info.loc[end_id]["lon"],
                        )
                        leg_path = [start_id, end_id]
                if not path_station_ids:
                    path_station_ids.extend(leg_path)
                else:
                    path_station_ids.extend(leg_path[1:])
                total_distance += leg_distance

            speed = total_distance / travel_hours if travel_hours > 0 else float("inf")
            if speed > max_speed_kmph:
                continue

            route_type = "standard"
            if speed < min_speed_kmph and depot is not None:
                route_type = "depot_return"

            moved_bikes = sum(item["amount"] for item in selected_pickups)
            routes.append(
                {
                    "timestamp": drop_event["timestamp"],
                    "origin_station_id": selected_pickups[0]["station_id"],
                    "origin_station_name": selected_pickups[0]["station_name"],
                    "destination_station_id": drop_event["station_id"],
                    "destination_station_name": drop_event["station_name"],
                    "moved_bikes": moved_bikes,
                    "direct_distance_km": total_distance,
                    "path_distance_km": total_distance,
                    "path_station_ids": path_station_ids,
                    "speed_kmph": speed,
                    "duration_hours": travel_hours,
                    "route_type": route_type,
                }
            )

    return routes


def _build_depot_route(
    drop: dict,
    depot_coords: Tuple[float, float],
    station_info: pd.DataFrame,
    max_speed_kmph: float,
) -> dict:
    if drop["station_id"] in station_info.index:
        dest = station_info.loc[drop["station_id"]]
        distance = _haversine_km(depot_coords[0], depot_coords[1], dest["lat"], dest["lon"])
    else:
        distance = 0.0
    travel_hours = max(distance / max_speed_kmph if max_speed_kmph > 0 else 0.25, 0.25)
    speed = distance / travel_hours if travel_hours > 0 else 0.0
    return {
        "timestamp": drop["timestamp"],
        "origin_station_id": "DEPOT",
        "origin_station_name": "Depot",
        "destination_station_id": drop["station_id"],
        "destination_station_name": drop["station_name"],
        "moved_bikes": drop.get("amount", 1),
        "direct_distance_km": distance,
        "path_distance_km": distance,
        "path_station_ids": ["DEPOT", drop["station_id"]],
        "speed_kmph": speed,
        "duration_hours": travel_hours,
        "route_type": "depot_delivery",
    }


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


if __name__ == "__main__":
    app()
