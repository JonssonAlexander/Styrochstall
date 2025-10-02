import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from styrstell.config import DataPaths
from styrstell.trips import (
    TripInferenceConfig,
    build_bike_timelines,
    build_edge_list,
    build_od_matrix,
    build_travel_time_distribution,
    export_edge_list_for_visualization,
    infer_trips_from_timelines,
    iter_free_bike_snapshots,
    iter_maps_bike_snapshots,
)


def _write_snapshot(folder: Path, bikes: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"data": {"bikes": bikes}}
    with (folder / "free_bike_status.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_trip_inference_simple(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    paths = DataPaths(root=data_root)
    raw_root = Path(paths.raw) / "GBFS"

    t0 = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=10)

    _write_snapshot(
        raw_root / t0.isoformat(),
        [
            {
                "bike_id": "bike1",
                "station_id": "A",
                "lat": 57.7,
                "lon": 11.9,
                "vehicle_type_id": "std",
                "is_reserved": False,
                "is_disabled": False,
            }
        ],
    )
    _write_snapshot(
        raw_root / t1.isoformat(),
        [
            {
                "bike_id": "bike1",
                "station_id": "B",
                "lat": 57.71,
                "lon": 11.91,
                "vehicle_type_id": "std",
                "is_reserved": False,
                "is_disabled": False,
            }
        ],
    )

    snapshots = list(iter_free_bike_snapshots(paths))
    assert len(snapshots) == 2
    timelines = build_bike_timelines(snapshots)
    assert len(timelines) == 2

    config = TripInferenceConfig(
        min_duration_seconds=0,
        max_duration_seconds=3600,
        min_distance_km=0.0,
        max_speed_kmph=100.0,
        ignore_nighttime=False,
    )
    trips = infer_trips_from_timelines(timelines, config)
    assert len(trips) == 1
    trip = trips.iloc[0]
    assert trip["origin_station_id"] == "A"
    assert trip["destination_station_id"] == "B"
    assert trip["duration_seconds"] == 600.0

    edge_list = build_edge_list(trips)
    assert len(edge_list) == 1
    edge = edge_list.iloc[0]
    assert edge["trip_count"] == 1
    assert abs(edge["mean_duration_minutes"] - 10.0) < 1e-6

    od_matrix = build_od_matrix(trips)
    assert od_matrix.loc["A", "B"] == 1

    travel_stats = build_travel_time_distribution(trips)
    assert not travel_stats.empty
    assert set(travel_stats.columns) == {"hour", "mean_minutes", "p50", "p80", "p95"}

    viz_edges = export_edge_list_for_visualization(edge_list)
    assert list(viz_edges.columns) == ["origin", "destination", "weight", "distance_km", "mean_duration_minutes"]
    assert viz_edges.iloc[0]["origin"] == "A"


def test_iter_maps_snapshots(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    paths = DataPaths(root=data_root)
    maps_root = Path(paths.raw) / "MAPS"

    snapshot_ts = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    folder = maps_root / snapshot_ts.isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "countries": [
            {
                "cities": [
                    {
                        "places": [
                            {
                                "uid": "S1",
                                "lat": 57.7,
                                "lng": 11.9,
                                "bike_list": [
                                    {"number": "B1", "state": "free", "bike_type": "std"},
                                    {"number": "B2", "state": "reserved", "bike_type": "std"},
                                ],
                            },
                            {
                                "uid": "S2",
                                "lat": 57.71,
                                "lng": 11.91,
                                "bike_list": [
                                    {"number": "B3", "state": "free", "bike_type": "std"},
                                ],
                            },
                        ]
                    }
                ]
            }
        ]
    }
    with (folder / "nextbike_live.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    snapshots = list(iter_maps_bike_snapshots(paths))
    assert len(snapshots) == 1
    timestamp, dataframe = snapshots[0]
    assert timestamp == snapshot_ts
    assert len(dataframe) == 2  # reserved bike filtered out
    assert set(dataframe["station_id"]) == {"S1", "S2"}
