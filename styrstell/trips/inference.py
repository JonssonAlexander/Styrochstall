"""Trip timeline construction and inference."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class TripInferenceConfig:
    """Parameters controlling trip inference heuristics."""

    min_duration_seconds: float = 60.0
    max_duration_seconds: float = 3 * 3600.0
    min_distance_km: float = 0.00
    max_speed_kmph: float = 35.0
    ignore_nighttime: bool = True
    nighttime_start_hour: int = 1
    nighttime_end_hour: int = 5
    nighttime_max_duration_seconds: float = 90 * 60.0


def build_bike_timelines(snapshots: Iterable[Tuple[datetime, pd.DataFrame]]) -> pd.DataFrame:
    """Stack snapshot observations into a bike timeline dataframe."""

    frames: List[pd.DataFrame] = []
    for timestamp, frame in snapshots:
        if frame.empty:
            continue
        local = frame.copy()
        ts = pd.to_datetime(timestamp, utc=True)
        local["timestamp"] = ts
        frames.append(local)
    if not frames:
        return pd.DataFrame(columns=["timestamp", "bike_id", "station_id", "lat", "lon"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[combined["station_id"].notna()].copy()
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined["bike_id"] = combined["bike_id"].astype(str)
    combined["station_id"] = combined["station_id"].astype(str)
    combined["lat"] = pd.to_numeric(combined.get("lat"), errors="coerce")
    combined["lon"] = pd.to_numeric(combined.get("lon"), errors="coerce")
    combined = combined.sort_values(["bike_id", "timestamp"]).reset_index(drop=True)
    return combined


def infer_trips_from_timelines(timelines: pd.DataFrame, config: TripInferenceConfig | None = None) -> pd.DataFrame:
    """Infer trips from bike timelines."""

    if timelines.empty:
        return pd.DataFrame(columns=_trip_columns())
    cfg = config or TripInferenceConfig()
    records: List[dict] = []
    for bike_id, group in timelines.groupby("bike_id", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < 2:
            continue
        segment_ids = (group["station_id"] != group["station_id"].shift()).cumsum()
        segment_first = group.groupby(segment_ids)["timestamp"].transform("first")
        segment_last = group.groupby(segment_ids)["timestamp"].transform("last")

        for idx in range(1, len(group)):
            current_station = group.loc[idx, "station_id"]
            previous_station = group.loc[idx - 1, "station_id"]
            if current_station == previous_station:
                continue
            prev_time = group.loc[idx - 1, "timestamp"]
            curr_time = group.loc[idx, "timestamp"]
            if pd.isna(prev_time) or pd.isna(curr_time):
                continue
            gap_seconds = (curr_time - prev_time).total_seconds()
            if gap_seconds <= 0:
                continue
            if gap_seconds < cfg.min_duration_seconds or gap_seconds > cfg.max_duration_seconds:
                continue
            # Nighttime heuristic
            if cfg.ignore_nighttime:
                hour = prev_time.tz_convert("UTC").hour if isinstance(prev_time, pd.Timestamp) else prev_time.hour
                in_night = _hour_in_range(hour, cfg.nighttime_start_hour, cfg.nighttime_end_hour)
                if in_night and gap_seconds > cfg.nighttime_max_duration_seconds:
                    continue

            origin_lat = group.loc[idx - 1, "lat"]
            origin_lon = group.loc[idx - 1, "lon"]
            dest_lat = group.loc[idx, "lat"]
            dest_lon = group.loc[idx, "lon"]
            distance_km = _haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
            if distance_km is not None and distance_km < cfg.min_distance_km:
                continue
            if distance_km is not None and gap_seconds > 0:
                speed_kmph = (distance_km / gap_seconds) * 3600.0
                if speed_kmph > cfg.max_speed_kmph:
                    continue
            else:
                speed_kmph = np.nan

            depart_last_seen = segment_last.loc[idx - 1]
            arrive_first_seen = segment_first.loc[idx]
            midpoint = depart_last_seen + (arrive_first_seen - depart_last_seen) / 2

            records.append(
                {
                    "bike_id": bike_id,
                    "origin_station_id": previous_station,
                    "destination_station_id": current_station,
                    "origin_last_seen_at": depart_last_seen,
                    "destination_first_seen_at": arrive_first_seen,
                    "estimated_departure_time": midpoint,
                    "estimated_arrival_time": midpoint,
                    "duration_seconds": gap_seconds,
                    "distance_km": distance_km,
                    "average_speed_kmph": speed_kmph,
                    "snapshot_gap_seconds": gap_seconds,
                    "snapshot_gap_count": 1,
                    "origin_lat": origin_lat,
                    "origin_lon": origin_lon,
                    "destination_lat": dest_lat,
                    "destination_lon": dest_lon,
                }
            )
    trips = pd.DataFrame(records, columns=_trip_columns())
    return trips.sort_values(["bike_id", "estimated_departure_time"]).reset_index(drop=True)


def _trip_columns() -> List[str]:
    return [
        "bike_id",
        "origin_station_id",
        "destination_station_id",
        "origin_last_seen_at",
        "destination_first_seen_at",
        "estimated_departure_time",
        "estimated_arrival_time",
        "duration_seconds",
        "distance_km",
        "average_speed_kmph",
        "snapshot_gap_seconds",
        "snapshot_gap_count",
        "origin_lat",
        "origin_lon",
        "destination_lat",
        "destination_lon",
    ]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float | None:
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return None
    lat1_rad, lon1_rad = np.radians([lat1, lon1])
    lat2_rad, lon2_rad = np.radians([lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    return float(earth_radius_km * c)


def _hour_in_range(hour: int, start: int, end: int) -> bool:
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end
