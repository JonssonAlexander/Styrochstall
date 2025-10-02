"""Simulation environment encapsulation for bike-share dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import simpy

from styrstell.config import SimulationConfig
from styrstell.simulation.policies import RentalPolicy


@dataclass
class StationState:
    """Mutable station inventory and availability state."""

    station_id: str
    capacity: int
    bikes: int
    last_update: float = 0.0
    stockout_minutes: float = 0.0
    dockout_minutes: float = 0.0

    def propagate(self, now: float) -> None:
        delta = max(now - self.last_update, 0.0)
        if delta > 0:
            if self.bikes <= 0:
                self.stockout_minutes += delta
            if self.capacity > 0 and self.bikes >= self.capacity:
                self.dockout_minutes += delta
            self.last_update = now

    @property
    def docks(self) -> int:
        return max(self.capacity - self.bikes, 0)


class SimulationEnvironment:
    """Container for the SimPy environment and associated data accessors."""

    def __init__(
        self,
        panel: pd.DataFrame,
        demand: pd.DataFrame,
        travel_time: pd.DataFrame,
        od_matrix: pd.DataFrame,
        stations: pd.DataFrame,
        sim_config: SimulationConfig,
        policy: RentalPolicy,
    ) -> None:
        self.env = simpy.Environment()
        self.sim_config = sim_config
        self.policy = policy
        self.random = np.random.default_rng(sim_config.random_seed)
        self.start_time = pd.to_datetime(demand.index.get_level_values("timestamp").min())
        self.horizon_minutes = int(sim_config.simulation_horizon.total_seconds() / 60)
        self.station_states = self._initialize_stations(panel, stations)
        self.lambda_lookup = self._build_lambda_lookup(demand)
        self.od_lookup = self._build_od_lookup(od_matrix)
        self.travel_lookup = self._build_travel_lookup(travel_time)
        self.distance_lookup = self._build_distance_lookup(stations)
        self.metrics: Dict[str, float] = {
            "completed_trips": 0.0,
            "truncated_trips": 0.0,
            "stockout_events": 0.0,
            "dockout_events": 0.0,
            "total_walk_minutes": 0.0,
            "total_rental_minutes": 0.0,
        }
        self.trip_log: List[Dict[str, object]] = []

    def current_timestamp(self) -> pd.Timestamp:
        return self.start_time + timedelta(minutes=self.env.now)

    def lambda_rate(self, station_id: str, timestamp: pd.Timestamp) -> float:
        series = self.lambda_lookup.get(station_id)
        if series is None or series.empty:
            return 0.0
        try:
            value = series.asof(timestamp)
        except ValueError:
            return 0.0
        return float(value) if value is not None and not np.isnan(value) else 0.0

    def sample_destination(self, origin: str, timestamp: pd.Timestamp) -> str:
        matrices = self.od_lookup.get(origin)
        if not matrices:
            return origin
        for start, end, probs, destinations in matrices:
            if start <= timestamp < end:
                if probs.sum() <= 0:
                    return origin
                choice = self.random.choice(len(destinations), p=probs)
                return destinations[choice]
        return origin

    def sample_travel_minutes(self, timestamp: pd.Timestamp) -> float:
        tod_str = timestamp.floor("5min").time().strftime("%H:%M:%S")
        buckets = self.travel_lookup.get(tod_str)
        if not buckets:
            # fallback to global distribution
            buckets = self.travel_lookup.get("__global__")
        if not buckets:
            return float(self.policy.max_duration(timestamp))
        durations, probs = buckets
        return float(self.random.choice(durations, p=probs))

    def nearest_station_with_dock(self, station_id: str) -> Optional[Tuple[str, float]]:
        neighbors = self.distance_lookup.get(station_id, [])
        for candidate, distance_km in neighbors:
            state = self.station_states.get(candidate)
            if state and state.docks > 0:
                return candidate, distance_km
        return None

    def record_trip(
        self,
        origin: str,
        destination: str,
        start_time: pd.Timestamp,
        duration_minutes: float,
        completed: bool,
        walking_minutes: float,
    ) -> None:
        self.metrics["total_rental_minutes"] += duration_minutes
        self.metrics["total_walk_minutes"] += walking_minutes
        if completed:
            self.metrics["completed_trips"] += 1
        else:
            self.metrics["truncated_trips"] += 1
        self.trip_log.append(
            {
                "origin": origin,
                "destination": destination,
                "start_time": start_time,
                "duration_minutes": duration_minutes,
                "completed": completed,
                "walking_minutes": walking_minutes,
            }
        )

    def _initialize_stations(self, panel: pd.DataFrame, stations: pd.DataFrame) -> Dict[str, StationState]:
        station_states: Dict[str, StationState] = {}
        first_snapshot = panel.reset_index().sort_values("timestamp").groupby("station_id").first()
        if not first_snapshot.empty:
            first_snapshot.index = first_snapshot.index.astype(str)
        for _, row in stations.iterrows():
            station_id = str(row["station_id"])
            capacity = int(row.get("capacity", 0) or 0)
            bikes = int(first_snapshot.loc[station_id]["num_bikes_available"]) if station_id in first_snapshot.index else 0
            station_states[station_id] = StationState(station_id=station_id, capacity=capacity, bikes=bikes)
        return station_states

    def _build_lambda_lookup(self, demand: pd.DataFrame) -> Dict[str, pd.Series]:
        lookup: Dict[str, pd.Series] = {}
        for station_id, group in demand.groupby(level="station_id"):
            series = group["lambda_departures"].copy().sort_index()
            series.index = pd.to_datetime(series.index.get_level_values("timestamp"), utc=True)
            lookup[str(station_id)] = series
        return lookup

    def _build_od_lookup(self, od: pd.DataFrame) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp, np.ndarray, np.ndarray]]]:
        lookup: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp, np.ndarray, np.ndarray]]] = {}
        if od.empty:
            return lookup
        for (origin, slot_start), group in od.groupby(["origin", "slot_start"]):
            destinations = group["destination"].astype(str).to_numpy()
            probs = group["probability"].astype(float).to_numpy()
            slot_end = pd.to_datetime(group["slot_end"].iloc[0]) if "slot_end" in group else pd.to_datetime(slot_start) + timedelta(minutes=60)
            lookup.setdefault(str(origin), []).append(
                (
                    pd.to_datetime(slot_start),
                    pd.to_datetime(slot_end),
                    probs / probs.sum() if probs.sum() > 0 else probs,
                    destinations,
                )
            )
        for origin in lookup:
            lookup[origin].sort(key=lambda item: item[0])
        return lookup

    def _build_travel_lookup(self, travel_time: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        lookup: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if travel_time.empty:
            return lookup
        for key, group in travel_time.groupby("time_bin"):
            durations = ((group["duration_bin_start"] + group["duration_bin_end"]) / 2.0).to_numpy(dtype=float)
            probs = group["probability"].to_numpy(dtype=float)
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
            lookup[str(key)] = (durations, probs)
        if len(travel_time) > 0:
            durations = ((travel_time["duration_bin_start"] + travel_time["duration_bin_end"]) / 2.0).to_numpy(dtype=float)
            probs = travel_time["probability"].to_numpy(dtype=float)
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
            lookup["__global__"] = (durations, probs)
        return lookup

    def _build_distance_lookup(self, stations: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        lat = np.radians(stations["lat"].to_numpy(dtype=float))
        lon = np.radians(stations["lon"].to_numpy(dtype=float))
        ids = stations["station_id"].astype(str).tolist()
        neighbors: Dict[str, List[Tuple[str, float]]] = {}
        for i, origin in enumerate(ids):
            distances = []
            for j, destination in enumerate(ids):
                if i == j:
                    continue
                distance_km = _haversine(lat[i], lon[i], lat[j], lon[j])
                distances.append((destination, distance_km))
            neighbors[origin] = sorted(distances, key=lambda item: item[1])
        return neighbors


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    earth_radius_km = 6371.0
    return 2 * earth_radius_km * np.arcsin(np.minimum(1.0, np.sqrt(a)))
