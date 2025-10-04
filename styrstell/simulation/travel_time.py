"""Travel time providers with optional Google Maps augmentation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import pandas as pd


def _time_key(timestamp: pd.Timestamp, granularity: str) -> str:
    ts = pd.to_datetime(timestamp, utc=True)
    return ts.floor(granularity).time().isoformat(timespec="minutes")


@dataclass
class TravelTimeObservation:
    minutes: float
    source: str


class TravelTimeProvider:
    """Protocol-like base class for travel time lookup implementations."""

    def get_minutes(self, origin: str, destination: str, departure: pd.Timestamp) -> TravelTimeObservation:
        raise NotImplementedError


class CachedTravelTimeProvider(TravelTimeProvider):
    """Lookup travel time from a cached table with optional fallback."""

    def __init__(
        self,
        cache: Optional[pd.DataFrame] = None,
        time_granularity: str = "15min",
        fallback_minutes: float = 15.0,
        coordinates: Optional[Mapping[str, Tuple[float, float]]] = None,
        speed_kmph: float = 17.0,
    ) -> None:
        self.time_granularity = time_granularity
        self.fallback_minutes = fallback_minutes
        self._cache: Dict[Tuple[str, str, str], float] = {}
        self.coordinates: Dict[str, Tuple[float, float]] = {}
        self.speed_kmph = speed_kmph
        if coordinates:
            self.coordinates = {
                str(station_id): (float(lat), float(lon))
                for station_id, (lat, lon) in coordinates.items()
            }
        if cache is not None and not cache.empty:
            self._ingest(cache)

    def _ingest(self, cache: pd.DataFrame) -> None:
        required = {"origin", "destination", "mean_minutes"}
        if not required.issubset(cache.columns):
            raise ValueError(f"Cache missing columns: {required - set(cache.columns)}")
        time_col = cache.columns.intersection(["time_bin", "hour", "time_key"])
        if time_col.empty:
            cache["time_key"] = "__global__"
        else:
            cache = cache.rename(columns={time_col[0]: "time_key"})
            cache["time_key"] = cache["time_key"].astype(str)
        for _, row in cache.iterrows():
            key = (str(row["origin"]), str(row["destination"]), row["time_key"])
            self._cache[key] = float(row["mean_minutes"])

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "origin": origin,
                "destination": destination,
                "time_key": time_key,
                "mean_minutes": minutes,
            }
            for (origin, destination, time_key), minutes in self._cache.items()
        ]
        return pd.DataFrame(rows)

    def get_minutes(self, origin: str, destination: str, departure: pd.Timestamp) -> TravelTimeObservation:
        origin = str(origin)
        destination = str(destination)
        if origin == destination:
            return TravelTimeObservation(minutes=0.0, source="self")

        time_key = _time_key(departure, self.time_granularity)
        for candidate in ((origin, destination, time_key), (origin, destination, "__global__")):
            value = self._cache.get(candidate)
            if value is not None:
                return TravelTimeObservation(minutes=float(value), source="cache")
        if self.coordinates:
            origin_coord = self.coordinates.get(origin)
            destination_coord = self.coordinates.get(destination)
            if origin_coord and destination_coord:
                minutes = _haversine_minutes(
                    origin_coord[0],
                    origin_coord[1],
                    destination_coord[0],
                    destination_coord[1],
                    self.speed_kmph,
                )
                self.update(origin, destination, departure, minutes)
                return TravelTimeObservation(minutes=float(minutes), source="haversine")

        return TravelTimeObservation(minutes=self.fallback_minutes, source="fallback")

    def update(self, origin: str, destination: str, departure: pd.Timestamp, minutes: float) -> None:
        key = (str(origin), str(destination), _time_key(departure, self.time_granularity))
        self._cache[key] = float(minutes)

    def save(self, path: Path) -> None:
        frame = self.to_frame()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


class GoogleMapsTravelTimeProvider(CachedTravelTimeProvider):
    """Augment cache with Google Maps Distance Matrix queries."""

    def __init__(
        self,
        api_key: str,
        cache_path: Optional[Path] = None,
        time_granularity: str = "15min",
        fallback_minutes: float = 15.0,
        coordinates: Optional[Mapping[str, Tuple[float, float]]] = None,
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path is not None else None
        initial_cache = None
        if self.cache_path is not None and self.cache_path.exists():
            initial_cache = pd.read_parquet(self.cache_path)
        super().__init__(
            initial_cache,
            time_granularity=time_granularity,
            fallback_minutes=fallback_minutes,
            coordinates=coordinates,
            speed_kmph=speed_kmph,
        )
        try:
            import googlemaps
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("googlemaps package is required for Maps travel time queries") from exc
        self._client = googlemaps.Client(key=api_key)

    def get_minutes(self, origin: str, destination: str, departure: pd.Timestamp) -> TravelTimeObservation:
        observation = super().get_minutes(origin, destination, departure)
        if observation.source != "fallback":
            return observation

        origin_coord = self.coordinates.get(str(origin))
        dest_coord = self.coordinates.get(str(destination))
        if origin_coord is None or dest_coord is None:
            return observation
        origin_str = f"{origin_coord[0]},{origin_coord[1]}"
        dest_str = f"{dest_coord[0]},{dest_coord[1]}"

        response = self._client.distance_matrix(
            origins=[origin_str],
            destinations=[dest_str],
            mode="bicycling",
            departure_time=departure.to_pydatetime(),
        )
        elements = response.get("rows", [{}])[0].get("elements", [{}])
        duration_seconds = elements[0].get("duration", {}).get("value")
        if duration_seconds is None:
            minutes = _haversine_minutes(
                origin_coord[0],
                origin_coord[1],
                dest_coord[0],
                dest_coord[1],
                self.speed_kmph,
            )
            self.update(origin, destination, departure, minutes)
            return TravelTimeObservation(minutes=float(minutes), source="haversine")
        minutes = duration_seconds / 60.0
        self.update(origin, destination, departure, minutes)
        if self.cache_path is not None:
            self.save(self.cache_path)
        return TravelTimeObservation(minutes=float(minutes), source="google_maps")


def ensure_positive_duration(minutes: float, minimum: float = 1.0) -> float:
    return max(minutes, minimum)


def _haversine_minutes(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    speed_kmph: float,
) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = radius * c
    if speed_kmph <= 0:
        speed_kmph = 1e-6
    hours = distance_km / speed_kmph
    minutes = hours * 60.0
    return minutes
