"""GBFS data ingestion with snapshot versioning."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import requests

from styrstell.config import DataPaths, GBFSFeedConfig
from styrstell.utils import ensure_directory


@dataclass(frozen=True)
class SnapshotMetadata:
    """Metadata describing a stored GBFS snapshot."""

    timestamp: datetime
    directory: Path
    files: Dict[str, Path]


class GBFSLoader:
    """Loader that fetches and materializes GBFS feeds as timestamped snapshots."""

    def __init__(self, data_paths: DataPaths, config: GBFSFeedConfig) -> None:
        self._data_paths = data_paths
        self._config = config
        self._raw_root = ensure_directory(data_paths.raw / "GBFS")

    def list_snapshots(self) -> Iterable[SnapshotMetadata]:
        """Yield available snapshots ordered by timestamp."""

        for directory in sorted(self._raw_root.iterdir()):
            if not directory.is_dir():
                continue
            try:
                ts = datetime.fromisoformat(directory.name)
            except ValueError:
                continue
            files = {f.stem: f for f in directory.glob("*.json")}
            yield SnapshotMetadata(timestamp=ts, directory=directory, files=files)

    def latest_snapshot(self) -> Optional[SnapshotMetadata]:
        """Return the most recent snapshot if one exists."""

        snapshots = list(self.list_snapshots())
        return snapshots[-1] if snapshots else None

    def fetch_snapshot(self, timestamp: Optional[datetime] = None) -> SnapshotMetadata:
        """Fetch GBFS feeds and persist them under a timestamped directory."""

        ts = timestamp or datetime.now(timezone.utc)
        label = ts.replace(microsecond=0).isoformat()
        snapshot_dir = ensure_directory(self._raw_root / label)
        files: Dict[str, Path] = {}
        for feed_name, url in self._iter_feeds().items():
            response = self._request_with_retries(url)
            payload = response.json()
            target = snapshot_dir / f"{feed_name}.json"
            target.write_text(response.text, encoding="utf-8")
            files[feed_name] = target
        return SnapshotMetadata(timestamp=ts, directory=snapshot_dir, files=files)

    def load_station_information(self, snapshot: Optional[SnapshotMetadata] = None) -> pd.DataFrame:
        """Load station metadata, backfilling missing fields where possible."""

        meta = snapshot or self.latest_snapshot()
        if meta is None or "station_information" not in meta.files:
            raise FileNotFoundError("No station_information snapshot available.")
        payload = self._read_json(meta.files["station_information"])
        data = payload.get("data", {}).get("stations", [])
        df = pd.json_normalize(data)
        expected = ["station_id", "name", "lat", "lon", "capacity"]
        for column in expected:
            if column not in df.columns:
                df[column] = None
        capacity = self._coerce_numeric(df.get("capacity"), df.index)
        vehicle_capacity = self._coerce_numeric(df.get("vehicle_capacity"), df.index)
        docks = self._coerce_numeric(df.get("docks"), df.index)
        df["capacity"] = capacity.fillna(vehicle_capacity).fillna(docks).fillna(0)
        df["station_id"] = df["station_id"].astype(str)
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(0).astype(int)
        return df

    def load_station_status(self, snapshot: Optional[SnapshotMetadata] = None) -> pd.DataFrame:
        """Load station status records, coercing types and tolerating missing keys."""

        meta = snapshot or self.latest_snapshot()
        if meta is None or "station_status" not in meta.files:
            raise FileNotFoundError("No station_status snapshot available.")
        payload = self._read_json(meta.files["station_status"])
        last_updated = payload.get("last_updated") or payload.get("ttl")
        timestamp = datetime.fromtimestamp(last_updated, tz=timezone.utc) if last_updated else datetime.now(timezone.utc)
        stations = payload.get("data", {}).get("stations", [])
        df = pd.json_normalize(stations)
        defaults = {
            "num_bikes_available": 0,
            "num_docks_available": 0,
            "is_installed": True,
            "is_renting": True,
            "is_returning": True,
        }
        for key, default in defaults.items():
            if key not in df.columns:
                df[key] = default
        df["station_id"] = df["station_id"].astype(str)
        df["last_reported"] = pd.to_datetime(df.get("last_reported"), unit="s", utc=True, errors="coerce")
        df["snapshot_ts"] = timestamp
        return df

    def _iter_feeds(self) -> Dict[str, str]:
        feeds = dict(self._config.feeds)
        if not self._config.include_free_bike_status:
            feeds.pop("free_bike_status", None)
        return feeds

    def _request_with_retries(self, url: str) -> requests.Response:
        for attempt in range(self._config.max_retries + 1):
            response = requests.get(url, timeout=self._config.request_timeout)
            if response.ok:
                return response
            if attempt == self._config.max_retries:
                response.raise_for_status()
        raise RuntimeError("Unreachable code path in _request_with_retries")

    @staticmethod
    def _coerce_numeric(series: Optional[pd.Series], index: pd.Index) -> pd.Series:
        if series is None:
            return pd.Series(float("nan"), index=index, dtype="float64")
        return pd.to_numeric(series, errors="coerce")

    @staticmethod
    def _read_json(path: Path) -> Dict[str, object]:
        import json

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
