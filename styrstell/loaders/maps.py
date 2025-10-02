"""Loader for the Nextbike maps API snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from styrstell.config import DataPaths, MapsAPIConfig
from styrstell.utils import ensure_directory


@dataclass(frozen=True)
class MapsSnapshotMetadata:
    """Metadata describing a stored maps API snapshot."""

    timestamp: datetime
    directory: Path
    file: Path


class NextbikeMapsLoader:
    """Persist Nextbike maps API responses under timestamped directories."""

    def __init__(self, data_paths: DataPaths, config: MapsAPIConfig) -> None:
        self._data_paths = data_paths
        self._config = config
        self._root = ensure_directory(data_paths.raw / "MAPS")

    def fetch_snapshot(self, timestamp: Optional[datetime] = None) -> MapsSnapshotMetadata:
        ts = (timestamp or datetime.now(timezone.utc)).replace(microsecond=0)
        directory = ensure_directory(self._root / ts.isoformat())
        response = self._request_with_retries(self._config.endpoint)
        target = directory / "nextbike_live.json"
        target.write_text(response.text, encoding="utf-8")
        return MapsSnapshotMetadata(timestamp=ts, directory=directory, file=target)

    def _request_with_retries(self, url: str) -> requests.Response:
        for attempt in range(self._config.max_retries + 1):
            response = requests.get(url, timeout=self._config.request_timeout)
            if response.ok:
                return response
            if attempt == self._config.max_retries:
                response.raise_for_status()
        raise RuntimeError("Unreachable code path in _request_with_retries")
