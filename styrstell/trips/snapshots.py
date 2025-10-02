"""Iterators for GBFS free bike status snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import pandas as pd

from styrstell.config import DataPaths

FREE_BIKE_FILENAME = "free_bike_status.json"
MAPS_FILENAME = "nextbike_live.json"


def iter_free_bike_snapshots(data_paths: DataPaths) -> Iterator[Tuple[datetime, pd.DataFrame]]:
    """Yield timestamped bike observations from GBFS snapshots.

    Parameters
    ----------
    data_paths:
        Configured data paths containing the raw GBFS directory.

    Yields
    ------
    tuple[datetime, pandas.DataFrame]
        UTC timestamp inferred from folder name and a dataframe with columns
        (`bike_id`, `station_id`, `lat`, `lon`, `vehicle_type_id`, `is_reserved`,
        `is_disabled`, plus any remaining fields present in the feed).
    """

    snapshot_root = Path(data_paths.raw) / "GBFS"
    if not snapshot_root.exists():
        return
    for folder in sorted(_snapshot_directories(snapshot_root)):
        timestamp = _parse_snapshot_timestamp(folder.name)
        file_path = folder / FREE_BIKE_FILENAME
        if not file_path.exists():
            continue
        df = _load_free_bike_status(file_path)
        if df.empty:
            continue
        yield timestamp, df


def iter_maps_bike_snapshots(data_paths: DataPaths) -> Iterator[Tuple[datetime, pd.DataFrame]]:
    """Yield timestamped bike observations from Nextbike maps API snapshots."""

    snapshot_root = Path(data_paths.raw) / "MAPS"
    if not snapshot_root.exists():
        return
    for folder in sorted(_snapshot_directories(snapshot_root)):
        timestamp = _parse_snapshot_timestamp(folder.name)
        file_path = folder / MAPS_FILENAME
        if not file_path.exists():
            continue
        df = _load_maps_snapshot(file_path)
        if df.empty:
            continue
        yield timestamp, df


def _snapshot_directories(root: Path) -> Iterable[Path]:
    for child in root.iterdir():
        if child.is_dir():
            yield child


def _parse_snapshot_timestamp(label: str) -> datetime:
    try:
        ts = datetime.fromisoformat(label)
    except ValueError:
        ts = datetime.fromisoformat(label.replace("Z", "+00:00")) if label.endswith("Z") else datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _load_free_bike_status(path: Path) -> pd.DataFrame:
    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    bikes = payload.get("data", {}).get("bikes", [])
    df = pd.json_normalize(bikes)
    if df.empty:
        return df
    columns = [
        "bike_id",
        "station_id",
        "lat",
        "lon",
        "vehicle_type_id",
        "is_reserved",
        "is_disabled",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df.assign(
        bike_id=df["bike_id"].astype(str),
        station_id=df["station_id"].astype(str),
        lat=pd.to_numeric(df["lat"], errors="coerce"),
        lon=pd.to_numeric(df["lon"], errors="coerce"),
        is_reserved=df["is_reserved"].fillna(False).astype(bool),
        is_disabled=df["is_disabled"].fillna(False).astype(bool),
    )
    df = df.loc[(~df["is_reserved"]) & (~df["is_disabled"]) & df["station_id"].notna(), :]
    return df.reset_index(drop=True)


def _load_maps_snapshot(path: Path) -> pd.DataFrame:
    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records: List[dict] = []
    for country in payload.get("countries", []):
        for city in country.get("cities", []):
            for place in city.get("places", []):
                station_uid = place.get("uid") or place.get("id")
                if station_uid is None:
                    continue
                station_id = str(station_uid)
                lat = place.get("lat")
                lon = place.get("lng")
                bike_list = place.get("bike_list") or []
                bike_numbers = place.get("bike_numbers")

                extracted: List[dict] = []
                for bike in bike_list:
                    number = bike.get("number") or bike.get("num") or bike.get("bike_number")
                    if number is None:
                        continue
                    state = str(bike.get("state") or "free").lower()
                    extracted.append(
                        {
                            "number": number,
                            "state": state,
                            "bike_type": bike.get("bike_type"),
                        }
                    )

                if not extracted and bike_numbers:
                    if isinstance(bike_numbers, str):
                        numbers = [item.strip() for item in bike_numbers.split(",") if item.strip()]
                    elif isinstance(bike_numbers, list):
                        numbers = [str(item) for item in bike_numbers if item]
                    else:
                        numbers = []
                    extracted.extend(
                        {
                            "number": number,
                            "state": "free",
                            "bike_type": place.get("bike_types") or None,
                        }
                        for number in numbers
                    )

                for bike in extracted:
                    state = bike.get("state", "free").lower()
                    if state in {"disabled", "reserved"}:
                        continue
                    number = bike.get("number")
                    if number is None:
                        continue
                    records.append(
                        {
                            "bike_id": str(number),
                            "station_id": station_id,
                            "lat": lat,
                            "lon": lon,
                            "vehicle_type_id": bike.get("bike_type"),
                            "state": state,
                        }
                    )
    if not records:
        return pd.DataFrame(columns=["bike_id", "station_id", "lat", "lon", "vehicle_type_id", "state"])
    df = pd.DataFrame.from_records(records)
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    return df
