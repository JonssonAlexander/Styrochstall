"""Compare GBFS station counts with Nextbike MAPS bike counts and create an interactive HTML animation."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Set

import pandas as pd
import plotly.express as px
import typer
import imageio.v3 as iio
import matplotlib.pyplot as plt

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
    output_html: Path = typer.Option(
        Path("reports/gbfs_vs_maps.html"),
        "--output-html",
        help="Output HTML path for the interactive comparison.",
    ),
    output_parquet: Path = typer.Option(
        Path("data/processed/gbfs_maps_station_counts.parquet"),
        "--output-parquet",
        help="Where to persist the tidy comparison table.",
    ),
    output_gif: Optional[Path] = typer.Option(
        None,
        "--output-gif",
        help="Optional animated GIF comparing feeds over time.",
    ),
) -> None:
    """Build an interactive animation comparing GBFS vs MAPS per-station bike counts."""

    calibration_cfg = cfg.load_calibration_config(config_path)
    data_paths = calibration_cfg.data
    gbfs_root = Path(data_paths.raw) / "GBFS"
    maps_root = Path(data_paths.raw) / "MAPS"

    if not gbfs_root.exists() or not maps_root.exists():
        raise FileNotFoundError("Both data/raw/GBFS and data/raw/MAPS directories are required.")

    timestamps = _aligned_timestamps(gbfs_root, maps_root)
    if not timestamps:
        raise RuntimeError("No aligned snapshots found between GBFS and MAPS.")

    records = []
    for label in timestamps:
        ts = pd.to_datetime(label).tz_localize("UTC") if _needs_localize(label) else pd.to_datetime(label)
        gbfs_counts = _load_gbfs_counts(gbfs_root / label)
        maps_counts = _load_maps_counts(maps_root / label)

        for station_id, count in gbfs_counts.items():
            records.append(
                {
                    "timestamp": ts,
                    "station_id": station_id,
                    "source": "GBFS",
                    "count": count,
                }
            )
        for station_id, count in maps_counts.items():
            records.append(
                {
                    "timestamp": ts,
                    "station_id": station_id,
                    "source": "MAPS",
                    "count": count,
                }
            )

    comparison = pd.DataFrame(records)
    comparison["station_id"] = comparison["station_id"].astype(str)
    comparison.sort_values(["timestamp", "station_id", "source"], inplace=True)

    station_info_path = data_paths.processed / "station_information.parquet"
    if station_info_path.exists():
        station_info = pd.read_parquet(station_info_path)
        station_info["station_id"] = station_info["station_id"].astype(str)
        station_info["station_label"] = (
            station_info.get("name")
            .fillna(station_info.get("short_name"))
            .fillna(station_info["station_id"])
        )
        comparison = comparison.merge(
            station_info[["station_id", "station_label"]],
            on="station_id",
            how="left",
        )
    else:
        comparison["station_label"] = comparison["station_id"]
    comparison["station_label"] = comparison["station_label"].fillna(comparison["station_id"])

    ensure_directory(output_parquet.parent)
    comparison.to_parquet(output_parquet, index=False)

    df_plot = comparison.copy()
    df_plot["timestamp_str"] = df_plot["timestamp"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M:%S")
    fig = px.bar(
        df_plot,
        x="station_label",
        y="count",
        color="source",
        animation_frame="timestamp_str",
        barmode="group",
        title="GBFS vs MAPS per-station bike counts",
        labels={"station_label": "Station", "count": "Bike count", "source": "Feed"},
    )
    fig.update_layout(transition={'duration': 500})

    ensure_directory(output_html.parent)
    fig.write_html(output_html, include_plotlyjs="cdn")
    typer.echo(f"Comparison table saved to {output_parquet}")
    typer.echo(f"Interactive HTML saved to {output_html}")

    if output_gif is not None:
        typer.echo("Rendering animated GIF ...")
        station_labels = (
            comparison[["station_id", "station_label"]]
            .drop_duplicates()
            .set_index("station_id")
            ["station_label"]
            .to_dict()
        )
        _export_gif(comparison, output_gif, station_labels)
        typer.echo(f"Animated GIF saved to {output_gif}")


def _aligned_timestamps(gbfs_root: Path, maps_root: Path) -> list[str]:
    gbfs_labels: Set[str] = {
        folder.name
        for folder in gbfs_root.iterdir()
        if folder.is_dir() and (folder / "station_status.json").exists()
    }
    maps_labels: Set[str] = {
        folder.name
        for folder in maps_root.iterdir()
        if folder.is_dir() and (folder / "nextbike_live.json").exists()
    }
    aligned = sorted(gbfs_labels & maps_labels)
    return aligned


def _needs_localize(label: str) -> bool:
    try:
        dt = pd.to_datetime(label)
    except Exception:
        return False
    return dt.tzinfo is None


def _load_gbfs_counts(snapshot_dir: Path) -> dict[str, int]:
    path = snapshot_dir / "station_status.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    stations = payload.get("data", {}).get("stations", [])
    counts: dict[str, int] = {}
    for station in stations:
        station_id = str(station.get("station_id"))
        if station_id in {"None", "nan", ""}:
            continue
        count = int(station.get("num_bikes_available") or 0)
        counts[station_id] = count
    return counts


def _load_maps_counts(snapshot_dir: Path) -> dict[str, int]:
    path = snapshot_dir / "nextbike_live.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    counts: dict[str, int] = {}
    for country in payload.get("countries", []):
        for city in country.get("cities", []):
            for place in city.get("places", []):
                station_uid = place.get("uid") or place.get("id")
                if station_uid is None:
                    continue
                station_id = str(station_uid)
                bike_list = place.get("bike_list") or []
                bike_numbers = place.get("bike_numbers")

                identifiers = []
                for bike in bike_list:
                    number = bike.get("number") or bike.get("num") or bike.get("bike_number")
                    if number:
                        state = str(bike.get("state") or "free").lower()
                        if state not in {"disabled", "reserved"}:
                            identifiers.append(str(number))
                if not identifiers and bike_numbers:
                    if isinstance(bike_numbers, str):
                        identifiers = [item.strip() for item in bike_numbers.split(",") if item.strip()]
                    elif isinstance(bike_numbers, list):
                        identifiers = [str(item) for item in bike_numbers if item]
                counts[station_id] = len(identifiers)
    return counts


def _export_gif(
    comparison: pd.DataFrame,
    output_gif: Path,
    station_labels: dict[str, str],
) -> None:
    ensure_directory(output_gif.parent)
    unique_stations = sorted(comparison["station_id"].unique())
    timestamps = sorted(comparison["timestamp"].unique())
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, ts in enumerate(timestamps):
            subset = comparison[comparison["timestamp"] == ts]
            pivot = (
                subset.pivot(index="station_id", columns="source", values="count")
                .reindex(unique_stations)
                .fillna(0)
            )
            pivot = pivot.reindex(columns=["GBFS", "MAPS"], fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 5))
            width = 0.35
            x = range(len(unique_stations))
            gbfs = pivot["GBFS"].to_numpy()
            maps = pivot["MAPS"].to_numpy()
            ax.bar([i - width / 2 for i in x], gbfs, width=width, label="GBFS")
            ax.bar([i + width / 2 for i in x], maps, width=width, label="MAPS")
            ax.set_xticks(list(x))
            labels = [station_labels.get(sid, sid) for sid in unique_stations]
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_ylabel("Bike count")
            ts_display = ts
            if hasattr(ts_display, "tzinfo") and ts_display.tzinfo is None:
                ts_display = ts_display.tz_localize("UTC")
            if hasattr(ts_display, "tz_convert"):
                ts_display = ts_display.tz_convert("UTC")
            ax.set_title(ts_display.strftime("%Y-%m-%d %H:%M:%S %Z"))
            ax.legend()
            fig.tight_layout()
            frame_path = Path(tmpdir) / f"frame_{idx:04d}.png"
            fig.savefig(frame_path, dpi=150)
            plt.close(fig)
            frames.append(iio.imread(frame_path))
    iio.imwrite(output_gif, frames, duration=0.5, loop=0)


if __name__ == "__main__":
    app()
