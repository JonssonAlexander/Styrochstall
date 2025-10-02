"""Utility to collect GBFS and maps API snapshots for Nextbike/Styr & Ställ."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from styrstell import config as cfg
from styrstell.loaders import GBFSLoader, NextbikeMapsLoader

app = typer.Typer(add_completion=False)


@app.command()
def run(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Optional calibration config to reuse endpoints and paths.",
    ),
    iterations: int = typer.Option(1, help="Number of snapshots to fetch."),
    interval_seconds: float = typer.Option(
        300.0, help="Delay between snapshots in seconds (ignored if iterations=1)."
    ),

) -> None:
    """Fetch timestamped GBFS and maps API snapshots for aligned snapshots."""

    calibration_cfg = cfg.load_calibration_config(config_path)
    gbfs_loader = GBFSLoader(calibration_cfg.data, calibration_cfg.gbfs)
    maps_loader = NextbikeMapsLoader(calibration_cfg.data, calibration_cfg.maps)
    typer.echo(
        "Collecting data from Nextbike GBFS feeds: "
        f"{', '.join(calibration_cfg.gbfs.feeds.values())}"
    )
    typer.echo(f"Collecting data from Nextbike maps API: {calibration_cfg.maps.endpoint}")
    for i in range(iterations):
        timestamp = datetime.now(timezone.utc).replace(microsecond=0)
        gbfs_meta = gbfs_loader.fetch_snapshot(timestamp)
        maps_meta = maps_loader.fetch_snapshot(timestamp)
        typer.echo(f"[{i + 1}/{iterations}] GBFS snapshot stored in {gbfs_meta.directory}")
        typer.echo(f"[{i + 1}/{iterations}] MAPS snapshot stored in {maps_meta.directory}")
        if i < iterations - 1:
            time.sleep(max(interval_seconds, 0.0))


if __name__ == "__main__":
    app()
