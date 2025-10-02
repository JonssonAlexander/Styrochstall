"""Quick data-quality and summary analysis for calibrated artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from styrstell import config as cfg
from styrstell.utils import ensure_directory, read_parquet

app = typer.Typer(add_completion=False)


@app.command()
def run(
    config_path: Optional[Path] = typer.Option(None, "--config", help="Calibration config to reuse paths."),
    output_path: Optional[Path] = typer.Option(
        Path("reports/analysis_summary.md"),
        "--output",
        help="Where to store the markdown summary (omit to print only).",
    ),
    max_rows: int = typer.Option(5, help="Number of head rows to capture per dataset."),
) -> None:
    """Summarise processed panels, flows, demand λ, travel-time bins, and OD matrices."""

    calib_cfg = cfg.load_calibration_config(config_path)
    data_paths = calib_cfg.data

    records: List[str] = []
    records.append("# Data Quality Summary\n")

    station_info_path = data_paths.processed / "station_information.parquet"
    panel_path = data_paths.processed / "station_panel.parquet"
    flows_path = data_paths.processed / "station_flows.parquet"
    lambda_path = data_paths.models / "lambda.parquet"
    travel_path = data_paths.models / "travel_time.parquet"
    od_path = data_paths.models / "od_matrix.parquet"

    _summarise_station_info(station_info_path, records, max_rows)
    _summarise_panel(panel_path, records)
    _summarise_flows(flows_path, records)
    _summarise_lambda(lambda_path, records)
    _summarise_travel_time(travel_path, records)
    _summarise_od(od_path, records)

    report = "\n".join(records)
    typer.echo(report)
    if output_path is not None:
        ensure_directory(output_path.parent)
        output_path.write_text(report, encoding="utf-8")
        typer.echo(f"Summary written to {output_path}")


def _summarise_station_info(path: Path, records: List[str], max_rows: int) -> None:
    records.append("## Station Information\n")
    if not path.exists():
        records.append(f"Missing: {path}\n")
        return
    df = read_parquet(path)
    records.append(f"Stations: {len(df)}\n")
    if "capacity" in df.columns:
        stats = df["capacity"].describe()[["min", "max", "mean", "50%"]]
        records.append("Capacity stats (min/median/mean/max): "
                       f"{stats['min']:.0f} / {stats['50%']:.0f} / {stats['mean']:.1f} / {stats['max']:.0f}\n")
    records.append(_render_head(df, max_rows))


def _summarise_panel(path: Path, records: List[str]) -> None:
    records.append("## Station Panel\n")
    if not path.exists():
        records.append(f"Missing: {path}\n")
        return
    panel = read_parquet(path)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    span = panel["timestamp"].agg(["min", "max"])
    records.append(
        f"Rows: {len(panel)} | Stations: {panel['station_id'].nunique()} | "
        f"Window: {span['min']} → {span['max']}\n"
    )
    availability_cols = [c for c in ["num_bikes_available", "num_docks_available"] if c in panel.columns]
    if availability_cols:
        stats = panel[availability_cols].describe().loc[["min", "max", "mean"]]
        records.append(stats.to_markdown())
        records.append("\n")


def _summarise_flows(path: Path, records: List[str]) -> None:
    records.append("## Station Flows\n")
    if not path.exists():
        records.append(f"Missing: {path}\n")
        return
    flows = read_parquet(path)
    flows["timestamp"] = pd.to_datetime(flows["timestamp"], utc=True)
    negatives = (flows[["departures_raw", "arrivals_raw"]] < 0).any(axis=1).sum()
    records.append(
        f"Rows: {len(flows)} | Negative counts: {negatives} | Rebalancing flags: {flows['is_rebalancing'].sum()}\n"
    )
    if {"departures", "arrivals"}.issubset(flows.columns):
        stats = flows[["departures", "arrivals"]].describe().loc[["mean", "max"]]
        records.append(stats.to_markdown())
        records.append("\n")


def _summarise_lambda(path: Path, records: List[str]) -> None:
    records.append("## Demand Intensity (λ)\n")
    if not path.exists():
        records.append(f"Missing: {path}\n")
        return
    df = read_parquet(path)
    summary = df[["lambda_departures", "lambda_arrivals"]].describe().loc[["mean", "max"]]
    records.append(summary.to_markdown())
    records.append("\n")
    top = (
        df.groupby("station_id")["lambda_departures"].max().sort_values(ascending=False).head(10)
    )
    records.append("Top departure intensities:\n")
    records.append(top.to_string())
    records.append("\n")


def _summarise_travel_time(path: Path, records: List[str]) -> None:
    records.append("## Travel-Time Distribution\n")
    if not path.exists():
        records.append(f"Missing: {path}\n")
        return
    df = read_parquet(path)
    coverage = df.groupby("time_bin")["probability"].sum()
    bad_bins = coverage[(coverage - 1).abs() > 0.05]
    records.append(
        f"Time bins: {coverage.size} | bins off by >5%: {bad_bins.size}\n"
    )
    records.append("First bins:\n")
    records.append(df.head(10).to_markdown(index=False))
    records.append("\n")


def _summarise_od(path: Path, records: List[str]) -> None:
    records.append("## OD Matrices\n")
    if not path.exists():
        records.append(f"Missing: {path}\n")
        return
    df = read_parquet(path)
    coverage = df.groupby(["slot_start", "origin"])["probability"].sum()
    off = coverage[(coverage - 1).abs() > 0.05]
    records.append(
        f"Slices: {coverage.size} | off-by->5%: {off.size}\n"
    )
    if not off.empty:
        records.append("Examples of mismatched probability sums:\n")
        records.append(off.head(5).to_string())
        records.append("\n")


def _render_head(df: pd.DataFrame, max_rows: int) -> str:
    head = df.head(max_rows)
    return head.to_markdown(index=False) + "\n"


if __name__ == "__main__":
    app()
