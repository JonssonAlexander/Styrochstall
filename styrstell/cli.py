"""Command-line interface for calibration and simulation workflows."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from styrstell import config as cfg
from styrstell.calibration import (
    calibrate_od_matrix,
    estimate_demand_intensity,
    estimate_od_matrix,
    estimate_travel_time_distribution,
)
from styrstell.features import build_station_panel, infer_station_flows
from styrstell.loaders import GBFSLoader
from styrstell.metrics import compute_policy_kpis
from styrstell.reporting import plot_policy_comparison
from styrstell.simulation.environment import SimulationEnvironment
from styrstell.simulation.policies import RentalPolicy, select_policy_parameter
from styrstell.simulation.processes import run_simulation as run_simulation_process
from styrstell.utils import ensure_directory, read_parquet, write_parquet
from styrstell.trips import (
    TripInferenceConfig,
    build_bike_timelines,
    build_edge_list,
    build_od_matrix,
    build_travel_time_distribution,
    build_maps_lambda,
    export_edge_list_for_visualization,
    infer_trips_from_timelines,
    iter_free_bike_snapshots,
    iter_maps_bike_snapshots,
)

app = typer.Typer(add_completion=False)


@app.command()
def calibrate(
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to calibration config JSON/YAML."),
    fetch: bool = typer.Option(True, "--fetch/--no-fetch", help="Fetch a fresh GBFS snapshot before calibrating."),
) -> None:
    """Fetch GBFS snapshots, derive flows, and calibrate demand/OD/travel models."""

    calibration_config = cfg.load_calibration_config(config_path)
    loader = GBFSLoader(calibration_config.data, calibration_config.gbfs)
    snapshots = list(loader.list_snapshots())
    if fetch or not snapshots:
        typer.echo("Fetching GBFS snapshot...")
        loader.fetch_snapshot()
        snapshots = list(loader.list_snapshots())
    typer.echo(f"Using {len(snapshots)} snapshots for calibration.")

    snapshot_with_info = next((s for s in reversed(snapshots) if "station_information" in s.files), None)
    if snapshot_with_info is None:
        raise FileNotFoundError("No station_information files found in available snapshots.")
    station_info = loader.load_station_information(snapshot_with_info)
    ensure_directory(calibration_config.data.processed)
    station_info_path = calibration_config.data.processed / "station_information.parquet"
    write_parquet(station_info, station_info_path)

    status_snapshots = [s for s in snapshots if "station_status" in s.files]
    if not status_snapshots:
        raise FileNotFoundError("No station_status files found in available snapshots.")
    skipped = len(snapshots) - len(status_snapshots)
    if skipped:
        typer.echo(f"Skipping {skipped} snapshots without station_status files.")
    status_frames = [loader.load_station_status(snapshot) for snapshot in status_snapshots]
    panel = build_station_panel(status_frames, calibration_config.features)
    panel_path = calibration_config.data.processed / "station_panel.parquet"
    write_parquet(panel.reset_index(), panel_path)

    flows = infer_station_flows(panel, calibration_config.features)
    flows_path = calibration_config.data.processed / "station_flows.parquet"
    write_parquet(flows.reset_index(), flows_path)

    typer.echo("Estimating demand intensities...")
    demand = estimate_demand_intensity(flows, calibration_config.demand)
    demand_path = calibration_config.data.models / "lambda.parquet"
    write_parquet(demand.reset_index(), demand_path)

    trips = _load_trip_samples(calibration_config)
    typer.echo("Estimating travel-time distributions...")
    travel_time = estimate_travel_time_distribution(trips, calibration_config.travel_time)
    travel_path = calibration_config.data.models / "travel_time.parquet"
    write_parquet(travel_time, travel_path)

    typer.echo("Calibrating time-varying OD matrices...")
    od = calibrate_od_matrix(demand, station_info, calibration_config.od)
    od_path = calibration_config.data.models / "od_matrix.parquet"
    write_parquet(od, od_path)

    typer.echo("Estimating aggregate OD probability matrix...")
    delta_df = (
        flows.reset_index()
        .merge(
            station_info[["station_id", "lat", "lon"]],
            on="station_id",
            how="left",
        )
        .dropna(subset=["lat", "lon"])
    )
    try:
        od_prob_result = estimate_od_matrix(
            delta_df[["timestamp", "station_id", "lat", "lon", "delta_bikes"]],
            beta=calibration_config.od.gravity_beta,
            window=f"{calibration_config.od.time_slot_minutes}min",
            tol=calibration_config.od.tolerance,
            max_iterations=calibration_config.od.max_iterations,
        )
        prob_path = calibration_config.data.models / "od_probability_matrix.parquet"
        flow_path = calibration_config.data.models / "od_flow_matrix.parquet"
        write_parquet(od_prob_result.probabilities, prob_path)
        write_parquet(od_prob_result.flows, flow_path)
    except ValueError as err:
        typer.echo(f"Skipping aggregate OD estimation: {err}")

    typer.echo("Calibration artifacts written to data/models/ and data/processed/.")


@app.command()
def simulate(
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to simulation config JSON/YAML."),
    output_dir: Path = typer.Option(Path("data/cache"), "--output-dir", help="Directory for KPI outputs."),
) -> None:
    """Run policy simulations and emit KPI tables and diagnostic plots."""

    sim_config = cfg.load_simulation_config(config_path)
    ensure_directory(output_dir)

    panel = _load_or_error(sim_config.features_path or sim_config.data.processed / "station_panel.parquet")
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel = panel.set_index(["timestamp", "station_id"])
    flows = _load_or_error(sim_config.data.processed / "station_flows.parquet")
    flows["timestamp"] = pd.to_datetime(flows["timestamp"], utc=True)
    flows = flows.set_index(["timestamp", "station_id"])
    demand = _load_or_error(sim_config.demand_model_path or sim_config.data.models / "lambda.parquet")
    demand["timestamp"] = pd.to_datetime(demand["timestamp"], utc=True)
    demand = demand.set_index(["timestamp", "station_id"])
    travel_time = _load_or_error(sim_config.travel_time_model_path or sim_config.data.models / "travel_time.parquet")
    od_matrix = _load_or_error(sim_config.od_model_path or sim_config.data.models / "od_matrix.parquet")
    stations = _load_or_error(sim_config.data.processed / "station_information.parquet")

    kpi_frames = []
    for policy_cfg in sim_config.policies:
        policy = select_policy_parameter(policy_cfg)
        typer.echo(f"Running simulation for policy: {policy.name}")
        sim_env = SimulationEnvironment(
            panel=panel,
            demand=demand,
            travel_time=travel_time,
            od_matrix=od_matrix,
            stations=stations,
            sim_config=sim_config,
            policy=policy,
        )
        metrics = run_simulation_process(sim_env)
        kpis = compute_policy_kpis(metrics, policy, sim_config)
        kpi_frames.append(kpis)
    kpi_table = pd.concat(kpi_frames, ignore_index=True)
    kpi_path = output_dir / "policy_kpis.parquet"
    write_parquet(kpi_table, kpi_path)
    typer.echo(f"KPI table saved to {kpi_path}")

    fig = plot_policy_comparison(kpi_table)
    fig_path = output_dir / "policy_comparison.png"
    fig.savefig(fig_path, dpi=150)
    typer.echo(f"Comparison plot saved to {fig_path}")


@app.command("infer-trips")
def infer_trips(
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to calibration config JSON/YAML."),
    min_duration_seconds: float = typer.Option(60.0, help="Minimum trip duration to accept."),
    max_duration_seconds: float = typer.Option(3 * 3600.0, help="Maximum trip duration to accept."),
    min_distance_km: float = typer.Option(0.05, help="Minimum travel distance to accept."),
    max_speed_kmph: float = typer.Option(35.0, help="Maximum permissible speed (km/h)."),
    ignore_nighttime: bool = typer.Option(True, "--ignore-nighttime/--allow-nighttime", help="Filter out long nighttime moves."),
    lambda_interval_minutes: int = typer.Option(60, "--lambda-interval-minutes", help="Interval (minutes) used when estimating MAPS demand."),
) -> None:
    """Infer observed trips from free bike status snapshots."""

    calibration_config = cfg.load_calibration_config(config_path)
    inference_cfg = TripInferenceConfig(
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
        min_distance_km=min_distance_km,
        max_speed_kmph=max_speed_kmph,
        ignore_nighttime=ignore_nighttime,
    )
    typer.echo("Loading free bike status snapshots...")
    snapshots = list(iter_free_bike_snapshots(calibration_config.data))
    timelines = build_bike_timelines(snapshots)
    typer.echo(f"Timeline observations: {len(timelines)}")
    trips = infer_trips_from_timelines(timelines, inference_cfg)
    typer.echo(f"Inferred trips: {len(trips)}")
    trips_path = calibration_config.data.processed / "trips.parquet"
    write_parquet(trips, trips_path)
    typer.echo(f"Trips saved to {trips_path}")


@app.command("infer-maps-trips")
def infer_maps_trips(
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to calibration config JSON/YAML."),
    min_duration_seconds: float = typer.Option(60.0, help="Minimum trip duration to accept."),
    max_duration_seconds: float = typer.Option(3 * 3600.0, help="Maximum trip duration to accept."),
    min_distance_km: float = typer.Option(0.05, help="Minimum travel distance to accept."),
    max_speed_kmph: float = typer.Option(35.0, help="Maximum permissible speed (km/h)."),
    ignore_nighttime: bool = typer.Option(True, "--ignore-nighttime/--allow-nighttime", help="Filter out long nighttime moves."),
) -> None:
    """Infer observed trips from Nextbike maps API snapshots."""

    calibration_config = cfg.load_calibration_config(config_path)
    inference_cfg = TripInferenceConfig(
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
        min_distance_km=min_distance_km,
        max_speed_kmph=max_speed_kmph,
        ignore_nighttime=ignore_nighttime,
    )
    typer.echo("Loading maps API snapshots...")
    snapshots = list(iter_maps_bike_snapshots(calibration_config.data))
    typer.echo(f"Snapshot count: {len(snapshots)}")
    timelines = build_bike_timelines(snapshots)
    typer.echo(f"Timeline observations: {len(timelines)}")
    trips = infer_trips_from_timelines(timelines, inference_cfg)
    typer.echo(f"Inferred trips: {len(trips)}")
    trips_path = calibration_config.data.processed / "maps_trips.parquet"
    write_parquet(trips, trips_path)
    typer.echo(f"Maps trips saved to {trips_path}")

    typer.echo("Estimating MAPS-based demand (lambda)...")
    maps_lambda = build_maps_lambda(trips, interval_minutes=lambda_interval_minutes)
    lambda_path = calibration_config.data.models / "maps_lambda.parquet"
    write_parquet(maps_lambda, lambda_path)
    typer.echo(f"MAPS lambda saved to {lambda_path}")


@app.command("observed-od")
def observed_od(
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to calibration config JSON/YAML."),
    trips_path: Optional[Path] = typer.Option(None, "--trips", help="Override path to trips.parquet."),
) -> None:
    """Build observed OD aggregates from inferred trips."""

    calibration_config = cfg.load_calibration_config(config_path)
    trips_file = trips_path or calibration_config.data.processed / "trips.parquet"
    if not trips_file.exists():
        raise FileNotFoundError(f"Trips file not found: {trips_file}")
    trips = read_parquet(trips_file)
    if trips.empty:
        typer.echo("Trips dataframe is empty; nothing to aggregate.")
        return
    typer.echo(f"Aggregating {len(trips)} trips...")
    edge_list = build_edge_list(trips)
    od_matrix = build_od_matrix(trips)
    probability_matrix = od_matrix.div(od_matrix.sum(axis=1).replace(0, 1), axis=0)
    travel_time_dist = build_travel_time_distribution(trips)
    edge_viz = export_edge_list_for_visualization(edge_list)

    trips = trips.copy()
    trips["estimated_departure_time"] = pd.to_datetime(trips["estimated_departure_time"], utc=True, errors="coerce")
    trips = trips.dropna(subset=["estimated_departure_time"])
    trips["slot_start"] = trips["estimated_departure_time"].dt.floor("1h")
    tidy_od = (
        trips.groupby(["slot_start", "origin_station_id", "destination_station_id"]).size().reset_index(name="flow")
    )
    if not tidy_od.empty:
        origin_totals = tidy_od.groupby(["slot_start", "origin_station_id"]) ["flow"].transform("sum")
        tidy_od["probability"] = tidy_od["flow"] / origin_totals.replace(0, np.nan)
        tidy_od["probability"] = tidy_od["probability"].fillna(0.0)
    else:
        tidy_od["probability"] = []
    tidy_od = tidy_od.rename(
        columns={
            "origin_station_id": "origin",
            "destination_station_id": "destination",
        }
    )
    tidy_od["slot_start"] = tidy_od["slot_start"].dt.tz_convert("UTC")
    tidy_od["slot_end"] = tidy_od["slot_start"] + pd.to_timedelta(1, unit="h")

    processed_root = calibration_config.data.processed
    models_root = calibration_config.data.models
    ensure_directory(processed_root)
    ensure_directory(models_root)

    edges_path = processed_root / "observed_edge_list.parquet"
    matrix_path = models_root / "observed_od_matrix.parquet"
    probability_path = models_root / "observed_od_probability.parquet"
    travel_path = models_root / "observed_travel_time.parquet"
    viz_path = models_root / "observed_edge_list_for_viz.parquet"
    tidy_od_path = models_root / "observed_od_tidy.parquet"

    write_parquet(edge_list, edges_path)
    write_parquet(od_matrix, matrix_path)
    write_parquet(probability_matrix, probability_path)
    write_parquet(travel_time_dist, travel_path)
    write_parquet(edge_viz, viz_path)
    write_parquet(tidy_od, tidy_od_path)

    typer.echo("Observed OD artifacts written:")
    typer.echo(f" - Edge list: {edges_path}")
    typer.echo(f" - Edge list (viz): {viz_path}")
    typer.echo(f" - OD matrix: {matrix_path}")
    typer.echo(f" - OD probability matrix: {probability_path}")
    typer.echo(f" - OD tidy table: {tidy_od_path}")
    typer.echo(f" - Travel time stats: {travel_path}")


def _load_trip_samples(config: cfg.CalibrationConfig) -> pd.DataFrame:
    trip_path = config.data.processed / "trip_samples.parquet"
    if trip_path.exists():
        return read_parquet(trip_path)
    typer.echo("No trip samples found; generating synthetic durations for bootstrapping.")
    synthetic = pd.DataFrame(
        {
            "duration_minutes": pd.Series([12, 18, 22, 28, 35, 45], dtype=float),
            "start_timestamp": pd.date_range("2024-01-01", periods=6, freq="1H", tz="UTC"),
        }
    )
    return synthetic


def _load_or_error(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return read_parquet(path)


def run_calibration() -> None:
    """Entry point for `styrstell-calibrate`."""

    typer.run(calibrate)


def run_simulation() -> None:
    """Entry point for `styrstell-simulate`."""

    typer.run(simulate)


def run_infer_trips() -> None:
    """Entry point for `styrstell-infer-trips`."""

    typer.run(infer_trips)


def run_infer_maps_trips() -> None:
    """Entry point for `styrstell-infer-maps-trips`."""

    typer.run(infer_maps_trips)


def run_observed_od() -> None:
    """Entry point for `styrstell-observed-od`."""

    typer.run(observed_od)


if __name__ == "__main__":
    app()
