"""Analyse observed OD matrices for absorbing states and strongly connected components."""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import typer

from styrstell.utils import ensure_directory

app = typer.Typer(add_completion=False)


@app.command()
def run(
    tidy_od_path: Path = typer.Option(
        Path("data/models/observed_od_tidy.parquet"),
        "--tidy-od",
        help="Parquet file containing tidy OD probabilities (slot_start, origin, destination, probability).",
    ),
    absorbing_threshold: float = typer.Option(
        0.9,
        "--absorbing-threshold",
        help="Minimum self-loop probability to consider a station absorbing.",
    ),
    edge_threshold: float = typer.Option(
        0.05,
        "--edge-threshold",
        help="Fraction of destinations per origin to retain when searching for strongly connected components.",
    ),
    absorbing_output: Optional[Path] = typer.Option(
        Path("data/processed/absorbing_states.parquet"),
        "--absorbing-output",
        help="Where to store the detected absorbing states (set to None to skip writing).",
    ),
    cycles_output: Optional[Path] = typer.Option(
        Path("data/processed/strongly_connected_components.parquet"),
        "--cycles-output",
        help="Where to store the strongly connected components (set to None to skip writing).",
    ),
    dependency_output: Optional[Path] = typer.Option(
        Path("data/processed/station_dependencies.parquet"),
        "--dependency-output",
        help="Where to store origin/destination dependency shares (set to None to skip writing).",
    ),
    flow_summary_output: Optional[Path] = typer.Option(
        Path("data/processed/station_flow_summary.parquet"),
        "--flow-summary-output",
        help="Where to store per-station inflow/outflow balance (set to None to skip writing).",
    ),
) -> None:
    """Compute absorbing states and strongly connected components from an OD probability table."""

    if not tidy_od_path.exists():
        raise FileNotFoundError(f"Tidy OD file not found: {tidy_od_path}")

    od = pd.read_parquet(tidy_od_path)
    required_cols = {"slot_start", "origin", "destination", "probability"}
    if not required_cols.issubset(od.columns):
        missing = required_cols - set(od.columns)
        raise KeyError(f"Tidy OD table is missing required columns: {missing}")

    typer.echo(f"Loaded {len(od)} OD transitions from {tidy_od_path}")

    absorbing_df = _detect_absorbing_states(od, absorbing_threshold)
    if absorbing_output is not None:
        ensure_directory(absorbing_output.parent)
        absorbing_df.to_parquet(absorbing_output, index=False)
        typer.echo(f"Absorbing-state summary written to {absorbing_output}")
    else:
        typer.echo(absorbing_df)

    components_df = _find_strong_components(od, edge_threshold)
    if cycles_output is not None:
        ensure_directory(cycles_output.parent)
        components_df.to_parquet(cycles_output, index=False)
        typer.echo(f"Strongly connected components written to {cycles_output}")
    else:
        typer.echo(components_df)

    dependencies_df = _compute_station_dependencies(od, components_df)
    if dependency_output is not None:
        ensure_directory(dependency_output.parent)
        dependencies_df.to_parquet(dependency_output, index=False)
        typer.echo(f"Station dependency summary written to {dependency_output}")
    else:
        typer.echo(dependencies_df)

    flow_summary_df = _summarize_station_flows(dependencies_df)
    if flow_summary_output is not None:
        ensure_directory(flow_summary_output.parent)
        flow_summary_df.to_parquet(flow_summary_output, index=False)
        typer.echo(f"Station flow balance written to {flow_summary_output}")
    else:
        typer.echo(flow_summary_df)


def _detect_absorbing_states(od: pd.DataFrame, threshold: float) -> pd.DataFrame:
    grouped = od.groupby(["origin", "destination"])  # aggregate duplicates if any
    aggregated = grouped["probability"].mean().reset_index()
    self_loops = aggregated[aggregated["origin"] == aggregated["destination"]].copy()
    absorbing = self_loops[self_loops["probability"] >= threshold].copy()
    absorbing = absorbing.rename(columns={"origin": "station_id", "probability": "self_probability"})

    # compute outgoing mass to verify stickiness
    outgoing = aggregated.groupby("origin")["probability"].sum().reset_index().rename(
        columns={"origin": "station_id", "probability": "outgoing_probability"}
    )
    absorbing = absorbing.merge(outgoing, on="station_id", how="left")
    absorbing["residual_probability"] = 1 - absorbing["self_probability"]
    absorbing = absorbing.sort_values("self_probability", ascending=False).reset_index(drop=True)
    return absorbing


def _find_strong_components(od: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if fraction <= 0:
        raise ValueError("edge_threshold must be a positive fraction of destinations to keep")

    aggregated = (
        od.groupby(["origin", "destination"]).agg(
            probability=("probability", "mean"),
            flow=("flow", "sum"),
        )
    ).reset_index()

    kept: List[pd.DataFrame] = []
    for _, group in aggregated.groupby("origin", sort=False):
        sorted_group = group.sort_values("probability", ascending=False)
        if fraction >= 1:
            keep = sorted_group
        else:
            n_dest = len(sorted_group)
            top_k = max(1, math.ceil(n_dest * fraction))
            keep = sorted_group.head(top_k)
        kept.append(keep)

    if kept:
        filtered = pd.concat(kept, ignore_index=True)
    else:
        filtered = aggregated.iloc[0:0].copy()

    origin_nodes = {str(val) for val in filtered["origin"]}
    destination_nodes = {str(val) for val in filtered["destination"]}
    nodes = sorted(origin_nodes.union(destination_nodes))
    adjacency: Dict[str, List[str]] = defaultdict(list)
    for _, row in filtered.iterrows():
        adjacency[str(row["origin"])].append(str(row["destination"]))

    components = _tarjan_scc(nodes, adjacency)
    data = [
        {
            "component_id": idx,
            "size": len(component),
            "stations": component,
        }
        for idx, component in enumerate(components)
        if len(component) > 1
    ]
    df = pd.DataFrame(data)
    return df.sort_values("size", ascending=False).reset_index(drop=True)


def _compute_station_dependencies(od: pd.DataFrame, components_df: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        od.groupby(["origin", "destination"]).agg(
            probability=("probability", "mean"),
            flow=("flow", "sum"),
        )
    ).reset_index()

    destination_inflow = (
        aggregated.groupby("destination")["probability"].sum().rename("destination_inflow")
    )
    origin_outflow = (
        aggregated.groupby("origin")["probability"].sum().rename("origin_outflow")
    )
    destination_trip_total = (
        aggregated.groupby("destination")["flow"].sum().rename("destination_trip_total")
    )
    origin_trip_total = (
        aggregated.groupby("origin")["flow"].sum().rename("origin_trip_total")
    )

    dependencies = aggregated.merge(destination_inflow, on="destination", how="left")
    dependencies = dependencies.merge(origin_outflow, on="origin", how="left")
    dependencies = dependencies.merge(destination_trip_total, on="destination", how="left")
    dependencies = dependencies.merge(origin_trip_total, on="origin", how="left")

    dependencies["destination_share"] = (
        dependencies["probability"] / dependencies["destination_inflow"].replace(0, pd.NA)
    ).fillna(0.0)
    dependencies["origin_share"] = (
        dependencies["probability"] / dependencies["origin_outflow"].replace(0, pd.NA)
    ).fillna(0.0)
    dependencies["destination_trip_share"] = (
        dependencies["flow"] / dependencies["destination_trip_total"].replace(0, pd.NA)
    ).fillna(0.0)
    dependencies["origin_trip_share"] = (
        dependencies["flow"] / dependencies["origin_trip_total"].replace(0, pd.NA)
    ).fillna(0.0)

    component_map: Dict[str, int] = {}
    if components_df is not None and not components_df.empty:
        for _, row in components_df.iterrows():
            component_id = int(row["component_id"])
            for station in row["stations"]:
                component_map[str(station)] = component_id

    dependencies["origin"] = dependencies["origin"].astype(str)
    dependencies["destination"] = dependencies["destination"].astype(str)
    dependencies["origin_component_id"] = (
        dependencies["origin"].map(component_map).astype("Int64")
    )
    dependencies["destination_component_id"] = (
        dependencies["destination"].map(component_map).astype("Int64")
    )
    dependencies["same_component"] = (
        dependencies["origin_component_id"].notna()
        & (dependencies["origin_component_id"] == dependencies["destination_component_id"])
    )

    return dependencies.sort_values(
        ["same_component", "destination_share"], ascending=[False, False]
    ).reset_index(drop=True)


def _summarize_station_flows(dependencies: pd.DataFrame) -> pd.DataFrame:
    if dependencies.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "incoming_probability",
                "outgoing_probability",
                "net_out_minus_in",
                "component_id",
                "component_inflow_probability",
                "component_inflow_share",
                "component_outflow_probability",
                "component_outflow_share",
                "primary_origin",
                "primary_origin_share",
            ]
        )

    incoming_prob = dependencies.groupby("destination")["probability"].sum()
    outgoing_prob = dependencies.groupby("origin")["probability"].sum()
    incoming_trips = dependencies.groupby("destination")["flow"].sum()
    outgoing_trips = dependencies.groupby("origin")["flow"].sum()

    same_component_mask = dependencies["same_component"]
    inbound_same = (
        dependencies.loc[same_component_mask].groupby("destination")["probability"].sum()
    )
    outbound_same = (
        dependencies.loc[same_component_mask].groupby("origin")["probability"].sum()
    )
    inbound_same_trips = (
        dependencies.loc[same_component_mask].groupby("destination")["flow"].sum()
    )
    outbound_same_trips = (
        dependencies.loc[same_component_mask].groupby("origin")["flow"].sum()
    )

    top_inflow = (
        dependencies.sort_values("destination_share", ascending=False)
        .drop_duplicates("destination")
        .set_index("destination")
    )

    stations = sorted(set(dependencies["origin"]).union(dependencies["destination"]))
    summary = pd.DataFrame({"station_id": stations})

    summary["incoming_probability"] = (
        summary["station_id"].map(incoming_prob).fillna(0.0)
    )
    summary["outgoing_probability"] = (
        summary["station_id"].map(outgoing_prob).fillna(0.0)
    )
    summary["incoming_trips"] = summary["station_id"].map(incoming_trips).fillna(0.0)
    summary["outgoing_trips"] = summary["station_id"].map(outgoing_trips).fillna(0.0)
    summary["net_trip_flow"] = summary["outgoing_trips"] - summary["incoming_trips"]
    summary["net_out_minus_in"] = (
        summary["outgoing_probability"] - summary["incoming_probability"]
    )

    summary["component_inflow_probability"] = (
        summary["station_id"].map(inbound_same).fillna(0.0)
    )
    summary["component_outflow_probability"] = (
        summary["station_id"].map(outbound_same).fillna(0.0)
    )
    summary["component_inflow_trips"] = (
        summary["station_id"].map(inbound_same_trips).fillna(0.0)
    )
    summary["component_outflow_trips"] = (
        summary["station_id"].map(outbound_same_trips).fillna(0.0)
    )

    incoming_array = summary["incoming_probability"].to_numpy(dtype=float)
    incoming_component = summary["component_inflow_probability"].to_numpy(dtype=float)
    outflow_array = summary["outgoing_probability"].to_numpy(dtype=float)
    outflow_component = summary["component_outflow_probability"].to_numpy(dtype=float)

    summary["component_inflow_share"] = np.divide(
        incoming_component,
        incoming_array,
        out=np.zeros_like(incoming_component, dtype=float),
        where=incoming_array > 0,
    )
    summary["component_outflow_share"] = np.divide(
        outflow_component,
        outflow_array,
        out=np.zeros_like(outflow_component, dtype=float),
        where=outflow_array > 0,
    )

    trip_in_array = summary["incoming_trips"].to_numpy(dtype=float)
    trip_in_component = summary["component_inflow_trips"].to_numpy(dtype=float)
    trip_out_array = summary["outgoing_trips"].to_numpy(dtype=float)
    trip_out_component = summary["component_outflow_trips"].to_numpy(dtype=float)

    summary["component_inflow_trip_share"] = np.divide(
        trip_in_component,
        trip_in_array,
        out=np.zeros_like(trip_in_component, dtype=float),
        where=trip_in_array > 0,
    )
    summary["component_outflow_trip_share"] = np.divide(
        trip_out_component,
        trip_out_array,
        out=np.zeros_like(trip_out_component, dtype=float),
        where=trip_out_array > 0,
    )

    if not top_inflow.empty:
        primary_origin_series = top_inflow["origin"]
        primary_share_series = top_inflow["destination_share"]
    else:
        primary_origin_series = pd.Series(dtype="object")
        primary_share_series = pd.Series(dtype="float64")

    summary["primary_origin"] = summary["station_id"].map(primary_origin_series)
    summary["primary_origin_share"] = summary["station_id"].map(primary_share_series)

    component_map: Dict[str, int] = {}
    for _, row in dependencies.iterrows():
        origin_component = row.get("origin_component_id")
        if not pd.isna(origin_component):
            component_map[str(row["origin"])] = int(origin_component)
        dest_component = row.get("destination_component_id")
        if not pd.isna(dest_component):
            component_map[str(row["destination"])] = int(dest_component)

    summary["component_id"] = (
        summary["station_id"].map(component_map).astype("Int64")
    )

    return summary.sort_values("station_id").reset_index(drop=True)


def _tarjan_scc(nodes: Iterable[str], adjacency: Dict[str, List[str]]) -> List[List[str]]:
    index = 0
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: Set[str] = set()
    components: List[List[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in adjacency.get(node, []):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] == indices[node]:
            component: List[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == node:
                    break
            components.append(component)

    for node in nodes:
        if node not in indices:
            strongconnect(node)

    return components


if __name__ == "__main__":
    app()
