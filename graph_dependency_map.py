"""Visualise component equilibria and inflow dependencies on a Folium map."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import folium
import pandas as pd
import typer
from branca.element import Element
from folium.features import RegularPolygonMarker

from graph_components_map import _generate_palette, _load_station_metadata
from styrstell.utils import ensure_directory

app = typer.Typer(add_completion=False)


@app.command()
def run(
    station_path: Path = typer.Option(
        Path("data/processed/station_information.parquet"),
        "--station-info",
        help="Parquet file with station metadata (station_id, name, lat, lon).",
    ),
    dependency_path: Path = typer.Option(
        Path("data/processed/station_dependencies.parquet"),
        "--dependency-data",
        help="Parquet file produced by graph_diagnostics with dependency shares.",
    ),
    flow_summary_path: Optional[Path] = typer.Option(
        Path("data/processed/station_flow_summary.parquet"),
        "--flow-summary",
        help="Optional parquet with per-station inflow/outflow summary (recomputed if missing).",
    ),
    share_threshold: float = typer.Option(
        0.2,
        "--share-threshold",
        help="Minimum share of a destination's inflow to draw within-component links.",
    ),
    cross_share_threshold: float = typer.Option(
        0.3,
        "--cross-share-threshold",
        help="Minimum share of a destination's inflow to draw cross-component dependency links.",
    ),
    focus_origin: Optional[str] = typer.Option(
        None,
        "--focus-origin",
        help="Optional station id or name to highlight its dependent destinations.",
    ),
    focus_share_threshold: float = typer.Option(
        0.3,
        "--focus-share-threshold",
        help="Minimum inflow share to show in the focus-origin layer.",
    ),
    output_path: Path = typer.Option(
        Path("reports/station_dependency_map.html"),
        "--output",
        help="Output HTML file for the dependency map.",
    ),
) -> None:
    """Render dependencies that sustain stations and component equilibria."""

    stations = _load_station_metadata(station_path)

    if not dependency_path.exists():
        raise FileNotFoundError(f"Dependency data not found: {dependency_path}")

    for label, value in {
        "share-threshold": share_threshold,
        "cross-share-threshold": cross_share_threshold,
        "focus-share-threshold": focus_share_threshold,
    }.items():
        if value <= 0:
            raise ValueError(f"{label} must be positive")
        if value > 1:
            raise ValueError(f"{label} must be ≤ 1")

    dependencies = pd.read_parquet(dependency_path)
    if dependencies.empty:
        raise ValueError("Dependency data is empty; run graph_diagnostics first.")

    dependencies = dependencies.copy()
    dependencies["origin"] = dependencies["origin"].astype(str)
    dependencies["destination"] = dependencies["destination"].astype(str)

    flow_summary: pd.DataFrame
    if flow_summary_path is not None and flow_summary_path.exists():
        flow_summary = pd.read_parquet(flow_summary_path)
    else:
        from graph_diagnostics import _summarize_station_flows  # lazy import to avoid circular CLI

        flow_summary = _summarize_station_flows(dependencies)

    if not flow_summary.empty:
        flow_summary = flow_summary.copy()
        flow_summary["station_id"] = flow_summary["station_id"].astype(str)
        flow_summary_lookup = flow_summary.set_index("station_id")
    else:
        flow_summary_lookup = pd.DataFrame()

    component_ids = _extract_component_ids(dependencies)
    component_colors = _build_component_palette(component_ids)
    component_station_map = _components_to_stations(dependencies)

    fmap = folium.Map(
        location=[stations["lat"].mean(), stations["lon"].mean()],
        zoom_start=12,
        tiles="cartodbpositron",
    )

    stations_layer = folium.FeatureGroup(name="Stations", show=True)
    for station_id, meta in stations.iterrows():
        tooltip = _build_station_tooltip(
            station_id,
            meta,
            stations,
            flow_summary_lookup,
        )
        comp_value = (
            flow_summary_lookup.at[station_id, "component_id"]
            if station_id in flow_summary_lookup.index
            else None
        )
        component_id = (
            int(comp_value) if comp_value is not None and pd.notna(comp_value) else None
        )

        color = component_colors.get(component_id, "#6c757d")
        folium.CircleMarker(
            location=(meta["lat"], meta["lon"]),
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=1,
            tooltip=tooltip,
        ).add_to(stations_layer)
    stations_layer.add_to(fmap)

    _add_component_layers(
        fmap,
        stations,
        dependencies,
        component_ids,
        component_colors,
        component_station_map,
        share_threshold,
    )

    _add_cross_component_layer(
        fmap,
        stations,
        dependencies,
        cross_share_threshold,
    )

    if focus_origin is not None:
        _add_focus_origin_layer(
            fmap,
            stations,
            dependencies,
            focus_origin,
            flow_summary_lookup,
            component_colors,
            focus_share_threshold,
        )

    _add_legend(
        fmap,
        share_threshold,
        cross_share_threshold,
        focus_share_threshold,
        focus_origin,
    )

    folium.LayerControl(collapsed=False).add_to(fmap)
    ensure_directory(output_path.parent)
    fmap.save(output_path)
    typer.echo(f"Dependency map saved to {output_path}")


def _extract_component_ids(dependencies: pd.DataFrame) -> list[int]:
    origin_components = dependencies.get("origin_component_id")
    destination_components = dependencies.get("destination_component_id")
    ids = set()
    if origin_components is not None:
        ids.update(c for c in origin_components.dropna().unique())
    if destination_components is not None:
        ids.update(c for c in destination_components.dropna().unique())
    return sorted(int(c) for c in ids)


def _build_component_palette(component_ids: list[int]) -> Dict[Optional[int], str]:
    if not component_ids:
        return {}
    colors = _generate_palette(len(component_ids))
    return dict(zip(component_ids, colors))


def _components_to_stations(dependencies: pd.DataFrame) -> Dict[int, List[str]]:
    lookup: Dict[int, List[str]] = {}
    for comp_col, station_col in (
        ("origin_component_id", "origin"),
        ("destination_component_id", "destination"),
    ):
        if comp_col not in dependencies.columns or station_col not in dependencies.columns:
            continue
        pairs = dependencies[[comp_col, station_col]].dropna(subset=[comp_col])
        for component_id, station_id in pairs.itertuples(index=False):
            cid = int(component_id)
            lookup.setdefault(cid, []).append(str(station_id))
    for cid, station_list in lookup.items():
        lookup[cid] = sorted(set(station_list))
    return lookup


def _mix_color(base: str, target: str, factor: float) -> str:
    factor = max(0.0, min(1.0, factor))
    base_rgb = tuple(int(base[i : i + 2], 16) for i in (1, 3, 5))
    target_rgb = tuple(int(target[i : i + 2], 16) for i in (1, 3, 5))
    blended = [
        int(round(base_rgb[idx] + (target_rgb[idx] - base_rgb[idx]) * factor))
        for idx in range(3)
    ]
    return "#" + "".join(f"{value:02x}" for value in blended)


def _cross_edge_color(share: float, threshold: float) -> str:
    ramp_start = "#fcae91"
    ramp_end = "#67000d"
    max_share = 0.75
    span = max(0.05, max_share - threshold)
    norm = 0.0 if share <= threshold else min((share - threshold) / span, 1.0)
    return _mix_color(ramp_start, ramp_end, norm)


def _build_station_tooltip(
    station_id: str,
    meta: pd.Series,
    stations: pd.DataFrame,
    flow_summary_lookup: pd.DataFrame,
) -> str:
    tooltip_lines = [f"<strong>{meta['station_name']}</strong>"]

    if station_id in flow_summary_lookup.index:
        summary = flow_summary_lookup.loc[station_id]
        incoming_val = summary.get("incoming_probability", pd.NA)
        outgoing_val = summary.get("outgoing_probability", pd.NA)
        component_id = summary.get("component_id")
        comp_in_val = summary.get("component_inflow_share", pd.NA)
        comp_out_val = summary.get("component_outflow_share", pd.NA)
        comp_in_trip_val = summary.get("component_inflow_trip_share", pd.NA)
        comp_out_trip_val = summary.get("component_outflow_trip_share", pd.NA)
        primary_origin = summary.get("primary_origin")
        primary_share_val = summary.get("primary_origin_share", pd.NA)
        incoming_trips_val = summary.get("incoming_trips", pd.NA)
        outgoing_trips_val = summary.get("outgoing_trips", pd.NA)

        incoming = float(incoming_val) if pd.notna(incoming_val) else float("nan")
        outgoing = float(outgoing_val) if pd.notna(outgoing_val) else float("nan")
        comp_in_share = (
            float(comp_in_val) if pd.notna(comp_in_val) else float("nan")
        )
        comp_out_share = (
            float(comp_out_val) if pd.notna(comp_out_val) else float("nan")
        )
        primary_share = (
            float(primary_share_val) if pd.notna(primary_share_val) else float("nan")
        )
        comp_in_trip_share = (
            float(comp_in_trip_val) if pd.notna(comp_in_trip_val) else float("nan")
        )
        comp_out_trip_share = (
            float(comp_out_trip_val) if pd.notna(comp_out_trip_val) else float("nan")
        )
        incoming_trips = (
            float(incoming_trips_val) if pd.notna(incoming_trips_val) else float("nan")
        )
        outgoing_trips = (
            float(outgoing_trips_val) if pd.notna(outgoing_trips_val) else float("nan")
        )

        tooltip_lines.append(
            f"Component: {component_id if pd.notna(component_id) else 'None'}"
        )
        tooltip_lines.append(f"Incoming P: {incoming:.3f}")
        tooltip_lines.append(f"Outgoing P: {outgoing:.3f}")
        tooltip_lines.append(f"Incoming trips: {incoming_trips:.0f}")
        tooltip_lines.append(f"Outgoing trips: {outgoing_trips:.0f}")
        tooltip_lines.append(f"Component inflow share: {comp_in_share:.2f}")
        tooltip_lines.append(f"Component outflow share: {comp_out_share:.2f}")
        tooltip_lines.append(
            f"Component inflow trip share: {comp_in_trip_share:.2f}"
        )
        tooltip_lines.append(
            f"Component outflow trip share: {comp_out_trip_share:.2f}"
        )
        if pd.notna(primary_origin):
            primary_origin_id = str(primary_origin)
            primary_name = (
                stations.loc[primary_origin_id, "station_name"]
                if primary_origin_id in stations.index
                else primary_origin_id
            )
            tooltip_lines.append(
                f"Primary inflow: {primary_name} ({primary_share:.2f})"
            )

    tooltip_lines.append("Values are probability mass fractions (0-1).")
    return "<br>".join(tooltip_lines)


def _add_component_layers(
    fmap: folium.Map,
    stations: pd.DataFrame,
    dependencies: pd.DataFrame,
    component_ids: list[int],
    component_colors: Dict[Optional[int], str],
    component_station_map: Dict[int, List[str]],
    share_threshold: float,
) -> None:
    for component_id in component_ids:
        layer = folium.FeatureGroup(
            name=f"Component {component_id} (≥{share_threshold:.0%} inflow share)",
            show=False,
        )

        component_edges = dependencies[
            (dependencies["origin_component_id"] == component_id)
            & (dependencies["destination_component_id"] == component_id)
            & (dependencies["destination_share"] >= share_threshold)
        ]

        color = component_colors.get(component_id, "#1f78b4")
        for _, dep in component_edges.iterrows():
            origin_id = dep["origin"]
            dest_id = dep["destination"]
            if origin_id not in stations.index or dest_id not in stations.index:
                continue

            origin_meta = stations.loc[origin_id]
            dest_meta = stations.loc[dest_id]
            share = float(dep["destination_share"])
            trip_share = float(dep.get("destination_trip_share", float("nan")))
            origin_trip_share = float(dep.get("origin_trip_share", float("nan")))
            trip_count = float(dep.get("flow", 0.0))
            tooltip = (
                f"{origin_meta['station_name']} → {dest_meta['station_name']}<br>"
                f"Trips: {trip_count:.0f}<br>"
                f"Dest inflow share (prob): {share:.2f}<br>"
                f"Dest inflow share (trips): {trip_share:.2f}<br>"
                f"Origin outflow share (prob): {dep['origin_share']:.2f}<br>"
                f"Origin outflow share (trips): {origin_trip_share:.2f}"
            )

            weight = 1.5 + min(share, 0.6) * 6
            folium.PolyLine(
                locations=[
                    (origin_meta["lat"], origin_meta["lon"]),
                    (dest_meta["lat"], dest_meta["lon"]),
                ],
                color=color,
                weight=weight,
                opacity=0.75,
                tooltip=tooltip,
            ).add_to(layer)

        station_ids = component_station_map.get(component_id, [])
        present_ids = [sid for sid in station_ids if sid in stations.index]
        if present_ids:
            coords = stations.loc[present_ids, ["lat", "lon"]]
            centroid_lat = coords["lat"].mean()
            centroid_lon = coords["lon"].mean()
            station_names = stations.loc[present_ids, "station_name"].tolist()
            names_html = "<br>".join(f"• {name}" for name in station_names)
            popup_html = (
                f"<strong>Component {component_id}</strong><br>{names_html}"
            )
            folium.Marker(
                location=(centroid_lat, centroid_lon),
                icon=folium.Icon(color="lightblue", icon="info-sign"),
                tooltip=f"Component {component_id} station list",
                popup=popup_html,
            ).add_to(layer)

        if len(component_edges) > 0 or present_ids:
            layer.add_to(fmap)


def _add_cross_component_layer(
    fmap: folium.Map,
    stations: pd.DataFrame,
    dependencies: pd.DataFrame,
    cross_share_threshold: float,
) -> None:
    cross_edges = dependencies[
        (~dependencies["same_component"])
        & (dependencies["destination_share"] >= cross_share_threshold)
    ]

    if cross_edges.empty:
        return

    layer = folium.FeatureGroup(
        name=f"Cross-component dependencies (≥{cross_share_threshold:.0%})",
        show=False,
    )
    for _, dep in cross_edges.iterrows():
        origin_id = dep["origin"]
        dest_id = dep["destination"]
        if origin_id not in stations.index or dest_id not in stations.index:
            continue

        origin_meta = stations.loc[origin_id]
        dest_meta = stations.loc[dest_id]
        share = float(dep["destination_share"])
        trip_share = float(dep.get("destination_trip_share", float("nan")))
        origin_trip_share = float(dep.get("origin_trip_share", float("nan")))
        trip_count = float(dep.get("flow", 0.0))
        tooltip = (
            f"{origin_meta['station_name']} → {dest_meta['station_name']}<br>"
            f"Trips: {trip_count:.0f}<br>"
            f"Dest inflow share (prob): {share:.2f}<br>"
            f"Dest inflow share (trips): {trip_share:.2f}<br>"
            f"Origin outflow share (prob): {dep['origin_share']:.2f}<br>"
            f"Origin outflow share (trips): {origin_trip_share:.2f}"
        )

        color = _cross_edge_color(share, cross_share_threshold)
        weight = 1.5 + min(share, 0.6) * 6
        folium.PolyLine(
            locations=[
                (origin_meta["lat"], origin_meta["lon"]),
                (dest_meta["lat"], dest_meta["lon"]),
            ],
            color=color,
            weight=weight,
            opacity=0.7,
            tooltip=tooltip,
        ).add_to(layer)

        angle = math.degrees(
            math.atan2(
                dest_meta["lat"] - origin_meta["lat"],
                dest_meta["lon"] - origin_meta["lon"],
            )
        )
        RegularPolygonMarker(
            location=(dest_meta["lat"], dest_meta["lon"]),
            number_of_sides=3,
            radius=6 + min(share, 0.5) * 10,
            rotation=angle + 90,
            color=color,
            fill=True,
            fill_color=color,
            opacity=0.8,
        ).add_to(layer)

    layer.add_to(fmap)


def _add_focus_origin_layer(
    fmap: folium.Map,
    stations: pd.DataFrame,
    dependencies: pd.DataFrame,
    focus_origin: str,
    flow_summary_lookup: pd.DataFrame,
    component_colors: Dict[Optional[int], str],
    focus_share_threshold: float,
) -> None:
    origin_id = _resolve_station_identifier(focus_origin, stations)
    if origin_id is None:
        raise ValueError(f"Could not find station matching '{focus_origin}'")

    origin_meta = stations.loc[origin_id]
    origin_name = origin_meta["station_name"]
    layer = folium.FeatureGroup(
        name=f"Dependencies from {origin_name} (≥{focus_share_threshold:.0%})",
        show=True,
    )

    folium.Marker(
        location=(origin_meta["lat"], origin_meta["lon"]),
        tooltip=f"{origin_name} (focus origin)",
        icon=folium.Icon(color="orange", icon="star", prefix="fa"),
    ).add_to(layer)

    focus_edges = dependencies[
        (dependencies["origin"] == origin_id)
        & (dependencies["destination_share"] >= focus_share_threshold)
    ]

    for _, dep in focus_edges.iterrows():
        dest_id = dep["destination"]
        if dest_id not in stations.index:
            continue

        dest_meta = stations.loc[dest_id]
        share = float(dep["destination_share"])
        trip_share = float(dep.get("destination_trip_share", float("nan")))
        origin_trip_share = float(dep.get("origin_trip_share", float("nan")))
        trip_count = float(dep.get("flow", 0.0))
        tooltip = (
            f"{origin_meta['station_name']} → {dest_meta['station_name']}<br>"
            f"Trips: {trip_count:.0f}<br>"
            f"Dest inflow share (prob): {share:.2f}<br>"
            f"Dest inflow share (trips): {trip_share:.2f}<br>"
            f"Origin outflow share (prob): {dep['origin_share']:.2f}<br>"
            f"Origin outflow share (trips): {origin_trip_share:.2f}"
        )

        folium.PolyLine(
            locations=[
                (origin_meta["lat"], origin_meta["lon"]),
                (dest_meta["lat"], dest_meta["lon"]),
            ],
            color="#ff6f61",
            weight=2 + share * 8,
            opacity=0.85,
            tooltip=tooltip,
        ).add_to(layer)

        folium.CircleMarker(
            location=(dest_meta["lat"], dest_meta["lon"]),
            radius=5 + min(share, 0.5) * 6,
            color="#ffbf00",
            fill=True,
            fill_color="#ffbf00",
            fill_opacity=0.75,
            weight=1,
            tooltip=tooltip,
        ).add_to(layer)

    layer.add_to(fmap)


def _add_legend(
    fmap: folium.Map,
    share_threshold: float,
    cross_share_threshold: float,
    focus_share_threshold: float,
    focus_origin: Optional[str],
) -> None:
    focus_line = (
        f"<li>Focus origin links ≥ {focus_share_threshold:.0%}; shows when a focus is selected.</li>"
        if focus_origin
        else ""
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; width: 260px; z-index: 9999; background: rgba(255, 255, 255, 0.92); padding: 12px 14px; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.2);">
      <h4 style="margin: 0 0 6px; font-size: 14px;">Dependency Map Legend</h4>
      <ul style="margin: 0; padding-left: 16px; font-size: 12px; line-height: 1.3;">
        <li>Marker colour indicates strongly connected component.</li>
        <li>Edge width scales with destination inflow share (capped at 60%).</li>
        <li>Component edges ≥ {share_threshold:.0%} keep the component colour.</li>
        <li>Cross-component edges ≥ {cross_share_threshold:.0%} use a red gradient with arrowheads.</li>
        {focus_line}
        <li>Tooltips report probability mass fractions (0-1) and trip totals.</li>
      </ul>
    </div>
    """
    fmap.get_root().html.add_child(Element(legend_html))


def _resolve_station_identifier(value: str, stations: pd.DataFrame) -> Optional[str]:
    candidate = str(value)
    if candidate in stations.index:
        return candidate

    lower_value = candidate.lower()
    matches = stations[stations["station_name"].str.lower() == lower_value]
    if not matches.empty:
        return matches.index[0]

    if "name" in stations.columns:
        matches = stations[stations["name"].str.lower() == lower_value]
        if not matches.empty:
            return matches.index[0]

    if "short_name" in stations.columns:
        matches = stations[stations["short_name"].str.lower() == lower_value]
        if not matches.empty:
            return matches.index[0]

    return None


if __name__ == "__main__":
    app()
