# Styr & Ställ Analysis and Simulation

This repository provides an end-to-end workflow to analyze Gothenburg's bike share (Styr & Ställ / Nextbike) using GBFS feeds, calibrate demand and travel models, and evaluate rental-time policies via discrete-event simulation.

## Project structure

- `data/` – raw snapshots, processed panels, cached intermediates, and calibrated models.
- `styrstell/` – Python package with loaders, feature engineering, calibration, simulation, metrics, and reporting utilities.
- `notebooks/` – exploratory and reporting notebooks for calibration and policy evaluation.
- `tests/` – lightweight regression tests and examples.

## Getting started

1. Create a virtual environment with Python 3.11+ and install dependencies:
   ```bash
   pip install -e .[dev]
   ```
2. Configure the run parameters in `styrstell/config.py`, or provide an override YAML/JSON file when invoking the CLI.
3. Calibrate demand, travel-time, and OD models (cached to `data/models/`):
   ```bash
   styrstell-calibrate --config configs/calibration.yaml
   ```
4. Simulate rental policies and output KPIs and plots:
   ```bash
   styrstell-simulate --config configs/simulation.yaml
   ```

The notebooks in `notebooks/` demonstrate calibration diagnostics and policy comparisons using the cached artifacts. Station identifiers in the GBFS feed are carried through transparently; when a specific station ID is not required (e.g., in examples or tests), stable placeholders are used and the analysis does not depend on their specific values.

## Data management

- Raw GBFS snapshots are stored under `data/raw/GBFS/<timestamp>/` with original filenames preserved.
- Normalized parquet tables live in `data/processed/` and are versioned by snapshot batch.
- Model parameters (λ, travel-time distributions, OD matrices) are serialized to `data/models/`.
- Derived dashboards and figures are written to `data/cache/` or `reports/`.

## Commands

- `styrstell-calibrate`: Fetch snapshots (unless cached), build time series features, estimate λ, travel-time distributions, time-varying OD matrices, and an aggregate OD probability matrix for sampling workflows.
- `styrstell-simulate`: Load calibrated artifacts, run the SimPy-based environment across policy scenarios, and emit KPI tables and comparison plots.
- `styrstell-infer-trips`: Reconstruct observed trips from stored `free_bike_status` snapshots and store them in `data/processed/trips.parquet`.
- `styrstell-infer-maps-trips`: Reconstruct observed trips from Nextbike maps API snapshots (bike lists per station) and save them to `data/processed/maps_trips.parquet`.
- `styrstell-observed-od`: Build observed edge lists, OD matrices (plus a row-normalised probability matrix), and travel-time distributions from the inferred trips for downstream exploration/visualisation.
- `python3.11 scripts/populate_data.py`: Grab one or multiple GBFS snapshots (via the public Nextbike endpoints) to pre-populate `data/raw/GBFS/` and `data/raw/MAPS/` before running calibration.
- `python3.11 scripts/analyze_data.py`: Generate a quick markdown/console summary of processed snapshots and calibrated models to validate data quality prior to simulation.
- `python3.11 scripts/plot_od_graph.py`: Render the estimated OD flows (top links) as a geographic graph with stations positioned by latitude/longitude (`--basemap` overlays OpenStreetMap tiles and needs `contextily` + `pyproj`; `--boundary path.geojson` draws a local GeoJSON outline with no extra deps).
- `python3.11 scripts/plot_od_map.py`: Build an interactive Leaflet map (HTML) with OD flows (`--boundary path.geojson` adds a local outline, `--annotate --label-top N` labels the top N stations by capacity using their real names/numbers).
- `python3.11 scripts/compare_gbfs_maps.py`: Compare per-station bike counts between the GBFS station status feed and MAPS bike lists, producing a tidy parquet table, an interactive Plotly animation, and (with `--output-gif`) a GIF timeline.
- `python3.11 scripts/detect_rebalancing.py`: Flag potential rebalancing movements by pairing large station count drops and spikes (writes spike events, inferred routes with Dijkstra, optional depot fallback, and a Folium map).
- `python3.11 scripts/plot_station_activity_map.py`: Create a Folium map showing per-station trip activity (departures, arrivals, or net flow) based on an inferred trips parquet.

Both commands accept a `--config` flag pointing to a JSON/YAML file matching the typed configuration schema in `styrstell/config.py`.

## Reproducibility & performance

Pipelines favor parquet caching, incremental fetches, and pragmatic resampling (per-minute or five-minute). Processing steps withstand schema drift, missing GBFS fields, and backfill the minimal requirements (e.g., capacities inferred from historical maxima). Subsampling options keep calibration/simulation laptop-friendly.

## Development

- Run `pytest` to execute unit tests.
- Run `mypy styrstell` for type checking.
- Notebooks expect an active ipykernel in the same environment (`python -m ipykernel install --user --name styrstell`).

## License

MIT License.
