Purpose

Assess whether reducing the maximum rental time (60 → 30/45 minutes or dynamic) is defensible given KPIs: completed trips, share of stockout/dockout minutes, average walking time to a free dock, bikes in circulation vs parked, and rebalancing needs.

Inputs

GBFS station_information.json (station_id, lat/lon, capacity).

GBFS station_status.json (num_bikes_available, num_docks_available, is_renting/is_returning, etc.).

Optional free_bike_status.json for free‑floating bikes (may be absent).

Basic geospatial helpers (e.g., haversine) for walking‑time estimates to nearby stations.

Note on IDs

If exact IDs do not matter for the analysis or simulation logic, explicitly note that ID choice does not matter and use stable placeholders.

Artifacts (indicative)

Raw snapshots: data/raw/GBFS/<timestamp>/*.json.

Processed panel: per station × time: state + inferred departures/arrivals + rebalancing flags.

Models: lambda_{s,t} estimates; g_t (travel‑time) by time‑of‑day; P_t (OD matrices) by time slot.

Core components

Loader: fetch/version GBFS; normalize schemas; write Parquet/Feather.

Feature builder: resample to a regular grid; infer departures/arrivals from deltas; flag rebalancing events.

Calibration:

lambda_{s,t}: rolling/windowed Poisson‑rate estimates.

g_t: empirical or parametric fit (e.g., gamma/lognormal) by time‑of‑day.

P_t: IPFP to match departures/arrivals per slot, with gravity prior (distance/attractiveness).

Simulation: discrete‑event engine with station capacities, user arrivals (driven by lambda_{s,t}), OD sampling via P_t, travel times sampled from g_t, and rental‑time policy logic (max time, optional grace). Include a simple rebalancing heuristic.

Evaluator: compute KPIs and check constraints; aggregate by time, station, and system.

Policies (examples)

max60 / max45 / max30: hard caps.

dynamic: max time as a function of time‑of‑day and/or congestion (e.g., 45 off‑peak, 30 peak).

grace=5min: overtime grace window without penalty.

rebalancing: threshold‑based relocation stub (extendable).

KPIs & constraints

Completed trips (absolute and per bike).

Stockout/dockout minutes per station and system‑wide (share of total time).

Average walking time to nearest available dock (requires proximity graph and availability model).

Bikes in use vs parked over time.

Rebalancing effort (moved bikes, distance if known).

Feasibility constraints: (dockout + stockout) ≤ 5%; mean walking time ≤ 3 minutes.

Decision logic

Run comparative simulations over the same period and random seeds for each policy. Select the policy that maximizes completed trips while satisfying feasibility constraints. Report trade‑offs, sensitivity analyses (e.g., grace window), and highlight operational impacts (rebalancing load, user walking time).