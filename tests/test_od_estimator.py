import numpy as np
import pandas as pd
import pytest

from styrstell.calibration.od_estimator import (
    aggregate_flows,
    estimate_od_matrix,
)


def _build_sample_data() -> pd.DataFrame:
    records = [
        {"timestamp": "2025-01-01T00:05:00Z", "station_id": "A", "lat": 57.7, "lon": 11.9, "delta_bikes": -3},
        {"timestamp": "2025-01-01T00:10:00Z", "station_id": "B", "lat": 57.71, "lon": 11.91, "delta_bikes": 2},
        {"timestamp": "2025-01-01T00:15:00Z", "station_id": "C", "lat": 57.72, "lon": 11.92, "delta_bikes": 1},
        {"timestamp": "2025-01-01T00:20:00Z", "station_id": "B", "lat": 57.71, "lon": 11.91, "delta_bikes": -1},
        {"timestamp": "2025-01-01T01:05:00Z", "station_id": "C", "lat": 57.72, "lon": 11.92, "delta_bikes": -2},
        {"timestamp": "2025-01-01T01:10:00Z", "station_id": "A", "lat": 57.7, "lon": 11.9, "delta_bikes": 3},
        {"timestamp": "2025-01-01T01:15:00Z", "station_id": "B", "lat": 57.71, "lon": 11.91, "delta_bikes": 1},
        {"timestamp": "2025-01-01T01:45:00Z", "station_id": "B", "lat": 57.71, "lon": 11.91, "delta_bikes": -2},
        {"timestamp": "2025-01-01T02:05:00Z", "station_id": "C", "lat": 57.72, "lon": 11.92, "delta_bikes": 1},
        {"timestamp": "2025-01-01T02:10:00Z", "station_id": "C", "lat": 57.72, "lon": 11.92, "delta_bikes": 1},
    ]
    return pd.DataFrame.from_records(records)


def test_aggregate_flows_balances_out_in():
    df = _build_sample_data()
    outflows, inflows, stations = aggregate_flows(df, window="1H")
    assert set(outflows.index) == {"A", "B", "C"}
    assert pytest.approx(outflows.sum(), rel=1e-6) == pytest.approx(inflows.sum(), rel=1e-6)
    assert outflows.loc["A"] == pytest.approx(3.0)
    assert inflows.loc["A"] == pytest.approx(3.0)


def test_estimate_od_matrix_returns_balanced_flows():
    df = _build_sample_data()
    result = estimate_od_matrix(df, beta=0.1, window="1H")
    flow_totals = result.flows.sum(axis=1)
    assert np.all(flow_totals >= 0)
    assert pytest.approx(flow_totals.sum(), rel=1e-6) == pytest.approx(result.flows.sum(axis=0).sum(), rel=1e-6)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_sampling_matches_trip_count():
    df = _build_sample_data()
    result = estimate_od_matrix(df, beta=0.2, window="1H")
    trips = result.sample_trips(500, random_state=np.random.default_rng(42))
    assert len(trips) == 500
    assert set(trips.columns) == {"origin", "destination"}
