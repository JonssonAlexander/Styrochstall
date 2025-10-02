"""Core simulation processes leveraging SimPy."""
from __future__ import annotations

from typing import Dict

import pandas as pd
from styrstell.simulation.environment import SimulationEnvironment, StationState


def run_simulation(sim_env: SimulationEnvironment) -> Dict[str, object]:
    """Execute the simulation for the configured horizon and return collected metrics."""

    for station_id in sim_env.station_states:
        sim_env.env.process(_station_arrivals(sim_env, station_id))
    if sim_env.sim_config.rebalancing.enabled:
        sim_env.env.process(_rebalancing_process(sim_env))
    sim_env.env.run(until=sim_env.horizon_minutes)
    for state in sim_env.station_states.values():
        state.propagate(sim_env.horizon_minutes)
    metrics = dict(sim_env.metrics)
    metrics["trip_log"] = sim_env.trip_log
    metrics["stockout_fraction"] = _fraction(sim_env.station_states, "stockout_minutes", sim_env.horizon_minutes)
    metrics["dockout_fraction"] = _fraction(sim_env.station_states, "dockout_minutes", sim_env.horizon_minutes)
    metrics["average_walk_minutes"] = (
        sim_env.metrics["total_walk_minutes"] / max(sim_env.metrics["completed_trips"], 1)
    )
    return metrics


def _station_arrivals(sim_env: SimulationEnvironment, station_id: str):
    while sim_env.env.now < sim_env.horizon_minutes:
        timestamp = sim_env.current_timestamp()
        lam = sim_env.lambda_rate(station_id, timestamp)
        if lam <= 1e-6:
            yield sim_env.env.timeout(1.0)
            continue
        interarrival = sim_env.random.exponential(1.0 / lam)
        yield sim_env.env.timeout(interarrival)
        if sim_env.env.now >= sim_env.horizon_minutes:
            break
        sim_env.env.process(_handle_rental(sim_env, station_id))


def _handle_rental(sim_env: SimulationEnvironment, origin_id: str):
    state = sim_env.station_states[origin_id]
    state.propagate(sim_env.env.now)
    if state.bikes <= 0:
        sim_env.metrics["stockout_events"] += 1
        return
    state.bikes -= 1
    state.last_update = sim_env.env.now
    timestamp = sim_env.current_timestamp()
    destination = sim_env.sample_destination(origin_id, timestamp)
    travel_minutes = sim_env.sample_travel_minutes(timestamp)
    allowed = sim_env.policy.max_duration(timestamp) + sim_env.policy.grace_period
    completed = travel_minutes <= allowed
    ride_minutes = min(travel_minutes, allowed)
    trip_start_time = timestamp
    yield sim_env.env.timeout(ride_minutes)
    yield from _complete_trip(sim_env, origin_id, destination, trip_start_time, ride_minutes, completed)


def _complete_trip(
    sim_env: SimulationEnvironment,
    origin: str,
    destination: str,
    start_time: pd.Timestamp,
    ride_minutes: float,
    completed: bool,
):
    walking_minutes = 0.0
    current_station = destination
    while True:
        state = sim_env.station_states[current_station]
        state.propagate(sim_env.env.now)
        if state.docks > 0:
            state.bikes += 1
            state.last_update = sim_env.env.now
            sim_env.record_trip(origin, current_station, start_time, ride_minutes, completed, walking_minutes)
            return
        sim_env.metrics["dockout_events"] += 1
        neighbor = sim_env.nearest_station_with_dock(current_station)
        if neighbor is None:
            sim_env.record_trip(origin, current_station, start_time, ride_minutes, False, walking_minutes)
            return
        next_station, distance_km = neighbor
        walk_time = (distance_km / sim_env.sim_config.walkers_speed_kmph) * 60.0
        walking_minutes += walk_time
        yield sim_env.env.timeout(walk_time)
        current_station = next_station


def _rebalancing_process(sim_env: SimulationEnvironment):
    interval = sim_env.sim_config.rebalancing.check_interval_minutes
    threshold = sim_env.sim_config.rebalancing.trigger_threshold
    batch = sim_env.sim_config.rebalancing.batch_size
    while True:
        yield sim_env.env.timeout(interval)
        if sim_env.env.now >= sim_env.horizon_minutes:
            break
        donor, receiver = _select_rebalance_pair(sim_env.station_states, threshold)
        if donor is None or receiver is None:
            continue
        donor.propagate(sim_env.env.now)
        receiver.propagate(sim_env.env.now)
        move_qty = min(batch, donor.bikes, max(receiver.capacity - receiver.bikes, 0))
        if move_qty <= 0:
            continue
        donor.bikes -= move_qty
        receiver.bikes += move_qty
        donor.last_update = sim_env.env.now
        receiver.last_update = sim_env.env.now


def _select_rebalance_pair(stations: Dict[str, StationState], threshold: int):
    donor = None
    receiver = None
    surplus = -float("inf")
    deficit = float("inf")
    for state in stations.values():
        available = state.bikes
        slack = state.capacity - state.bikes
        if available - slack > threshold and available > 0:
            if available - slack > surplus:
                donor = state
                surplus = available - slack
        if slack - available > threshold and slack > 0:
            if slack - available < deficit:
                receiver = state
                deficit = slack - available
    return donor, receiver


def _fraction(stations: Dict[str, StationState], attribute: str, horizon: float) -> float:
    total = sum(getattr(state, attribute) for state in stations.values())
    return total / (len(stations) * max(horizon, 1.0))
