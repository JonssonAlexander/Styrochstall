"""Monte Carlo simulation over Markov transitions and travel times."""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .markov import MarkovChain
from .travel_time import TravelTimeObservation, TravelTimeProvider, ensure_positive_duration

SimulationScenario = Literal["demand_only", "demand_with_refill"]


@dataclass
class SimulationOutcome:
    scenario: SimulationScenario
    replication: int
    station_id: str
    stockout_time: Optional[pd.Timestamp]
    stockout_occurred: bool
    truncated_departures: int


class MonteCarloSimulator:
    """Runs Monte Carlo simulations using MAPS-based Markov transitions."""

    def __init__(
        self,
        chain: MarkovChain,
        departures: pd.DataFrame,
        initial_inventory: pd.DataFrame,
        travel_provider: TravelTimeProvider,
        rebalancing_events: pd.DataFrame,
        time_granularity: str,
        low_stock_threshold: int = 1,
        demand_mode: str = "fixed",
        random_seed: Optional[int] = None,
    ) -> None:
        self.chain = chain
        self.time_bins = sorted(pd.to_datetime(chain.time_bins))
        self.time_granularity = time_granularity.lower()
        self.bin_offset = pd.to_timedelta(self.time_granularity)
        self.low_stock_threshold = low_stock_threshold
        self.travel_provider = travel_provider
        self.rng = np.random.default_rng(random_seed)
        demand_mode = demand_mode.lower()
        if demand_mode not in {"fixed", "poisson"}:
            raise ValueError("demand_mode must be 'fixed' or 'poisson'")
        self.demand_mode = demand_mode

        departures = departures.copy()
        departures["time_bin"] = pd.to_datetime(departures["time_bin"], utc=True)
        departures["station_id"] = departures["station_id"].astype(str)
        self.lambda_lookup: Dict[Tuple[pd.Timestamp, str], float] = {
            (row.time_bin, row.station_id): float(row.departures)
            for row in departures.itertuples()
        }

        inventory = initial_inventory.copy()
        inventory["station_id"] = inventory["station_id"].astype(str)
        self.initial_bikes: Dict[str, int] = {
            row.station_id: int(row.bikes) if not pd.isna(row.bikes) else 0
            for row in inventory.itertuples()
        }
        self.capacity: Dict[str, Optional[int]] = {
            row.station_id: int(row.capacity) if hasattr(row, "capacity") and not pd.isna(row.capacity) else None
            for row in inventory.itertuples()
        }
        # Ensure every chain station has an inventory entry, default 0
        for station_id in chain.station_ids:
            self.initial_bikes.setdefault(station_id, 0)
            self.capacity.setdefault(station_id, None)

        rebalancing_events = rebalancing_events.copy()
        rebalancing_events["timestamp"] = pd.to_datetime(
            rebalancing_events["timestamp"], utc=True
        )
        rebalancing_events["station_id"] = rebalancing_events["station_id"].astype(str)
        rebalancing_events["delta"] = rebalancing_events["delta"].astype(float)
        self.rebalancing_events = rebalancing_events.sort_values("timestamp").reset_index(drop=True)

    def _departure_rate(self, time_bin: pd.Timestamp, station_id: str) -> float:
        return self.lambda_lookup.get((time_bin, station_id), 0.0)

    def _sample_departure_count(self, lam: float) -> int:
        if lam <= 0:
            return 0
        if self.demand_mode == "fixed":
            return int(round(lam))
        return int(self.rng.poisson(lam))

    def _draw_destination(self, station_id: str, when: pd.Timestamp) -> str:
        probs = self.chain.transition_vector(station_id, when)
        if probs.sum() <= 0:
            return station_id
        destinations = probs.index.to_numpy()
        return str(self.rng.choice(destinations, p=probs.to_numpy()))

    def simulate(
        self,
        num_replications: int,
        scenario: SimulationScenario,
        log_inventory: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        outcomes: List[SimulationOutcome] = []
        inventory_records: List[dict] = [] if log_inventory else None
        if not self.time_bins:
            raise ValueError("No time bins available in the Markov chain.")

        for rep in range(num_replications):
            rep_outcomes, rep_inventory = self._simulate_single(rep, scenario, log_inventory)
            outcomes.extend(rep_outcomes)
            if log_inventory and rep_inventory is not None:
                inventory_records.extend(rep_inventory)

        records = [
            {
                "scenario": outcome.scenario,
                "replication": outcome.replication,
                "station_id": outcome.station_id,
                "stockout_time": outcome.stockout_time,
                "stockout_occurred": outcome.stockout_occurred,
                "truncated_departures": outcome.truncated_departures,
            }
            for outcome in outcomes
        ]
        outcome_df = pd.DataFrame(records)
        inventory_df = pd.DataFrame(inventory_records) if log_inventory else None
        return outcome_df, inventory_df

    def _simulate_single(
        self,
        replication: int,
        scenario: SimulationScenario,
        log_inventory: bool,
    ) -> Tuple[List[SimulationOutcome], Optional[List[dict]]]:
        state = {station: int(bikes) for station, bikes in self.initial_bikes.items()}
        truncated_departures = {station: 0 for station in state}
        stockout_time: Dict[str, Optional[pd.Timestamp]] = {station: None for station in state}

        inventory_records: List[dict] = [] if log_inventory else None

        arrival_heap: List[Tuple[int, int, str]] = []
        event_counter = 0

        reb_events = self.rebalancing_events
        reb_index = 0
        total_reb = len(reb_events)

        def check_stockout(station: str, timestamp: pd.Timestamp) -> None:
            if stockout_time[station] is not None:
                return
            if state[station] <= self.low_stock_threshold:
                stockout_time[station] = timestamp

        start_time = pd.Timestamp(self.time_bins[0])
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize("UTC")
        else:
            start_time = start_time.tz_convert("UTC")
        for station in state:
            check_stockout(station, start_time)
            if log_inventory:
                inventory_records.append(
                    {
                        "scenario": scenario,
                        "replication": replication,
                        "station_id": station,
                        "time_bin": start_time,
                        "bikes": state[station],
                    }
                )

        for bin_start in self.time_bins:
            bin_start = pd.Timestamp(bin_start)
            if bin_start.tzinfo is None:
                bin_start = bin_start.tz_localize("UTC")
            else:
                bin_start = bin_start.tz_convert("UTC")
            bin_end = bin_start + self.bin_offset

            # Process any arrivals scheduled up to the start of the bin
            arrival_heap = self._drain_arrivals(arrival_heap, state, check_stockout, bin_start)

            # Apply rebalancing events up to the start of the bin (only in refill scenario)
            if scenario == "demand_with_refill":
                reb_index = self._apply_rebalancing(
                    reb_events,
                    reb_index,
                    total_reb,
                    state,
                    check_stockout,
                    cutoff=bin_start,
                )

            # Generate departures for this bin
            for station in self.chain.station_ids:
                lam = self._departure_rate(bin_start, station)
                departures = self._sample_departure_count(lam)
                if departures <= 0:
                    continue
                for _ in range(departures):
                    departure_time = bin_start + pd.to_timedelta(
                        self.rng.uniform(0, self.bin_offset.total_seconds()), unit="s"
                    )
                    if state[station] <= 0:
                        truncated_departures[station] += 1
                        check_stockout(station, departure_time)
                        continue
                    state[station] -= 1
                    check_stockout(station, departure_time)
                    destination = self._draw_destination(station, bin_start)
                    observation: TravelTimeObservation = self.travel_provider.get_minutes(
                        station, destination, departure_time
                    )
                    travel_minutes = ensure_positive_duration(observation.minutes)
                    arrival_time = departure_time + pd.to_timedelta(travel_minutes, unit="m")
                    event_counter += 1
                    heapq.heappush(
                        arrival_heap,
                        (arrival_time.value, event_counter, destination),
                    )

            # Apply rebalancing events within the bin end
            if scenario == "demand_with_refill":
                reb_index = self._apply_rebalancing(
                    reb_events,
                    reb_index,
                    total_reb,
                    state,
                    check_stockout,
                    cutoff=bin_end,
                )

            # Process arrivals up to the end of the bin
            arrival_heap = self._drain_arrivals(arrival_heap, state, check_stockout, bin_end)

            if log_inventory:
                for station in state:
                    inventory_records.append(
                        {
                            "scenario": scenario,
                            "replication": replication,
                            "station_id": station,
                            "time_bin": bin_end,
                            "bikes": state[station],
                        }
                    )

        # Drain any remaining arrivals after the horizon
        arrival_heap = self._drain_arrivals(
            arrival_heap,
            state,
            check_stockout,
            pd.Timestamp.max.tz_localize("UTC"),
        )

        outcomes: List[SimulationOutcome] = []
        for station in self.chain.station_ids:
            outcomes.append(
                SimulationOutcome(
                    scenario=scenario,
                    replication=replication,
                    station_id=station,
                    stockout_time=stockout_time[station],
                    stockout_occurred=stockout_time[station] is not None,
                    truncated_departures=truncated_departures[station],
                )
            )
        return outcomes, inventory_records

    def _apply_rebalancing(
        self,
        reb_events: pd.DataFrame,
        index: int,
        total: int,
        state: Dict[str, int],
        check_stockout,
        cutoff: pd.Timestamp,
    ) -> int:
        while index < total and reb_events.iloc[index].timestamp <= cutoff:
            row = reb_events.iloc[index]
            station_id = row.station_id
            delta = float(row.delta)
            if station_id not in state:
                state[station_id] = 0
            state[station_id] += int(delta)
            capacity = self.capacity.get(station_id)
            if capacity is not None:
                state[station_id] = min(state[station_id], capacity)
            check_stockout(station_id, row.timestamp)
            index += 1
        return index

    def _drain_arrivals(
        self,
        arrival_heap: List[Tuple[int, int, str]],
        state: Dict[str, int],
        check_stockout,
        cutoff: pd.Timestamp,
    ) -> List[Tuple[int, int, str]]:
        while arrival_heap and arrival_heap[0][0] <= cutoff.value:
            _, _, station_id = heapq.heappop(arrival_heap)
            if station_id not in state:
                state[station_id] = 0
            state[station_id] += 1
            capacity = self.capacity.get(station_id)
            if capacity is not None and state[station_id] > capacity:
                state[station_id] = capacity
        return arrival_heap


def aggregate_stockout_probabilities(
    outcomes: pd.DataFrame,
    chain: MarkovChain,
    time_granularity: str,
    *,
    cumulative: bool = True,
    inventory_log: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute stockout probabilities (and optional mean bikes) per station/time bin."""

    if outcomes.empty:
        return pd.DataFrame(columns=["scenario", "time_bin", "station_id", "stockout_probability"])

    time_bins = sorted(pd.to_datetime(chain.time_bins))
    offset = pd.to_timedelta(time_granularity.lower())

    inventory_lookup: Dict[Tuple[str, str, pd.Timestamp], float] = {}
    if inventory_log is not None and not inventory_log.empty:
        inventory = inventory_log.copy()
        inventory["time_bin"] = pd.to_datetime(inventory["time_bin"], utc=True)
        inventory["station_id"] = inventory["station_id"].astype(str)
        grouped_inventory = inventory.groupby(["scenario", "station_id", "time_bin"])  # type: ignore[arg-type]
        inventory_lookup = {
            (scenario, station_id, time_bin): float(group["bikes"].mean())
            for (scenario, station_id, time_bin), group in grouped_inventory
        }

    rows = []
    grouped = outcomes.groupby(["scenario", "station_id"])
    for (scenario, station_id), group in grouped:
        total = len(group)
        stock_times = group["stockout_time"].to_list()
        for bin_start in time_bins:
            bin_start_ts = pd.Timestamp(bin_start)
            if bin_start_ts.tzinfo is None:
                bin_start_ts = bin_start_ts.tz_localize("UTC")
            else:
                bin_start_ts = bin_start_ts.tz_convert("UTC")
            bin_end = bin_start_ts + offset
            if cumulative:
                count = sum(pd.notna(st) and st <= bin_end for st in stock_times)
            else:
                count = sum(pd.notna(st) and bin_start_ts <= st < bin_end for st in stock_times)
            probability = count / total if total > 0 else 0.0
            rows.append(
                {
                    "scenario": scenario,
                    "station_id": station_id,
                    "time_bin": bin_start_ts,
                    "stockout_probability": probability,
                    "mean_bikes": inventory_lookup.get((scenario, station_id, bin_start_ts)),
                }
            )

    return pd.DataFrame(rows)
