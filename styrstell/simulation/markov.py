"""Utilities for building and sampling Markov transition models."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd

from styrstell.simulation.travel_time import (
    TravelTimeObservation,
    TravelTimeProvider,
    ensure_positive_duration,
)


@dataclass
class MarkovChain:
    """Sparse time-dependent transition model for station-to-station flows."""

    transitions: pd.DataFrame
    station_ids: List[str]
    time_bins: List[pd.Timestamp]
    metadata: dict

    def transition_vector(self, origin: str, when: pd.Timestamp | None = None) -> pd.Series:
        """Return the transition probabilities for a station at a given time."""

        if when is None:
            when = self.time_bins[0]
        when = pd.to_datetime(when)
        filtered = self.transitions[
            (self.transitions["origin"] == str(origin))
            & (self.transitions["time_bin"] == when)
        ]
        if filtered.empty:
            return pd.Series(
                1.0 / len(self.station_ids), index=self.station_ids, dtype=float
            )
        return (
            filtered.set_index("destination")["probability"]
            .reindex(self.station_ids, fill_value=0.0)
            .astype(float)
        )

    def iter_transitions(self) -> Iterator[pd.DataFrame]:
        """Iterate over transition slices grouped by time bin."""

        for time_bin, frame in self.transitions.groupby("time_bin", sort=True):
            yield frame.reset_index(drop=True)

    def to_parquet(self, path: Path) -> None:
        """Serialize the transition table alongside metadata."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = path.with_suffix(".json")
        self.transitions.to_parquet(path, index=False)
        metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    @classmethod
    def from_parquet(cls, path: Path) -> "MarkovChain":
        path = Path(path)
        transitions = pd.read_parquet(path)
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        station_ids = sorted(
            set(transitions["origin"].astype(str)).union(transitions["destination"].astype(str))
        )
        time_bins = sorted(pd.to_datetime(transitions["time_bin"].unique()))
        return cls(transitions, station_ids, time_bins, metadata)


class MarkovChainBuilder:
    """Construct a time-dependent Markov chain from observed OD data."""

    def __init__(
        self,
        time_granularity: str | None = "1H",
        smoothing: float = 0.05,
        ensure_self_loops: bool = True,
    ) -> None:
        self.time_granularity = time_granularity.lower() if time_granularity else None
        self.smoothing = smoothing
        self.ensure_self_loops = ensure_self_loops

    def fit(
        self,
        od: pd.DataFrame,
        station_ids: Optional[Sequence[str]] = None,
    ) -> MarkovChain:
        """Build the transition table from a tidy OD probability dataset."""

        if od.empty:
            raise ValueError("Observed OD table is empty; cannot build Markov chain")

        frame = od.copy()
        frame["origin"] = frame["origin"].astype(str)
        frame["destination"] = frame["destination"].astype(str)
        frame["slot_start"] = pd.to_datetime(frame["slot_start"], utc=True)

        if self.time_granularity:
            frame["time_bin"] = frame["slot_start"].dt.floor(self.time_granularity)
        else:
            frame["time_bin"] = frame["slot_start"].min()

        observed_ids = set(frame["origin"].unique()).union(frame["destination"].unique())
        if station_ids is None:
            station_ids = sorted(observed_ids)
        else:
            station_ids = sorted(set(str(s) for s in station_ids).union(observed_ids))

        transitions = self._build_transitions(frame, station_ids)
        time_bins = sorted(transitions["time_bin"].drop_duplicates())
        metadata = {
            "time_granularity": self.time_granularity,
            "smoothing": self.smoothing,
            "ensure_self_loops": self.ensure_self_loops,
            "station_count": len(station_ids),
            "time_bin_count": len(time_bins),
        }
        return MarkovChain(transitions, list(station_ids), time_bins, metadata)

    def _build_transitions(
        self,
        frame: pd.DataFrame,
        station_ids: Sequence[str],
    ) -> pd.DataFrame:
        records: List[dict] = []
        station_index = pd.Index(station_ids, dtype="object")

        for (time_bin, origin), group in frame.groupby(["time_bin", "origin"], sort=True):
            observed_flow = (
                group.groupby("destination")["flow"].sum().reindex(station_index, fill_value=0.0)
            )
            observed_prob = observed_flow.copy()
            total_flow = observed_flow.sum()
            if total_flow > 0:
                observed_prob = observed_prob / total_flow
            else:
                observed_prob[:] = 0.0

            smoothed = observed_flow + self.smoothing
            if self.ensure_self_loops and smoothed.loc[origin] <= 0:
                smoothed.loc[origin] += self.smoothing if self.smoothing > 0 else 1e-6

            total = smoothed.sum()
            if total <= 0:
                # fallback to uniform distribution if no data at all
                probabilities = pd.Series(
                    1.0 / len(station_index), index=station_index, dtype=float
                )
            else:
                probabilities = smoothed / total

            for destination in station_index:
                records.append(
                    {
                        "time_bin": time_bin,
                        "origin": origin,
                        "destination": destination,
                        "probability": float(probabilities.loc[destination]),
                        "observed_probability": float(observed_prob.loc[destination]),
                        "observed_flow": float(observed_flow.loc[destination]),
                    }
                )

        transitions = pd.DataFrame.from_records(records)
        return transitions.sort_values(["time_bin", "origin", "destination"]).reset_index(drop=True)


@dataclass
class SimulatedTrip:
    path_index: int
    step_index: int
    origin: str
    destination: str
    departure_time: pd.Timestamp
    arrival_time: pd.Timestamp
    travel_minutes: float
    selection_probability: float
    travel_source: str


class MarkovSimulator:
    """Sample trips from a Markov chain with an external travel time provider."""

    def __init__(
        self,
        chain: MarkovChain,
        travel_provider: TravelTimeProvider,
        random_seed: Optional[int] = None,
    ) -> None:
        self.chain = chain
        self.travel_provider = travel_provider
        self.rng = np.random.default_rng(random_seed)

    def simulate_paths(
        self,
        origins: Sequence[str],
        steps: int,
        start_time: pd.Timestamp,
    ) -> pd.DataFrame:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if not origins:
            raise ValueError("origins cannot be empty")

        start_time = pd.to_datetime(start_time, utc=True)
        records: List[SimulatedTrip] = []
        station_index = pd.Index(self.chain.station_ids, dtype="object")

        for path_idx, initial_origin in enumerate(origins):
            current_station = str(initial_origin)
            if current_station not in station_index:
                raise KeyError(f"Origin {current_station} not present in Markov chain")
            current_time = start_time

            for step_idx in range(steps):
                time_bin = self._select_time_bin(current_time)
                probs = self.chain.transition_vector(current_station, time_bin)
                if probs.sum() <= 0:
                    raise RuntimeError(f"No outbound probability mass for {current_station} at {time_bin}")
                destination = self.rng.choice(probs.index.to_numpy(), p=probs.to_numpy())
                selection_probability = float(probs.loc[destination])
                observation = self.travel_provider.get_minutes(current_station, destination, current_time)
                travel_minutes = ensure_positive_duration(observation.minutes)
                arrival_time = current_time + pd.to_timedelta(travel_minutes, unit="minutes")

                records.append(
                    SimulatedTrip(
                        path_index=path_idx,
                        step_index=step_idx,
                        origin=current_station,
                        destination=str(destination),
                        departure_time=current_time,
                        arrival_time=arrival_time,
                        travel_minutes=float(travel_minutes),
                        selection_probability=selection_probability,
                        travel_source=observation.source,
                    )
                )

                current_station = str(destination)
                current_time = arrival_time

        trips_frame = pd.DataFrame([trip.__dict__ for trip in records])
        return trips_frame

    def _select_time_bin(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        ts = pd.to_datetime(timestamp, utc=True)
        eligible = [t for t in self.chain.time_bins if t <= ts]
        if not eligible:
            return self.chain.time_bins[0]
        return eligible[-1]
