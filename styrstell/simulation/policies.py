"""Simulation policies and helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

from styrstell.config import PolicyConfig


@dataclass
class RentalPolicy:
    """Wrapper around a policy configuration with sampling utilities."""

    config: PolicyConfig
    _profile: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if not self.config.dynamic_profile:
            return
        profile: Dict[str, int] = self.config.dynamic_profile
        for window, limit in profile.items():
            start_str, end_str = window.split("-")
            start = pd.to_datetime(start_str).time()
            end = pd.to_datetime(end_str).time()
            today = pd.Timestamp.today().normalize()
            self._profile.append((today + pd.Timedelta(hours=start.hour, minutes=start.minute), today + pd.Timedelta(hours=end.hour, minutes=end.minute), limit))

    def max_duration(self, timestamp: pd.Timestamp) -> int:
        if not self._profile:
            return self.config.max_rental_minutes
        ts_time = timestamp.time()
        for start, end, limit in self._profile:
            if start.time() <= ts_time < end.time():
                return limit
        return self.config.max_rental_minutes

    @property
    def grace_period(self) -> int:
        return self.config.grace_period_minutes

    @property
    def name(self) -> str:
        return self.config.name


def select_policy_parameter(config: PolicyConfig) -> RentalPolicy:
    """Instantiate a rental policy from configuration."""

    return RentalPolicy(config=config)
