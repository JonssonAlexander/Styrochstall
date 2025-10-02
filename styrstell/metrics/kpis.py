"""Key performance indicator aggregation."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from styrstell.config import SimulationConfig
from styrstell.simulation.policies import RentalPolicy


def compute_policy_kpis(metrics: Dict[str, object], policy: RentalPolicy, config: SimulationConfig) -> pd.DataFrame:
    """Aggregate simulation metrics into a tidy KPI table for a single policy."""

    completed = float(metrics.get("completed_trips", 0.0))
    truncated = float(metrics.get("truncated_trips", 0.0))
    stockout_fraction = float(metrics.get("stockout_fraction", 0.0))
    dockout_fraction = float(metrics.get("dockout_fraction", 0.0))
    walk_minutes = float(metrics.get("average_walk_minutes", 0.0))
    feasible = (
        stockout_fraction + dockout_fraction <= config.policy_constraint_stockout
        and walk_minutes <= config.policy_constraint_walk_minutes
    )
    return pd.DataFrame(
        {
            "policy": [policy.name],
            "completed_trips": [completed],
            "truncated_trips": [truncated],
            "stockout_fraction": [stockout_fraction],
            "dockout_fraction": [dockout_fraction],
            "avg_walk_minutes": [walk_minutes],
            "constraint_threshold_stockout": [config.policy_constraint_stockout],
            "constraint_threshold_walk": [config.policy_constraint_walk_minutes],
            "feasible": [feasible],
        }
    )
