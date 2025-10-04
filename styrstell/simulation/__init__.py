"""Discrete-event simulation primitives (SimPy optional)."""

from typing import Optional

try:  # pragma: no cover - optional dependency guard
    from .environment import SimulationEnvironment
except ModuleNotFoundError:  # pragma: no cover
    SimulationEnvironment = None  # type: ignore

from .policies import RentalPolicy, select_policy_parameter

try:  # pragma: no cover
    from .processes import run_simulation
except ModuleNotFoundError:  # pragma: no cover
    run_simulation = None  # type: ignore

__all__ = [
    "SimulationEnvironment",
    "RentalPolicy",
    "select_policy_parameter",
    "run_simulation",
]
