"""Typed configuration models for Styr & Ställ workflows."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


class DataPaths(BaseModel):
    """Filesystem layout for raw, processed, and model artifacts."""

    root: Path = Field(default=Path("data"))
    raw: Path = Field(default=Path("raw"))
    processed: Path = Field(default=Path("processed"))
    models: Path = Field(default=Path("models"))
    cache: Path = Field(default=Path("cache"))

    @validator("raw", "processed", "models", "cache", pre=True, always=True)
    def _resolve(cls, value: Path, values: Dict[str, Path]) -> Path:  # type: ignore[arg-type]
        root: Path = values.get("root", Path("data"))
        value_path = Path(value)
        return value_path if value_path.is_absolute() else root / value_path


class GBFSFeedConfig(BaseModel):
    """GBFS feed endpoints and loader controls."""

    auto_discovery_url: str = Field(
        default="https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_zg/gbfs.json",
        description="Root GBFS feed providing per-language endpoints.",
    )
    feeds: Dict[str, str] = Field(
        default_factory=lambda: {
            "station_information": "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_zg/station_information.json",
            "station_status": "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_zg/station_status.json",
            "free_bike_status": "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_zg/free_bike_status.json",
        }
    )
    request_timeout: float = Field(default=10.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)
    snapshot_frequency_minutes: int = Field(default=5, ge=1)
    min_fields: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "station_information": ["station_id", "name", "lat", "lon"],
            "station_status": ["station_id", "num_bikes_available", "num_docks_available"],
        }
    )
    include_free_bike_status: bool = Field(default=True)


class MapsAPIConfig(BaseModel):
    """Configuration for the Nextbike maps API endpoint."""

    endpoint: str = Field(
        default="https://api.nextbike.net/maps/nextbike-live.json?city=658&format=json",
        description="Nextbike maps API endpoint returning station bike lists.",
    )
    request_timeout: float = Field(default=10.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)


class ResampleConfig(BaseModel):
    """Controls for deriving tidy time-series panels."""

    frequency: Literal["1min", "5min"] = Field(default="5min")
    fill_method: Literal["pad", "nearest", "none"] = Field(default="pad")
    drop_unobserved_stations: bool = Field(default=False)


class FeatureConfig(BaseModel):
    """Feature engineering options when deriving departures/arrivals."""

    resample: ResampleConfig = Field(default_factory=ResampleConfig)
    rebalancing_threshold: int = Field(
        default=5,
        description="Absolute change in bikes within one interval considered rebalancing if exceeded.",
    )
    smoothing_window: int = Field(
        default=3,
        description="Rolling window (in intervals) used to smooth inferred departures/arrivals.",
    )


class DemandCalibrationConfig(BaseModel):
    """Demand intensity estimation controls."""

    method: Literal["moving_average", "kernel"] = Field(default="moving_average")
    window_intervals: int = Field(default=12, ge=1)
    bandwidth: Optional[int] = Field(default=None, ge=1)


class TravelTimeCalibrationConfig(BaseModel):
    """Travel-time distribution calibration settings."""

    bin_minutes: int = Field(default=5, ge=1)
    max_minutes: int = Field(default=90, ge=10)
    smoothing: float = Field(default=0.01, ge=0.0)


class ODCalibrationConfig(BaseModel):
    """Origin-Destination (OD) matrix calibration settings."""

    time_slot_minutes: int = Field(default=60, ge=5)
    max_iterations: int = Field(default=100, ge=1)
    tolerance: float = Field(default=1e-5, ge=1e-8)
    gravity_beta: float = Field(default=0.5, ge=0.0)
    nearest_neighbor_k: int = Field(default=5, ge=1)


class CalibrationConfig(BaseModel):
    """Top-level calibration configuration."""

    data: DataPaths = Field(default_factory=DataPaths)
    gbfs: GBFSFeedConfig = Field(default_factory=GBFSFeedConfig)
    maps: MapsAPIConfig = Field(default_factory=MapsAPIConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    demand: DemandCalibrationConfig = Field(default_factory=DemandCalibrationConfig)
    travel_time: TravelTimeCalibrationConfig = Field(default_factory=TravelTimeCalibrationConfig)
    od: ODCalibrationConfig = Field(default_factory=ODCalibrationConfig)
    cache: bool = Field(default=True)
    overwrite: bool = Field(default=False)


class PolicyConfig(BaseModel):
    """Parameters defining a user-facing rental policy."""

    name: str
    max_rental_minutes: int = Field(default=60, ge=5)
    grace_period_minutes: int = Field(default=0, ge=0)
    dynamic_profile: Optional[Dict[str, int]] = Field(
        default=None,
        description="Optional mapping from time-of-day window (HH:MM-HH:MM) to max rental minutes.",
    )


class RebalancingPolicyConfig(BaseModel):
    """Simple repositioning policy parameters."""

    enabled: bool = Field(default=False)
    trigger_threshold: int = Field(
        default=5,
        description="Trigger repositioning when docks or bikes deviate from target by this amount.",
    )
    batch_size: int = Field(default=10, ge=1)
    check_interval_minutes: int = Field(default=30, ge=5)


class SimulationConfig(BaseModel):
    """Simulation runtime configuration."""

    data: DataPaths = Field(default_factory=DataPaths)
    features_path: Optional[Path] = Field(default=None)
    demand_model_path: Optional[Path] = Field(default=None)
    travel_time_model_path: Optional[Path] = Field(default=None)
    od_model_path: Optional[Path] = Field(default=None)
    policies: List[PolicyConfig] = Field(
        default_factory=lambda: [
            PolicyConfig(name="max60", max_rental_minutes=60),
            PolicyConfig(name="max45", max_rental_minutes=45),
            PolicyConfig(name="max30", max_rental_minutes=30),
        ]
    )
    rebalancing: RebalancingPolicyConfig = Field(default_factory=RebalancingPolicyConfig)
    simulation_horizon: timedelta = Field(default=timedelta(hours=24))
    warm_up_minutes: int = Field(default=30, ge=0)
    random_seed: int = Field(default=42)
    walkers_speed_kmph: float = Field(default=4.8, ge=0.1)
    policy_constraint_stockout: float = Field(default=0.05, ge=0.0, le=1.0)
    policy_constraint_walk_minutes: float = Field(default=3.0, ge=0.0)


def load_calibration_config(path: Optional[Path] = None) -> CalibrationConfig:
    """Load calibration configuration from disk or return defaults."""

    if path is None:
        return CalibrationConfig()
    data = _load_json_or_yaml(path)
    return CalibrationConfig.model_validate(data)


def load_simulation_config(path: Optional[Path] = None) -> SimulationConfig:
    """Load simulation configuration from disk or return defaults."""

    if path is None:
        return SimulationConfig()
    data = _load_json_or_yaml(path)
    return SimulationConfig.model_validate(data)


def _load_json_or_yaml(path: Path) -> Dict[str, object]:
    if path.suffix in {".json"}:
        import json

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    raise ValueError(f"Unsupported config format: {path}")
