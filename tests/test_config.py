import pytest

from styrstell import config


def test_default_calibration_config():
    cfg = config.load_calibration_config(None)
    assert cfg.gbfs.snapshot_frequency_minutes in (1, 5)
    assert cfg.features.resample.frequency in {"1min", "5min"}


def test_default_simulation_config():
    cfg = config.load_simulation_config(None)
    assert cfg.simulation_horizon.total_seconds() > 0
    assert len(cfg.policies) >= 1
