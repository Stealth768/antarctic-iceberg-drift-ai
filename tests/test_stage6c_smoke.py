"""
Stage 6C Synthetic Smoke Test.

Validates that the complete pipeline (GLORYS boundary handling, multi-file loading,
physics evaluation, baseline comparison) works end-to-end before real data is obtained.

Tests use synthetic ERA5 + GLORYS covering a tiny spatio-temporal window with known values.
"""

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.copernicus import CopernicusLoader
from src.data.era5 import ERA5Loader
from src.data.environment import HistoricalEnvironmentProvider
from src.data.iceberg import BYUConsolidatedDatabaseLoader
from src.evaluation.historical_pairs import EvaluationPair, build_evaluation_pairs
from src.evaluation.physics_evaluator import IcebergPhysicsEvaluator, create_default_iceberg_properties
from src.evaluation.baseline_evaluator import ConstantVelocityBaselineEvaluator
from src.evaluation.real_physics_benchmark import open_environmental_sources


@pytest.fixture
def synthetic_smoke_era5():
    """Create tiny ERA5 dataset covering 2020-01-01 to 2020-01-03 in A23A region."""
    times = pd.date_range("2020-01-01", "2020-01-03", freq="1D")  # naive timestamps
    lats = np.linspace(-77, -75, 5)
    lons = np.linspace(-43, -41, 5)
    
    ds = xr.Dataset(
        data_vars={
            "u10": (("time", "latitude", "longitude"), np.random.uniform(-2, 2, (3, 5, 5))),
            "v10": (("time", "latitude", "longitude"), np.random.uniform(-2, 2, (3, 5, 5))),
            "t2m": (("time", "latitude", "longitude"), np.full((3, 5, 5), 260.0)),
            "msl": (("time", "latitude", "longitude"), np.full((3, 5, 5), 100000.0)),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds


@pytest.fixture
def synthetic_smoke_glorys():
    """
    Create tiny GLORYS dataset covering 2020-01-01 to 2020-01-03 in A23A region.
    Include depths 0.494025, 186.125595, and 222.475204 to test interpolation.
    """
    times = pd.date_range("2020-01-01", "2020-01-03", freq="1D")  # naive timestamps
    lats = np.linspace(-77, -75, 5)
    lons = np.linspace(-43, -41, 5)
    depths = np.array([0.494025, 186.125595, 222.475204])
    
    ds = xr.Dataset(
        data_vars={
            "uo": (("time", "depth", "latitude", "longitude"), 
                   np.random.uniform(-0.1, 0.1, (3, 3, 5, 5))),
            "vo": (("time", "depth", "latitude", "longitude"), 
                   np.random.uniform(-0.1, 0.1, (3, 3, 5, 5))),
            "thetao": (("time", "depth", "latitude", "longitude"), 
                       np.full((3, 3, 5, 5), 271.15)),
        },
        coords={
            "time": times,
            "depth": depths,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds


def test_smoke_era5_loader_integration(synthetic_smoke_era5):
    """Smoke Test 1: ERA5Loader opens, queries, and returns finite values."""
    loader = ERA5Loader(synthetic_smoke_era5)
    
    result = loader.get_forcing(
        timestamp=pd.Timestamp("2020-01-01T12:00:00"),
        latitude=-76.0,
        longitude=-42.0,
        method="linear",
    )
    
    assert set(result.keys()) == {"wind_u", "wind_v", "temperature", "pressure"}
    assert all(np.isfinite(v) for v in result.values())
    print("✓ ERA5Loader: OK")


def test_smoke_glorys_loader_boundary_interpolation(synthetic_smoke_glorys):
    """Smoke Test 2: CopernicusLoader interpolates 200 m boundary correctly."""
    loader = CopernicusLoader(synthetic_smoke_glorys)
    
    result = loader.get_ocean_currents(
        timestamp=pd.Timestamp("2020-01-01"),
        latitude=-76.0,
        longitude=-42.0,
        draft_meters=200.0,
        method="linear",
    )
    
    assert set(result.keys()) == {"ocean_u", "ocean_v", "sst"}
    assert all(np.isfinite(v) for v in result.values())
    print("✓ CopernicusLoader boundary interpolation: OK")


def test_smoke_historical_provider_integration(synthetic_smoke_era5, synthetic_smoke_glorys):
    """Smoke Test 3: HistoricalEnvironmentProvider integrates ERA5 + GLORYS."""
    provider = HistoricalEnvironmentProvider(
        era5_source=synthetic_smoke_era5,
        copernicus_source=synthetic_smoke_glorys,
        default_sea_ice_concentration=0.05,
        draft_meters=200.0,
    )
    
    env = provider.get_environment(
        timestamp=pd.Timestamp("2020-01-01T12:00:00"),
        latitude=-76.0,
        longitude=-42.0,
    )
    
    expected_keys = {
        "ocean_u", "ocean_v", "sst",
        "wind_u", "wind_v", "temperature", "pressure",
        "sea_ice_concentration",
    }
    assert set(env.keys()) == expected_keys
    assert all(np.isfinite(v) for v in env.values())
    print("✓ HistoricalEnvironmentProvider: OK")


def test_smoke_physics_evaluator_integration(synthetic_smoke_era5, synthetic_smoke_glorys):
    """Smoke Test 4: IcebergPhysicsEvaluator runs a full 3-day simulation."""
    provider = HistoricalEnvironmentProvider(
        era5_source=synthetic_smoke_era5,
        copernicus_source=synthetic_smoke_glorys,
        default_sea_ice_concentration=0.05,
        draft_meters=200.0,
    )
    
    evaluator = IcebergPhysicsEvaluator(
        environment_provider=provider,
        default_properties=create_default_iceberg_properties(),
        dt_seconds=3600.0,  # 1-hour steps for speed
    )
    
    pair = EvaluationPair(
        iceberg_id="SMOKE_TEST",
        prediction_time=pd.Timestamp("2020-01-01"),
        previous_observation_time=pd.Timestamp("2019-12-31"),
        target_time=pd.Timestamp("2020-01-04"),
        horizon_days=3,
        initial_latitude=-76.0,
        initial_longitude=-42.0,
        previous_latitude=-76.1,
        previous_longitude=-42.1,
        target_latitude=-75.9,
        target_longitude=-41.9,
        initial_position_source="synthetic",
        previous_position_source="synthetic",
        target_position_source="synthetic",
        initial_is_raw=True,
        previous_is_raw=True,
        target_is_raw=True,
        initial_vx_mps=0.05,
        initial_vy_mps=0.03,
        initial_speed_mps=0.058,
        initial_bearing_deg=30.0,
    )
    
    result = evaluator.evaluate_pair(pair)
    
    assert result.iceberg_id == "SMOKE_TEST"
    assert result.horizon_days == 3
    assert np.isfinite(result.predicted_latitude)
    assert np.isfinite(result.predicted_longitude)
    assert np.isfinite(result.geodesic_error_km)
    assert result.geodesic_error_km >= 0.0
    print("✓ IcebergPhysicsEvaluator (3-day simulation): OK")


def test_smoke_baseline_evaluator_integration():
    """Smoke Test 5: ConstantVelocityBaselineEvaluator computes baseline."""
    evaluator = ConstantVelocityBaselineEvaluator()
    
    pair = EvaluationPair(
        iceberg_id="SMOKE_TEST",
        prediction_time=pd.Timestamp("2020-01-01"),
        previous_observation_time=pd.Timestamp("2019-12-31"),
        target_time=pd.Timestamp("2020-01-04"),
        horizon_days=3,
        initial_latitude=-76.0,
        initial_longitude=-42.0,
        previous_latitude=-76.1,
        previous_longitude=-42.1,
        target_latitude=-75.9,
        target_longitude=-41.9,
        initial_position_source="synthetic",
        previous_position_source="synthetic",
        target_position_source="synthetic",
        initial_is_raw=True,
        previous_is_raw=True,
        target_is_raw=True,
        initial_vx_mps=0.05,
        initial_vy_mps=0.03,
        initial_speed_mps=0.058,
        initial_bearing_deg=30.0,
    )
    
    result = evaluator.evaluate_pair(pair)
    
    assert result.iceberg_id == "SMOKE_TEST"
    assert result.horizon_days == 3
    assert np.isfinite(result.geodesic_error_km)
    print("✓ ConstantVelocityBaselineEvaluator: OK")


def test_smoke_multi_file_environmental_loading(synthetic_smoke_era5, synthetic_smoke_glorys, tmp_path):
    """Smoke Test 6: Multi-file environmental loading via open_environmental_sources."""
    # Save two ERA5 chunks
    era5_1 = synthetic_smoke_era5.isel(time=slice(0, 2))
    era5_2 = synthetic_smoke_era5.isel(time=slice(1, 3))
    
    era5_1_path = tmp_path / "era5_2020_01_01_02.nc"
    era5_2_path = tmp_path / "era5_2020_01_02_03.nc"
    
    era5_1.to_netcdf(era5_1_path)
    era5_2.to_netcdf(era5_2_path)
    
    # Load combined dataset
    combined_era5 = open_environmental_sources([era5_1_path, era5_2_path])
    
    assert "u10" in combined_era5.data_vars
    assert combined_era5.sizes["time"] >= 3
    print("✓ Multi-file environmental loading: OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
