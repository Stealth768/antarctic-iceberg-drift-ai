"""
Tests for Phase 1 Step 1.1: Physics-Residual ML Dataset Pipeline.

Verifies:
1. Feature extraction (kinematic, atmospheric, oceanic, sea-ice).
2. Projected-coordinate residual calculation (residual_x = x_truth - x_physics).
3. Explicit handling of missing environmental data (no fake/invented values).
4. Strict chronological splitting with zero future data leakage.
5. Strict temporal non-overlap between train, validation, and test partitions.
6. Deterministic dataset construction.
7. End-to-end dataset builder execution on real A-23a ground truth data.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import SyntheticEnvironment
from src.data.environment import MissingDataError
from src.data.copernicus import CopernicusLoader
from src.data.era5 import ERA5Loader
from src.data.nsidc import NSIDCLoader
from src.data.observations import IcebergObservationLoader
from src.data.environment import CompositeEnvironmentProvider
from src.models.iceberg_physics import (
    CoordinateHandler,
    IcebergProperties,
    IcebergState,
    simulate_iceberg,
)
from src.models.ml.features import (
    ALL_FEATURE_NAMES,
    CORE_FEATURE_NAMES,
    ResidualFeatureExtractor,
    TARGET_NAMES,
)
from src.models.ml.dataset import (
    ResidualDataset,
    ResidualSample,
    build_partition_from_samples,
    build_residual_dataset,
    build_residual_samples_from_trajectory,
    chronological_split,
)


def test_feature_extractor_core_values():
    """Verify feature extractor extracts all core environmental & kinematic features."""
    env = SyntheticEnvironment(
        wind_u=5.0,
        wind_v=12.0,
        ocean_u=0.3,
        ocean_v=-0.4,
        sea_ice_concentration=0.85,
    )
    extractor = ResidualFeatureExtractor(crs="EPSG:3412", include_static_properties=False)

    feats = extractor.extract_features(
        timestamp="2020-01-01 12:00:00",
        latitude=-65.0,
        longitude=-45.0,
        physics_vx=0.1,
        physics_vy=0.2,
        dt_seconds=3600.0,
        environment_provider=env,
    )

    assert set(feats.keys()) == set(CORE_FEATURE_NAMES)
    assert feats["wind_u"] == 5.0
    assert feats["wind_v"] == 12.0
    assert feats["wind_speed"] == pytest.approx(13.0, abs=1e-4)  # 5-12-13 triangle
    assert feats["ocean_u"] == 0.3
    assert feats["ocean_v"] == -0.4
    assert feats["ocean_speed"] == pytest.approx(0.5, abs=1e-4)  # 3-4-5 triangle
    assert feats["sea_ice_concentration"] == 0.85
    assert feats["physics_vx"] == 0.1
    assert feats["physics_vy"] == 0.2
    assert feats["physics_speed"] == pytest.approx(np.hypot(0.1, 0.2), abs=1e-4)
    assert feats["dt_seconds"] == 3600.0


def test_projected_residual_calculation():
    """Verify projected residual calculation matches x_truth - x_physics in EPSG:3412."""
    extractor = ResidualFeatureExtractor(crs="EPSG:3412")
    handler = CoordinateHandler(crs="EPSG:3412")

    # Known point
    lon_gt, lat_gt = -45.0, -75.0
    x_gt, y_gt = handler.to_projected(lon_gt, lat_gt)

    # Simulated position offset by 1500m in X and -2500m in Y
    x_sim = x_gt - 1500.0
    y_sim = y_gt + 2500.0

    res_x, res_y = extractor.calculate_projected_residual(
        truth_longitude=lon_gt,
        truth_latitude=lat_gt,
        physics_x_m=x_sim,
        physics_y_m=y_sim,
    )

    assert res_x == pytest.approx(1500.0, abs=1e-3)
    assert res_y == pytest.approx(-2500.0, abs=1e-3)


def test_missing_environmental_values_handled_explicitly():
    """Verify that missing/NaN environmental values raise MissingDataError and are not faked."""
    class BrokenEnvironment(SyntheticEnvironment):
        def get_environment(self, timestamp, latitude, longitude):
            # Missing ocean_u (NaN)
            return {
                "wind_u": 5.0,
                "wind_v": 5.0,
                "ocean_u": np.nan,
                "ocean_v": 0.1,
                "sea_ice_concentration": 0.5,
                "temperature": 260.0,
                "pressure": 100000.0,
                "sst": 271.0,
            }

    broken_env = BrokenEnvironment()
    extractor = ResidualFeatureExtractor(crs="EPSG:3412")

    # Direct feature extraction must raise MissingDataError
    with pytest.raises(MissingDataError, match="Missing ocean current forcing"):
        extractor.extract_features(
            timestamp="2020-01-01",
            latitude=-75.0,
            longitude=-45.0,
            physics_vx=0.0,
            physics_vy=0.0,
            dt_seconds=0.0,
            environment_provider=broken_env,
        )

    # Trajectory extraction with on_missing='skip' must skip cleanly
    times = pd.date_range("2020-01-01", periods=5, freq="1h")
    df_sim = pd.DataFrame({
        "timestamp": times,
        "latitude": np.linspace(-75, -74.9, 5),
        "longitude": np.linspace(-45, -44.9, 5),
        "vx_mps": 0.05,
        "vy_mps": 0.02,
    })
    df_truth = df_sim[["timestamp", "latitude", "longitude"]].copy()

    samples_skipped = build_residual_samples_from_trajectory(
        df_sim=df_sim,
        df_truth=df_truth,
        environment_provider=broken_env,
        on_missing="skip",
    )
    assert len(samples_skipped) == 0

    # With on_missing='raise' it must raise immediately
    with pytest.raises(MissingDataError):
        build_residual_samples_from_trajectory(
            df_sim=df_sim,
            df_truth=df_truth,
            environment_provider=broken_env,
            on_missing="raise",
        )


def test_chronological_splitting_no_leakage():
    """Verify strict chronological ordering and non-overlapping partitions."""
    times = pd.date_range("2020-01-01", periods=100, freq="1h")
    dummy_feats = {name: 1.0 for name in CORE_FEATURE_NAMES}

    samples = [
        ResidualSample(
            timestamp=t,
            iceberg_id="TEST_BERG",
            features=dummy_feats,
            residual_x_m=float(i),
            residual_y_m=float(i * 2),
            physics_x_m=1000.0,
            physics_y_m=2000.0,
            truth_x_m=1000.0 + float(i),
            truth_y_m=2000.0 + float(i * 2),
            dt_seconds=float(i * 3600),
        )
        for i, t in enumerate(times)
    ]

    dataset = chronological_split(samples, train_frac=0.60, val_frac=0.20, test_frac=0.20)

    assert dataset.total_samples == 100
    assert len(dataset.train) == 60
    assert len(dataset.val) == 20
    assert len(dataset.test) == 20

    # Strict chronological non-overlap assertion
    max_train_t = dataset.train.timestamps.max()
    min_val_t = dataset.val.timestamps.min()
    max_val_t = dataset.val.timestamps.max()
    min_test_t = dataset.test.timestamps.min()

    assert max_train_t < min_val_t
    assert max_val_t < min_test_t

    # Verify target shapes and contents
    assert dataset.train.X.shape == (60, len(CORE_FEATURE_NAMES))
    assert dataset.train.y.shape == (60, 2)
    assert dataset.val.y[0, 0] == 60.0  # i=60 residual_x_m
    assert dataset.test.y[0, 0] == 80.0  # i=80 residual_x_m


def test_chronological_split_explicit_timestamps():
    """Verify chronological split using explicit cutoff timestamps."""
    times = pd.date_range("2020-01-01", periods=50, freq="1h")
    dummy_feats = {name: 1.0 for name in CORE_FEATURE_NAMES}

    samples = [
        ResidualSample(
            timestamp=t,
            iceberg_id="TEST_BERG",
            features=dummy_feats,
            residual_x_m=0.0,
            residual_y_m=0.0,
            physics_x_m=0.0,
            physics_y_m=0.0,
            truth_x_m=0.0,
            truth_y_m=0.0,
            dt_seconds=0.0,
        )
        for t in times
    ]

    # Split at hour 20 and hour 35
    t_train_end = times[20]
    t_val_end = times[35]

    dataset = chronological_split(
        samples,
        train_end_time=t_train_end,
        val_end_time=t_val_end,
    )

    assert dataset.train.timestamps.max() <= t_train_end
    assert dataset.val.timestamps.min() > t_train_end
    assert dataset.val.timestamps.max() <= t_val_end
    assert dataset.test.timestamps.min() > t_val_end


def test_deterministic_dataset_construction():
    """Verify building dataset multiple times produces identical arrays."""
    times = pd.date_range("2020-01-01", periods=20, freq="1h")
    env = SyntheticEnvironment(wind_u=2.0, wind_v=3.0, ocean_u=0.1, ocean_v=0.1)

    df_sim = pd.DataFrame({
        "timestamp": times,
        "latitude": np.linspace(-75.0, -74.8, 20),
        "longitude": np.linspace(-45.0, -44.8, 20),
        "vx_mps": 0.05,
        "vy_mps": 0.02,
        "x_m": np.linspace(1000, 2000, 20),
        "y_m": np.linspace(2000, 3000, 20),
    })
    df_truth = df_sim[["timestamp", "latitude", "longitude"]].copy()

    ds1 = build_residual_dataset(df_sim, df_truth, env)
    ds2 = build_residual_dataset(df_sim, df_truth, env)

    assert np.array_equal(ds1.train.X, ds2.train.X)
    assert np.array_equal(ds1.train.y, ds2.train.y)
    assert np.array_equal(ds1.val.X, ds2.val.X)
    assert np.array_equal(ds1.test.X, ds2.test.X)


def test_real_data_pipeline_end_to_end():
    """Verify end-to-end dataset construction using real A23A ground-truth and real test NetCDFs."""
    obs_path = Path("data/raw/observations/a23a_ground_truth.csv")
    assert obs_path.exists(), "Missing real observation file a23a_ground_truth.csv"

    obs_loader = IcebergObservationLoader(obs_path)
    df_truth = obs_loader.load_track()

    glorys_path = Path("data/raw/glorys_test/glorys_a23a_test.nc")
    era5_path = Path("data/raw/era5_test/era5_a23a_real_200001.nc")
    nsidc_path = Path("data/raw/nsidc_test/nsidc_a23a_test.nc")

    env_provider = CompositeEnvironmentProvider(
        nsidc_loader=NSIDCLoader(source=nsidc_path),
        era5_loader=ERA5Loader(source=era5_path),
        copernicus_loader=CopernicusLoader(source=glorys_path),
    )

    coord_handler = CoordinateHandler(crs="EPSG:3412")
    x0, y0 = coord_handler.to_projected(
        longitude=df_truth["longitude"].iloc[0],
        latitude=df_truth["latitude"].iloc[0],
    )
    init_state = IcebergState(x_m=x0, y_m=y0, vx_mps=0.0, vy_mps=0.0)

    props_cal = IcebergProperties(
        mass_kg=1e12,
        length_m=5000.0,
        width_m=2500.0,
        draft_m=200.0,
        air_drag_coefficient=0.2000,
        water_drag_coefficient=1.0065,
    )

    start_time = df_truth["timestamp"].iloc[0]
    end_time = df_truth["timestamp"].iloc[-1]
    duration_sec = (end_time - start_time).total_seconds()

    df_sim = simulate_iceberg(
        initial_state=init_state,
        start_time=start_time,
        duration_seconds=duration_sec,
        dt_seconds=600.0,
        environment_provider=env_provider,
        iceberg_properties=props_cal,
        crs="EPSG:3412",
    )

    dataset = build_residual_dataset(
        df_sim=df_sim,
        df_truth=df_truth,
        environment_provider=env_provider,
        iceberg_id="A23A",
        iceberg_properties=props_cal,
        train_frac=0.60,
        val_frac=0.20,
        test_frac=0.20,
    )

    assert dataset.total_samples == 73
    assert len(dataset.train) == 44
    assert len(dataset.val) == 15
    assert len(dataset.test) == 14

    # Assert no NaNs anywhere in matrices
    assert np.all(np.isfinite(dataset.train.X))
    assert np.all(np.isfinite(dataset.train.y))
    assert np.all(np.isfinite(dataset.val.X))
    assert np.all(np.isfinite(dataset.val.y))
    assert np.all(np.isfinite(dataset.test.X))
    assert np.all(np.isfinite(dataset.test.y))

    # Assert features and targets
    assert dataset.feature_names == ALL_FEATURE_NAMES
    assert dataset.target_names == TARGET_NAMES
