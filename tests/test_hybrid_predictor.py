"""Integration coverage for the physics-plus-residual hybrid predictor."""

from pathlib import Path

import numpy as np
import pytest

from src.data.copernicus import CopernicusLoader
from src.data.environment import CompositeEnvironmentProvider
from src.data.era5 import ERA5Loader
from src.data.nsidc import NSIDCLoader
from src.data.observations import IcebergObservationLoader
from src.metrics.trajectory import calculate_trajectory_metrics
from src.models.iceberg_physics import (
    CoordinateHandler,
    IcebergProperties,
    IcebergState,
    simulate_iceberg,
)
from src.models.ml.dataset import build_residual_dataset
from src.models.ml.hybrid_predictor import (
    HybridIcebergPredictor,
    constant_velocity_baseline,
)
from src.models.ml.residual_model import RidgeResidualModel


@pytest.fixture
def real_a23a_context():
    """Build the real-data integration context, or skip when data is absent."""
    obs_path = Path("data/raw/observations/a23a_ground_truth.csv")
    glorys_path = Path("data/raw/glorys_test/glorys_a23a_test.nc")
    era5_path = Path("data/raw/era5_test/era5_a23a_real_200001.nc")
    nsidc_path = Path("data/raw/nsidc_test/nsidc_a23a_test.nc")

    required_paths = (obs_path, glorys_path, era5_path, nsidc_path)
    if not all(path.exists() for path in required_paths):
        pytest.skip("Real A23A observation and environmental files are not available")

    df_truth = IcebergObservationLoader(obs_path).load_track()
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
    props = IcebergProperties(
        mass_kg=1e12,
        length_m=5000.0,
        width_m=2500.0,
        draft_m=200.0,
        air_drag_coefficient=0.2,
        water_drag_coefficient=1.0065,
    )
    start_time = df_truth["timestamp"].iloc[0]
    duration_sec = (df_truth["timestamp"].iloc[-1] - start_time).total_seconds()
    df_sim = simulate_iceberg(
        initial_state=init_state,
        start_time=start_time,
        duration_seconds=duration_sec,
        dt_seconds=600.0,
        environment_provider=env_provider,
        iceberg_properties=props,
        crs="EPSG:3412",
    )
    dataset = build_residual_dataset(
        df_sim=df_sim,
        df_truth=df_truth,
        environment_provider=env_provider,
        iceberg_id="A23A",
        iceberg_properties=props,
        train_frac=0.60,
        val_frac=0.20,
        test_frac=0.20,
    )
    return {
        "dataset": dataset,
        "df_truth": df_truth,
        "env_provider": env_provider,
        "init_state": init_state,
        "props": props,
        "start_time": start_time,
        "duration_sec": duration_sec,
    }


def test_three_way_comparison_on_real_a23a(real_a23a_context):
    """Compare constant velocity, physics-only, and hybrid on real A23A test data."""
    context = real_a23a_context
    dataset = context["dataset"]
    
    # Train Ridge model
    ridge_model = RidgeResidualModel(alpha=10.0, random_state=42)
    ridge_model.fit(dataset.train.X, dataset.train.y)
    
    # Create predictor
    predictor = HybridIcebergPredictor(
        residual_model=ridge_model,
        residual_bound_m=1000.0,
    )
    
    # Constant velocity baseline
    df_cv = constant_velocity_baseline(context["df_truth"])
    metrics_cv = calculate_trajectory_metrics(df_cv, context["df_truth"])
    
    # Physics-only
    df_physics = predictor.physics_only(
        initial_state=context["init_state"],
        start_time=context["start_time"],
        duration_seconds=context["duration_sec"],
        dt_seconds=600.0,
        environment_provider=context["env_provider"],
        iceberg_properties=context["props"],
    )
    metrics_physics = calculate_trajectory_metrics(df_physics, context["df_truth"])
    
    # Hybrid
    df_hybrid = predictor.predict(
        initial_state=context["init_state"],
        start_time=context["start_time"],
        duration_seconds=context["duration_sec"],
        dt_seconds=600.0,
        environment_provider=context["env_provider"],
        iceberg_properties=context["props"],
        apply_ml_correction=True,
    )
    metrics_hybrid = calculate_trajectory_metrics(df_hybrid, context["df_truth"])
    
    # Verify all models produce valid finite metrics.
    assert np.isfinite(metrics_cv["rmse_km"])
    assert np.isfinite(metrics_physics["rmse_km"])
    assert np.isfinite(metrics_hybrid["rmse_km"])

    # Hybrid should improve on the physics-only trajectory.
    assert metrics_hybrid["rmse_km"] <= metrics_physics["rmse_km"]
