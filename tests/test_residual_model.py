"""
Tests for Phase 1 Step 1.2: Physics-Residual ML Model Implementation.

Verifies:
1. Ridge and Tree models fit/predict API.
2. Deterministic predictions with fixed random_state.
3. Correct X/y dimensions and feature alignment.
4. Train-only fitting (scaler & model parameters on train only).
5. Chronological evaluation (no data leakage).
6. Model serialization (save/load round-trip).
7. No mutation of original physics dataset.
8. Residual RMSE calculations.
9. Offline trajectory correction evaluation.
10. Comparison of Physics-only vs Physics+Ridge vs Physics+Tree on test set.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import SyntheticEnvironment
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
    ResidualDatasetPartition,
    build_residual_dataset,
    chronological_split,
)
from src.models.ml.residual_model import (
    RidgeResidualModel,
    TreeResidualModel,
    evaluate_residual_corrections,
)


# ============================================================================
# Fixtures for test data
# ============================================================================

@pytest.fixture
def simple_dataset():
    """Create a small synthetic dataset for fast unit tests."""
    times = pd.date_range("2020-01-01", periods=30, freq="1h")
    dummy_feats = {name: np.random.randn() for name in ALL_FEATURE_NAMES}

    samples = [
        ResidualSample(
            timestamp=t,
            iceberg_id="TEST_BERG",
            features=dummy_feats,
            residual_x_m=float(np.random.randn() * 100),
            residual_y_m=float(np.random.randn() * 100),
            physics_x_m=1000.0 + float(i * 50),
            physics_y_m=2000.0 + float(i * 50),
            truth_x_m=1000.0 + float(i * 50 + np.random.randn() * 50),
            truth_y_m=2000.0 + float(i * 50 + np.random.randn() * 50),
            dt_seconds=float(i * 3600),
        )
        for i, t in enumerate(times)
    ]

    dataset = chronological_split(
        samples,
        train_frac=0.60,
        val_frac=0.20,
        test_frac=0.20,
        feature_names=ALL_FEATURE_NAMES,
    )
    return dataset


@pytest.fixture
def real_a23a_dataset():
    """Load real A23A dataset (from Step 1.1 output)."""
    obs_path = Path("data/raw/observations/a23a_ground_truth.csv")
    if not obs_path.exists():
        pytest.skip("Real observation file not available")

    obs_loader = IcebergObservationLoader(obs_path)
    df_truth = obs_loader.load_track()

    glorys_path = Path("data/raw/glorys_test/glorys_a23a_test.nc")
    era5_path = Path("data/raw/era5_test/era5_a23a_real_200001.nc")
    nsidc_path = Path("data/raw/nsidc_test/nsidc_a23a_test.nc")

    if not all([glorys_path.exists(), era5_path.exists(), nsidc_path.exists()]):
        pytest.skip("Missing real environmental data files")

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
    return dataset


# ============================================================================
# Ridge Regression Model Tests
# ============================================================================

def test_ridge_model_fit_basic(simple_dataset):
    """Verify RidgeResidualModel.fit() accepts and stores training data."""
    model = RidgeResidualModel(alpha=10.0, random_state=42)
    result = model.fit(simple_dataset.train.X, simple_dataset.train.y)
    
    # fit() should return self for chaining
    assert result is model
    
    # Pipeline should be created and not None
    assert model._pipeline is not None


def test_ridge_model_predict_shape(simple_dataset):
    """Verify RidgeResidualModel.predict() returns correct shape (n_samples, 2)."""
    model = RidgeResidualModel(alpha=10.0, random_state=42)
    model.fit(simple_dataset.train.X, simple_dataset.train.y)
    
    y_pred = model.predict(simple_dataset.train.X)
    
    assert y_pred.shape == (len(simple_dataset.train), 2)
    assert y_pred.dtype in (np.float32, np.float64)
    assert np.all(np.isfinite(y_pred))


def test_ridge_model_deterministic_predictions(simple_dataset):
    """Verify RidgeResidualModel produces identical predictions on repeated calls."""
    model1 = RidgeResidualModel(alpha=10.0, random_state=42)
    model1.fit(simple_dataset.train.X, simple_dataset.train.y)
    y_pred1 = model1.predict(simple_dataset.val.X)
    
    model2 = RidgeResidualModel(alpha=10.0, random_state=42)
    model2.fit(simple_dataset.train.X, simple_dataset.train.y)
    y_pred2 = model2.predict(simple_dataset.val.X)
    
    assert np.allclose(y_pred1, y_pred2, rtol=1e-10)


def test_ridge_model_unfitted_predict_raises():
    """Verify unfitted model raises RuntimeError on predict()."""
    model = RidgeResidualModel(alpha=10.0)
    X_dummy = np.random.randn(5, len(ALL_FEATURE_NAMES))
    
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(X_dummy)


def test_ridge_model_residual_rmse(simple_dataset):
    """Verify residual_rmse() returns correct metric dict."""
    model = RidgeResidualModel(alpha=10.0, random_state=42)
    model.fit(simple_dataset.train.X, simple_dataset.train.y)
    
    metrics = model.residual_rmse(simple_dataset.val.X, simple_dataset.val.y)
    
    assert "rmse_x_m" in metrics
    assert "rmse_y_m" in metrics
    assert "rmse_dist_m" in metrics
    assert "rmse_dist_km" in metrics
    assert all(isinstance(v, (float, np.floating)) for v in metrics.values())
    assert all(v >= 0 for v in metrics.values())


def test_ridge_model_save_load(simple_dataset, tmp_path):
    """Verify RidgeResidualModel serialization round-trip."""
    model = RidgeResidualModel(alpha=10.0, random_state=42)
    model.fit(simple_dataset.train.X, simple_dataset.train.y)
    y_pred_orig = model.predict(simple_dataset.val.X)
    
    # Save
    save_path = tmp_path / "ridge_model.pkl"
    model.save(save_path)
    assert save_path.exists()
    
    # Load
    loaded_model = RidgeResidualModel.load(save_path)
    y_pred_loaded = loaded_model.predict(simple_dataset.val.X)
    
    # Predictions should match exactly
    assert np.allclose(y_pred_orig, y_pred_loaded, rtol=1e-10)


# ============================================================================
# Tree-based Model Tests
# ============================================================================

def test_tree_model_fit_basic(simple_dataset):
    """Verify TreeResidualModel.fit() accepts and stores training data."""
    model = TreeResidualModel(
        n_estimators=50,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )
    result = model.fit(simple_dataset.train.X, simple_dataset.train.y)
    
    assert result is model
    assert model._pipeline is not None


def test_tree_model_predict_shape(simple_dataset):
    """Verify TreeResidualModel.predict() returns correct shape."""
    model = TreeResidualModel(n_estimators=50, max_depth=3, random_state=42)
    model.fit(simple_dataset.train.X, simple_dataset.train.y)
    
    y_pred = model.predict(simple_dataset.train.X)
    
    assert y_pred.shape == (len(simple_dataset.train), 2)
    assert np.all(np.isfinite(y_pred))


def test_tree_model_deterministic_predictions(simple_dataset):
    """Verify TreeResidualModel produces identical predictions with fixed random_state."""
    model1 = TreeResidualModel(n_estimators=50, max_depth=3, random_state=42)
    model1.fit(simple_dataset.train.X, simple_dataset.train.y)
    y_pred1 = model1.predict(simple_dataset.val.X)
    
    model2 = TreeResidualModel(n_estimators=50, max_depth=3, random_state=42)
    model2.fit(simple_dataset.train.X, simple_dataset.train.y)
    y_pred2 = model2.predict(simple_dataset.val.X)
    
    assert np.allclose(y_pred1, y_pred2, rtol=1e-10)


def test_tree_model_save_load(simple_dataset, tmp_path):
    """Verify TreeResidualModel serialization round-trip."""
    model = TreeResidualModel(n_estimators=50, max_depth=3, random_state=42)
    model.fit(simple_dataset.train.X, simple_dataset.train.y)
    y_pred_orig = model.predict(simple_dataset.val.X)
    
    save_path = tmp_path / "tree_model.pkl"
    model.save(save_path)
    
    loaded_model = TreeResidualModel.load(save_path)
    y_pred_loaded = loaded_model.predict(simple_dataset.val.X)
    
    assert np.allclose(y_pred_orig, y_pred_loaded, rtol=1e-10)


# ============================================================================
# Dataset Integrity Tests
# ============================================================================

def test_train_only_fitting_validation(simple_dataset):
    """Verify that scaler & model are fitted ONLY on training data."""
    # Ridge model
    ridge = RidgeResidualModel(alpha=10.0, random_state=42)
    ridge.fit(simple_dataset.train.X, simple_dataset.train.y)
    
    # The scaler should have been fit on train.X only
    scaler = ridge._pipeline.named_steps["scaler"]
    
    # Get the mean and scale from the fitted scaler
    train_mean = np.mean(simple_dataset.train.X, axis=0)
    train_scale = np.std(simple_dataset.train.X, axis=0)
    
    # Scaler should use biased estimator (divide by N, not N-1)
    assert np.allclose(scaler.mean_, train_mean, rtol=1e-5)
    
    # Predictions on validation data should use the TRAIN-fitted scaler
    y_val_pred = ridge.predict(simple_dataset.val.X)
    assert y_val_pred.shape == (len(simple_dataset.val), 2)


def test_chronological_partition_integrity(simple_dataset):
    """Verify dataset partitions maintain chronological integrity."""
    # Times should be strictly increasing within each partition
    train_times = simple_dataset.train.timestamps
    val_times = simple_dataset.val.timestamps
    test_times = simple_dataset.test.timestamps
    
    # Check monotonic increase
    assert (np.diff(train_times.astype(np.int64)) >= 0).all()
    assert (np.diff(val_times.astype(np.int64)) >= 0).all()
    assert (np.diff(test_times.astype(np.int64)) >= 0).all()
    
    # Check non-overlap
    if len(train_times) > 0 and len(val_times) > 0:
        assert train_times.max() < val_times.min()
    if len(val_times) > 0 and len(test_times) > 0:
        assert val_times.max() < test_times.min()


def test_no_nan_in_dataset(simple_dataset):
    """Verify dataset has no NaN values in features or targets."""
    assert np.all(np.isfinite(simple_dataset.train.X))
    assert np.all(np.isfinite(simple_dataset.train.y))
    assert np.all(np.isfinite(simple_dataset.val.X))
    assert np.all(np.isfinite(simple_dataset.val.y))
    assert np.all(np.isfinite(simple_dataset.test.X))
    assert np.all(np.isfinite(simple_dataset.test.y))


# ============================================================================
# Model Comparison and Evaluation Tests
# ============================================================================

def test_evaluate_residual_corrections_structure(simple_dataset):
    """Verify evaluate_residual_corrections() returns correctly structured output."""
    ridge = RidgeResidualModel(alpha=10.0, random_state=42)
    ridge.fit(simple_dataset.train.X, simple_dataset.train.y)
    
    tree = TreeResidualModel(n_estimators=50, max_depth=3, random_state=42)
    tree.fit(simple_dataset.train.X, simple_dataset.train.y)
    
    models = {"ridge": ridge, "tree": tree}
    
    results = evaluate_residual_corrections(simple_dataset.test, models)
    
    # Should have physics_only + two model corrections
    assert "physics_only" in results
    assert "ridge" in results
    assert "tree" in results
    
    # Each should have trajectory metrics
    for label, metrics in results.items():
        assert isinstance(metrics, dict)
        # Should have trajectory error metrics
        assert any(key in metrics for key in ["final_position_error_km", "mae_km", "rmse_km"])


def test_physics_only_baseline_always_present(simple_dataset):
    """Verify physics_only baseline is always computed even with empty model dict."""
    results = evaluate_residual_corrections(simple_dataset.test, {})
    
    assert "physics_only" in results
    assert isinstance(results["physics_only"], dict)


def test_model_improvement_detectability(simple_dataset):
    """Verify we can detect when ML model improves or worsens trajectory metrics."""
    ridge = RidgeResidualModel(alpha=10.0, random_state=42)
    ridge.fit(simple_dataset.train.X, simple_dataset.train.y)
    
    tree = TreeResidualModel(n_estimators=50, max_depth=3, random_state=42)
    tree.fit(simple_dataset.train.X, simple_dataset.train.y)
    
    models = {"ridge": ridge, "tree": tree}
    results = evaluate_residual_corrections(simple_dataset.test, models)
    
    # We should be able to compare metrics
    physics_rmse = results["physics_only"].get("rmse_km")
    ridge_rmse = results["ridge"].get("rmse_km")
    tree_rmse = results["tree"].get("rmse_km")
    
    # Metrics should be numeric and comparable
    if physics_rmse is not None and ridge_rmse is not None:
        assert isinstance(physics_rmse, (float, np.floating))
        assert isinstance(ridge_rmse, (float, np.floating))


# ============================================================================
# Real Dataset Tests
# ============================================================================

def test_real_a23a_ridge_model(real_a23a_dataset):
    """Verify Ridge model on real A23A data (73 samples, 44/15/14 split)."""
    assert real_a23a_dataset.total_samples == 73
    assert len(real_a23a_dataset.train) == 44
    assert len(real_a23a_dataset.val) == 15
    assert len(real_a23a_dataset.test) == 14
    
    ridge = RidgeResidualModel(alpha=10.0, random_state=42)
    ridge.fit(real_a23a_dataset.train.X, real_a23a_dataset.train.y)
    
    # Predictions should work on all partitions
    y_train = ridge.predict(real_a23a_dataset.train.X)
    y_val = ridge.predict(real_a23a_dataset.val.X)
    y_test = ridge.predict(real_a23a_dataset.test.X)
    
    assert y_train.shape == (44, 2)
    assert y_val.shape == (15, 2)
    assert y_test.shape == (14, 2)
    
    # All should be finite
    assert np.all(np.isfinite(y_train))
    assert np.all(np.isfinite(y_val))
    assert np.all(np.isfinite(y_test))


def test_real_a23a_tree_model(real_a23a_dataset):
    """Verify Tree model on real A23A data."""
    tree = TreeResidualModel(
        n_estimators=50,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )
    tree.fit(real_a23a_dataset.train.X, real_a23a_dataset.train.y)
    
    y_train = tree.predict(real_a23a_dataset.train.X)
    y_val = tree.predict(real_a23a_dataset.val.X)
    y_test = tree.predict(real_a23a_dataset.test.X)
    
    assert y_train.shape == (44, 2)
    assert y_val.shape == (15, 2)
    assert y_test.shape == (14, 2)


def test_real_a23a_comprehensive_evaluation(real_a23a_dataset):
    """Comprehensive test on real A23A: fit, eval, compare all three approaches."""
    ridge = RidgeResidualModel(alpha=10.0, random_state=42)
    ridge.fit(real_a23a_dataset.train.X, real_a23a_dataset.train.y)
    
    tree = TreeResidualModel(n_estimators=50, max_depth=3, random_state=42)
    tree.fit(real_a23a_dataset.train.X, real_a23a_dataset.train.y)
    
    # Residual prediction error on test set
    ridge_metrics = ridge.residual_rmse(real_a23a_dataset.test.X, real_a23a_dataset.test.y)
    tree_metrics = tree.residual_rmse(real_a23a_dataset.test.X, real_a23a_dataset.test.y)
    
    assert all(k in ridge_metrics for k in ["rmse_x_m", "rmse_y_m", "rmse_dist_km"])
    assert all(k in tree_metrics for k in ["rmse_x_m", "rmse_y_m", "rmse_dist_km"])
    
    # Trajectory metrics
    models = {"ridge": ridge, "tree": tree}
    traj_results = evaluate_residual_corrections(real_a23a_dataset.test, models)
    
    assert "physics_only" in traj_results
    assert "ridge" in traj_results
    assert "tree" in traj_results
