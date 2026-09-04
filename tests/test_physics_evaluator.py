"""
Integration & Regression Tests for Stage 6A Physics Historical Evaluation Harness.

NOTE ON DATA INTEGRITY:
All tests in this suite utilize synthetic environment generators and synthetic
trajectories. They are explicitly designated as SMOKE / INTEGRATION tests
to verify system mechanics, coordinate consistency, and temporal safeguards.
They do NOT constitute empirical scientific validation.
"""

from typing import Any, Dict, Union
import numpy as np
import pandas as pd
import pytest

from src.data.environment import EnvironmentProvider, HistoricalIntegrityViolationError
from src.data.synthetic import SyntheticEnvironment
from src.evaluation.baseline_evaluator import (
    ConstantVelocityBaselineEvaluator,
    HorizonMetrics,
)
from src.evaluation.historical_pairs import EvaluationPair, build_evaluation_pairs
from src.evaluation.physics_evaluator import (
    IcebergPhysicsEvaluator,
    PhysicsEvaluationReport,
    PhysicsPredictionResult,
    create_default_iceberg_properties,
)
from src.models.iceberg_physics import CoordinateHandler, NumericalInstabilityError


@pytest.fixture
def synthetic_env() -> SyntheticEnvironment:
    """Deterministic synthetic ocean/atmosphere environment for testing."""
    return SyntheticEnvironment(
        ocean_u=0.10,   # 0.10 m/s eastward current
        ocean_v=0.00,
        wind_u=8.00,    # 8.0 m/s eastward wind
        wind_v=0.00,
        sea_ice_concentration=0.15,
    )


@pytest.fixture
def sample_evaluation_pair_3d() -> EvaluationPair:
    """Deterministic 3-day evaluation pair."""
    t0 = pd.Timestamp("2020-01-01 00:00:00+00:00")
    return EvaluationPair(
        iceberg_id="SYNTH_BERG_01",
        prediction_time=t0,
        previous_observation_time=t0 - pd.Timedelta(days=1),
        target_time=t0 + pd.Timedelta(days=3),
        horizon_days=3,
        initial_latitude=-68.0,
        initial_longitude=0.0,
        previous_latitude=-68.0,
        previous_longitude=-0.1,
        target_latitude=-67.8,
        target_longitude=0.3,
        initial_position_source="ascat",
        previous_position_source="ascat",
        target_position_source="ascat",
        initial_is_raw=True,
        previous_is_raw=True,
        target_is_raw=True,
        initial_vx_mps=0.05,
        initial_vy_mps=0.02,
        initial_speed_mps=float(np.hypot(0.05, 0.02)),
        initial_bearing_deg=68.2,
    )


@pytest.fixture
def sample_evaluation_pair_4d() -> EvaluationPair:
    """Deterministic 4-day evaluation pair."""
    t0 = pd.Timestamp("2020-01-01 00:00:00+00:00")
    return EvaluationPair(
        iceberg_id="SYNTH_BERG_01",
        prediction_time=t0,
        previous_observation_time=t0 - pd.Timedelta(days=1),
        target_time=t0 + pd.Timedelta(days=4),
        horizon_days=4,
        initial_latitude=-68.0,
        initial_longitude=0.0,
        previous_latitude=-68.0,
        previous_longitude=-0.1,
        target_latitude=-67.7,
        target_longitude=0.4,
        initial_position_source="ascat",
        previous_position_source="ascat",
        target_position_source="ascat",
        initial_is_raw=True,
        previous_is_raw=True,
        target_is_raw=True,
        initial_vx_mps=0.05,
        initial_vy_mps=0.02,
        initial_speed_mps=float(np.hypot(0.05, 0.02)),
        initial_bearing_deg=68.2,
    )


@pytest.fixture
def multi_point_synthetic_trajectory() -> pd.DataFrame:
    """10-day synthetic trajectory for multi-step trajectory evaluation."""
    coord = CoordinateHandler("EPSG:3412")
    x0, y0 = 0.0, 2000000.0  # Approx -72 deg S, 0 deg E
    records = []
    base_time = pd.Timestamp("2020-01-01 00:00:00+00:00")
    for d in range(10):
        t = base_time + pd.Timedelta(days=d)
        dt_sec = d * 86400.0
        x = x0 + 0.15 * dt_sec
        y = y0 + 0.05 * dt_sec
        lon, lat = coord.to_geographic(x, y)
        records.append({
            "iceberg_id": "SYNTH_TRACK",
            "timestamp": t,
            "latitude": lat,
            "longitude": lon,
            "length_km": 30.0,
            "width_km": 15.0,
            "position_source": "ascat",
            "is_raw_observation": True,
            "is_interpolated": False,
            "time_delta_days": 1.0 if d > 0 else np.nan,
        })
    return pd.DataFrame(records)


def test_synthetic_3_day_physics_evaluation(synthetic_env, sample_evaluation_pair_3d):
    """
    Test 1: Run 3-day physics evaluation using synthetic forcing.
    Verifies that the evaluator integrates the ODE and produces a valid result.
    """
    evaluator = IcebergPhysicsEvaluator(
        environment_provider=synthetic_env,
        dt_seconds=900.0,  # 15-minute integration timestep for fast smoke test
    )
    result = evaluator.evaluate_pair(sample_evaluation_pair_3d)

    assert isinstance(result, PhysicsPredictionResult)
    assert result.horizon_days == 3
    assert result.iceberg_id == "SYNTH_BERG_01"
    assert result.target_time == sample_evaluation_pair_3d.prediction_time + pd.Timedelta(days=3)


def test_synthetic_4_day_physics_evaluation(synthetic_env, sample_evaluation_pair_4d):
    """
    Test 2: Run 4-day physics evaluation using synthetic forcing.
    Verifies that the evaluator integrates the ODE across 4 full calendar days.
    """
    evaluator = IcebergPhysicsEvaluator(
        environment_provider=synthetic_env,
        dt_seconds=900.0,
    )
    result = evaluator.evaluate_pair(sample_evaluation_pair_4d)

    assert isinstance(result, PhysicsPredictionResult)
    assert result.horizon_days == 4
    assert result.target_time == sample_evaluation_pair_4d.prediction_time + pd.Timedelta(days=4)


def test_finite_prediction_coordinates(synthetic_env, sample_evaluation_pair_3d):
    """
    Test 3: Predicted coordinates and velocities must be finite and physically plausible.
    """
    evaluator = IcebergPhysicsEvaluator(
        environment_provider=synthetic_env,
        dt_seconds=900.0,
    )
    result = evaluator.evaluate_pair(sample_evaluation_pair_3d)

    assert np.isfinite(result.predicted_latitude)
    assert np.isfinite(result.predicted_longitude)
    assert np.isfinite(result.predicted_vx_mps)
    assert np.isfinite(result.predicted_vy_mps)
    assert np.isfinite(result.predicted_speed_mps)

    # Southern hemisphere polar coordinates check
    assert -90.0 <= result.predicted_latitude <= -50.0
    assert -180.0 <= result.predicted_longitude <= 180.0
    assert result.predicted_speed_mps >= 0.0


def test_finite_geodesic_error(synthetic_env, sample_evaluation_pair_3d):
    """
    Test 4: Geodesic error calculation on WGS84 ellipsoid must return a non-negative finite float.
    """
    evaluator = IcebergPhysicsEvaluator(
        environment_provider=synthetic_env,
        dt_seconds=900.0,
    )
    result = evaluator.evaluate_pair(sample_evaluation_pair_3d)

    assert np.isfinite(result.geodesic_error_km)
    assert result.geodesic_error_km >= 0.0


def test_historical_cutoff_violation_is_rejected(sample_evaluation_pair_3d):
    """
    Test 5: Historical integrity cutoff violation must be strictly rejected.
    Two violation modes:
    A) Prediction origin time exceeds provider cutoff.
    B) Forecast integration attempts to query environment beyond provider cutoff.
    """
    # Case A: Origin time T (2020-01-01) is after provider cutoff (2019-12-31)
    env_past_cutoff = SyntheticEnvironment(max_allowed_timestamp="2019-12-31 00:00:00+00:00")
    evaluator_a = IcebergPhysicsEvaluator(environment_provider=env_past_cutoff, dt_seconds=900.0)

    with pytest.raises(HistoricalIntegrityViolationError, match="exceeds environment provider"):
        evaluator_a.evaluate_pair(sample_evaluation_pair_3d)

    # Case B: Origin time is valid (2020-01-01), but cutoff is at 2020-01-02 (mid-forecast)
    env_mid_forecast = SyntheticEnvironment(max_allowed_timestamp="2020-01-02 00:00:00+00:00")
    evaluator_b = IcebergPhysicsEvaluator(environment_provider=env_mid_forecast, dt_seconds=900.0)

    with pytest.raises(HistoricalIntegrityViolationError, match="Temporal leak detected"):
        evaluator_b.evaluate_pair(sample_evaluation_pair_3d)


def test_future_target_does_not_influence_initial_state(synthetic_env, sample_evaluation_pair_3d):
    """
    Test 6: Future target coordinate modifications change geodesic error, but must NEVER
    influence predicted state or trajectory evolution.
    """
    evaluator = IcebergPhysicsEvaluator(
        environment_provider=synthetic_env,
        dt_seconds=900.0,
    )
    result1 = evaluator.evaluate_pair(sample_evaluation_pair_3d)

    # Create a corrupted pair with different future target position
    corrupted_pair = EvaluationPair(
        iceberg_id=sample_evaluation_pair_3d.iceberg_id,
        prediction_time=sample_evaluation_pair_3d.prediction_time,
        previous_observation_time=sample_evaluation_pair_3d.previous_observation_time,
        target_time=sample_evaluation_pair_3d.target_time,
        horizon_days=sample_evaluation_pair_3d.horizon_days,
        initial_latitude=sample_evaluation_pair_3d.initial_latitude,
        initial_longitude=sample_evaluation_pair_3d.initial_longitude,
        previous_latitude=sample_evaluation_pair_3d.previous_latitude,
        previous_longitude=sample_evaluation_pair_3d.previous_longitude,
        target_latitude=-60.0,   # Modified target
        target_longitude=30.0,   # Modified target
        initial_position_source="ascat",
        previous_position_source="ascat",
        target_position_source="ascat",
        initial_is_raw=True,
        previous_is_raw=True,
        target_is_raw=True,
        initial_vx_mps=sample_evaluation_pair_3d.initial_vx_mps,
        initial_vy_mps=sample_evaluation_pair_3d.initial_vy_mps,
        initial_speed_mps=sample_evaluation_pair_3d.initial_speed_mps,
        initial_bearing_deg=sample_evaluation_pair_3d.initial_bearing_deg,
    )

    result2 = evaluator.evaluate_pair(corrupted_pair)

    # Physical trajectory MUST be 100% identical
    assert result1.predicted_latitude == pytest.approx(result2.predicted_latitude, abs=1e-9)
    assert result1.predicted_longitude == pytest.approx(result2.predicted_longitude, abs=1e-9)
    assert result1.predicted_vx_mps == pytest.approx(result2.predicted_vx_mps, abs=1e-9)
    assert result1.predicted_vy_mps == pytest.approx(result2.predicted_vy_mps, abs=1e-9)

    # Only evaluation error relative to the target changes
    assert result1.geodesic_error_km != result2.geodesic_error_km


class CorruptedEnvironmentProvider(EnvironmentProvider):
    """Test provider that deliberately injects NaN or infinite environmental forcing."""

    def get_environment(self, timestamp: Any, latitude: float, longitude: float) -> Dict[str, float]:
        return {
            "ocean_u": np.nan,  # Corrupted NaN current
            "ocean_v": 0.1,
            "wind_u": 5.0,
            "wind_v": 0.0,
            "sea_ice_concentration": 0.0,
            "sst": -1.0,
            "temperature": -15.0,
            "pressure": 98500.0,
        }


def test_invalid_non_finite_environmental_values_fail_clearly(sample_evaluation_pair_3d):
    """
    Test 7: Non-finite environmental data must trigger NumericalInstabilityError,
    preventing silent numerical corruption or invalid predictions.
    """
    corrupted_env = CorruptedEnvironmentProvider()
    evaluator = IcebergPhysicsEvaluator(
        environment_provider=corrupted_env,
        dt_seconds=900.0,
    )

    with pytest.raises(NumericalInstabilityError):
        evaluator.evaluate_pair(sample_evaluation_pair_3d)


def test_physics_report_produces_same_metric_schema_as_baseline_report(
    synthetic_env, multi_point_synthetic_trajectory
):
    """
    Test 8: PhysicsEvaluationReport must conform to the exact metric schema
    produced by BaselineEvaluationReport.
    """
    physics_evaluator = IcebergPhysicsEvaluator(
        environment_provider=synthetic_env,
        dt_seconds=1800.0,  # Fast step for multi-point trajectory
    )
    baseline_evaluator = ConstantVelocityBaselineEvaluator()

    physics_report = physics_evaluator.evaluate_trajectory(
        multi_point_synthetic_trajectory, horizons=[3, 4]
    )
    baseline_report = baseline_evaluator.evaluate_trajectory(
        multi_point_synthetic_trajectory, horizons=[3, 4]
    )

    assert isinstance(physics_report, PhysicsEvaluationReport)

    # Compare summary table columns
    p_summary = physics_report.summary_table()
    b_summary = baseline_report.summary_table()

    assert list(p_summary.columns) == list(b_summary.columns)
    assert len(p_summary) == 2  # Horizons 3 and 4

    # Check metrics fields
    m3 = physics_report.overall_metrics[3]
    assert isinstance(m3, HorizonMetrics)
    assert m3.num_cases > 0
    assert np.isfinite(m3.mean_error_km)
    assert np.isfinite(m3.median_error_km)
    assert np.isfinite(m3.rmse_km)
    assert np.isfinite(m3.p90_error_km)
    assert np.isfinite(m3.max_error_km)
    assert np.isfinite(m3.min_error_km)
