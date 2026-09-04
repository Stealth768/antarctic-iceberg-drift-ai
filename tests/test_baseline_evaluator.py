"""
Unit tests for Constant-Velocity Baseline Evaluator (Stage 5B).

Verifies:
1. Constant velocity prediction on synthetic trajectory (zero error when motion is truly linear).
2. Geodesic error calculation accuracy.
3. T+3 day horizon evaluation.
4. T+4 day horizon evaluation.
5. Strict absence of future data leakage into prediction.
6. Safe handling of empty, single-point, or invalid trajectory inputs.
7. Correct calculation of aggregate metrics (mean, median, RMSE, P90, max, min).
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.baseline_evaluator import (
    BaselineEvaluationReport,
    BaselinePredictionResult,
    ConstantVelocityBaselineEvaluator,
    HorizonMetrics,
    compute_horizon_metrics,
)
from src.evaluation.historical_pairs import (
    EvaluationPair,
    build_evaluation_pairs,
    calculate_geodesic_error_km,
)
from src.models.iceberg_physics import CoordinateHandler


@pytest.fixture
def evaluator() -> ConstantVelocityBaselineEvaluator:
    return ConstantVelocityBaselineEvaluator(crs="EPSG:3412")


@pytest.fixture
def linear_synthetic_trajectory() -> pd.DataFrame:
    """
    Synthetic trajectory where an iceberg moves at EXACT constant velocity in projected space:
    vx = 0.20 m/s, vy = 0.0 m/s over 10 consecutive daily observations.
    """
    coord = CoordinateHandler(crs="EPSG:3412")
    x0, y0 = -1000000.0, 500000.0
    vx = 0.20
    vy = 0.0

    records = []
    base_time = pd.Timestamp("2020-01-01 00:00:00+00:00")
    for d in range(10):
        t = base_time + pd.Timedelta(days=d)
        dt_sec = d * 86400.0
        x = x0 + vx * dt_sec
        y = y0 + vy * dt_sec
        lon, lat = coord.to_geographic(x, y)
        records.append({
            "iceberg_id": "SYNTH_LINEAR",
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


def test_constant_velocity_prediction_synthetic(evaluator, linear_synthetic_trajectory):
    """
    Test 1: When real motion is exactly constant velocity, the baseline predictor
    should produce near-zero geodesic displacement error (within numerical floating precision).
    """
    report = evaluator.evaluate_trajectory(linear_synthetic_trajectory, horizons=[3, 4])

    m3 = report.overall_metrics[3]
    m4 = report.overall_metrics[4]

    assert m3.num_cases > 0
    assert m4.num_cases > 0

    # Geodesic errors should be negligible (< 1 meter = 0.001 km)
    assert m3.mean_error_km < 0.01
    assert m3.max_error_km < 0.01
    assert m4.mean_error_km < 0.01
    assert m4.max_error_km < 0.01


def test_geodesic_error_calculation(evaluator):
    """
    Test 2: Geodesic error is calculated accurately on WGS84 ellipsoid.
    """
    # Create an artificial EvaluationPair where actual target is 111.57 km away from prediction
    t0 = pd.Timestamp("2020-01-01 00:00:00+00:00")
    pair = EvaluationPair(
        iceberg_id="TEST",
        prediction_time=t0,
        previous_observation_time=t0 - pd.Timedelta(days=1),
        target_time=t0 + pd.Timedelta(days=3),
        horizon_days=3,
        initial_latitude=-70.0,
        initial_longitude=0.0,
        previous_latitude=-70.0,
        previous_longitude=0.0,
        target_latitude=-71.0,  # 1 degree south
        target_longitude=0.0,
        initial_position_source="ascat",
        previous_position_source="ascat",
        target_position_source="ascat",
        initial_is_raw=True,
        previous_is_raw=True,
        target_is_raw=True,
        initial_vx_mps=0.0,  # Zero velocity => predicted position remains (-70, 0)
        initial_vy_mps=0.0,
        initial_speed_mps=0.0,
        initial_bearing_deg=0.0,
    )

    res = evaluator.evaluate_pair(pair)
    assert res.predicted_latitude == pytest.approx(-70.0, abs=1e-5)
    assert res.predicted_longitude == pytest.approx(0.0, abs=1e-5)

    # 1 degree latitude near -70S is ~111.57 km
    assert 111.0 < res.geodesic_error_km < 112.0


def test_t_plus_3_evaluation(evaluator, linear_synthetic_trajectory):
    """
    Test 3: T+3 day horizon evaluates exactly at elapsed time dt = 259,200 s.
    """
    report = evaluator.evaluate_trajectory(linear_synthetic_trajectory, horizons=[3])
    preds = report.predictions

    assert len(preds) > 0
    for p in preds:
        assert p.horizon_days == 3
        expected_target_time = p.prediction_time + pd.Timedelta(days=3)
        assert p.target_time == expected_target_time


def test_t_plus_4_evaluation(evaluator, linear_synthetic_trajectory):
    """
    Test 4: T+4 day horizon evaluates exactly at elapsed time dt = 345,600 s.
    """
    report = evaluator.evaluate_trajectory(linear_synthetic_trajectory, horizons=[4])
    preds = report.predictions

    assert len(preds) > 0
    for p in preds:
        assert p.horizon_days == 4
        expected_target_time = p.prediction_time + pd.Timedelta(days=4)
        assert p.target_time == expected_target_time


def test_no_future_leakage_during_evaluation(evaluator, linear_synthetic_trajectory):
    """
    Test 5: Future target position modification changes the evaluated error,
    but does NOT alter predicted_latitude or predicted_longitude.
    """
    # 1. Evaluate baseline trajectory
    report1 = evaluator.evaluate_trajectory(linear_synthetic_trajectory, horizons=[3])
    pred1 = report1.predictions[0]

    # 2. Corrupt future target of the same trajectory
    corrupted_df = linear_synthetic_trajectory.copy()
    # Modify target at Day 5 (index 4)
    corrupted_df.loc[4, "latitude"] = -55.0
    corrupted_df.loc[4, "longitude"] = 20.0

    report2 = evaluator.evaluate_trajectory(corrupted_df, horizons=[3])
    pred2 = report2.predictions[0]

    # Predictions MUST be identical
    assert pred1.predicted_latitude == pred2.predicted_latitude
    assert pred1.predicted_longitude == pred2.predicted_longitude
    assert pred1.initial_vx_mps == pred2.initial_vx_mps

    # Only the evaluated error changes
    assert pred1.geodesic_error_km != pred2.geodesic_error_km


def test_empty_and_invalid_cases_handled_safely(evaluator):
    """
    Test 6: Empty DataFrames, single observations, or empty pairs return zero cases cleanly.
    """
    empty_df = pd.DataFrame()
    report_empty = evaluator.evaluate_trajectory(empty_df)
    assert len(report_empty.predictions) == 0
    assert report_empty.summary_table().empty

    # Single-observation trajectory (cannot estimate velocity)
    single_df = pd.DataFrame([{
        "iceberg_id": "ONE",
        "timestamp": pd.Timestamp("2020-01-01 00:00:00+00:00"),
        "latitude": -70.0,
        "longitude": -50.0,
        "is_raw_observation": True,
    }])
    report_single = evaluator.evaluate_trajectory(single_df)
    assert len(report_single.predictions) == 0

    # compute_horizon_metrics with empty list
    m_empty = compute_horizon_metrics([], horizon_days=3)
    assert m_empty.num_cases == 0
    assert m_empty.mean_error_km == 0.0


def test_metrics_computation_p90_rmse():
    """
    Test 7: Verify statistical metrics (mean, median, RMSE, P90, max, min).
    """
    errors = [10.0, 20.0, 30.0, 40.0, 50.0]
    m = compute_horizon_metrics(errors, horizon_days=3)

    assert m.num_cases == 5
    assert m.mean_error_km == 30.0
    assert m.median_error_km == 30.0
    assert m.min_error_km == 10.0
    assert m.max_error_km == 50.0
    # RMSE of [10, 20, 30, 40, 50] = sqrt((100 + 400 + 900 + 1600 + 2500) / 5) = sqrt(1100) = 33.1662
    expected_rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    assert pytest.approx(m.rmse_km, rel=1e-5) == expected_rmse
    # 90th percentile of [10, 20, 30, 40, 50]
    expected_p90 = np.percentile(errors, 90)
    assert pytest.approx(m.p90_error_km, rel=1e-5) == expected_p90


def test_multi_iceberg_evaluate_trajectory_produces_separate_metrics(evaluator, linear_synthetic_trajectory):
    """
    Test 8: Regression test proving evaluate_trajectory() produces distinct, independent
    per-iceberg metrics when given a DataFrame containing multiple iceberg IDs.
    """
    coord = CoordinateHandler(crs="EPSG:3412")
    x0, y0 = 500000.0, -800000.0
    vx2, vy2 = 0.0, 0.40  # Northward motion at 0.4 m/s

    records2 = []
    base_time = pd.Timestamp("2020-01-01 00:00:00+00:00")
    for d in range(10):
        t = base_time + pd.Timedelta(days=d)
        dt_sec = d * 86400.0
        x = x0 + vx2 * dt_sec
        y = y0 + vy2 * dt_sec
        lon, lat = coord.to_geographic(x, y)
        records2.append({
            "iceberg_id": "BERG_TWO",
            "timestamp": t,
            "latitude": lat,
            "longitude": lon,
            "length_km": 20.0,
            "width_km": 10.0,
            "position_source": "ascat",
            "is_raw_observation": True,
            "is_interpolated": False,
            "time_delta_days": 1.0 if d > 0 else np.nan,
        })
    df2 = pd.DataFrame(records2)

    # Combine both icebergs into one trajectory DataFrame
    combined_df = pd.concat([linear_synthetic_trajectory, df2], ignore_index=True)

    report = evaluator.evaluate_trajectory(combined_df, horizons=[3, 4])

    # Must contain entries for BOTH icebergs
    assert "SYNTH_LINEAR" in report.per_iceberg_metrics
    assert "BERG_TWO" in report.per_iceberg_metrics

    # Check that both icebergs have valid cases
    assert report.per_iceberg_metrics["SYNTH_LINEAR"][3].num_cases > 0
    assert report.per_iceberg_metrics["BERG_TWO"][3].num_cases > 0

    # Total cases in overall metrics must equal the sum of per-iceberg cases
    n_total_3 = report.overall_metrics[3].num_cases
    n_sum_3 = (
        report.per_iceberg_metrics["SYNTH_LINEAR"][3].num_cases
        + report.per_iceberg_metrics["BERG_TWO"][3].num_cases
    )
    assert n_total_3 == n_sum_3


def test_duplicate_raw_timestamps_raise_validation_error(linear_synthetic_trajectory):
    """
    Test 9: build_evaluation_pairs() must detect duplicate raw observation timestamps
    and raise a clear ValueError rather than silently overwriting records.
    """
    corrupted_df = linear_synthetic_trajectory.copy()
    # Duplicate the timestamp of row 2 on row 3
    corrupted_df.loc[3, "timestamp"] = corrupted_df.loc[2, "timestamp"]

    with pytest.raises(ValueError, match="Duplicate raw observation timestamps found"):
        build_evaluation_pairs(corrupted_df, horizons=[3, 4])


def test_calendar_day_target_normalization():
    """
    Test 10: Verify that YYYYJJJ-derived timestamps are normalized to exact calendar-day
    boundaries (00:00:00 UTC) such that T + 3 days and T + 4 days target exact calendar-day timestamps.
    """
    from src.data.iceberg import parse_byu_date

    # 1. Verify parse_byu_date produces exact midnight UTC timestamps
    dates_raw = [2020001, 2020002, 2020004, 2020005]  # Days 1, 2, 4, 5
    parsed = [parse_byu_date(d) for d in dates_raw]

    for t in parsed:
        assert t.hour == 0
        assert t.minute == 0
        assert t.second == 0
        assert t.microsecond == 0
        assert t.tzinfo is not None

    # 2. Check that T + 3d from Day 1 lands exactly on Day 4
    t_start = parsed[0]  # 2020-01-01 00:00:00+00:00
    target_3d = t_start + pd.Timedelta(days=3)  # 2020-01-04 00:00:00+00:00
    assert target_3d == parsed[2]

    # 3. Check that T + 4d from Day 1 lands exactly on Day 5
    target_4d = t_start + pd.Timedelta(days=4)  # 2020-01-05 00:00:00+00:00
    assert target_4d == parsed[3]

    # 4. In build_evaluation_pairs, verify pairs form exact calendar-day links
    df = pd.DataFrame([
        {"iceberg_id": "CAL_TEST", "timestamp": parsed[0], "latitude": -70.0, "longitude": 0.0, "is_raw_observation": True, "position_source": "ascat"},
        {"iceberg_id": "CAL_TEST", "timestamp": parsed[1], "latitude": -70.01, "longitude": 0.01, "is_raw_observation": True, "position_source": "ascat"},
        {"iceberg_id": "CAL_TEST", "timestamp": parsed[2], "latitude": -70.04, "longitude": 0.04, "is_raw_observation": True, "position_source": "ascat"},
        {"iceberg_id": "CAL_TEST", "timestamp": parsed[3], "latitude": -70.05, "longitude": 0.05, "is_raw_observation": True, "position_source": "ascat"},
    ])

    pairs = build_evaluation_pairs(df, horizons=[3])
    assert len(pairs) == 1
    p = pairs[0]
    # Starting at Day 2 (index 1), T+3d is Day 5 (index 3)
    assert p.prediction_time == parsed[1]
    assert p.target_time == parsed[3]
    assert (p.target_time - p.prediction_time).total_seconds() == 3 * 86400.0
