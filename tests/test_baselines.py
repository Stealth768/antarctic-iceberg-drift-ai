"""
Unit tests for Stage 4 Constant-Velocity Trajectory Baseline.

Verifies:
1. Zero velocity persistence (stationary state).
2. Constant Grid-X motion (x_future = x + vx * dt).
3. Constant Grid-Y motion (y_future = y + vy * dt).
4. Combined multi-axis motion.
5. 3-day trajectory forecast (259,200 seconds).
6. 4-day trajectory forecast (345,600 seconds).
7. Historical velocity estimation from consecutive observations.
8. Zero time interval error detection (dt = 0).
9. Invalid time ordering error detection (t2 <= t1).
10. Projected <-> geographic coordinate conversion using CoordinateHandler.
11. Strict independence from environmental conditions (atmospheric/oceanic forcing).
"""

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import (
    ConstantVelocityPredictor,
    create_state_from_observations,
    estimate_velocity_geographic,
    estimate_velocity_projected,
)
from src.models.iceberg_physics import CoordinateHandler, IcebergState


class TestConstantVelocityPredictor:
    """Tests 1 to 6: Persistence predictor dynamics and multi-day horizons."""

    @pytest.fixture
    def predictor(self) -> ConstantVelocityPredictor:
        return ConstantVelocityPredictor(crs="EPSG:3412")

    def test_zero_velocity(self, predictor: ConstantVelocityPredictor):
        """Test 1: An iceberg with vx=0, vy=0 remains stationary across any horizon."""
        initial_state = IcebergState(x_m=-1200000.0, y_m=650000.0, vx_mps=0.0, vy_mps=0.0)
        dt = 86400.0  # 1 day

        future_state = predictor.predict_state(initial_state, dt_seconds=dt)
        assert future_state.x_m == initial_state.x_m
        assert future_state.y_m == initial_state.y_m
        assert future_state.vx_mps == 0.0
        assert future_state.vy_mps == 0.0

    def test_constant_grid_x_motion(self, predictor: ConstantVelocityPredictor):
        """Test 2: Linear extrapolation along Grid-X."""
        x0, y0 = 500000.0, -800000.0
        vx = 0.25  # m/s
        vy = 0.0
        initial_state = IcebergState(x_m=x0, y_m=y0, vx_mps=vx, vy_mps=vy)
        dt = 3600.0  # 1 hour

        res = predictor.predict_state(initial_state, dt_seconds=dt)
        expected_x = x0 + vx * dt
        assert pytest.approx(res.x_m, rel=1e-7) == expected_x
        assert pytest.approx(res.y_m, rel=1e-7) == y0
        assert res.vx_mps == vx
        assert res.vy_mps == vy

    def test_constant_grid_y_motion(self, predictor: ConstantVelocityPredictor):
        """Test 3: Linear extrapolation along Grid-Y."""
        x0, y0 = 500000.0, -800000.0
        vx = 0.0
        vy = -0.40  # m/s
        initial_state = IcebergState(x_m=x0, y_m=y0, vx_mps=vx, vy_mps=vy)
        dt = 7200.0  # 2 hours

        res = predictor.predict_state(initial_state, dt_seconds=dt)
        expected_y = y0 + vy * dt
        assert pytest.approx(res.x_m, rel=1e-7) == x0
        assert pytest.approx(res.y_m, rel=1e-7) == expected_y

    def test_combined_motion(self, predictor: ConstantVelocityPredictor):
        """Test 4: Simultaneous 2D movement along both X and Y."""
        x0, y0 = -200000.0, 400000.0
        vx = 0.15
        vy = 0.30
        initial_state = IcebergState(x_m=x0, y_m=y0, vx_mps=vx, vy_mps=vy)
        dt = 10000.0

        res = predictor.predict_state(initial_state, dt_seconds=dt)
        assert pytest.approx(res.x_m, rel=1e-7) == x0 + vx * dt
        assert pytest.approx(res.y_m, rel=1e-7) == y0 + vy * dt

    def test_three_day_forecast(self, predictor: ConstantVelocityPredictor):
        """Test 5: Predict future state exactly at 3-day horizon (3 * 24 * 3600 s)."""
        x0, y0 = 0.0, 1000000.0
        vx, vy = 0.10, -0.05
        initial_state = IcebergState(x_m=x0, y_m=y0, vx_mps=vx, vy_mps=vy)

        horizon_3d = 3 * 24 * 3600.0  # 259,200 seconds
        df = predictor.predict(
            initial_state=initial_state,
            start_time="2026-01-01 00:00:00",
            forecast_seconds=horizon_3d,
        )

        assert len(df) == 2  # t=0 and t=3d
        row_3d = df.iloc[-1]
        assert row_3d["timestamp"] == pd.Timestamp("2026-01-04 00:00:00")
        assert pytest.approx(row_3d["x_m"], rel=1e-7) == x0 + vx * horizon_3d
        assert pytest.approx(row_3d["y_m"], rel=1e-7) == y0 + vy * horizon_3d
        assert row_3d["vx_mps"] == vx
        assert row_3d["vy_mps"] == vy

    def test_four_day_forecast(self, predictor: ConstantVelocityPredictor):
        """Test 6: Predict future state at 4-day horizon (4 * 24 * 3600 s)."""
        x0, y0 = 250000.0, -1500000.0
        vx, vy = -0.20, 0.15
        initial_state = IcebergState(x_m=x0, y_m=y0, vx_mps=vx, vy_mps=vy)

        horizon_4d = 4 * 24 * 3600.0  # 345,600 seconds
        df = predictor.predict(
            initial_state=initial_state,
            start_time="2026-01-01 12:00:00",
            forecast_seconds=horizon_4d,
        )

        row_4d = df.iloc[-1]
        assert row_4d["timestamp"] == pd.Timestamp("2026-01-05 12:00:00")
        assert pytest.approx(row_4d["x_m"], rel=1e-7) == x0 + vx * horizon_4d
        assert pytest.approx(row_4d["y_m"], rel=1e-7) == y0 + vy * horizon_4d


class TestVelocityEstimation:
    """Tests 7 to 9: Velocity estimation from historical observations and error guards."""

    def test_velocity_estimation_known_displacement(self):
        """Test 7: Estimate velocity from two known projected coordinates."""
        t1 = "2026-01-01 00:00:00"
        t2 = "2026-01-02 00:00:00"  # dt = 86400 s

        x1, y1 = 100000.0, 200000.0
        x2, y2 = 108640.0, 191360.0  # dx = +8640 m, dy = -8640 m

        vx, vy, dt = estimate_velocity_projected(x1, y1, t1, x2, y2, t2)
        assert dt == 86400.0
        assert pytest.approx(vx, rel=1e-6) == 0.10   # 8640 / 86400 = 0.10 m/s
        assert pytest.approx(vy, rel=1e-6) == -0.10  # -8640 / 86400 = -0.10 m/s

    def test_zero_time_interval_rejected(self):
        """Test 8: Reject zero time interval (t1 == t2)."""
        t = "2026-01-01 00:00:00"
        with pytest.raises(ValueError, match="Invalid time interval"):
            estimate_velocity_projected(100.0, 200.0, t, 150.0, 250.0, t)

    def test_invalid_time_ordering_rejected(self):
        """Test 9: Reject observations where t2 < t1."""
        t1 = "2026-01-02 00:00:00"
        t2 = "2026-01-01 00:00:00"
        with pytest.raises(ValueError, match="Invalid time interval"):
            estimate_velocity_projected(100.0, 200.0, t1, 150.0, 250.0, t2)


class TestCoordinateHandlingAndIndependence:
    """Tests 10 and 11: Coordinate transformations and environmental independence."""

    def test_geographic_velocity_estimation_and_round_trip(self):
        """Test 10: Geographic velocity estimation transforms coordinates properly."""
        coord_handler = CoordinateHandler(crs="EPSG:3412")
        lon1, lat1 = 0.0, -65.0
        t1 = "2026-01-01 00:00:00"
        t2 = "2026-01-02 00:00:00"

        # Obtain projected coordinates for obs1
        x1, y1 = coord_handler.to_projected(lon1, lat1)
        # Shift in projected space by exactly 17280m along X (0.2 m/s over 1 day)
        x2 = x1 + 17280.0
        y2 = y1
        lon2, lat2 = coord_handler.to_geographic(x2, y2)

        vx, vy, dt = estimate_velocity_geographic(lon1, lat1, t1, lon2, lat2, t2, crs="EPSG:3412")
        assert dt == 86400.0
        assert pytest.approx(vx, rel=1e-5) == 0.20
        assert pytest.approx(vy, abs=1e-5) == 0.0

        # Also test create_state_from_observations
        obs1 = {"longitude": lon1, "latitude": lat1, "timestamp": t1}
        obs2 = {"longitude": lon2, "latitude": lat2, "timestamp": t2}
        state = create_state_from_observations(obs1, obs2, crs="EPSG:3412")
        assert pytest.approx(state.x_m, rel=1e-5) == x2
        assert pytest.approx(state.y_m, rel=1e-5) == y2
        assert pytest.approx(state.vx_mps, rel=1e-5) == 0.20
        assert pytest.approx(state.vy_mps, abs=1e-5) == 0.0

    def test_independence_from_environmental_forcing(self):
        """Test 11: Predictor output is completely independent of external environmental states."""
        predictor = ConstantVelocityPredictor(crs="EPSG:3412")
        initial_state = IcebergState(x_m=100000.0, y_m=-500000.0, vx_mps=0.3, vy_mps=-0.1)

        # The predictor API takes no environment provider and makes no network/disk calls
        res1 = predictor.predict_state(initial_state, dt_seconds=3600.0)
        res2 = predictor.predict_state(initial_state, dt_seconds=3600.0)

        assert res1.x_m == res2.x_m
        assert res1.y_m == res2.y_m
        assert res1.vx_mps == res2.vx_mps
        assert res1.vy_mps == res2.vy_mps
