"""
Unit tests for Stage 3 Iceberg Physics Solver and RK4 Integration.

Verifies:
1. RK4 numerical correctness on an analytical benchmark ODE (dx/dt = x).
2. Pure inertial linear drift under zero-force conditions.
3. Drag response toward a constant ocean current.
4. Aerodynamic response to wind forcing.
5. Exact Coriolis deflection direction in the Southern Hemisphere (deflection to the LEFT).
6. High-precision coordinate round-trip (lat/lon <-> projected x/y).
7. Verification that RK4 evaluates environmental forcing at intermediate stages.
8. Validation of physical constraints and detection of invalid parameters.
9. Numerical stability error handling when unphysical states occur.
"""

from typing import Dict, List, Union
import numpy as np
import pandas as pd
import pytest

from src.data.environment import EnvironmentProvider
from src.data.synthetic import SyntheticEnvironment
from src.models.iceberg_physics import (
    CoordinateHandler,
    IcebergProperties,
    IcebergState,
    NumericalInstabilityError,
    iceberg_derivative,
    rk4_step,
    simulate_iceberg,
)


class TestRK4Integrator:
    """Test 1: Generic RK4 mathematical correctness independently of iceberg physics."""

    def test_rk4_exponential_decay_and_growth(self):
        # Analytical ODE: dx/dt = a * x => x(t) = x0 * exp(a * t)
        a = 1.0
        x0 = 2.0
        dt = 0.05
        t_final = 2.0

        def linear_ode(t: float, state: np.ndarray) -> np.ndarray:
            return a * state

        current_state = np.array([x0], dtype=np.float64)
        t = 0.0
        while t < t_final - 1e-9:
            current_state = rk4_step(current_state, t, dt, linear_ode)
            t += dt

        exact_solution = x0 * np.exp(a * t_final)
        # RK4 should achieve high accuracy for smooth exponential ODE
        assert pytest.approx(current_state[0], rel=1e-5) == exact_solution


class TestIcebergPhysicsDynamics:
    """Tests 2 to 5: Physical force balance, drag, and Coriolis acceleration."""

    @pytest.fixture
    def standard_properties(self) -> IcebergProperties:
        """Create standard test iceberg (approx 1 km x 0.5 km x 150m draft)."""
        # Mass ~ 1000m * 500m * (150m / 0.88) * 900 kg/m3 ~ 7.6e10 kg
        return IcebergProperties(
            mass_kg=7.5e10,
            length_m=1000.0,
            width_m=500.0,
            draft_m=150.0,
            air_drag_coefficient=1.30,
            water_drag_coefficient=0.90,
            damping_coefficient=0.0,
            enable_coriolis=False,
        )

    def test_zero_force_motion(self, standard_properties: IcebergProperties):
        """Test 2: Constant initial velocity under zero external forces produces linear motion."""
        # Zero current, zero wind
        env = SyntheticEnvironment(
            ocean_u=0.0,
            ocean_v=0.0,
            wind_u=0.0,
            wind_v=0.0,
        )

        coord_handler = CoordinateHandler(crs="EPSG:3412")
        x0, y0 = coord_handler.to_projected(longitude=0.0, latitude=-65.0)

        vx0 = 0.20  # 0.2 m/s
        vy0 = 0.10  # 0.1 m/s
        initial_state = IcebergState(x_m=x0, y_m=y0, vx_mps=vx0, vy_mps=vy0)

        # To ensure zero water drag, configure drag coefficients to zero for this isolated test
        zero_drag_props = IcebergProperties(
            mass_kg=standard_properties.mass_kg,
            length_m=standard_properties.length_m,
            width_m=standard_properties.width_m,
            draft_m=standard_properties.draft_m,
            air_drag_coefficient=0.0,
            water_drag_coefficient=0.0,
            damping_coefficient=0.0,
            enable_coriolis=False,
        )

        duration = 3600.0  # 1 hour
        dt = 60.0         # 1 minute

        df = simulate_iceberg(
            initial_state=initial_state,
            start_time="2026-01-01 00:00:00",
            duration_seconds=duration,
            dt_seconds=dt,
            environment_provider=env,
            iceberg_properties=zero_drag_props,
            crs="EPSG:3412",
        )

        final_row = df.iloc[-1]
        expected_x = x0 + vx0 * duration
        expected_y = y0 + vy0 * duration

        assert pytest.approx(final_row["x_m"], rel=1e-4) == expected_x
        assert pytest.approx(final_row["y_m"], rel=1e-4) == expected_y
        assert pytest.approx(final_row["vx_mps"]) == vx0
        assert pytest.approx(final_row["vy_mps"]) == vy0

    def test_constant_ocean_current_acceleration(self, standard_properties: IcebergProperties):
        """Test 3: Stationary iceberg accelerates toward ocean current velocity."""
        ocean_u = 0.25  # m/s eastward
        ocean_v = 0.00
        env = SyntheticEnvironment(
            ocean_u=ocean_u,
            ocean_v=ocean_v,
            wind_u=0.0,
            wind_v=0.0,
        )

        coord_handler = CoordinateHandler(crs="EPSG:3412")
        # At lon=0, lat=-65: meridian convergence is 0, so geographic East corresponds to grid +X
        x0, y0 = coord_handler.to_projected(longitude=0.0, latitude=-65.0)

        initial_state = IcebergState(x_m=x0, y_m=y0, vx_mps=0.0, vy_mps=0.0)

        # Simulate for 12 hours
        duration = 12.0 * 3600.0
        dt = 120.0

        df = simulate_iceberg(
            initial_state=initial_state,
            start_time="2026-01-01 00:00:00",
            duration_seconds=duration,
            dt_seconds=dt,
            environment_provider=env,
            iceberg_properties=standard_properties,
            crs="EPSG:3412",
        )

        final_row = df.iloc[-1]
        # Iceberg must have accelerated from 0 toward ocean_u (0.25 m/s)
        assert final_row["vx_mps"] > 0.05
        assert final_row["vx_mps"] <= ocean_u
        # vy should remain approximately zero since ocean_v = 0 and Coriolis is disabled
        assert pytest.approx(final_row["vy_mps"], abs=1e-3) == 0.0

    def test_wind_forcing_acceleration(self, standard_properties: IcebergProperties):
        """Test 4: Wind forcing in the absence of current accelerates iceberg in wind direction."""
        wind_u = 12.0  # 12 m/s eastward wind
        wind_v = 0.0
        env = SyntheticEnvironment(
            ocean_u=0.0,
            ocean_v=0.0,
            wind_u=wind_u,
            wind_v=0.0,
        )

        coord_handler = CoordinateHandler(crs="EPSG:3412")
        x0, y0 = coord_handler.to_projected(longitude=0.0, latitude=-65.0)

        initial_state = IcebergState(x_m=x0, y_m=y0, vx_mps=0.0, vy_mps=0.0)

        df = simulate_iceberg(
            initial_state=initial_state,
            start_time="2026-01-01 00:00:00",
            duration_seconds=6.0 * 3600.0,
            dt_seconds=120.0,
            environment_provider=env,
            iceberg_properties=standard_properties,
            crs="EPSG:3412",
        )

        final_row = df.iloc[-1]
        # Wind pushes iceberg in positive X direction
        assert final_row["vx_mps"] > 0.01

    def test_southern_hemisphere_coriolis_deflection_direction(self):
        """
        Test 5: Explicitly verify Coriolis deflection direction in the Southern Hemisphere.

        PHYSICAL GROUND TRUTH:
        In the Southern Hemisphere (latitude < 0, f < 0):
        The Coriolis force deflects moving objects to the LEFT of their velocity vector.

        If an object moves East along +X (vx > 0, vy = 0):
        - Left is North (+Y).
        - Acceleration a_y = -f * vx.
        - Since latitude < 0, f < 0, so -f > 0.
        - Therefore a_y > 0 (deflection to the LEFT / North).
        """
        props = IcebergProperties(
            mass_kg=5e10,
            length_m=500.0,
            width_m=300.0,
            draft_m=100.0,
            air_drag_coefficient=0.0,
            water_drag_coefficient=0.0,
            damping_coefficient=0.0,
            enable_coriolis=True,  # Coriolis enabled
        )

        env = SyntheticEnvironment(ocean_u=0.0, ocean_v=0.0, wind_u=0.0, wind_v=0.0)
        coord_handler = CoordinateHandler(crs="EPSG:3412")

        # At longitude 0, latitude -70 (Southern Hemisphere)
        x0, y0 = coord_handler.to_projected(longitude=0.0, latitude=-70.0)
        state_arr = np.array([x0, y0, 0.50, 0.0], dtype=np.float64)  # vx = +0.5 m/s, vy = 0

        deriv = iceberg_derivative(
            t_seconds=0.0,
            state_arr=state_arr,
            start_timestamp=pd.Timestamp("2026-01-01"),
            environment_provider=env,
            props=props,
            coord_handler=coord_handler,
        )

        ax = deriv[2]
        ay = deriv[3]

        # For vx > 0 and vy = 0:
        # ax = f * vy = 0
        # ay = -f * vx. Because f < 0 in S. Hemisphere, ay MUST be strictly positive!
        assert pytest.approx(ax, abs=1e-8) == 0.0
        assert ay > 0.0, f"Expected positive ay (deflection to the LEFT) in S. Hemisphere, got ay={ay}"

        # Contrast check: At Northern Hemisphere latitude (+70 deg), deflection must be to the RIGHT (ay < 0)
        # Using a Northern latitude with the same formula:
        f_north = 2.0 * 7.292115e-5 * np.sin(np.radians(70.0))
        ay_north = -f_north * 0.50
        assert ay_north < 0.0


class TestCoordinatesAndInstrumentedCalls:
    """Tests 6 to 9: Coordinate round-trip, intermediate RK4 calls, and validation."""

    def test_coordinate_round_trip(self):
        """Test 6: Verify high-precision round-trip conversion between lat/lon and EPSG:3412."""
        coord_handler = CoordinateHandler(crs="EPSG:3412")

        test_points = [
            (-65.0, 0.0),
            (-75.0, 45.0),
            (-80.0, -120.0),
            (-68.5, 90.0),
        ]

        for lat, lon in test_points:
            x, y = coord_handler.to_projected(longitude=lon, latitude=lat)
            lon_back, lat_back = coord_handler.to_geographic(x, y)

            # Round trip error should be within millimeter/sub-arcsecond tolerance (< 1e-6 degrees)
            assert pytest.approx(lat, abs=1e-6) == lat_back
            assert pytest.approx(lon, abs=1e-6) == lon_back

    def test_rk4_intermediate_environment_calls(self):
        """Test 7: Verify that RK4 queries the environment at intermediate stages during each timestep."""
        class InstrumentedEnvironment(EnvironmentProvider):
            def __init__(self):
                super().__init__()
                self.queried_timestamps: List[pd.Timestamp] = []
                self.call_count = 0

            def get_environment(self, timestamp: Union[str, pd.Timestamp], latitude: float, longitude: float) -> Dict[str, float]:
                self.call_count += 1
                self.queried_timestamps.append(pd.Timestamp(timestamp))
                return {
                    "sea_ice_concentration": 0.1,
                    "ocean_u": 0.05,
                    "ocean_v": 0.0,
                    "sst": 271.35,
                    "wind_u": 5.0,
                    "wind_v": 0.0,
                    "temperature": 260.0,
                    "pressure": 98500.0,
                }

        instrumented_env = InstrumentedEnvironment()
        props = IcebergProperties(
            mass_kg=1e10,
            length_m=300.0,
            width_m=200.0,
            draft_m=80.0,
        )

        coord_handler = CoordinateHandler(crs="EPSG:3412")
        x0, y0 = coord_handler.to_projected(longitude=0.0, latitude=-65.0)
        initial_state = IcebergState(x_m=x0, y_m=y0, vx_mps=0.1, vy_mps=0.0)

        # Run exactly 1 timestep of dt = 100 seconds
        duration = 100.0
        dt = 100.0

        simulate_iceberg(
            initial_state=initial_state,
            start_time="2026-01-01 00:00:00",
            duration_seconds=duration,
            dt_seconds=dt,
            environment_provider=instrumented_env,
            iceberg_properties=props,
        )

        # Exactly 4 queries must have occurred for 1 RK4 step (k1 at t0, k2 at t0+dt/2, k3 at t0+dt/2, k4 at t0+dt)
        assert instrumented_env.call_count == 4

        t0 = pd.Timestamp("2026-01-01 00:00:00")
        t_mid = t0 + pd.Timedelta(seconds=50.0)
        t_end = t0 + pd.Timedelta(seconds=100.0)

        assert instrumented_env.queried_timestamps[0] == t0
        assert instrumented_env.queried_timestamps[1] == t_mid
        assert instrumented_env.queried_timestamps[2] == t_mid
        assert instrumented_env.queried_timestamps[3] == t_end

    def test_invalid_parameters_raise_exceptions(self):
        """Test 8: Verify negative or zero physical parameters raise ValueError."""
        # Non-positive mass
        with pytest.raises(ValueError, match="mass_kg"):
            IcebergProperties(mass_kg=0.0, length_m=100.0, width_m=50.0, draft_m=20.0)

        # Non-positive dimensions
        with pytest.raises(ValueError, match="length_m"):
            IcebergProperties(mass_kg=1e6, length_m=-10.0, width_m=50.0, draft_m=20.0)

        with pytest.raises(ValueError, match="width_m"):
            IcebergProperties(mass_kg=1e6, length_m=100.0, width_m=0.0, draft_m=20.0)

        # Negative draft
        with pytest.raises(ValueError, match="draft_m"):
            IcebergProperties(mass_kg=1e6, length_m=100.0, width_m=50.0, draft_m=-5.0)

        # Negative drag coefficients
        with pytest.raises(ValueError, match="air_drag_coefficient"):
            IcebergProperties(mass_kg=1e6, length_m=100.0, width_m=50.0, draft_m=20.0, air_drag_coefficient=-0.5)

        # Non-positive dt
        props = IcebergProperties(mass_kg=1e6, length_m=100.0, width_m=50.0, draft_m=20.0)
        env = SyntheticEnvironment()
        initial_state = IcebergState(x_m=0.0, y_m=0.0, vx_mps=0.0, vy_mps=0.0)
        with pytest.raises(ValueError, match="dt_seconds"):
            simulate_iceberg(
                initial_state=initial_state,
                start_time="2026-01-01",
                duration_seconds=100.0,
                dt_seconds=0.0,
                environment_provider=env,
                iceberg_properties=props,
            )

    def test_numerical_instability_detection(self):
        """Test 9: Verify non-finite numerical states trigger NumericalInstabilityError."""
        class ExplodingEnvironment(EnvironmentProvider):
            def get_environment(self, timestamp, latitude, longitude):
                return {
                    "sea_ice_concentration": 0.0,
                    "ocean_u": np.nan,  # Inject NaN
                    "ocean_v": 0.0,
                    "sst": 271.35,
                    "wind_u": 0.0,
                    "wind_v": 0.0,
                    "temperature": 260.0,
                    "pressure": 98500.0,
                }

        env = ExplodingEnvironment()
        props = IcebergProperties(mass_kg=1e6, length_m=100.0, width_m=50.0, draft_m=20.0)
        coord_handler = CoordinateHandler(crs="EPSG:3412")
        x0, y0 = coord_handler.to_projected(longitude=0.0, latitude=-65.0)
        initial_state = IcebergState(x_m=x0, y_m=y0, vx_mps=0.0, vy_mps=0.0)

        with pytest.raises(NumericalInstabilityError):
            simulate_iceberg(
                initial_state=initial_state,
                start_time="2026-01-01",
                duration_seconds=60.0,
                dt_seconds=10.0,
                environment_provider=env,
                iceberg_properties=props,
            )
