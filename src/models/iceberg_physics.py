"""
Physics-Based Iceberg Drift Simulation Module.

Implements a transparent, dimensionally consistent, simplified numerical simulator
for Antarctic iceberg trajectories. Solves the horizontal momentum conservation ODE
using a generic 4th-order Runge-Kutta (RK4) integrator in a projected Cartesian plane
(EPSG:3412 / EPSG:3031), consuming atmospheric and oceanic forcing from EnvironmentProvider.

MOMENTUM CONSERVATION EQUATION:
    m * (dv/dt) = F_water + F_air + F_coriolis - F_damping

FORCES & ACCELERATIONS:
1. Water Drag (Quadratic):
    F_water = 0.5 * rho_water * Cd_water * A_underwater * |v_rel_water| * v_rel_water
    where v_rel_water = v_ocean - v_iceberg

2. Air Drag (Quadratic):
    F_air   = 0.5 * rho_air * Cd_air * A_above_water * |v_rel_air| * v_rel_air
    where v_rel_air = v_wind - v_iceberg

3. Coriolis Force (Southern Hemisphere):
    a_coriolis_x =  f * vy
    a_coriolis_y = -f * vx
    where f = 2 * Omega * sin(latitude), Omega = 7.292115e-5 rad/s.
    In the Southern Hemisphere (latitude < 0), f < 0, deflecting motion to the LEFT.
    F_coriolis = m * a_coriolis

4. Linear Damping (Prototype resistance / residual loss):
    F_damping = c * v_iceberg (directed opposite to motion)

LIMITATIONS (PROTOTYPE ASSUMPTIONS):
This is a research baseline model. It does NOT simulate:
- Iceberg rotation / moment of inertia
- Wave radiation / wave drift forces
- Detailed 3D keel geometry and profile variations
- Mechanical fracture / calving / grounding processes
- Thermodynamic ablation / melting / mass loss
- Detailed sea-ice packet pressure or pack-ice damping
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyproj

from src.data.environment import EnvironmentProvider

logger = logging.getLogger(__name__)

# Earth's angular rotation rate in rad/s (sidereal day = 86164.0905 s)
EARTH_OMEGA_RAD_PER_S: float = 7.292115e-5


class PhysicsSimulationError(Exception):
    """Base exception for iceberg physics simulation errors."""
    pass


class NumericalInstabilityError(PhysicsSimulationError):
    """Raised when the simulation encounters non-finite (NaN / inf) values or diverging state."""
    pass


@dataclass(frozen=True)
class IcebergState:
    """
    State vector for an iceberg in projected Cartesian coordinates.

    Attributes:
        x_m: Projected horizontal position (Eastings in meters).
        y_m: Projected vertical position (Northings in meters).
        vx_mps: Horizontal velocity component in grid X direction (m/s).
        vy_mps: Vertical velocity component in grid Y direction (m/s).
    """
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x_m, self.y_m, self.vx_mps, self.vy_mps], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> IcebergState:
        return cls(x_m=float(arr[0]), y_m=float(arr[1]), vx_mps=float(arr[2]), vy_mps=float(arr[3]))


@dataclass
class IcebergProperties:
    """
    Physical characteristics and hydrodynamic parameters of a modeled iceberg.

    All drag and damping parameters are explicit prototype configurations and
    must not be treated as calibrated operational constants without validation.

    Attributes:
        mass_kg: Total iceberg mass (kg). Must be > 0.
        length_m: Characteristic waterline length (m). Must be > 0.
        width_m: Characteristic waterline width (m). Must be > 0.
        draft_m: Submerged keel draft (m). Must be >= 0.
        air_drag_coefficient: Form drag coefficient for exposed sail. (Prototype: ~1.3).
        water_drag_coefficient: Form drag coefficient for submerged keel. (Prototype: ~0.9).
        damping_coefficient: Prototype linear velocity damping (N*s/m). Must be >= 0.
        water_density_kg_per_m3: Density of polar seawater (kg/m^3, default 1025.0).
        air_density_kg_per_m3: Surface air density (kg/m^3, default 1.25).
        ice_density_kg_per_m3: Glacial ice density (kg/m^3, default 900.0).
        freeboard_m: Height of exposed sail above waterline (m). If None, estimated via isostasy.
        enable_coriolis: Toggle to enable/disable Coriolis acceleration for isolated testing.
    """
    mass_kg: float
    length_m: float
    width_m: float
    draft_m: float
    air_drag_coefficient: float = 1.30
    water_drag_coefficient: float = 0.90
    damping_coefficient: float = 0.00
    water_density_kg_per_m3: float = 1025.0
    air_density_kg_per_m3: float = 1.25
    ice_density_kg_per_m3: float = 900.0
    freeboard_m: Optional[float] = None
    enable_coriolis: bool = True

    def __post_init__(self) -> None:
        if self.mass_kg <= 0:
            raise ValueError(f"mass_kg must be strictly positive, got {self.mass_kg}")
        if self.length_m <= 0:
            raise ValueError(f"length_m must be strictly positive, got {self.length_m}")
        if self.width_m <= 0:
            raise ValueError(f"width_m must be strictly positive, got {self.width_m}")
        if self.draft_m < 0:
            raise ValueError(f"draft_m must be non-negative, got {self.draft_m}")
        if self.air_drag_coefficient < 0:
            raise ValueError(f"air_drag_coefficient must be non-negative, got {self.air_drag_coefficient}")
        if self.water_drag_coefficient < 0:
            raise ValueError(f"water_drag_coefficient must be non-negative, got {self.water_drag_coefficient}")
        if self.damping_coefficient < 0:
            raise ValueError(f"damping_coefficient must be non-negative, got {self.damping_coefficient}")

        # Compute isostatic freeboard height if not explicitly supplied:
        # Archimedes principle: Total_height = Draft * (rho_water / rho_ice)
        # Freeboard = Total_height - Draft = Draft * (rho_water - rho_ice) / rho_ice
        if self.freeboard_m is None:
            delta_rho = max(0.0, self.water_density_kg_per_m3 - self.ice_density_kg_per_m3)
            self.freeboard_m = float(self.draft_m * (delta_rho / self.ice_density_kg_per_m3))

        # Approximate isotropic projected cross-sectional area using geometric mean width
        self.effective_width_m = float(np.sqrt(self.length_m * self.width_m))
        self.underwater_area_m2 = float(self.effective_width_m * self.draft_m)
        self.above_water_area_m2 = float(self.effective_width_m * self.freeboard_m)


# ==============================================================================
# Numerical Integration & Coordinate Helpers
# ==============================================================================

def rk4_step(
    state: np.ndarray,
    t: float,
    dt: float,
    derivative_function: Callable[[float, np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Generic explicit fourth-order Runge-Kutta step for an arbitrary state vector.

    Args:
        state: State vector at time t.
        t: Current integration time (seconds).
        dt: Timestep (seconds).
        derivative_function: Callable f(t, state) -> dstate/dt.

    Returns:
        New state vector at time t + dt.
    """
    k1 = derivative_function(t, state)
    k2 = derivative_function(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = derivative_function(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = derivative_function(t + dt, state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class CoordinateHandler:
    """
    Bidirectional coordinate transformer between WGS84 geographic (lat/lon)
    and Antarctic projected coordinates (EPSG:3412 / EPSG:3031).
    Also calculates meridian convergence to rotate East/North velocities to grid X/Y.
    """

    def __init__(self, crs: str = "EPSG:3412"):
        self.crs_str = crs
        try:
            self.proj = pyproj.Proj(self.crs_str)
            self.geo_to_proj = pyproj.Transformer.from_crs("EPSG:4326", self.crs_str, always_xy=True)
            self.proj_to_geo = pyproj.Transformer.from_crs(self.crs_str, "EPSG:4326", always_xy=True)
        except Exception as e:
            raise ValueError(f"Invalid coordinate reference system '{crs}': {e}") from e

    def to_projected(self, longitude: float, latitude: float) -> Tuple[float, float]:
        """Convert (lon, lat) degrees to (x, y) meters."""
        x, y = self.geo_to_proj.transform(longitude, latitude)
        return float(x), float(y)

    def to_geographic(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """Convert (x, y) meters to (lon, lat) degrees."""
        lon, lat = self.proj_to_geo.transform(x_m, y_m)
        return float(lon), float(lat)

    def rotate_vector_to_grid(self, u_east: float, v_north: float, lon: float, lat: float) -> Tuple[float, float]:
        """
        Rotate geographic velocity vector (u_east, v_north) into projected grid coordinates (vx, vy).

        Uses meridian convergence angle gamma:
            vx =  u_east * cos(gamma) - v_north * sin(gamma)
            vy =  u_east * sin(gamma) + v_north * cos(gamma)
        """
        factors = self.proj.get_factors(lon, lat)
        gamma = np.radians(factors.meridian_convergence)
        cos_g = np.cos(gamma)
        sin_g = np.sin(gamma)

        vx = u_east * cos_g - v_north * sin_g
        vy = u_east * sin_g + v_north * cos_g
        return float(vx), float(vy)

    def rotate_grid_to_vector(self, vx: float, vy: float, lon: float, lat: float) -> Tuple[float, float]:
        """
        Rotate projected grid velocity (vx, vy) back to geographic coordinates (u_east, v_north).
        """
        factors = self.proj.get_factors(lon, lat)
        gamma = np.radians(factors.meridian_convergence)
        cos_g = np.cos(gamma)
        sin_g = np.sin(gamma)

        u_east = vx * cos_g + vy * sin_g
        v_north = -vx * sin_g + vy * cos_g
        return float(u_east), float(v_north)


# ==============================================================================
# Iceberg Derivative Function & Simulator
# ==============================================================================

def iceberg_derivative(
    t_seconds: float,
    state_arr: np.ndarray,
    start_timestamp: pd.Timestamp,
    environment_provider: EnvironmentProvider,
    props: IcebergProperties,
    coord_handler: CoordinateHandler,
) -> np.ndarray:
    """
    Computes [dx/dt, dy/dt, dvx/dt, dvy/dt] for an iceberg state vector.

    Evaluates environmental forcing dynamically at the exact intermediate time
    and intermediate spatial position specified by the numerical integrator.
    """
    if not np.isfinite(state_arr).all():
        current_time = start_timestamp + pd.Timedelta(seconds=t_seconds)
        raise NumericalInstabilityError(
            f"Non-finite state encountered in derivative evaluation at timestamp={current_time}: {state_arr}"
        )

    x_m, y_m, vx_mps, vy_mps = state_arr

    # 1. Transform projected position to latitude/longitude
    lon_deg, lat_deg = coord_handler.to_geographic(x_m, y_m)

    # 2. Advance query timestamp
    current_time = start_timestamp + pd.Timedelta(seconds=t_seconds)

    # 3. Query environmental conditions at intermediate state
    env_data = environment_provider.get_environment(current_time, lat_deg, lon_deg)
    u_ocean = env_data["ocean_u"]
    v_ocean = env_data["ocean_v"]
    u_wind = env_data["wind_u"]
    v_wind = env_data["wind_v"]

    if not np.isfinite([u_ocean, v_ocean, u_wind, v_wind]).all():
        raise NumericalInstabilityError(
            f"Non-finite environmental forcing retrieved at timestamp={current_time}, "
            f"lat={lat_deg:.4f}, lon={lon_deg:.4f}: {env_data}"
        )

    # 4. Rotate environmental vectors to grid orientation
    u_ocean_grid, v_ocean_grid = coord_handler.rotate_vector_to_grid(u_ocean, v_ocean, lon_deg, lat_deg)
    u_wind_grid, v_wind_grid = coord_handler.rotate_vector_to_grid(u_wind, v_wind, lon_deg, lat_deg)

    # 5. Water drag force
    # Relative velocity of water relative to iceberg: v_rel_water = v_ocean - v_iceberg
    v_rel_water_x = u_ocean_grid - vx_mps
    v_rel_water_y = v_ocean_grid - vy_mps
    speed_rel_water = np.hypot(v_rel_water_x, v_rel_water_y)

    drag_water_coeff = 0.5 * props.water_density_kg_per_m3 * props.water_drag_coefficient * props.underwater_area_m2
    F_water_x = drag_water_coeff * speed_rel_water * v_rel_water_x
    F_water_y = drag_water_coeff * speed_rel_water * v_rel_water_y

    # 6. Wind drag force
    # Relative velocity of air relative to iceberg: v_rel_air = v_wind - v_iceberg
    v_rel_air_x = u_wind_grid - vx_mps
    v_rel_air_y = v_wind_grid - vy_mps
    speed_rel_air = np.hypot(v_rel_air_x, v_rel_air_y)

    drag_air_coeff = 0.5 * props.air_density_kg_per_m3 * props.air_drag_coefficient * props.above_water_area_m2
    F_air_x = drag_air_coeff * speed_rel_air * v_rel_air_x
    F_air_y = drag_air_coeff * speed_rel_air * v_rel_air_y

    # 7. Coriolis acceleration
    # f = 2 * Omega * sin(lat). In Southern Hemisphere, lat < 0 -> f < 0.
    # a_coriolis_x =  f * vy
    # a_coriolis_y = -f * vx
    if props.enable_coriolis:
        f_coriolis = 2.0 * EARTH_OMEGA_RAD_PER_S * np.sin(np.radians(lat_deg))
        F_coriolis_x = props.mass_kg * (f_coriolis * vy_mps)
        F_coriolis_y = props.mass_kg * (-f_coriolis * vx_mps)
    else:
        F_coriolis_x = 0.0
        F_coriolis_y = 0.0

    # 8. Linear Damping (opposing velocity)
    F_damping_x = props.damping_coefficient * vx_mps
    F_damping_y = props.damping_coefficient * vy_mps

    # 9. Total force and acceleration (m * a = F)
    F_total_x = F_water_x + F_air_x + F_coriolis_x - F_damping_x
    F_total_y = F_water_y + F_air_y + F_coriolis_y - F_damping_y

    ax_mps2 = F_total_x / props.mass_kg
    ay_mps2 = F_total_y / props.mass_kg

    out_deriv = np.array([vx_mps, vy_mps, ax_mps2, ay_mps2], dtype=np.float64)
    if not np.isfinite(out_deriv).all():
        raise NumericalInstabilityError(
            f"Non-finite acceleration calculated at timestamp={current_time}: {out_deriv}"
        )

    return out_deriv


def simulate_iceberg(
    initial_state: IcebergState,
    start_time: Union[str, pd.Timestamp, datetime],
    duration_seconds: float,
    dt_seconds: float,
    environment_provider: EnvironmentProvider,
    iceberg_properties: IcebergProperties,
    crs: str = "EPSG:3412",
) -> pd.DataFrame:
    """
    Simulate iceberg trajectory forward in time using 4th-order Runge-Kutta integration.

    Args:
        initial_state: Starting position and velocity in projected Cartesian meters.
        start_time: Start timestamp (UTC).
        duration_seconds: Total simulation duration in seconds (must be >= 0).
        dt_seconds: Numerical integration time-step in seconds (must be > 0).
        environment_provider: Provider supplying environmental forcing.
        iceberg_properties: Physical mass, dimensions, and drag parameters.
        crs: Coordinate reference system for numerical calculations (default EPSG:3412).

    Returns:
        DataFrame containing trajectory points with columns:
        [timestamp, x_m, y_m, latitude, longitude, vx_mps, vy_mps, speed_mps]
    """
    if dt_seconds <= 0:
        raise ValueError(f"dt_seconds must be strictly positive, got {dt_seconds}")
    if duration_seconds < 0:
        raise ValueError(f"duration_seconds must be non-negative, got {duration_seconds}")

    t_start = pd.Timestamp(start_time)
    coord_handler = CoordinateHandler(crs=crs)

    current_state = initial_state.to_array()
    if not np.isfinite(current_state).all():
        raise NumericalInstabilityError(f"Initial state contains non-finite values: {current_state}")

    t = 0.0
    num_steps = int(np.floor(duration_seconds / dt_seconds))

    # Initial state record
    lon0, lat0 = coord_handler.to_geographic(current_state[0], current_state[1])
    speed0 = float(np.hypot(current_state[2], current_state[3]))

    records: List[Dict[str, Any]] = [{
        "timestamp": t_start,
        "x_m": float(current_state[0]),
        "y_m": float(current_state[1]),
        "latitude": lat0,
        "longitude": lon0,
        "vx_mps": float(current_state[2]),
        "vy_mps": float(current_state[3]),
        "speed_mps": speed0,
    }]

    # Wrapper for derivative function
    def deriv(eval_t: float, eval_state: np.ndarray) -> np.ndarray:
        return iceberg_derivative(
            t_seconds=eval_t,
            state_arr=eval_state,
            start_timestamp=t_start,
            environment_provider=environment_provider,
            props=iceberg_properties,
            coord_handler=coord_handler,
        )

    # Integration loop
    for step in range(num_steps):
        current_state = rk4_step(current_state, t, dt_seconds, deriv)
        t += dt_seconds
        current_time = t_start + pd.Timedelta(seconds=t)

        if not np.isfinite(current_state).all():
            raise NumericalInstabilityError(
                f"Numerical instability encountered at step {step + 1}, timestamp={current_time}. "
                f"State: {current_state}"
            )

        lon, lat = coord_handler.to_geographic(current_state[0], current_state[1])
        speed = float(np.hypot(current_state[2], current_state[3]))

        records.append({
            "timestamp": current_time,
            "x_m": float(current_state[0]),
            "y_m": float(current_state[1]),
            "latitude": lat,
            "longitude": lon,
            "vx_mps": float(current_state[2]),
            "vy_mps": float(current_state[3]),
            "speed_mps": speed,
        })

    return pd.DataFrame(records)
