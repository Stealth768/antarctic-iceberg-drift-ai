"""
Constant-Velocity / Persistence Iceberg Trajectory Baseline.

Implements a deterministic, zero-acceleration persistence reference model.
Assumes velocity remains constant over the forecast horizon:
    x(t + dt) = x(t) + vx * dt
    y(t + dt) = y(t) + vy * dt

Serves as an independent, transparent benchmark against which the Stage 3
momentum-conservation physics solver and future spatiotemporal ML models are evaluated.
This baseline strictly does not query atmospheric, oceanic, or sea-ice environmental data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.models.iceberg_physics import CoordinateHandler, IcebergState

logger = logging.getLogger(__name__)


@dataclass
class ConstantVelocityPredictor:
    """
    Persistence trajectory predictor assuming constant velocity over time.

    Operates in projected Cartesian coordinates (meters, meters per second)
    using CoordinateHandler (default EPSG:3412).
    """
    crs: str = "EPSG:3412"

    def __post_init__(self) -> None:
        self.coord_handler = CoordinateHandler(crs=self.crs)

    def predict_state(self, initial_state: IcebergState, dt_seconds: float) -> IcebergState:
        """
        Advance an iceberg state by dt_seconds assuming zero acceleration.

        Args:
            initial_state: Starting state (x, y in meters; vx, vy in m/s).
            dt_seconds: Elapsed time in seconds.

        Returns:
            Projected IcebergState at t + dt.
        """
        if not np.isfinite(initial_state.to_array()).all():
            raise ValueError(f"Non-finite values in initial_state: {initial_state}")
        if not np.isfinite(dt_seconds):
            raise ValueError(f"Non-finite dt_seconds: {dt_seconds}")

        x_future = initial_state.x_m + initial_state.vx_mps * dt_seconds
        y_future = initial_state.y_m + initial_state.vy_mps * dt_seconds

        return IcebergState(
            x_m=float(x_future),
            y_m=float(y_future),
            vx_mps=float(initial_state.vx_mps),
            vy_mps=float(initial_state.vy_mps),
        )

    def predict(
        self,
        initial_state: IcebergState,
        start_time: Union[str, pd.Timestamp, datetime],
        forecast_seconds: Union[float, Sequence[float]],
    ) -> pd.DataFrame:
        """
        Predict future positions at one or more forecast horizons.

        Args:
            initial_state: Starting position and velocity.
            start_time: Start timestamp (UTC).
            forecast_seconds: Single duration or sequence of horizons in seconds.

        Returns:
            DataFrame containing predicted trajectory points:
            [timestamp, x_m, y_m, longitude, latitude, vx_mps, vy_mps, speed_mps]
        """
        t_start = pd.Timestamp(start_time)
        if isinstance(forecast_seconds, (int, float)):
            horizons = [float(forecast_seconds)]
        else:
            horizons = [float(h) for h in forecast_seconds]

        # Always ensure horizons are sorted and non-negative
        for h in horizons:
            if h < 0:
                raise ValueError(f"Forecast horizon must be non-negative, got {h}")
            if not np.isfinite(h):
                raise ValueError(f"Non-finite horizon: {h}")

        records: List[Dict[str, Any]] = []

        # Include starting state if 0.0 in horizons or if horizons empty
        if 0.0 not in horizons:
            eval_horizons = [0.0] + sorted(horizons)
        else:
            eval_horizons = sorted(horizons)

        for h in eval_horizons:
            pred_state = self.predict_state(initial_state, h)
            pred_time = t_start + pd.Timedelta(seconds=h)
            lon, lat = self.coord_handler.to_geographic(pred_state.x_m, pred_state.y_m)
            speed = float(np.hypot(pred_state.vx_mps, pred_state.vy_mps))

            records.append({
                "timestamp": pred_time,
                "x_m": pred_state.x_m,
                "y_m": pred_state.y_m,
                "longitude": lon,
                "latitude": lat,
                "vx_mps": pred_state.vx_mps,
                "vy_mps": pred_state.vy_mps,
                "speed_mps": speed,
            })

        return pd.DataFrame(records)


# ==============================================================================
# Historical Velocity Estimation Helpers
# ==============================================================================

def estimate_velocity_projected(
    x1_m: float,
    y1_m: float,
    t1: Union[str, pd.Timestamp, datetime],
    x2_m: float,
    y2_m: float,
    t2: Union[str, pd.Timestamp, datetime],
) -> Tuple[float, float, float]:
    """
    Estimate constant velocity from two consecutive projected observations.

    vx = (x2 - x1) / dt
    vy = (y2 - y1) / dt

    Args:
        x1_m: Projected X coordinate at time t1 (meters).
        y1_m: Projected Y coordinate at time t1 (meters).
        t1: Timestamp of first observation.
        x2_m: Projected X coordinate at time t2 (meters).
        y2_m: Projected Y coordinate at time t2 (meters).
        t2: Timestamp of second observation.

    Returns:
        Tuple of (vx_mps, vy_mps, dt_seconds).
    """
    time1 = pd.Timestamp(t1)
    time2 = pd.Timestamp(t2)

    dt_sec = (time2 - time1).total_seconds()
    if dt_sec <= 0:
        raise ValueError(
            f"Invalid time interval: t2 ({time2}) must be strictly greater than t1 ({time1}). "
            f"Elapsed dt = {dt_sec} seconds."
        )

    coords = [x1_m, y1_m, x2_m, y2_m]
    if not np.isfinite(coords).all():
        raise ValueError(f"Coordinates must be finite real numbers, got {coords}")

    vx = (x2_m - x1_m) / dt_sec
    vy = (y2_m - y1_m) / dt_sec

    return float(vx), float(vy), float(dt_sec)


def estimate_velocity_geographic(
    lon1_deg: float,
    lat1_deg: float,
    t1: Union[str, pd.Timestamp, datetime],
    lon2_deg: float,
    lat2_deg: float,
    t2: Union[str, pd.Timestamp, datetime],
    crs: str = "EPSG:3412",
) -> Tuple[float, float, float]:
    """
    Estimate velocity by projecting two geographic (lon, lat) observations.

    Transforms geographic coordinates to projected plane meters before computing:
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt

    Does NOT compute velocity by subtracting degrees latitude/longitude.

    Returns:
        Tuple of (vx_mps, vy_mps, dt_seconds).
    """
    coord_handler = CoordinateHandler(crs=crs)
    x1, y1 = coord_handler.to_projected(longitude=lon1_deg, latitude=lat1_deg)
    x2, y2 = coord_handler.to_projected(longitude=lon2_deg, latitude=lat2_deg)

    return estimate_velocity_projected(x1, y1, t1, x2, y2, t2)


def create_state_from_observations(
    obs1: Dict[str, Any],
    obs2: Dict[str, Any],
    crs: str = "EPSG:3412",
) -> IcebergState:
    """
    Construct an IcebergState at the timestamp of the second observation (t2).

    Accepts dicts containing either ('x_m', 'y_m', 'timestamp') or ('longitude', 'latitude', 'timestamp').

    Returns:
        IcebergState located at (x2, y2) with estimated velocity (vx, vy).
    """
    coord_handler = CoordinateHandler(crs=crs)

    t1 = obs1["timestamp"]
    t2 = obs2["timestamp"]

    if "x_m" in obs1 and "y_m" in obs1:
        x1, y1 = float(obs1["x_m"]), float(obs1["y_m"])
    else:
        x1, y1 = coord_handler.to_projected(longitude=obs1["longitude"], latitude=obs1["latitude"])

    if "x_m" in obs2 and "y_m" in obs2:
        x2, y2 = float(obs2["x_m"]), float(obs2["y_m"])
    else:
        x2, y2 = coord_handler.to_projected(longitude=obs2["longitude"], latitude=obs2["latitude"])

    vx, vy, _ = estimate_velocity_projected(x1, y1, t1, x2, y2, t2)

    return IcebergState(x_m=x2, y_m=y2, vx_mps=vx, vy_mps=vy)
