"""
Feature Extraction and Residual Calculation Module for Physics-Residual ML.

Extracts kinematic, environmental, and spatial features for supervised learning
of iceberg trajectory prediction error residuals:
    residual_x = x_truth - x_physics
    residual_y = y_truth - y_physics
in projected Antarctic coordinates (default EPSG:3412, meters).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.data.environment import EnvironmentProvider, MissingDataError
from src.models.iceberg_physics import CoordinateHandler, IcebergProperties

logger = logging.getLogger(__name__)

# Standard core features available from environmental provider + physics state
CORE_FEATURE_NAMES: List[str] = [
    "wind_u",
    "wind_v",
    "wind_speed",
    "ocean_u",
    "ocean_v",
    "ocean_speed",
    "sea_ice_concentration",
    "physics_vx",
    "physics_vy",
    "physics_speed",
    "dt_seconds",
]

# Static physical iceberg properties (included when available)
STATIC_PROPERTY_NAMES: List[str] = [
    "length_m",
    "width_m",
    "draft_m",
    "mass_kg",
    "air_drag_coefficient",
    "water_drag_coefficient",
]

ALL_FEATURE_NAMES: List[str] = CORE_FEATURE_NAMES + STATIC_PROPERTY_NAMES
TARGET_NAMES: List[str] = ["residual_x_m", "residual_y_m"]


class ResidualFeatureExtractor:
    """
    Extracts features and computes projected trajectory residuals.
    """

    def __init__(
        self,
        crs: str = "EPSG:3412",
        include_static_properties: bool = True,
    ) -> None:
        self.crs = crs
        self.include_static_properties = include_static_properties
        self.coord_handler = CoordinateHandler(crs=self.crs)
        self.feature_names = list(ALL_FEATURE_NAMES) if include_static_properties else list(CORE_FEATURE_NAMES)

    def extract_features(
        self,
        timestamp: Union[str, pd.Timestamp],
        latitude: float,
        longitude: float,
        physics_vx: float,
        physics_vy: float,
        dt_seconds: float,
        environment_provider: EnvironmentProvider,
        iceberg_properties: Optional[IcebergProperties] = None,
    ) -> Dict[str, float]:
        """
        Query environmental provider and combine with kinematic/physical state.

        Args:
            timestamp: Query timestamp.
            latitude: Iceberg latitude (degrees).
            longitude: Iceberg longitude (degrees).
            physics_vx: Simulated x-velocity in projected plane (m/s).
            physics_vy: Simulated y-velocity in projected plane (m/s).
            dt_seconds: Elapsed prediction duration from origin T (seconds).
            environment_provider: Provider for atmospheric, oceanic, and sea-ice forcing.
            iceberg_properties: Optional physical dimensions of the iceberg.

        Returns:
            Dictionary mapping feature names to numerical float values.

        Raises:
            MissingDataError: If any required environmental variable is missing or NaN.
        """
        t = pd.Timestamp(timestamp)

        # 1. Query environmental conditions
        try:
            env = environment_provider.get_environment(
                timestamp=t,
                latitude=latitude,
                longitude=longitude,
            )
        except Exception as e:
            raise MissingDataError(f"Failed to query environment at {t}, ({latitude}, {longitude}): {e}") from e

        # 2. Extract and validate atmospheric variables
        wind_u = env.get("wind_u")
        wind_v = env.get("wind_v")
        if wind_u is None or wind_v is None or np.isnan(wind_u) or np.isnan(wind_v):
            raise MissingDataError(f"Missing atmospheric wind forcing at {t}, ({latitude}, {longitude})")
        wind_u_flt = float(wind_u)
        wind_v_flt = float(wind_v)
        wind_speed = float(np.hypot(wind_u_flt, wind_v_flt))

        # 3. Extract and validate ocean current variables
        ocean_u = env.get("ocean_u")
        ocean_v = env.get("ocean_v")
        if ocean_u is None or ocean_v is None or np.isnan(ocean_u) or np.isnan(ocean_v):
            raise MissingDataError(f"Missing ocean current forcing at {t}, ({latitude}, {longitude})")
        ocean_u_flt = float(ocean_u)
        ocean_v_flt = float(ocean_v)
        ocean_speed = float(np.hypot(ocean_u_flt, ocean_v_flt))

        # 4. Extract sea ice concentration
        sic = env.get("sea_ice_concentration")
        if sic is None or np.isnan(sic):
            raise MissingDataError(f"Missing sea ice concentration at {t}, ({latitude}, {longitude})")
        sic_flt = float(np.clip(sic, 0.0, 1.0))

        # 5. Kinematic features
        pvx_flt = float(physics_vx)
        pvy_flt = float(physics_vy)
        physics_speed = float(np.hypot(pvx_flt, pvy_flt))
        dt_flt = float(dt_seconds)

        features: Dict[str, float] = {
            "wind_u": wind_u_flt,
            "wind_v": wind_v_flt,
            "wind_speed": wind_speed,
            "ocean_u": ocean_u_flt,
            "ocean_v": ocean_v_flt,
            "ocean_speed": ocean_speed,
            "sea_ice_concentration": sic_flt,
            "physics_vx": pvx_flt,
            "physics_vy": pvy_flt,
            "physics_speed": physics_speed,
            "dt_seconds": dt_flt,
        }

        # 6. Static iceberg properties (if requested and provided)
        if self.include_static_properties:
            if iceberg_properties is not None:
                features["length_m"] = float(iceberg_properties.length_m)
                features["width_m"] = float(iceberg_properties.width_m)
                features["draft_m"] = float(iceberg_properties.draft_m)
                features["mass_kg"] = float(iceberg_properties.mass_kg)
                features["air_drag_coefficient"] = float(iceberg_properties.air_drag_coefficient)
                features["water_drag_coefficient"] = float(iceberg_properties.water_drag_coefficient)
            else:
                # Default prototype parameters
                features["length_m"] = 5000.0
                features["width_m"] = 2500.0
                features["draft_m"] = 200.0
                features["mass_kg"] = 1e12
                features["air_drag_coefficient"] = 0.2000
                features["water_drag_coefficient"] = 1.0065

        return features

    def calculate_projected_residual(
        self,
        truth_longitude: float,
        truth_latitude: float,
        physics_x_m: float,
        physics_y_m: float,
    ) -> Tuple[float, float]:
        """
        Calculate Cartesian displacement error residual in EPSG:3412 meters:
            residual_x = x_truth - x_physics
            residual_y = y_truth - y_physics

        Args:
            truth_longitude: Observed ground truth longitude (degrees).
            truth_latitude: Observed ground truth latitude (degrees).
            physics_x_m: Simulated x-coordinate (meters).
            physics_y_m: Simulated y-coordinate (meters).

        Returns:
            Tuple of (residual_x_m, residual_y_m).
        """
        x_truth, y_truth = self.coord_handler.to_projected(truth_longitude, truth_latitude)
        res_x = float(x_truth - physics_x_m)
        res_y = float(y_truth - physics_y_m)
        return res_x, res_y
