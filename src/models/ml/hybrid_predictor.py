"""
Hybrid Physics + ML Iceberg Trajectory Predictor for Antarctic Drift.

Combines calibrated physics simulation with learned Ridge residual corrections
to improve trajectory predictions. The ML correction is bounded to prevent
pathological extrapolation and ensure numerical stability.

Architecture:
    1. Run calibrated physics simulation (existing simulate_iceberg)
    2. Extract features at physics output times
    3. Predict residuals using trained Ridge model
    4. Apply bounded corrections in projected coordinates
    5. Convert back to geographic coordinates
    6. Return trajectory compatible with existing evaluation infrastructure

The hybrid model is intended as a post-processing correction to the physics
baseline, not a replacement. The physics simulation remains the primary model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.environment import EnvironmentProvider
from src.models.iceberg_physics import (
    CoordinateHandler,
    IcebergProperties,
    IcebergState,
    simulate_iceberg,
)
from src.models.ml.features import ResidualFeatureExtractor
from src.models.ml.residual_model import ResidualModel

logger = logging.getLogger(__name__)


class HybridIcebergPredictor:
    """
    Physics-informed hybrid trajectory predictor with learned ML residual correction.

    The hybrid predictor combines:
    1. Calibrated physics simulation (deterministic, physical baseline)
    2. Ridge regression residual correction (learned from data)
    3. Explicit correction bounds (numerical safety)

    The correction is applied in projected coordinates (EPSG:3412) and bounded
    to the statistics of the training/validation residual distribution to prevent
    the ML model from extrapolating pathologically.

    Attributes:
        residual_model: Trained ResidualModel (typically Ridge) for residual prediction.
        residual_bound_m: Maximum allowed residual correction vector magnitude (meters).
        crs: Projected coordinate system (default EPSG:3412).
        feature_extractor: Feature extraction pipeline for residual inputs.
    """

    def __init__(
        self,
        residual_model: ResidualModel,
        residual_bound_m: float = 1000.0,
        crs: str = "EPSG:3412",
        feature_extractor: Optional[ResidualFeatureExtractor] = None,
    ) -> None:
        """
        Initialize the hybrid predictor with a trained residual model.

        Args:
            residual_model: Fitted ResidualModel (e.g., RidgeResidualModel).
            residual_bound_m: Maximum allowed residual correction vector magnitude in meters.
                Default 1000m is derived from train/val residual statistics (P95 ≈ 900m).
                Set lower to be more conservative; higher to allow larger corrections.
            crs: Projected coordinate system string (default EPSG:3412).
            feature_extractor: Feature extraction pipeline. If None, a default
                ResidualFeatureExtractor with static properties is created.

        Raises:
            ValueError: If residual_bound_m <= 0.
        """
        if residual_bound_m <= 0:
            raise ValueError(f"residual_bound_m must be positive, got {residual_bound_m}")

        self.residual_model = residual_model
        self.residual_bound_m = residual_bound_m
        self.crs = crs
        self.coord_handler = CoordinateHandler(crs=crs)

        if feature_extractor is None:
            self.feature_extractor = ResidualFeatureExtractor(
                crs=crs,
                include_static_properties=True,
            )
        else:
            self.feature_extractor = feature_extractor

        logger.info(
            f"HybridIcebergPredictor initialized with residual_bound_m={residual_bound_m} m, "
            f"crs={crs}"
        )

    def predict(
        self,
        initial_state: IcebergState,
        start_time: pd.Timestamp,
        duration_seconds: float,
        dt_seconds: float,
        environment_provider: EnvironmentProvider,
        iceberg_properties: IcebergProperties,
        apply_ml_correction: bool = True,
    ) -> pd.DataFrame:
        """
        Predict hybrid trajectory combining physics and ML residual correction.

        Steps:
        1. Run calibrated physics simulation (no corrections).
        2. Extract features at each time step.
        3. Predict residuals using trained ML model.
        4. Clip corrections to safety bounds.
        5. Apply bounded corrections to physics coordinates.
        6. Convert back to lat/lon.

        Args:
            initial_state: Starting iceberg position and velocity (EPSG:3412 projected).
            start_time: Simulation start timestamp.
            duration_seconds: Total simulation duration in seconds.
            dt_seconds: Physics integration timestep in seconds.
            environment_provider: Provider for atmospheric/oceanic forcing.
            iceberg_properties: Physical properties of the iceberg.
            apply_ml_correction: If False, return physics-only trajectory (useful for baselines).

        Returns:
            DataFrame with columns [timestamp, latitude, longitude] representing
            the hybrid trajectory. Compatible with calculate_trajectory_metrics().
        """
        # Step 1: Run physics simulation
        logger.info("Running calibrated physics simulation...")
        df_physics = simulate_iceberg(
            initial_state=initial_state,
            start_time=start_time,
            duration_seconds=duration_seconds,
            dt_seconds=dt_seconds,
            environment_provider=environment_provider,
            iceberg_properties=iceberg_properties,
            crs=self.crs,
        )

        if not apply_ml_correction:
            logger.info("ML correction disabled; returning physics-only trajectory")
            return df_physics[["timestamp", "latitude", "longitude"]].copy()

        logger.info(f"Physics simulation produced {len(df_physics)} trajectory points")

        # Step 2: Extract features at each physics time step
        logger.info("Extracting features for ML residual prediction...")
        features_list = []
        times_to_process = []

        for idx, row in df_physics.iterrows():
            t = pd.Timestamp(row["timestamp"])
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            vx = float(row.get("vx_mps", 0.0))
            vy = float(row.get("vy_mps", 0.0))
            dt_elapsed = float((t - start_time).total_seconds())

            try:
                feats = self.feature_extractor.extract_features(
                    timestamp=t,
                    latitude=lat,
                    longitude=lon,
                    physics_vx=vx,
                    physics_vy=vy,
                    dt_seconds=dt_elapsed,
                    environment_provider=environment_provider,
                    iceberg_properties=iceberg_properties,
                )
                features_list.append(feats)
                times_to_process.append(idx)
            except Exception as e:
                logger.warning(f"Failed to extract features at {t}: {e}. Skipping residual correction.")
                features_list.append(None)
                times_to_process.append(idx)

        # Step 3: Predict residuals for all points with valid features
        logger.info("Predicting residuals with Ridge model...")
        X_data = []
        valid_indices = []

        for i, feats in enumerate(features_list):
            if feats is not None:
                row_feat = [float(feats.get(f, np.nan)) for f in self.feature_extractor.feature_names]
                if not np.any(np.isnan(row_feat)):
                    X_data.append(row_feat)
                    valid_indices.append(i)

        if len(X_data) == 0:
            logger.warning("No valid features extracted; returning physics-only trajectory")
            return df_physics[["timestamp", "latitude", "longitude"]].copy()

        X_array = np.asarray(X_data, dtype=np.float32)
        y_pred = self.residual_model.predict(X_array)  # shape (n_valid, 2)

        # Step 4: Apply bounded corrections
        logger.info(f"Applying bounded residual corrections (bound={self.residual_bound_m} m)...")
        y_corrected = np.asarray(y_pred, dtype=float).copy()

        magnitudes = np.linalg.norm(y_corrected, axis=1)
        scale = np.ones_like(magnitudes)

        mask = magnitudes > self.residual_bound_m
        scale[mask] = self.residual_bound_m / magnitudes[mask]

        y_corrected *= scale[:, None]

        # Step 5: Build corrected trajectory
        df_corrected = df_physics.copy()
        
        for out_idx, in_idx in enumerate(valid_indices):
            x_phys = float(df_physics.iloc[in_idx]["x_m"])
            y_phys = float(df_physics.iloc[in_idx]["y_m"])

            x_corr = x_phys + y_corrected[out_idx, 0]
            y_corr = y_phys + y_corrected[out_idx, 1]

            lon_corr, lat_corr = self.coord_handler.to_geographic(x_corr, y_corr)
            df_corrected.at[in_idx, "latitude"] = lat_corr
            df_corrected.at[in_idx, "longitude"] = lon_corr

        logger.info(f"Hybrid trajectory constructed with {len(valid_indices)}/{len(df_physics)} corrections applied")

        return df_corrected[["timestamp", "latitude", "longitude"]].copy()

    def physics_only(
        self,
        initial_state: IcebergState,
        start_time: pd.Timestamp,
        duration_seconds: float,
        dt_seconds: float,
        environment_provider: EnvironmentProvider,
        iceberg_properties: IcebergProperties,
    ) -> pd.DataFrame:
        """
        Convenience method: return physics-only trajectory (no ML correction).

        Equivalent to predict(..., apply_ml_correction=False).
        """
        return self.predict(
            initial_state=initial_state,
            start_time=start_time,
            duration_seconds=duration_seconds,
            dt_seconds=dt_seconds,
            environment_provider=environment_provider,
            iceberg_properties=iceberg_properties,
            apply_ml_correction=False,
        )


def constant_velocity_baseline(
    df_truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a constant-velocity baseline trajectory.

    Uses the first observation as position and velocity, then extrapolates
    linearly forward in time. Useful for comparison against physics and hybrid.

    Args:
        df_truth: Ground-truth observation trajectory with columns
            [timestamp, latitude, longitude].

    Returns:
        DataFrame with constant-velocity prediction at same timestamps.
    """
    if len(df_truth) < 2:
        raise ValueError("Need at least 2 truth observations to compute velocity")

    # Initial position and time
    t0 = pd.Timestamp(df_truth["timestamp"].iloc[0])
    lat0 = float(df_truth["latitude"].iloc[0])
    lon0 = float(df_truth["longitude"].iloc[0])

    # Compute velocity from first two observations
    t1 = pd.Timestamp(df_truth["timestamp"].iloc[1])
    lat1 = float(df_truth["latitude"].iloc[1])
    lon1 = float(df_truth["longitude"].iloc[1])

    dt_sec = (t1 - t0).total_seconds()
    dlat_dt = (lat1 - lat0) / dt_sec if dt_sec > 0 else 0.0
    dlon_dt = (lon1 - lon0) / dt_sec if dt_sec > 0 else 0.0

    # Extrapolate
    times = pd.to_datetime(df_truth["timestamp"])
    elapsed_seconds = (times - t0).dt.total_seconds()
    lats = lat0 + dlat_dt * elapsed_seconds
    lons = lon0 + dlon_dt * elapsed_seconds

    return pd.DataFrame({
        "timestamp": times,
        "latitude": lats,
        "longitude": lons,
    })
