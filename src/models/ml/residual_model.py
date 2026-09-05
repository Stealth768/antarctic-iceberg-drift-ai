"""
Physics-Residual ML Model for Antarctic Iceberg Drift.

Implements supervised regression models that predict the displacement error
between the calibrated physics trajectory and the observed ground truth:

    residual_x = x_truth - x_physics   (meters, EPSG:3412)
    residual_y = y_truth - y_physics   (meters, EPSG:3412)

The predicted correction can be added to the physics output to obtain an
adjusted position estimate, but integration into simulate_iceberg() is
intentionally deferred — this module only provides offline evaluation.

Models
------
RidgeResidualModel
    StandardScaler + Ridge regression (alpha=10.0).  Preferred for N≈73.

TreeResidualModel
    StandardScaler + RandomForestRegressor with conservative depth/leaf
    constraints to limit overfitting at small sample counts.

Serialization
-------------
Both models persist via joblib.  save(path) / load(path) round-trip the
fitted pipeline including the scaler.

Offline trajectory correction evaluation
-----------------------------------------
evaluate_residual_corrections() applies predicted residuals to the physics
projected coordinates, reprojects to WGS-84, and calls
calculate_trajectory_metrics() for each model — returning a comparable
dict of metrics for Physics-only, Physics+Ridge, and Physics+Tree.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics.trajectory import calculate_trajectory_metrics
from src.models.iceberg_physics import CoordinateHandler
from src.models.ml.dataset import ResidualDatasetPartition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class ResidualModel(ABC):
    """
    Abstract base for two-output residual regression models.

    Subclasses must wrap a scikit-learn estimator and expose a consistent
    fit / predict / save / load API.  Pre-processing (e.g. StandardScaler)
    is entirely internal to each subclass.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ResidualModel":
        """
        Fit the model on training features and targets.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target matrix of shape (n_samples, 2) — [residual_x_m, residual_y_m].

        Returns:
            self (for chaining).
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict residuals.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) — predicted [residual_x_m, residual_y_m].
        """

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted model to disk (including any fitted scaler)."""

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "ResidualModel":
        """Restore a previously saved model from disk."""

    def residual_rmse(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compute RMSE of raw residual predictions.

        Returns a dict with keys: rmse_x_m, rmse_y_m, rmse_dist_m, rmse_dist_km.
        The distance RMSE is sqrt(mean(||pred - truth||²)).
        """
        y_pred = self.predict(X)
        diff_x = y_pred[:, 0] - y[:, 0]
        diff_y = y_pred[:, 1] - y[:, 1]
        rmse_x = float(np.sqrt(np.mean(diff_x ** 2)))
        rmse_y = float(np.sqrt(np.mean(diff_y ** 2)))
        dist_err = np.hypot(diff_x, diff_y)
        rmse_dist = float(np.sqrt(np.mean(dist_err ** 2)))
        return {
            "rmse_x_m": rmse_x,
            "rmse_y_m": rmse_y,
            "rmse_dist_m": rmse_dist,
            "rmse_dist_km": rmse_dist / 1000.0,
        }


# ---------------------------------------------------------------------------
# Ridge regression model
# ---------------------------------------------------------------------------

class RidgeResidualModel(ResidualModel):
    """
    Regularized linear regression (Ridge, alpha=10) with StandardScaler.

    Preferred for the extremely small N≈73 dataset: low variance, no
    decision-tree overfitting, closed-form solution.

    Scaler parameters are fitted **only** on training data (fit() call).
    predict() transforms with the already-fitted scaler.
    """

    def __init__(self, alpha: float = 10.0, random_state: int = 42) -> None:
        self.alpha = alpha
        self.random_state = random_state
        self._pipeline: Optional[Pipeline] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeResidualModel":
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha, random_state=self.random_state)),
        ])
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return np.asarray(self._pipeline.predict(X), dtype=np.float64)

    def save(self, path: Union[str, Path]) -> None:
        if self._pipeline is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        joblib.dump({"type": "RidgeResidualModel", "pipeline": self._pipeline,
                     "alpha": self.alpha, "random_state": self.random_state}, path)
        logger.info(f"RidgeResidualModel saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RidgeResidualModel":
        state = joblib.load(path)
        if state.get("type") != "RidgeResidualModel":
            raise ValueError(f"Checkpoint type mismatch: expected RidgeResidualModel, got {state.get('type')}")
        model = cls(alpha=state["alpha"], random_state=state["random_state"])
        model._pipeline = state["pipeline"]
        logger.info(f"RidgeResidualModel loaded from {path}")
        return model


# ---------------------------------------------------------------------------
# Shallow tree-based model
# ---------------------------------------------------------------------------

class TreeResidualModel(ResidualModel):
    """
    Shallow RandomForestRegressor with StandardScaler pre-processing.

    Conservative depth/leaf hyperparameters reduce overfitting at small N,
    but Ridge is expected to generalise better for this dataset size.

    Hyperparameter defaults (conservative for N≈73):
        n_estimators=50, max_depth=3, min_samples_split=4, min_samples_leaf=2

    The scaler is fitted ONLY inside fit() and is not re-fitted on predict().
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 3,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._pipeline: Optional[Pipeline] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TreeResidualModel":
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", rf),
        ])
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return np.asarray(self._pipeline.predict(X), dtype=np.float64)

    def save(self, path: Union[str, Path]) -> None:
        if self._pipeline is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        joblib.dump({
            "type": "TreeResidualModel",
            "pipeline": self._pipeline,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }, path)
        logger.info(f"TreeResidualModel saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TreeResidualModel":
        state = joblib.load(path)
        if state.get("type") != "TreeResidualModel":
            raise ValueError(f"Checkpoint type mismatch: expected TreeResidualModel, got {state.get('type')}")
        model = cls(
            n_estimators=state["n_estimators"],
            max_depth=state["max_depth"],
            min_samples_split=state["min_samples_split"],
            min_samples_leaf=state["min_samples_leaf"],
            random_state=state["random_state"],
        )
        model._pipeline = state["pipeline"]
        logger.info(f"TreeResidualModel loaded from {path}")
        return model


# ---------------------------------------------------------------------------
# Offline trajectory correction evaluation
# ---------------------------------------------------------------------------

def _corrected_trajectory_dataframe(
    partition: ResidualDatasetPartition,
    y_pred: np.ndarray,
    crs: str = "EPSG:3412",
) -> pd.DataFrame:
    """
    Build a lat/lon trajectory DataFrame by applying predicted residuals to
    the physics projected coordinates stored in partition.metadata.

    Steps:
      1.  x_corrected = physics_x_m + pred_residual_x
          y_corrected = physics_y_m + pred_residual_y
      2.  Convert (x_corrected, y_corrected) → (lon, lat) via CoordinateHandler.to_geographic()
      3.  Return DataFrame with columns: timestamp, latitude, longitude

    Args:
        partition: A ResidualDatasetPartition (val or test) with metadata.
        y_pred:    Predicted residuals, shape (n_samples, 2).
        crs:       Projected CRS used by the physics model (default EPSG:3412).

    Returns:
        DataFrame with columns [timestamp, latitude, longitude].
    """
    ch = CoordinateHandler(crs=crs)
    meta = partition.metadata

    physics_x = meta["physics_x_m"].values
    physics_y = meta["physics_y_m"].values

    x_corrected = physics_x + y_pred[:, 0]
    y_corrected = physics_y + y_pred[:, 1]

    lons: List[float] = []
    lats: List[float] = []
    for xc, yc in zip(x_corrected, y_corrected):
        lon_c, lat_c = ch.to_geographic(float(xc), float(yc))
        lons.append(lon_c)
        lats.append(lat_c)

    return pd.DataFrame({
        "timestamp": partition.timestamps,
        "latitude": lats,
        "longitude": lons,
    })


def _physics_only_trajectory_dataframe(
    partition: ResidualDatasetPartition,
    crs: str = "EPSG:3412",
) -> pd.DataFrame:
    """
    Build a lat/lon trajectory DataFrame using ONLY the physics projected
    coordinates (no ML correction) stored in partition.metadata.
    """
    ch = CoordinateHandler(crs=crs)
    meta = partition.metadata

    lons: List[float] = []
    lats: List[float] = []
    for xp, yp in zip(meta["physics_x_m"].values, meta["physics_y_m"].values):
        lon_p, lat_p = ch.to_geographic(float(xp), float(yp))
        lons.append(lon_p)
        lats.append(lat_p)

    return pd.DataFrame({
        "timestamp": partition.timestamps,
        "latitude": lats,
        "longitude": lons,
    })


def _truth_trajectory_dataframe(
    partition: ResidualDatasetPartition,
    crs: str = "EPSG:3412",
) -> pd.DataFrame:
    """
    Build a lat/lon trajectory DataFrame from the ground-truth projected
    coordinates stored in partition.metadata.
    """
    ch = CoordinateHandler(crs=crs)
    meta = partition.metadata

    lons: List[float] = []
    lats: List[float] = []
    for xt, yt in zip(meta["truth_x_m"].values, meta["truth_y_m"].values):
        lon_t, lat_t = ch.to_geographic(float(xt), float(yt))
        lons.append(lon_t)
        lats.append(lat_t)

    return pd.DataFrame({
        "timestamp": partition.timestamps,
        "latitude": lats,
        "longitude": lons,
    })


def evaluate_residual_corrections(
    partition: ResidualDatasetPartition,
    models: Dict[str, ResidualModel],
    crs: str = "EPSG:3412",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate trajectory metrics for physics-only and each physics+ML correction.

    For each model in *models*, predicted residuals are added to the physics
    projected coordinates and re-projected to WGS-84.  The corrected trajectory
    is then compared against ground-truth observations via
    calculate_trajectory_metrics().

    The key "physics_only" is always included (no model correction).

    Args:
        partition: A ResidualDatasetPartition (typically the test partition).
        models:    Dict mapping model label → fitted ResidualModel.
        crs:       Projected CRS (default EPSG:3412).

    Returns:
        Dict mapping label → trajectory metrics dict.
        Labels: "physics_only", and one per model key.
        Metric keys per label: final_position_error_km, mae_km, rmse_km,
            max_error_km, mean_heading_error_deg, matched_timestamps.
    """
    df_truth = _truth_trajectory_dataframe(partition, crs=crs)
    df_physics = _physics_only_trajectory_dataframe(partition, crs=crs)

    results: Dict[str, Dict[str, float]] = {}

    # Physics-only baseline
    results["physics_only"] = calculate_trajectory_metrics(df_physics, df_truth)

    # Each ML correction
    for label, model in models.items():
        y_pred = model.predict(partition.X)
        df_corrected = _corrected_trajectory_dataframe(partition, y_pred, crs=crs)
        results[label] = calculate_trajectory_metrics(df_corrected, df_truth)

    return results
