"""
Physics-Residual ML Dataset Builder and Chronological Partitioner.

Constructs training, validation, and test datasets for residual learning:
    residual_x = x_truth - x_physics
    residual_y = y_truth - y_physics
Enforces strict chronological splitting to prevent future information leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.data.environment import EnvironmentProvider, MissingDataError
from src.models.iceberg_physics import CoordinateHandler, IcebergProperties
from src.models.ml.features import (
    ALL_FEATURE_NAMES,
    CORE_FEATURE_NAMES,
    ResidualFeatureExtractor,
    TARGET_NAMES,
)

logger = logging.getLogger(__name__)


@dataclass
class ResidualSample:
    """Individual data point containing state, environmental features, and target residuals."""
    timestamp: pd.Timestamp
    iceberg_id: str
    features: Dict[str, float]
    residual_x_m: float
    residual_y_m: float
    physics_x_m: float
    physics_y_m: float
    truth_x_m: float
    truth_y_m: float
    dt_seconds: float

    @property
    def residual_distance_m(self) -> float:
        """Euclidean distance of the displacement error in meters."""
        return float(np.hypot(self.residual_x_m, self.residual_y_m))

    @property
    def residual_distance_km(self) -> float:
        """Euclidean distance of the displacement error in kilometers."""
        return self.residual_distance_m / 1000.0


@dataclass
class ResidualDatasetPartition:
    """A single split (train, validation, or test) formatted for ML models."""
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    timestamps: pd.DatetimeIndex
    metadata: pd.DataFrame

    def __len__(self) -> int:
        return len(self.X)

    @property
    def num_samples(self) -> int:
        return len(self.X)

    @property
    def num_features(self) -> int:
        return self.X.shape[1] if len(self.X.shape) > 1 else 0

    def to_dataframe(self) -> pd.DataFrame:
        """Export features, targets, and metadata as a unified pandas DataFrame."""
        df_feat = pd.DataFrame(self.X, columns=self.feature_names, index=self.timestamps)
        for i, target_col in enumerate(self.target_names):
            df_feat[target_col] = self.y[:, i]
        for col in self.metadata.columns:
            if col not in df_feat.columns:
                df_feat[col] = self.metadata[col].values
        return df_feat


@dataclass
class ResidualDataset:
    """Complete container holding chronological train, validation, and test splits."""
    train: ResidualDatasetPartition
    val: ResidualDatasetPartition
    test: ResidualDatasetPartition
    feature_names: List[str]
    target_names: List[str]
    split_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_samples(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def summary(self) -> Dict[str, Any]:
        """Summary statistics of sample counts, temporal spans, and residual distributions."""
        def partition_stats(part: ResidualDatasetPartition) -> Dict[str, Any]:
            if len(part) == 0:
                return {"count": 0, "start": None, "end": None, "mean_residual_km": np.nan}
            res_km = np.hypot(part.y[:, 0], part.y[:, 1]) / 1000.0
            return {
                "count": len(part),
                "start": str(part.timestamps.min()),
                "end": str(part.timestamps.max()),
                "mean_residual_km": float(np.mean(res_km)),
                "median_residual_km": float(np.median(res_km)),
                "max_residual_km": float(np.max(res_km)),
            }

        return {
            "total_samples": self.total_samples,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "train": partition_stats(self.train),
            "val": partition_stats(self.val),
            "test": partition_stats(self.test),
            "split_info": self.split_info,
        }


def build_residual_samples_from_trajectory(
    df_sim: pd.DataFrame,
    df_truth: pd.DataFrame,
    environment_provider: EnvironmentProvider,
    iceberg_id: str = "A23A",
    iceberg_properties: Optional[IcebergProperties] = None,
    feature_extractor: Optional[ResidualFeatureExtractor] = None,
    on_missing: str = "skip",
) -> List[ResidualSample]:
    """
    Extract aligned feature-target residual samples along a matched trajectory.

    Args:
        df_sim: Simulation trajectory DataFrame (timestamp, latitude, longitude, vx_mps, vy_mps, x_m, y_m).
        df_truth: Ground-truth trajectory DataFrame (timestamp, latitude, longitude).
        environment_provider: Historical/Composite provider for atmospheric/oceanic fields.
        iceberg_id: Identifier string for the iceberg.
        iceberg_properties: Physical dimensions of the iceberg.
        feature_extractor: Custom feature extractor (defaults to standard ResidualFeatureExtractor).
        on_missing: Handling policy for missing environmental data ('skip' or 'raise').

    Returns:
        List of ResidualSample objects sorted chronologically.
    """
    if feature_extractor is None:
        feature_extractor = ResidualFeatureExtractor(crs="EPSG:3412", include_static_properties=True)

    coord_handler = feature_extractor.coord_handler

    merged = pd.merge(
        df_sim,
        df_truth,
        on="timestamp",
        suffixes=("_sim", "_truth"),
        how="inner",
    ).sort_values("timestamp").reset_index(drop=True)

    if merged.empty:
        raise ValueError("No matching timestamps between simulated and ground-truth trajectories.")

    t0 = pd.Timestamp(merged["timestamp"].iloc[0])
    samples: List[ResidualSample] = []
    skipped_count = 0

    for _, row in merged.iterrows():
        t = pd.Timestamp(row["timestamp"])
        dt_seconds = float((t - t0).total_seconds())

        lat_sim = float(row["latitude_sim"])
        lon_sim = float(row["longitude_sim"])
        vx_sim = float(row.get("vx_mps", 0.0))
        vy_sim = float(row.get("vy_mps", 0.0))

        # Simulation projected coordinates
        if "x_m" in row and "y_m" in row and np.isfinite(row["x_m"]) and np.isfinite(row["y_m"]):
            x_sim = float(row["x_m"])
            y_sim = float(row["y_m"])
        else:
            x_sim, y_sim = coord_handler.to_projected(lon_sim, lat_sim)

        # Ground truth projected coordinates
        lat_truth = float(row["latitude_truth"])
        lon_truth = float(row["longitude_truth"])
        x_truth, y_truth = coord_handler.to_projected(lon_truth, lat_truth)

        # Residual targets
        residual_x = float(x_truth - x_sim)
        residual_y = float(y_truth - y_sim)

        # Extract features at simulated position and current time
        try:
            feats = feature_extractor.extract_features(
                timestamp=t,
                latitude=lat_sim,
                longitude=lon_sim,
                physics_vx=vx_sim,
                physics_vy=vy_sim,
                dt_seconds=dt_seconds,
                environment_provider=environment_provider,
                iceberg_properties=iceberg_properties,
            )
        except MissingDataError as exc:
            if on_missing == "raise":
                raise
            skipped_count += 1
            logger.debug(f"Skipping sample at {t} due to missing environmental data: {exc}")
            continue

        sample = ResidualSample(
            timestamp=t,
            iceberg_id=iceberg_id,
            features=feats,
            residual_x_m=residual_x,
            residual_y_m=residual_y,
            physics_x_m=x_sim,
            physics_y_m=y_sim,
            truth_x_m=x_truth,
            truth_y_m=y_truth,
            dt_seconds=dt_seconds,
        )
        samples.append(sample)

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} samples along trajectory due to missing environmental data.")

    return samples


def build_partition_from_samples(
    samples: Sequence[ResidualSample],
    feature_names: Sequence[str],
) -> ResidualDatasetPartition:
    """Convert a sequence of ResidualSample objects into arrays and a partition object."""
    if len(samples) == 0:
        return ResidualDatasetPartition(
            X=np.empty((0, len(feature_names)), dtype=np.float32),
            y=np.empty((0, 2), dtype=np.float32),
            feature_names=list(feature_names),
            target_names=list(TARGET_NAMES),
            timestamps=pd.DatetimeIndex([]),
            metadata=pd.DataFrame(),
        )

    X_list: List[List[float]] = []
    y_list: List[List[float]] = []
    timestamps: List[pd.Timestamp] = []
    meta_records: List[Dict[str, Any]] = []

    for s in samples:
        row_feat = [float(s.features.get(f, np.nan)) for f in feature_names]
        X_list.append(row_feat)
        y_list.append([s.residual_x_m, s.residual_y_m])
        timestamps.append(s.timestamp)
        meta_records.append({
            "iceberg_id": s.iceberg_id,
            "physics_x_m": s.physics_x_m,
            "physics_y_m": s.physics_y_m,
            "truth_x_m": s.truth_x_m,
            "truth_y_m": s.truth_y_m,
            "dt_seconds": s.dt_seconds,
            "residual_distance_km": s.residual_distance_km,
        })

    X_arr = np.asarray(X_list, dtype=np.float32)
    y_arr = np.asarray(y_list, dtype=np.float32)
    ts_idx = pd.DatetimeIndex(timestamps)
    df_meta = pd.DataFrame(meta_records, index=ts_idx)

    return ResidualDatasetPartition(
        X=X_arr,
        y=y_arr,
        feature_names=list(feature_names),
        target_names=list(TARGET_NAMES),
        timestamps=ts_idx,
        metadata=df_meta,
    )


def chronological_split(
    samples: Sequence[ResidualSample],
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    train_end_time: Optional[Union[str, pd.Timestamp]] = None,
    val_end_time: Optional[Union[str, pd.Timestamp]] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> ResidualDataset:
    """
    Split samples strictly chronologically into train, validation, and test partitions.
    Guarantees that no future trajectory information leaks into the training partition.

    Args:
        samples: List of ResidualSample objects.
        train_frac: Fraction of samples for training (default 0.60).
        val_frac: Fraction of samples for validation (default 0.20).
        test_frac: Fraction of samples for testing (default 0.20).
        train_end_time: Optional explicit timestamp cutoff for end of training window.
        val_end_time: Optional explicit timestamp cutoff for end of validation window.
        feature_names: Feature names to include in X matrices.

    Returns:
        ResidualDataset with train, val, and test partitions.
    """
    if len(samples) == 0:
        raise ValueError("Cannot split empty sample list.")

    # Sort strictly by timestamp
    sorted_samples = sorted(samples, key=lambda s: s.timestamp)

    if feature_names is None:
        feature_names = list(sorted_samples[0].features.keys())

    if train_end_time is not None and val_end_time is not None:
        t_train_end = pd.Timestamp(train_end_time)
        t_val_end = pd.Timestamp(val_end_time)

        if t_train_end >= t_val_end:
            raise ValueError(f"train_end_time ({t_train_end}) must be before val_end_time ({t_val_end})")

        train_samples = [s for s in sorted_samples if s.timestamp <= t_train_end]
        val_samples = [s for s in sorted_samples if t_train_end < s.timestamp <= t_val_end]
        test_samples = [s for s in sorted_samples if s.timestamp > t_val_end]
        split_method = "explicit_timestamps"
    else:
        total = len(sorted_samples)
        norm_sum = train_frac + val_frac + test_frac
        if abs(norm_sum - 1.0) > 1e-4:
            raise ValueError(f"Fractions must sum to 1.0, got {norm_sum}")

        n_train = max(1, int(round(total * train_frac)))
        n_val = max(1, int(round(total * val_frac)))

        # Ensure bounds
        if n_train + n_val >= total:
            n_train = max(1, total - 2)
            n_val = 1

        train_samples = sorted_samples[:n_train]
        val_samples = sorted_samples[n_train:n_train + n_val]
        test_samples = sorted_samples[n_train + n_val:]
        split_method = "fractions"

    # Validation of chronological integrity
    if len(train_samples) > 0 and len(val_samples) > 0:
        max_train_t = max(s.timestamp for s in train_samples)
        min_val_t = min(s.timestamp for s in val_samples)
        if max_train_t >= min_val_t:
            raise ValueError(
                f"Chronological leak detected: Train max timestamp ({max_train_t}) >= Val min timestamp ({min_val_t})"
            )

    if len(val_samples) > 0 and len(test_samples) > 0:
        max_val_t = max(s.timestamp for s in val_samples)
        min_test_t = min(s.timestamp for s in test_samples)
        if max_val_t >= min_test_t:
            raise ValueError(
                f"Chronological leak detected: Val max timestamp ({max_val_t}) >= Test min timestamp ({min_test_t})"
            )

    train_part = build_partition_from_samples(train_samples, feature_names)
    val_part = build_partition_from_samples(val_samples, feature_names)
    test_part = build_partition_from_samples(test_samples, feature_names)

    split_info = {
        "split_method": split_method,
        "train_range": (str(train_part.timestamps.min()) if len(train_part) else None,
                        str(train_part.timestamps.max()) if len(train_part) else None),
        "val_range": (str(val_part.timestamps.min()) if len(val_part) else None,
                      str(val_part.timestamps.max()) if len(val_part) else None),
        "test_range": (str(test_part.timestamps.min()) if len(test_part) else None,
                       str(test_part.timestamps.max()) if len(test_part) else None),
        "train_count": len(train_part),
        "val_count": len(val_part),
        "test_count": len(test_part),
    }

    return ResidualDataset(
        train=train_part,
        val=val_part,
        test=test_part,
        feature_names=list(feature_names),
        target_names=list(TARGET_NAMES),
        split_info=split_info,
    )


def build_residual_dataset(
    df_sim: pd.DataFrame,
    df_truth: pd.DataFrame,
    environment_provider: EnvironmentProvider,
    iceberg_id: str = "A23A",
    iceberg_properties: Optional[IcebergProperties] = None,
    feature_extractor: Optional[ResidualFeatureExtractor] = None,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    train_end_time: Optional[Union[str, pd.Timestamp]] = None,
    val_end_time: Optional[Union[str, pd.Timestamp]] = None,
    on_missing: str = "skip",
) -> ResidualDataset:
    """
    High-level factory function: extracts residual samples and partitions them chronologically.
    """
    samples = build_residual_samples_from_trajectory(
        df_sim=df_sim,
        df_truth=df_truth,
        environment_provider=environment_provider,
        iceberg_id=iceberg_id,
        iceberg_properties=iceberg_properties,
        feature_extractor=feature_extractor,
        on_missing=on_missing,
    )

    return chronological_split(
        samples=samples,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        train_end_time=train_end_time,
        val_end_time=val_end_time,
        feature_names=feature_extractor.feature_names if feature_extractor else None,
    )
