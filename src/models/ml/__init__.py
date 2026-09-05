"""
Machine Learning & Hybrid Physics-ML Module for Antarctic Iceberg Drift.
"""

from src.models.ml.features import (
    ALL_FEATURE_NAMES,
    CORE_FEATURE_NAMES,
    STATIC_PROPERTY_NAMES,
    TARGET_NAMES,
    ResidualFeatureExtractor,
)
from src.models.ml.dataset import (
    ResidualDataset,
    ResidualDatasetPartition,
    ResidualSample,
    build_partition_from_samples,
    build_residual_dataset,
    build_residual_samples_from_trajectory,
    chronological_split,
)

__all__ = [
    "CORE_FEATURE_NAMES",
    "STATIC_PROPERTY_NAMES",
    "ALL_FEATURE_NAMES",
    "TARGET_NAMES",
    "ResidualFeatureExtractor",
    "ResidualSample",
    "ResidualDatasetPartition",
    "ResidualDataset",
    "build_residual_samples_from_trajectory",
    "build_partition_from_samples",
    "chronological_split",
    "build_residual_dataset",
]
