"""
Environmental Data Module.

Exports unified data abstraction layer, dataset loaders, and synthetic environmental providers.
"""

from src.data.copernicus import CopernicusLoader
from src.data.environment import (
    CompositeEnvironmentProvider,
    CoordinateOutOfBoundsError,
    EnvironmentalDataError,
    EnvironmentProvider,
    EnvironmentState,
    HistoricalEnvironmentProvider,
    HistoricalIntegrityViolationError,
    IncompatibleDatasetError,
    MissingDataError,
)
from src.data.era5 import ERA5Loader
from src.data.iceberg import (
    BYUConsolidatedDatabaseLoader,
    IcebergDatabaseLoader,
    create_synthetic_iceberg_track,
    parse_byu_date,
)
from src.data.nsidc import NSIDCLoader
from src.data.synthetic import (
    OceanicVortex,
    SyntheticEnvironment,
    create_synthetic_copernicus_dataset,
    create_synthetic_era5_dataset,
    create_synthetic_nsidc_dataset,
)

__all__ = [
    "EnvironmentProvider",
    "EnvironmentState",
    "CompositeEnvironmentProvider",
    "HistoricalEnvironmentProvider",
    "EnvironmentalDataError",
    "MissingDataError",
    "CoordinateOutOfBoundsError",
    "IncompatibleDatasetError",
    "HistoricalIntegrityViolationError",
    "NSIDCLoader",
    "ERA5Loader",
    "CopernicusLoader",
    "IcebergDatabaseLoader",
    "BYUConsolidatedDatabaseLoader",
    "parse_byu_date",
    "SyntheticEnvironment",
    "OceanicVortex",
    "create_synthetic_iceberg_track",
    "create_synthetic_nsidc_dataset",
    "create_synthetic_era5_dataset",
    "create_synthetic_copernicus_dataset",
]
