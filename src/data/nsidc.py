"""
NSIDC Antarctic Sea-Ice Concentration Data Loader.

Handles NetCDF/CDR daily polar stereographic sea-ice concentration files (EPSG:3412 / EPSG:3031).
Detects coordinate systems, handles quality flag masks, standardizes units to [0.0, 1.0],
and exposes robust temporal and spatial selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from src.data.environment import (
    CoordinateOutOfBoundsError,
    IncompatibleDatasetError,
    MissingDataError,
)

logger = logging.getLogger(__name__)

# Standard known variable names for NSIDC sea-ice products
CANDIDATE_SIC_VARS = [
    "cdr_seaice_conc",
    "goddard_merged_seaice_conc",
    "seaice_conc_cdr",
    "nsidc_nt_seaice_conc",
    "nsidc_bt_seaice_conc",
    "sea_ice_concentration",
    "sic",
    "ice_conc",
]

# NSIDC CDR flag values for non-ice conditions
# 251: pole hole, 252: unused, 253: coastline, 254: land, 255: missing
FLAG_VALUES = {251, 252, 253, 254, 255}


class NSIDCLoader:
    """
    Reader and spatio-temporal extractor for NSIDC Sea Ice Concentration datasets.
    """

    def __init__(
        self,
        source: Union[str, Path, xr.Dataset],
        variable_name: Optional[str] = None,
        crs: str = "EPSG:3412",
        auto_mask_flags: bool = True,
    ):
        """
        Args:
            source: Filepath to NetCDF file, or pre-loaded xarray.Dataset.
            variable_name: Explicit variable name. If None, auto-detected from metadata.
            crs: Coordinate Reference System string (default EPSG:3412, NSIDC South Polar Stereographic).
            auto_mask_flags: Whether to convert NSIDC land/missing flag codes to NaN upfront.
        """
        self.crs_str = crs
        self.auto_mask_flags = auto_mask_flags

        if isinstance(source, (str, Path)):
            try:
                self.ds = xr.open_dataset(source)
            except Exception as e:
                raise IncompatibleDatasetError(f"Failed to open NSIDC NetCDF source '{source}': {e}") from e
        elif isinstance(source, xr.Dataset):
            self.ds = source
        else:
            raise TypeError(f"Expected path or xr.Dataset, got {type(source)}")

        self.time_dim = self._detect_coord(["time", "valid_time", "tdim", "date"])
        self.x_dim = self._detect_coord(["x", "lon", "longitude", "xc"])
        self.y_dim = self._detect_coord(["y", "lat", "latitude", "yc"])

        self.var_name = self._detect_variable(variable_name)
        self.is_projected = self.x_dim == "x" and self.y_dim == "y"

        # Pre-process flags and scale factors so interpolation works correctly
        if self.auto_mask_flags:
            self._preprocess_dataset()

        # Initialize pyproj transformer from WGS84 (lat/lon) to grid CRS if projected
        if self.is_projected:
            try:
                self.transformer = pyproj.Transformer.from_crs("EPSG:4326", self.crs_str, always_xy=True)
            except Exception as e:
                raise IncompatibleDatasetError(f"Invalid CRS specification '{crs}': {e}") from e
        else:
            self.transformer = None

    def _detect_coord(self, candidates: List[str]) -> str:
        """Find the matching coordinate or dimension name in the dataset."""
        for name in candidates:
            if name in self.ds.coords or name in self.ds.dims:
                return name
        raise IncompatibleDatasetError(
            f"Could not identify coordinate among candidates {candidates}. "
            f"Available coordinates: {list(self.ds.coords.keys())}, dims: {list(self.ds.sizes.keys())}"
        )

    def _detect_variable(self, preferred_name: Optional[str]) -> str:
        """Locate the sea ice concentration variable using preferred name or metadata heuristics."""
        if preferred_name:
            if preferred_name in self.ds.data_vars:
                return preferred_name
            raise IncompatibleDatasetError(
                f"Requested variable '{preferred_name}' not found in dataset. "
                f"Available variables: {list(self.ds.data_vars.keys())}"
            )

        for candidate in CANDIDATE_SIC_VARS:
            if candidate in self.ds.data_vars:
                logger.info(f"Auto-detected NSIDC sea ice variable: '{candidate}'")
                return candidate

        # Metadata attribute search by standard_name
        for var_name, var in self.ds.data_vars.items():
            std_name = var.attrs.get("standard_name", "").lower()
            if std_name in ("sea_ice_area_fraction", "sea_ice_concentration"):
                logger.info(f"Detected sea ice variable by standard_name: '{var_name}'")
                return var_name

        raise IncompatibleDatasetError(
            f"No valid sea ice concentration variable detected. Available: {list(self.ds.data_vars.keys())}"
        )

    def _preprocess_dataset(self) -> None:
        """Pre-mask flag values and normalize scaling to avoid linear interpolation artifacts."""
        da = self.ds[self.var_name]

        # Mask flags
        mask = da.isin(list(FLAG_VALUES))
        da = da.where(~mask, np.nan)

        # Rescale if stored as integers (0–1000 or 0–100)
        max_val = float(da.max(skipna=True).values) if da.notnull().any() else 0.0
        if max_val > 100.0:
            da = da / 1000.0
        elif max_val > 1.01:
            da = da / 100.0

        self.ds[self.var_name] = da

    def get_sic(
        self,
        timestamp: Union[str, pd.Timestamp],
        latitude: float,
        longitude: float,
        method: str = "nearest",
    ) -> float:
        """
        Query sea-ice concentration [0.0, 1.0] at a given timestamp and lat/lon coordinate.

        Args:
            timestamp: Target query date/time.
            latitude: Latitude (-90 to +90 degrees).
            longitude: Longitude (-180 to +180 or 0 to 360 degrees).
            method: Spatial interpolation method ("nearest" or "linear").

        Returns:
            Fractional sea ice concentration [0.0, 1.0].
            Returns np.nan if in land mask, open ocean pole hole, or missing.
        """
        t = pd.Timestamp(timestamp)

        # 1. Temporal selection
        if self.time_dim in self.ds.coords and len(self.ds[self.time_dim]) > 0:
            time_slice = self.ds[self.var_name].sel({self.time_dim: t}, method="nearest")
        else:
            time_slice = self.ds[self.var_name]

        # 2. Spatial coordinate resolution
        if self.is_projected:
            x_proj, y_proj = self.transformer.transform(longitude, latitude)

            x_min, x_max = float(self.ds[self.x_dim].min()), float(self.ds[self.x_dim].max())
            y_min, y_max = float(self.ds[self.y_dim].min()), float(self.ds[self.y_dim].max())
            margin = 50000.0  # 50 km tolerance
            
            if not (x_min - margin <= x_proj <= x_max + margin and y_min - margin <= y_proj <= y_max + margin):
                raise CoordinateOutOfBoundsError(
                    f"Projected point ({x_proj:.1f}, {y_proj:.1f}) for lat/lon ({latitude}, {longitude}) "
                    f"is outside grid bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]"
                )

            val_raw = time_slice.sel({self.x_dim: x_proj, self.y_dim: y_proj}, method=method).values
        else:
            lon_query = longitude
            ds_lons = self.ds[self.x_dim].values
            if (ds_lons >= 0).all() and lon_query < 0:
                lon_query = lon_query % 360.0

            val_raw = time_slice.sel({self.y_dim: latitude, self.x_dim: lon_query}, method=method).values

        val = float(np.asarray(val_raw).squeeze())

        if np.isnan(val) or val < 0.0 or val > 1.0:
            return np.nan

        return float(val)

    def select_temporal(
        self,
        start_time: Union[str, pd.Timestamp],
        end_time: Union[str, pd.Timestamp],
    ) -> xr.Dataset:
        """Extract temporal subset of the dataset."""
        t_start = pd.Timestamp(start_time)
        t_end = pd.Timestamp(end_time)
        return self.ds.sel({self.time_dim: slice(t_start, t_end)})

    def close(self) -> None:
        """Close underlying xarray Dataset."""
        if hasattr(self, "ds"):
            self.ds.close()