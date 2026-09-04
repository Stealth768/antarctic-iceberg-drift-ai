"""
ERA5 Atmospheric Reanalysis Data Loader.

Extracts and standardizes atmospheric surface forcing:
- 10m Eastward wind component (u10, m/s)
- 10m Northward wind component (v10, m/s)
- 2m Air temperature (t2m, Kelvin)
- Mean sea-level pressure (msl, Pa)
Supports hourly-to-daily temporal resampling, coordinate aliasing, and geographic interpolation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from src.data.environment import (
    CoordinateOutOfBoundsError,
    HistoricalIntegrityViolationError,
    IncompatibleDatasetError,
    MissingDataError,
)

logger = logging.getLogger(__name__)

# Standard ERA5 variable name mappings
VAR_ALIASES = {
    "wind_u": ["u10", "10u", "u_wind", "eastward_wind_at_10m", "u"],
    "wind_v": ["v10", "10v", "v_wind", "northward_wind_at_10m", "v"],
    "temperature": ["t2m", "2t", "temp_2m", "air_temperature", "t"],
    "pressure": ["msl", "mean_sea_level_pressure", "surface_pressure", "sp"],
}


class ERA5Loader:
    """
    Reader and interpolator for ERA5 atmospheric reanalysis NetCDF datasets.
    """

    def __init__(
        self,
        source: Union[str, Path, xr.Dataset],
        variable_mapping: Optional[Dict[str, str]] = None,
        max_allowed_timestamp: Optional[Union[str, datetime, pd.Timestamp]] = None,
    ):
        """
        Args:
            source: NetCDF file path or xarray.Dataset.
            variable_mapping: Explicit dict mapping {"wind_u": "...", "wind_v": "...", ...}.
            max_allowed_timestamp: Optional historical cutoff timestamp to prevent temporal leaks.
        """
        if isinstance(source, (str, Path)):
            try:
                self.ds = xr.open_dataset(source)
            except Exception as e:
                raise IncompatibleDatasetError(f"Failed to open ERA5 NetCDF source '{source}': {e}") from e
        elif isinstance(source, xr.Dataset):
            self.ds = source
        else:
            raise TypeError(f"Expected path or xr.Dataset, got {type(source)}")

        self.max_allowed_timestamp = (
            pd.Timestamp(max_allowed_timestamp) if max_allowed_timestamp is not None else None
        )

        self.time_dim = self._detect_coord(["time", "valid_time", "date"])
        self.lat_dim = self._detect_coord(["latitude", "lat"])
        self.lon_dim = self._detect_coord(["longitude", "lon"])

        self.var_map = self._resolve_variables(variable_mapping)

    def _detect_coord(self, candidates: List[str]) -> str:
        for name in candidates:
            if name in self.ds.coords or name in self.ds.dims:
                return name
        raise IncompatibleDatasetError(
            f"Could not find coordinate from {candidates}. "
            f"Available coordinates: {list(self.ds.coords.keys())}, dims: {list(self.ds.sizes.keys())}"
        )

    def _resolve_variables(self, explicit_mapping: Optional[Dict[str, str]]) -> Dict[str, str]:
        resolved = {}
        for canonical_name, candidates in VAR_ALIASES.items():
            if explicit_mapping and canonical_name in explicit_mapping:
                explicit_var = explicit_mapping[canonical_name]
                if explicit_var not in self.ds.data_vars:
                    raise IncompatibleDatasetError(f"Variable '{explicit_var}' not found in ERA5 dataset.")
                resolved[canonical_name] = explicit_var
                continue

            found = None
            for cand in candidates:
                if cand in self.ds.data_vars:
                    found = cand
                    break
            if found is None:
                raise IncompatibleDatasetError(
                    f"ERA5 dataset missing required variable '{canonical_name}' "
                    f"(looked for {candidates}). Available: {list(self.ds.data_vars.keys())}"
                )
            resolved[canonical_name] = found

        return resolved

    def _normalize_longitude(self, lon: float) -> float:
        """Align query longitude with dataset longitude range ([0, 360] vs [-180, 180])."""
        ds_lons = self.ds[self.lon_dim].values
        if (ds_lons >= 0).all() and lon < 0:
            return float(lon % 360.0)
        if (ds_lons <= 180).all() and (ds_lons >= -180).all() and lon > 180:
            return float(((lon + 180) % 360) - 180)
        return float(lon)

    def get_forcing(
        self,
        timestamp: Union[str, pd.Timestamp],
        latitude: float,
        longitude: float,
        method: str = "nearest",
    ) -> Dict[str, float]:
        """
        Query atmospheric forcing at the given spatio-temporal coordinate.

        Supports method="nearest" or method="linear" (spatial interpolation).
        Enforces strict causal temporal indexing: never accesses observations later than timestamp.

        Returns:
            Dict containing 'wind_u', 'wind_v', 'temperature', 'pressure'.
        """
        t = pd.Timestamp(timestamp)

        # 1. Historical cutoff check
        if self.max_allowed_timestamp is not None:
            max_t = pd.Timestamp(self.max_allowed_timestamp)
            if t > max_t:
                raise HistoricalIntegrityViolationError(
                    f"Temporal leak detected: Query timestamp {t} exceeds "
                    f"historical cutoff {self.max_allowed_timestamp}."
                )

        norm_lon = self._normalize_longitude(longitude)

        # 2. Bounds check
        lat_min, lat_max = float(self.ds[self.lat_dim].min()), float(self.ds[self.lat_dim].max())
        if not (lat_min - 0.5 <= latitude <= lat_max + 0.5):
            raise CoordinateOutOfBoundsError(
                f"Latitude {latitude} is outside ERA5 domain [{lat_min:.2f}, {lat_max:.2f}]"
            )

        # 3. Causal temporal indexing: select strictly observations at or before t
        time_arr = self.ds[self.time_dim].values
        t_naive = t.tz_localize(None) if t.tzinfo is not None else t
        t_dt64 = np.datetime64(t_naive)

        valid_mask = time_arr <= t_dt64
        if not np.any(valid_mask):
            earliest_t = pd.Timestamp(time_arr.min())
            raise MissingDataError(
                f"No ERA5 observations available at or before query timestamp {t}. "
                f"Earliest observation in dataset is {earliest_t}."
            )

        latest_past_time = time_arr[valid_mask].max()
        time_slice = self.ds.sel({self.time_dim: latest_past_time})

        # 4. Spatial extraction / interpolation
        if method == "linear":
            subset = time_slice.interp({self.lat_dim: latitude, self.lon_dim: norm_lon}, method="linear")
        else:
            subset = time_slice.sel({self.lat_dim: latitude, self.lon_dim: norm_lon}, method="nearest")

        out = {}
        for canonical, actual in self.var_map.items():
            val = float(subset[actual].values)
            if np.isnan(val):
                raise MissingDataError(
                    f"ERA5 variable '{canonical}' ({actual}) is NaN at time={t}, lat={latitude}, lon={norm_lon}"
                )
            out[canonical] = val

        return out

    def resample_temporal(self, freq: str = "1D") -> ERA5Loader:
        """Aggregate/resample hourly data to daily or custom frequency."""
        resampled_ds = self.ds.resample({self.time_dim: freq}).mean()
        return ERA5Loader(resampled_ds, variable_mapping=self.var_map)

    def select_spatial(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> xr.Dataset:
        """Extract spatial bounding box."""
        if self.ds[self.lat_dim].values[0] > self.ds[self.lat_dim].values[-1]:
            lat_slice = slice(lat_max, lat_min)
        else:
            lat_slice = slice(lat_min, lat_max)

        norm_lon_min = self._normalize_longitude(lon_min)
        norm_lon_max = self._normalize_longitude(lon_max)
        lon_slice = slice(norm_lon_min, norm_lon_max)

        return self.ds.sel({self.lat_dim: lat_slice, self.lon_dim: lon_slice})
