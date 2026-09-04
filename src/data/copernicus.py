"""
Copernicus Marine GLORYS Ocean Physics Reanalysis Data Loader.

Extracts and standardizes ocean forcing:
- Eastward sea water velocity (uo, m/s)
- Northward sea water velocity (vo, m/s)
- Sea water potential temperature / SST (thetao, Kelvin or °C)

CRITICAL REQUIREMENT:
Extracts near-surface currents (or depth-integrates over specified iceberg draft),
preventing accidental ingestion of deep-ocean/abyssal velocity profiles.
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

VAR_ALIASES = {
    "ocean_u": ["uo", "u_ocean", "eastward_sea_water_velocity", "u"],
    "ocean_v": ["vo", "v_ocean", "northward_sea_water_velocity", "v"],
    "sst": ["thetao", "sst", "temperature", "sea_surface_temperature", "to"],
}


class CopernicusLoader:
    """
    Reader and surface/keel current extractor for Copernicus Marine GLORYS ocean reanalyses.
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
            variable_mapping: Optional dict mapping {"ocean_u": "...", "ocean_v": "...", "sst": "..."}.
            max_allowed_timestamp: Optional historical cutoff timestamp to prevent temporal leaks.
        """
        if isinstance(source, (str, Path)):
            try:
                self.ds = xr.open_dataset(source)
            except Exception as e:
                raise IncompatibleDatasetError(f"Failed to open Copernicus NetCDF source '{source}': {e}") from e
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
        self.depth_dim = self._detect_optional_coord(["depth", "deptho", "lev", "z"])

        self.var_map = self._resolve_variables(variable_mapping)

    def _detect_coord(self, candidates: List[str]) -> str:
        for name in candidates:
            if name in self.ds.coords or name in self.ds.dims:
                return name
        raise IncompatibleDatasetError(
            f"Could not identify coordinate from {candidates}. "
            f"Available coords: {list(self.ds.coords.keys())}, dims: {list(self.ds.sizes.keys())}"
        )

    def _detect_optional_coord(self, candidates: List[str]) -> Optional[str]:
        for name in candidates:
            if name in self.ds.coords or name in self.ds.dims:
                return name
        return None

    def _resolve_variables(self, explicit_mapping: Optional[Dict[str, str]]) -> Dict[str, str]:
        resolved = {}
        for canonical_name, candidates in VAR_ALIASES.items():
            if explicit_mapping and canonical_name in explicit_mapping:
                explicit_var = explicit_mapping[canonical_name]
                if explicit_var not in self.ds.data_vars:
                    raise IncompatibleDatasetError(f"Variable '{explicit_var}' not found in Copernicus dataset.")
                resolved[canonical_name] = explicit_var
                continue

            found = None
            for cand in candidates:
                if cand in self.ds.data_vars:
                    found = cand
                    break
            if found is None:
                raise IncompatibleDatasetError(
                    f"Copernicus dataset missing required variable '{canonical_name}' "
                    f"(searched for {candidates}). Available: {list(self.ds.data_vars.keys())}"
                )
            resolved[canonical_name] = found

        return resolved

    def _normalize_longitude(self, lon: float) -> float:
        ds_lons = self.ds[self.lon_dim].values
        if (ds_lons >= 0).all() and lon < 0:
            return float(lon % 360.0)
        if (ds_lons <= 180).all() and (ds_lons >= -180).all() and lon > 180:
            return float(((lon + 180) % 360) - 180)
        return float(lon)

    def get_ocean_currents(
        self,
        timestamp: Union[str, pd.Timestamp],
        latitude: float,
        longitude: float,
        depth_m: float = 0.0,
        draft_meters: Optional[float] = None,
        method: str = "nearest",
    ) -> Dict[str, float]:
        """
        Query ocean current and temperature at the given coordinate.

        Args:
            timestamp: Target query time.
            latitude: Latitude (-90 to +90).
            longitude: Longitude (-180 to +180 or 0 to 360).
            depth_m: Specific depth in meters (default 0.0 for near-surface).
            draft_meters: If provided, computes a trapezoidal depth-average over [0, draft_meters].
            method: Spatial extraction method ("nearest" or "linear").

        Returns:
            Dict containing:
                - 'ocean_u': Eastward current velocity (m/s), surface or depth-averaged.
                - 'ocean_v': Northward current velocity (m/s), surface or depth-averaged.
                - 'sst': Ocean temperature (Kelvin). Represents surface SST when draft_meters is
                         None or 0; represents depth-averaged potential temperature across the submerged
                         keel draft when draft_meters > 0.
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
                f"Latitude {latitude} is outside Copernicus ocean domain [{lat_min:.2f}, {lat_max:.2f}]"
            )

        # 3. Causal temporal indexing: select strictly observations at or before t
        time_arr = self.ds[self.time_dim].values
        t_naive = t.tz_localize(None) if t.tzinfo is not None else t
        t_dt64 = np.datetime64(t_naive)

        valid_mask = time_arr <= t_dt64
        if not np.any(valid_mask):
            earliest_t = pd.Timestamp(time_arr.min())
            raise MissingDataError(
                f"No Copernicus ocean observations available at or before query timestamp {t}. "
                f"Earliest observation in dataset is {earliest_t}."
            )

        latest_past_time = time_arr[valid_mask].max()
        time_sub = self.ds.sel({self.time_dim: latest_past_time})

        # 4. Depth selection and draft integration
        if self.depth_dim is not None:
            depth_vals = np.asarray(self.ds[self.depth_dim].values, dtype=np.float64)
            abs_depths = np.abs(depth_vals)
            max_available_depth = float(np.max(abs_depths))

            if draft_meters is not None and draft_meters < 0:
                raise ValueError(f"draft_meters must be non-negative, got {draft_meters}")

            if draft_meters is not None and draft_meters > 0:
                # If requested draft exceeds available depth in the dataset, raise MissingDataError
                if draft_meters > max_available_depth:
                    raise MissingDataError(
                        f"Requested iceberg draft {draft_meters:.1f}m exceeds maximum available "
                        f"ocean depth {max_available_depth:.1f}m in Copernicus dataset."
                    )

                exact_idx = np.where(np.isclose(abs_depths, draft_meters, rtol=0.0, atol=1e-6))[0]
                shallower_indices = np.where(abs_depths < draft_meters)[0]
                deeper_indices = np.where(abs_depths > draft_meters)[0]

                if len(exact_idx) > 0:
                    matching_indices = np.where(abs_depths <= draft_meters)[0]
                elif len(shallower_indices) == 0 or len(deeper_indices) == 0:
                    raise MissingDataError(
                        f"Cannot represent requested draft boundary {draft_meters:.1f}m: "
                        "the Copernicus subset lacks a bracketing depth level."
                    )
                else:
                    shallow_idx = shallower_indices[np.argmax(abs_depths[shallower_indices])]
                    deep_idx = deeper_indices[np.argmin(abs_depths[deeper_indices])]
                    matching_indices = shallower_indices

                    shallow_depth = abs_depths[shallow_idx]
                    deep_depth = abs_depths[deep_idx]
                    weight = (draft_meters - shallow_depth) / (deep_depth - shallow_depth)
                    shallow = time_sub.isel({self.depth_dim: shallow_idx})
                    deeper = time_sub.isel({self.depth_dim: deep_idx})
                    endpoint = shallow + weight * (deeper - shallow)
                    endpoint = endpoint.expand_dims({self.depth_dim: [draft_meters]})
                    sub_levels = xr.concat(
                        [time_sub.isel({self.depth_dim: matching_indices}), endpoint],
                        dim=self.depth_dim,
                    )
                    sub_depths = np.append(
                        np.sort(abs_depths[matching_indices]), draft_meters
                    )

                if len(exact_idx) == 0:
                    delta_z = float(sub_depths[-1] - sub_depths[0])
                    if delta_z <= 0:
                        raise MissingDataError(
                            f"Cannot integrate requested draft boundary {draft_meters:.1f}m "
                            "over the available Copernicus depth levels."
                        )
                    depth_sub = sub_levels.assign_coords({self.depth_dim: sub_depths})
                    integral = depth_sub.integrate(coord=self.depth_dim)
                    depth_sub = integral / delta_z
                elif len(matching_indices) == 0:
                    shallowest_idx = int(np.argmin(abs_depths))
                    depth_sub = time_sub.isel({self.depth_dim: shallowest_idx})
                elif len(matching_indices) == 1:
                    depth_sub = time_sub.isel({self.depth_dim: matching_indices[0]})
                else:
                    sorted_indices = matching_indices[np.argsort(abs_depths[matching_indices])]
                    sub_levels = time_sub.isel({self.depth_dim: sorted_indices})
                    sub_depths = np.abs(sub_levels[self.depth_dim].values)
                    delta_z = float(sub_depths[-1] - sub_depths[0])

                    if abs(delta_z) > 0:
                        integral = sub_levels.integrate(coord=self.depth_dim)
                        depth_sub = integral / delta_z
                    else:
                        depth_sub = sub_levels.isel({self.depth_dim: 0})
            else:
                # Near-surface behavior when no draft is requested:
                # Select the level closest to depth_m (default 0.0 for near-surface)
                target_depth = float(depth_m) if depth_m is not None else 0.0
                closest_idx = int(np.argmin(np.abs(abs_depths - target_depth)))
                depth_sub = time_sub.isel({self.depth_dim: closest_idx})
        else:
            depth_sub = time_sub

        # 5. Spatial selection / interpolation
        if method == "linear":
            subset = depth_sub.interp({self.lat_dim: latitude, self.lon_dim: norm_lon}, method="linear")
        else:
            subset = depth_sub.sel({self.lat_dim: latitude, self.lon_dim: norm_lon}, method="nearest")

        out = {}
        for canonical, actual in self.var_map.items():
            val = float(subset[actual].values)
            if np.isnan(val):
                raise MissingDataError(
                    f"Copernicus variable '{canonical}' ({actual}) is NaN at time={t}, lat={latitude}, lon={norm_lon}"
                )
            out[canonical] = val

        # Standardize temperature to Kelvin
        if out["sst"] < 100.0:
            out["sst"] += 273.15

        return out
