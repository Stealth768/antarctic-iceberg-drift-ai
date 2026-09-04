"""
Unified Environmental Data Abstraction Layer.

Defines standard interfaces, dataclasses, and composite providers for querying
sea-ice, atmospheric, and oceanographic environmental variables across Antarctica.
Forecasting and routing modules depend exclusively on this abstraction rather than
directly coupling to specific file formats or raw data providers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EnvironmentalDataError(Exception):
    """Base exception for environmental data operations."""
    pass


class MissingDataError(EnvironmentalDataError):
    """Raised when environmental data is missing or undefined at the requested spatio-temporal coordinate."""
    pass


class CoordinateOutOfBoundsError(EnvironmentalDataError):
    """Raised when queried coordinates lie outside the spatial domain of the provider."""
    pass


class IncompatibleDatasetError(EnvironmentalDataError):
    """Raised when a dataset does not meet required variables, dimensions, or coordinate specifications."""
    pass


class HistoricalIntegrityViolationError(EnvironmentalDataError):
    """Raised when an environmental query attempts to access data beyond the simulation forecast cutoff timestamp."""
    pass


@dataclass(frozen=True)
class EnvironmentState:
    """
    Environmental state at a specific spatio-temporal coordinate.

    Attributes:
        sea_ice_concentration: Fractional sea-ice concentration [0.0, 1.0], unitless.
        ocean_u: Eastward near-surface ocean current velocity, m/s.
        ocean_v: Northward near-surface ocean current velocity, m/s.
        sst: Sea surface temperature, Kelvin.
        wind_u: Eastward 10-meter wind velocity, m/s.
        wind_v: Northward 10-meter wind velocity, m/s.
        temperature: 2-meter atmospheric air temperature, Kelvin.
        pressure: Mean sea-level atmospheric pressure, Pascals (Pa).
        timestamp: Query timestamp, UTC.
        latitude: Latitude in degrees North [-90.0, 90.0].
        longitude: Longitude in degrees East [-180.0, 180.0].
    """
    sea_ice_concentration: float
    ocean_u: float
    ocean_v: float
    sst: float
    wind_u: float
    wind_v: float
    temperature: float
    pressure: float
    timestamp: Optional[pd.Timestamp] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Return clean dictionary representation conforming to the SIH26059 interface."""
        return {
            "sea_ice_concentration": float(self.sea_ice_concentration),
            "ocean_u": float(self.ocean_u),
            "ocean_v": float(self.ocean_v),
            "sst": float(self.sst),
            "wind_u": float(self.wind_u),
            "wind_v": float(self.wind_v),
            "temperature": float(self.temperature),
            "pressure": float(self.pressure),
        }


class EnvironmentProvider(ABC):
    """
    Abstract Base Class for environmental data providers.

    All real (NSIDC, ERA5, Copernicus) and synthetic environmental data sources
    must implement this interface to be consumable by the drift physics solver
    and polar navigation routing engine.
    """

    def __init__(self, max_allowed_timestamp: Optional[Union[str, datetime, pd.Timestamp]] = None):
        """
        Args:
            max_allowed_timestamp: Optional cutoff timestamp for historical replay integrity.
                                  Queries with timestamp > max_allowed_timestamp will be rejected.
        """
        self.max_allowed_timestamp = (
            pd.Timestamp(max_allowed_timestamp) if max_allowed_timestamp is not None else None
        )

    def _check_temporal_integrity(self, timestamp: pd.Timestamp) -> None:
        """Enforce that future observations cannot leak into the model feature set."""
        if self.max_allowed_timestamp is not None and timestamp > self.max_allowed_timestamp:
            raise HistoricalIntegrityViolationError(
                f"Temporal leak detected: Query timestamp {timestamp} exceeds "
                f"forecast initialization cutoff {self.max_allowed_timestamp}."
            )

    @abstractmethod
    def get_environment(
        self,
        timestamp: Union[str, datetime, pd.Timestamp],
        latitude: float,
        longitude: float,
    ) -> Dict[str, float]:
        """
        Retrieve environmental conditions at the given time and geographical location.

        Args:
            timestamp: Query timestamp.
            latitude: Latitude in degrees (-90 to +90).
            longitude: Longitude in degrees (-180 to +180 or 0 to 360).

        Returns:
            Dictionary with keys:
                sea_ice_concentration, ocean_u, ocean_v, sst,
                wind_u, wind_v, temperature, pressure
        """
        pass

    def get_state(
        self,
        timestamp: Union[str, datetime, pd.Timestamp],
        latitude: float,
        longitude: float,
    ) -> EnvironmentState:
        """Retrieve full EnvironmentState dataclass instance."""
        t = pd.Timestamp(timestamp)
        data = self.get_environment(t, latitude, longitude)
        return EnvironmentState(
            sea_ice_concentration=data["sea_ice_concentration"],
            ocean_u=data["ocean_u"],
            ocean_v=data["ocean_v"],
            sst=data["sst"],
            wind_u=data["wind_u"],
            wind_v=data["wind_v"],
            temperature=data["temperature"],
            pressure=data["pressure"],
            timestamp=t,
            latitude=latitude,
            longitude=longitude,
        )


class CompositeEnvironmentProvider(EnvironmentProvider):
    """
    Combines individual data loaders (NSIDC, ERA5, Copernicus) into a unified
    environmental provider conforming to the EnvironmentProvider contract.
    """

    def __init__(
        self,
        nsidc_loader: Optional[Any] = None,
        era5_loader: Optional[Any] = None,
        copernicus_loader: Optional[Any] = None,
        max_allowed_timestamp: Optional[Union[str, datetime, pd.Timestamp]] = None,
    ):
        super().__init__(max_allowed_timestamp=max_allowed_timestamp)
        self.nsidc_loader = nsidc_loader
        self.era5_loader = era5_loader
        self.copernicus_loader = copernicus_loader

    def get_environment(
        self,
        timestamp: Union[str, datetime, pd.Timestamp],
        latitude: float,
        longitude: float,
    ) -> Dict[str, float]:
        t = pd.Timestamp(timestamp)
        self._check_temporal_integrity(t)

        # 1. Sea ice concentration from NSIDC loader
        if self.nsidc_loader is not None:
            sic = self.nsidc_loader.get_sic(t, latitude, longitude)
            if np.isnan(sic):
                raise MissingDataError(
                    f"NSIDC sea ice concentration is NaN at (lat={latitude}, lon={longitude}, time={t})"
                )
        else:
            raise MissingDataError("No NSIDC loader configured to provide sea_ice_concentration.")

        # 2. Atmospheric fields from ERA5 loader
        if self.era5_loader is not None:
            atm = self.era5_loader.get_forcing(t, latitude, longitude)
            for k in ("wind_u", "wind_v", "temperature", "pressure"):
                if np.isnan(atm.get(k, np.nan)):
                    raise MissingDataError(f"ERA5 variable '{k}' is NaN at (lat={latitude}, lon={longitude}, time={t})")
            wind_u = atm["wind_u"]
            wind_v = atm["wind_v"]
            temperature = atm["temperature"]
            pressure = atm["pressure"]
        else:
            raise MissingDataError("No ERA5 loader configured to provide atmospheric forcing.")

        # 3. Ocean current fields from Copernicus loader
        if self.copernicus_loader is not None:
            ocean = self.copernicus_loader.get_ocean_currents(t, latitude, longitude)
            for k in ("ocean_u", "ocean_v", "sst"):
                if np.isnan(ocean.get(k, np.nan)):
                    raise MissingDataError(f"Copernicus variable '{k}' is NaN at (lat={latitude}, lon={longitude}, time={t})")
            ocean_u = ocean["ocean_u"]
            ocean_v = ocean["ocean_v"]
            sst = ocean["sst"]
        else:
            raise MissingDataError("No Copernicus loader configured to provide ocean currents.")

        return {
            "sea_ice_concentration": float(sic),
            "ocean_u": float(ocean_u),
            "ocean_v": float(ocean_v),
            "sst": float(sst),
            "wind_u": float(wind_u),
            "wind_v": float(wind_v),
            "temperature": float(temperature),
            "pressure": float(pressure),
        }


class HistoricalEnvironmentProvider(EnvironmentProvider):
    """
    Real/historical environmental data provider for polar navigation and iceberg drift modeling.

    Integrates:
    - ERA5 atmospheric reanalysis (u10, v10, t2m, msl)
    - Copernicus Marine GLORYS ocean reanalysis (uo, vo, thetao)
    - Optional NSIDC Sea Ice Concentration (or default open-water SIC)

    Enforces strict historical integrity:
    1. Rejects queries after max_allowed_timestamp (HistoricalIntegrityViolationError).
    2. Causal temporal indexing: observations from timestamps later than the query time
       are NEVER accessed or forward-filled into the past.
    3. Spatial interpolation: supports bilinear ('linear') or nearest-neighbor ('nearest')
       spatial extraction, with automated longitude wrapping and coordinate orientation.
    """

    def __init__(
        self,
        era5_source: Any,
        copernicus_source: Any,
        nsidc_source: Optional[Any] = None,
        default_sea_ice_concentration: float = 0.0,
        draft_meters: Optional[float] = 200.0,
        spatial_interpolation: str = "linear",
        max_allowed_timestamp: Optional[Union[str, datetime, pd.Timestamp]] = None,
    ):
        """
        Args:
            era5_source: NetCDF file path, xr.Dataset, or ERA5Loader instance.
            copernicus_source: NetCDF file path, xr.Dataset, or CopernicusLoader instance.
            nsidc_source: Optional NetCDF path, xr.Dataset, or NSIDCLoader instance.
            default_sea_ice_concentration: Fallback SIC when NSIDC is not configured (default 0.0).
            draft_meters: Characteristic submerged iceberg keel draft for depth-averaged currents (m).
            spatial_interpolation: Spatial interpolation method ('linear' or 'nearest').
            max_allowed_timestamp: Optional historical replay cutoff timestamp.
        """
        super().__init__(max_allowed_timestamp=max_allowed_timestamp)

        # Setup ERA5 loader
        if hasattr(era5_source, "get_forcing"):
            self.era5_loader = era5_source
        else:
            from src.data.era5 import ERA5Loader
            self.era5_loader = ERA5Loader(era5_source, max_allowed_timestamp=self.max_allowed_timestamp)

        # Setup Copernicus loader
        if hasattr(copernicus_source, "get_ocean_currents"):
            self.copernicus_loader = copernicus_source
        else:
            from src.data.copernicus import CopernicusLoader
            self.copernicus_loader = CopernicusLoader(
                copernicus_source, max_allowed_timestamp=self.max_allowed_timestamp
            )

        # Setup NSIDC loader
        if nsidc_source is not None:
            if hasattr(nsidc_source, "get_sic"):
                self.nsidc_loader = nsidc_source
            else:
                from src.data.nsidc import NSIDCLoader
                self.nsidc_loader = NSIDCLoader(
                    nsidc_source, max_allowed_timestamp=self.max_allowed_timestamp
                )
        else:
            self.nsidc_loader = None

        self.default_sea_ice_concentration = float(default_sea_ice_concentration)
        self.draft_meters = draft_meters
        self.spatial_interpolation = spatial_interpolation

    def get_environment(
        self,
        timestamp: Union[str, datetime, pd.Timestamp],
        latitude: float,
        longitude: float,
    ) -> Dict[str, float]:
        """
        Retrieve standardized environmental conditions at the given spatio-temporal coordinate.

        Returns:
            Dict containing:
                - ocean_u: Ocean eastward velocity (m/s), surface or depth-averaged.
                - ocean_v: Ocean northward velocity (m/s), surface or depth-averaged.
                - sst: Ocean temperature (K). Represents near-surface SST when draft_meters is None
                       or 0; represents depth-averaged potential temperature across the submerged draft
                       when draft_meters > 0 (retained under key 'sst' for API compatibility).
                - wind_u: 10m eastward wind (m/s).
                - wind_v: 10m northward wind (m/s).
                - temperature: 2m air temperature (K).
                - pressure: Mean sea-level pressure (Pa).
                - sea_ice_concentration: Fractional sea ice concentration [0, 1].
        """
        t = pd.Timestamp(timestamp)
        self._check_temporal_integrity(t)

        # 1. Atmospheric forcing from ERA5
        atm = self.era5_loader.get_forcing(
            timestamp=t,
            latitude=latitude,
            longitude=longitude,
            method=self.spatial_interpolation,
        )

        # 2. Ocean currents from Copernicus
        ocean = self.copernicus_loader.get_ocean_currents(
            timestamp=t,
            latitude=latitude,
            longitude=longitude,
            depth_m=0.0,
            draft_meters=self.draft_meters,
            method=self.spatial_interpolation,
        )

        # 3. Sea ice concentration
        if self.nsidc_loader is not None:
            sic = float(self.nsidc_loader.get_sic(t, latitude, longitude))
            if np.isnan(sic):
                raise MissingDataError(
                    f"NSIDC sea ice concentration is NaN at time={t}, lat={latitude}, lon={longitude}"
                )
        else:
            sic = self.default_sea_ice_concentration

        return {
            "sea_ice_concentration": sic,
            "ocean_u": float(ocean["ocean_u"]),
            "ocean_v": float(ocean["ocean_v"]),
            "sst": float(ocean["sst"]),
            "wind_u": float(atm["wind_u"]),
            "wind_v": float(atm["wind_v"]),
            "temperature": float(atm["temperature"]),
            "pressure": float(atm["pressure"]),
        }
