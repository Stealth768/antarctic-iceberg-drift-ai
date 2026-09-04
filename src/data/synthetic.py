"""
Deterministic Synthetic Environmental Provider and Fixture Generator.

Provides mathematically rigorous, reproducible test conditions (constant fields,
linear spatial gradients, and Antarctic gyre vortexes) to validate physics solvers
and route planning algorithms without requiring multi-gigabyte satellite/reanalysis downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from src.data.environment import EnvironmentProvider, EnvironmentState


@dataclass
class OceanicVortex:
    """
    Representation of an idealized circular oceanic eddy or gyre (e.g., Weddell/Ross Gyre).

    Attributes:
        center_lat: Center latitude in degrees North.
        center_lon: Center longitude in degrees East.
        radius_km: Characteristic radius of the eddy core in kilometers.
        max_tangential_velocity_mps: Maximum tangential swirl velocity (positive = cyclonic / clockwise in S. Hemisphere).
    """
    center_lat: float
    center_lon: float
    radius_km: float
    max_tangential_velocity_mps: float


class SyntheticEnvironment(EnvironmentProvider):
    """
    Deterministic environmental condition generator conforming to EnvironmentProvider.
    """

    def __init__(
        self,
        sea_ice_concentration: float = 0.20,
        ocean_u: float = 0.05,
        ocean_v: float = -0.02,
        sst: float = 271.35,  # ~ -1.8 °C (freezing point of polar seawater)
        wind_u: float = 5.00,
        wind_v: float = 2.00,
        temperature: float = 263.15,  # ~ -10.0 °C
        pressure: float = 98500.0,  # 985 hPa typical Antarctic circumpolar low
        sic_gradient_lat: float = 0.0,  # delta SIC per degree south of ref_lat
        sic_ref_lat: float = -60.0,
        vortex: Optional[OceanicVortex] = None,
        max_allowed_timestamp: Optional[Union[str, datetime, pd.Timestamp]] = None,
    ):
        """
        Args:
            sea_ice_concentration: Base fractional SIC [0.0, 1.0].
            ocean_u: Base eastward current velocity (m/s).
            ocean_v: Base northward current velocity (m/s).
            sst: Sea surface temperature (Kelvin).
            wind_u: 10m eastward wind velocity (m/s).
            wind_v: 10m northward wind velocity (m/s).
            temperature: 2m air temperature (Kelvin).
            pressure: Mean sea-level pressure (Pa).
            sic_gradient_lat: Rate of change of SIC per degree southward.
            sic_ref_lat: Reference latitude where base SIC applies.
            vortex: Optional oceanic vortex component.
            max_allowed_timestamp: Cutoff timestamp for historical forecast replay.
        """
        super().__init__(max_allowed_timestamp=max_allowed_timestamp)
        self.base_sic = float(sea_ice_concentration)
        self.base_ocean_u = float(ocean_u)
        self.base_ocean_v = float(ocean_v)
        self.base_sst = float(sst)
        self.base_wind_u = float(wind_u)
        self.base_wind_v = float(wind_v)
        self.base_temp = float(temperature)
        self.base_pressure = float(pressure)
        self.sic_gradient_lat = float(sic_gradient_lat)
        self.sic_ref_lat = float(sic_ref_lat)
        self.vortex = vortex

    def get_environment(
        self,
        timestamp: Union[str, datetime, pd.Timestamp],
        latitude: float,
        longitude: float,
    ) -> Dict[str, float]:
        t = pd.Timestamp(timestamp)
        self._check_temporal_integrity(t)

        # 1. Compute SIC with optional southward latitudinal gradient
        delta_lat_south = max(0.0, self.sic_ref_lat - latitude)
        sic = self.base_sic + self.sic_gradient_lat * delta_lat_south
        sic = float(np.clip(sic, 0.0, 1.0))

        # 2. Compute Ocean current with optional vortex perturbation
        ocean_u = self.base_ocean_u
        ocean_v = self.base_ocean_v

        if self.vortex is not None:
            rad_lat = np.radians(self.vortex.center_lat)
            dy_km = (latitude - self.vortex.center_lat) * 111.139
            dx_km = (longitude - self.vortex.center_lon) * 111.139 * np.cos(rad_lat)
            dist_km = np.hypot(dx_km, dy_km)

            if dist_km > 1e-3:
                r_norm = dist_km / max(self.vortex.radius_km, 1e-3)
                v_tangent = self.vortex.max_tangential_velocity_mps * r_norm * np.exp(-0.5 * (r_norm**2))
                theta = np.arctan2(dy_km, dx_km)
                ocean_u += float(v_tangent * np.sin(theta))
                ocean_v += float(-v_tangent * np.cos(theta))

        return {
            "sea_ice_concentration": sic,
            "ocean_u": float(ocean_u),
            "ocean_v": float(ocean_v),
            "sst": self.base_sst,
            "wind_u": self.base_wind_u,
            "wind_v": self.base_wind_v,
            "temperature": self.base_temp,
            "pressure": self.base_pressure,
        }


def create_synthetic_nsidc_dataset(
    start_date: str = "2026-01-01",
    num_days: int = 5,
    nx: int = 15,
    ny: int = 15,
    include_flags: bool = True,
) -> xr.Dataset:
    times = pd.date_range(start=start_date, periods=num_days, freq="1D")
    x_coords = np.linspace(-1500, -1150, nx) * 1000.0
    y_coords = np.linspace(500, 850, ny) * 1000.0

    base_sic = np.linspace(0.1, 0.95, ny)[:, None] * np.ones((1, nx))
    data = np.zeros((num_days, ny, nx), dtype=np.float32)
    for t in range(num_days):
        data[t, :, :] = base_sic + 0.02 * np.sin(t)

    if include_flags:
        data[:, 0, 0] = np.nan
        data[:, -1, -1] = np.nan

    ds = xr.Dataset(
        data_vars={
            "cdr_seaice_conc": (
                ("time", "y", "x"),
                data,
                {
                    "standard_name": "sea_ice_area_fraction",
                    "long_name": "NOAA/NSIDC Climate Data Record Sea Ice Concentration",
                    "units": "1",
                    "valid_range": [0.0, 1.0],
                },
            )
        },
        coords={
            "time": times,
            "y": y_coords,
            "x": x_coords,
        },
        attrs={
            "title": "Synthetic NSIDC Sea Ice Concentration Dataset for Testing",
            "crs": "EPSG:3412",
        },
    )
    return ds


def create_synthetic_era5_dataset(
    start_date: str = "2026-01-01 00:00:00",
    num_hours: int = 24,
    lats: Optional[np.ndarray] = None,
    lons: Optional[np.ndarray] = None,
) -> xr.Dataset:
    times = pd.date_range(start=start_date, periods=num_hours, freq="1h")
    if lats is None:
        lats = np.linspace(-75.0, -60.0, 7)
    if lons is None:
        lons = np.linspace(20.0, 40.0, 9)

    nt = len(times)
    nlat = len(lats)
    nlon = len(lons)

    u10 = 5.0 + np.sin(np.linspace(0, 2 * np.pi, nt))[:, None, None] * np.ones((1, nlat, nlon))
    v10 = 2.0 * np.ones((nt, nlat, nlon))
    t2m_lat = 265.0 - (lats - (-60.0)) * 0.5  # shape (nlat,)
    t2m = np.broadcast_to(t2m_lat[None, :, None], (nt, nlat, nlon)).copy()
    msl = 98500.0 * np.ones((nt, nlat, nlon))

    ds = xr.Dataset(
        data_vars={
            "u10": (("time", "latitude", "longitude"), u10, {"units": "m s**-1", "long_name": "10 metre U wind component"}),
            "v10": (("time", "latitude", "longitude"), v10, {"units": "m s**-1", "long_name": "10 metre V wind component"}),
            "t2m": (("time", "latitude", "longitude"), t2m, {"units": "K", "long_name": "2 metre temperature"}),
            "msl": (("time", "latitude", "longitude"), msl, {"units": "Pa", "long_name": "Mean sea level pressure"}),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "title": "Synthetic ERA5 Atmospheric Reanalysis for Testing",
        },
    )
    return ds


def create_synthetic_copernicus_dataset(
    start_date: str = "2026-01-01",
    num_days: int = 5,
    depths: Optional[np.ndarray] = None,
    lats: Optional[np.ndarray] = None,
    lons: Optional[np.ndarray] = None,
) -> xr.Dataset:
    times = pd.date_range(start=start_date, periods=num_days, freq="1D")
    if depths is None:
        depths = np.array([0.5, 5.0, 15.0, 50.0, 100.0, 500.0])
    if lats is None:
        lats = np.linspace(-75.0, -60.0, 7)
    if lons is None:
        lons = np.linspace(20.0, 40.0, 9)

    nt = len(times)
    ndepth = len(depths)
    nlat = len(lats)
    nlon = len(lons)

    depth_decay = np.exp(-depths / 100.0)[None, :, None, None]
    uo = 0.10 * depth_decay * np.ones((nt, ndepth, nlat, nlon))
    vo = -0.04 * depth_decay * np.ones((nt, ndepth, nlat, nlon))
    thetao = 271.35 * np.ones((nt, ndepth, nlat, nlon))

    ds = xr.Dataset(
        data_vars={
            "uo": (("time", "depth", "latitude", "longitude"), uo, {"units": "m s-1", "long_name": "Eastward velocity"}),
            "vo": (("time", "depth", "latitude", "longitude"), vo, {"units": "m s-1", "long_name": "Northward velocity"}),
            "thetao": (("time", "depth", "latitude", "longitude"), thetao, {"units": "degrees_C", "long_name": "Sea water potential temperature"}),
        },
        coords={
            "time": times,
            "depth": depths,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "title": "Synthetic Copernicus Marine GLORYS Ocean Reanalysis for Testing",
        },
    )
    return ds
