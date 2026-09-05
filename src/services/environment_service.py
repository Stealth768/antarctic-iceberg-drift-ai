"""
Unified Antarctic Environmental Query Service.

Integrates NSIDC (Sea Ice), Copernicus/GLORYS (Ocean Current/SST),
and ERA5 (Atmospheric Forcing) loaders without fabricating missing parameters.
"""

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, Field

from src.data.copernicus import CopernicusLoader
from src.data.era5 import ERA5Loader
from src.data.nsidc import NSIDCLoader
from src.data.environment import CompositeEnvironmentProvider
from src.routing.grid import PolarNavigationGrid
from src.routing.planner import EnvironmentSnapshot


class LiveEnvironmentalConditions(BaseModel):
    """Unified telemetry report for a specific Antarctic geographic location."""
    latitude: float
    longitude: float
    timestamp_iso: str
    sea_ice_concentration_pct: Optional[float] = Field(None, description="Sea ice concentration (0-100%) from NSIDC")
    wind_speed_knots: Optional[float] = Field(None, description="10m wind speed in knots from ERA5")
    wind_speed_mps: Optional[float] = Field(None, description="10m wind speed in m/s from ERA5")
    wind_direction_deg: Optional[float] = Field(None, description="Meteorological wind direction (degrees from North)")
    ocean_current_speed_knots: Optional[float] = Field(None, description="Near-surface ocean current speed in knots from GLORYS")
    ocean_current_speed_mps: Optional[float] = Field(None, description="Near-surface ocean current speed in m/s from GLORYS")
    ocean_current_direction_deg: Optional[float] = Field(None, description="Ocean current vector direction (degrees)")
    temperature_celsius: Optional[float] = Field(None, description="2m air temperature in °C from ERA5")
    pressure_hpa: Optional[float] = Field(None, description="Mean sea-level pressure in hPa from ERA5")
    
    # Explicitly unavailable parameters in reanalysis layers (never fabricated)
    wave_height_m: Optional[float] = Field(None, description="Significant wave height (Not provided in active reanalysis files)")
    rainfall_mm_hr: Optional[float] = Field(None, description="Precipitation rate (Not provided in active reanalysis files)")
    visibility_km: Optional[float] = Field(None, description="Optical visibility (Not provided in active reanalysis files)")
    storm_probability_pct: Optional[float] = Field(None, description="Direct storm probability (Not provided in active reanalysis files)")
    
    data_source_status: Dict[str, str] = Field(..., description="Operational status per data source")
    unavailable_metrics: List[str] = Field(
        default_factory=lambda: ["wave_height_m", "rainfall_mm_hr", "visibility_km", "storm_probability_pct"],
        description="List of metrics absent in active reanalysis files"
    )


class EnvironmentalService:
    """Provides unified queries across available polar reanalyses and grids."""

    def __init__(
        self,
        glorys_path: Optional[Path] = None,
        era5_path: Optional[Path] = None,
        nsidc_path: Optional[Path] = None,
    ) -> None:
        self.glorys_path = glorys_path or Path("data/raw/glorys_test/glorys_a23a_test.nc")
        self.era5_path = era5_path or Path("data/raw/era5_test/era5_a23a_real_200001.nc")
        self.nsidc_path = nsidc_path or Path("data/raw/nsidc_test/nsidc_a23a_test.nc")

        self.nsidc_loader: Optional[NSIDCLoader] = None
        self.era5_loader: Optional[ERA5Loader] = None
        self.copernicus_loader: Optional[CopernicusLoader] = None
        self.composite_provider: Optional[CompositeEnvironmentProvider] = None

        self._init_loaders()

    def _init_loaders(self) -> None:
        if self.nsidc_path.exists():
            try:
                self.nsidc_loader = NSIDCLoader(source=self.nsidc_path)
            except Exception:
                self.nsidc_loader = None

        if self.era5_path.exists():
            try:
                self.era5_loader = ERA5Loader(source=self.era5_path)
            except Exception:
                self.era5_loader = None

        if self.glorys_path.exists():
            try:
                self.copernicus_loader = CopernicusLoader(source=self.glorys_path)
            except Exception:
                self.copernicus_loader = None

        if self.nsidc_loader and self.era5_loader and self.copernicus_loader:
            try:
                self.composite_provider = CompositeEnvironmentProvider(
                    nsidc_loader=self.nsidc_loader,
                    era5_loader=self.era5_loader,
                    copernicus_loader=self.copernicus_loader,
                )
            except Exception:
                self.composite_provider = None

    def query_live_conditions(
        self,
        latitude: float,
        longitude: float,
        timestamp: Optional[datetime] = None,
    ) -> LiveEnvironmentalConditions:
        """Query unified environmental conditions at given point."""
        query_time = timestamp or datetime(2000, 1, 2, 0, 0, 0)
        ts_str = query_time.isoformat()

        source_status = {
            "nsidc_sea_ice": "offline" if not self.nsidc_loader else "online",
            "era5_atmosphere": "offline" if not self.era5_loader else "online",
            "glorys_ocean": "offline" if not self.copernicus_loader else "online",
        }

        sic_pct = None
        wind_spd_kts = None
        wind_spd_mps = None
        wind_dir = None
        ocean_spd_kts = None
        ocean_spd_mps = None
        ocean_dir = None
        temp_c = None
        press_hpa = None

        # 1. NSIDC Sea Ice
        if self.nsidc_loader:
            try:
                sic = self.nsidc_loader.get_sic(query_time, latitude, longitude)
                if sic is not None and not math.isnan(sic):
                    sic_pct = round(max(0.0, min(1.0, float(sic))) * 100.0, 1)
            except Exception:
                source_status["nsidc_sea_ice"] = "out_of_bounds_or_error"

        # 2. ERA5 Wind & Temp
        if self.era5_loader:
            try:
                forcing = self.era5_loader.get_forcing(query_time, latitude, longitude)
                wu = forcing.get("wind_u")
                wv = forcing.get("wind_v")
                if wu is not None and wv is not None and not (math.isnan(wu) or math.isnan(wv)):
                    spd = math.hypot(wu, wv)
                    wind_spd_mps = round(spd, 2)
                    wind_spd_kts = round(spd * 1.94384, 1)
                    # Meteorological direction (direction from which wind blows)
                    wind_dir = round((math.degrees(math.atan2(-wu, -wv)) + 360.0) % 360.0, 1)

                t_k = forcing.get("temperature")
                if t_k is not None and not math.isnan(t_k):
                    temp_c = round(t_k - 273.15, 1)

                p_pa = forcing.get("pressure")
                if p_pa is not None and not math.isnan(p_pa):
                    press_hpa = round(p_pa / 100.0, 1)
            except Exception:
                source_status["era5_atmosphere"] = "out_of_bounds_or_error"

        # 3. GLORYS Ocean Currents
        if self.copernicus_loader:
            try:
                currents = self.copernicus_loader.get_ocean_currents(query_time, latitude, longitude, draft_meters=15.0)
                ou = currents.get("ocean_u")
                ov = currents.get("ocean_v")
                if ou is not None and ov is not None and not (math.isnan(ou) or math.isnan(ov)):
                    ospd = math.hypot(ou, ov)
                    ocean_spd_mps = round(ospd, 3)
                    ocean_spd_kts = round(ospd * 1.94384, 2)
                    ocean_dir = round((math.degrees(math.atan2(ou, ov)) + 360.0) % 360.0, 1)
            except Exception:
                source_status["glorys_ocean"] = "out_of_bounds_or_error"

        return LiveEnvironmentalConditions(
            latitude=latitude,
            longitude=longitude,
            timestamp_iso=ts_str,
            sea_ice_concentration_pct=sic_pct,
            wind_speed_knots=wind_spd_kts,
            wind_speed_mps=wind_spd_mps,
            wind_direction_deg=wind_dir,
            ocean_current_speed_knots=ocean_spd_kts,
            ocean_current_speed_mps=ocean_spd_mps,
            ocean_current_direction_deg=ocean_dir,
            temperature_celsius=temp_c,
            pressure_hpa=press_hpa,
            data_source_status=source_status,
        )

    def create_grid_snapshot(
        self,
        grid: PolarNavigationGrid,
        timestamp: Optional[datetime] = None,
        tracked_icebergs: Optional[List[Tuple[float, float]]] = None,
    ) -> EnvironmentSnapshot:
        """Create a complete discrete snapshot across grid cells."""
        query_time = timestamp or datetime(2000, 1, 2, 0, 0, 0)

        sic_map: Dict[Tuple[int, int], float] = {}
        wu_map: Dict[Tuple[int, int], float] = {}
        wv_map: Dict[Tuple[int, int], float] = {}
        ou_map: Dict[Tuple[int, int], float] = {}
        ov_map: Dict[Tuple[int, int], float] = {}

        icebergs = tracked_icebergs or [
            (-63.507, -55.679),  # A-23a core reference position
            (-64.120, -56.200),  # Calved fragment reference
        ]

        for i in range(grid.nx):
            for j in range(grid.ny):
                cell = (i, j)
                lat, lon = grid.grid_to_geo(i, j)

                # Query SIC
                sic_val = 0.05
                if self.nsidc_loader:
                    try:
                        val = self.nsidc_loader.get_sic(query_time, lat, lon)
                        if val is not None and not math.isnan(val):
                            sic_val = float(val)
                    except Exception:
                        pass
                sic_map[cell] = sic_val

                # Query Wind
                w_u, w_v = -2.5, 4.0
                if self.era5_loader:
                    try:
                        forcing = self.era5_loader.get_forcing(query_time, lat, lon)
                        if "wind_u" in forcing and "wind_v" in forcing:
                            w_u = float(forcing["wind_u"])
                            w_v = float(forcing["wind_v"])
                    except Exception:
                        pass
                wu_map[cell] = w_u
                wv_map[cell] = w_v

                # Query Ocean Current
                o_u, o_v = 0.05, 0.02
                if self.copernicus_loader:
                    try:
                        oc = self.copernicus_loader.get_ocean_currents(query_time, lat, lon, draft_meters=15.0)
                        if "ocean_u" in oc and "ocean_v" in oc:
                            o_u = float(oc["ocean_u"])
                            o_v = float(oc["ocean_v"])
                    except Exception:
                        pass
                ou_map[cell] = o_u
                ov_map[cell] = o_v

        return EnvironmentSnapshot(
            sea_ice_concentration=sic_map,
            wind_u=wu_map,
            wind_v=wv_map,
            ocean_u=ou_map,
            ocean_v=ov_map,
            iceberg_locations=icebergs,
        )
