"""
Unit tests for Environmental Data Abstraction Layer and Data Loaders.

Verifies:
- Data loading, coordinate detection, and variable aliasing.
- Spatial/temporal extraction and nearest-neighbour interpolation.
- Missing and flag value handling (ensuring no silent replacement with zero).
- Temporal replay integrity checks (blocking future data leakage).
- BYU/NIC iceberg trajectory parsing and irregular interval tracking.
- Synthetic environmental provider reproducibility and spatial gradients.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.copernicus import CopernicusLoader
from src.data.environment import (
    CompositeEnvironmentProvider,
    CoordinateOutOfBoundsError,
    HistoricalIntegrityViolationError,
    IncompatibleDatasetError,
    MissingDataError,
)
from src.data.era5 import ERA5Loader
from src.data.iceberg import IcebergDatabaseLoader, create_synthetic_iceberg_track
from src.data.nsidc import NSIDCLoader
from src.data.synthetic import (
    OceanicVortex,
    SyntheticEnvironment,
    create_synthetic_copernicus_dataset,
    create_synthetic_era5_dataset,
    create_synthetic_nsidc_dataset,
)


class TestSyntheticEnvironment:
    """Tests for the Stage 2 Synthetic Environment Provider."""

    def test_constant_environment_query(self):
        env = SyntheticEnvironment(
            sea_ice_concentration=0.35,
            ocean_u=0.12,
            ocean_v=-0.04,
            sst=271.35,
            wind_u=8.0,
            wind_v=1.5,
            temperature=260.0,
            pressure=98000.0,
        )

        res = env.get_environment("2026-01-01 12:00:00", latitude=-65.0, longitude=-45.0)

        assert isinstance(res, dict)
        assert set(res.keys()) == {
            "sea_ice_concentration",
            "ocean_u",
            "ocean_v",
            "sst",
            "wind_u",
            "wind_v",
            "temperature",
            "pressure",
        }
        assert pytest.approx(res["sea_ice_concentration"]) == 0.35
        assert pytest.approx(res["ocean_u"]) == 0.12
        assert pytest.approx(res["ocean_v"]) == -0.04
        assert pytest.approx(res["sst"]) == 271.35
        assert pytest.approx(res["wind_u"]) == 8.0
        assert pytest.approx(res["wind_v"]) == 1.5
        assert pytest.approx(res["temperature"]) == 260.0
        assert pytest.approx(res["pressure"]) == 98000.0

    def test_get_state_dataclass(self):
        env = SyntheticEnvironment(sea_ice_concentration=0.5)
        state = env.get_state("2026-01-01", latitude=-66.0, longitude=20.0)

        assert state.sea_ice_concentration == 0.5
        assert state.latitude == -66.0
        assert state.longitude == 20.0
        assert state.timestamp == pd.Timestamp("2026-01-01")
        as_dict = state.to_dict()
        assert as_dict["sea_ice_concentration"] == 0.5

    def test_sic_spatial_gradient(self):
        # Base SIC at -60°S is 0.2. Gradient is 0.05 per degree further south.
        env = SyntheticEnvironment(
            sea_ice_concentration=0.20,
            sic_ref_lat=-60.0,
            sic_gradient_lat=0.05,
        )

        res_60 = env.get_environment("2026-01-01", latitude=-60.0, longitude=0.0)
        res_65 = env.get_environment("2026-01-01", latitude=-65.0, longitude=0.0)
        res_70 = env.get_environment("2026-01-01", latitude=-70.0, longitude=0.0)

        assert pytest.approx(res_60["sea_ice_concentration"]) == 0.20
        # 5 degrees south: 0.20 + 5 * 0.05 = 0.45
        assert pytest.approx(res_65["sea_ice_concentration"]) == 0.45
        # 10 degrees south: 0.20 + 10 * 0.05 = 0.70
        assert pytest.approx(res_70["sea_ice_concentration"]) == 0.70

    def test_oceanic_vortex(self):
        vortex = OceanicVortex(
            center_lat=-65.0,
            center_lon=-50.0,
            radius_km=100.0,
            max_tangential_velocity_mps=0.5,
        )
        env = SyntheticEnvironment(
            ocean_u=0.0,
            ocean_v=0.0,
            vortex=vortex,
        )

        # Center of vortex has zero tangential velocity
        center_res = env.get_environment("2026-01-01", latitude=-65.0, longitude=-50.0)
        assert pytest.approx(center_res["ocean_u"], abs=1e-3) == 0.0
        assert pytest.approx(center_res["ocean_v"], abs=1e-3) == 0.0

        # East of center: in S. Hemisphere clockwise flow, velocity should be southward (negative v)
        east_res = env.get_environment("2026-01-01", latitude=-65.0, longitude=-48.5)
        assert east_res["ocean_v"] < -0.1

    def test_historical_integrity_cutoff(self):
        env = SyntheticEnvironment(max_allowed_timestamp="2026-01-05 00:00:00")

        # Query before or at cutoff is allowed
        valid_res = env.get_environment("2026-01-04 12:00:00", latitude=-65.0, longitude=0.0)
        assert valid_res is not None

        # Query beyond cutoff must fail to prevent temporal data leakage
        with pytest.raises(HistoricalIntegrityViolationError):
            env.get_environment("2026-01-06 00:00:00", latitude=-65.0, longitude=0.0)


class TestNSIDCLoader:
    """Tests for NSIDC sea-ice concentration loader."""

    def test_load_and_query_synthetic_nsidc(self):
        ds = create_synthetic_nsidc_dataset()
        loader = NSIDCLoader(ds, crs="EPSG:3412")

        assert loader.var_name == "cdr_seaice_conc"
        assert loader.is_projected is True

        # Query a coordinate that transforms to valid inner grid bounds
        # Approximate point in Antarctic quadrant corresponding to dataset's x, y
        # We can also verify bounds handling:
        out_of_bounds_lat = 0.0  # Equator is far out of bounds for Antarctic polar stereographic
        with pytest.raises(CoordinateOutOfBoundsError):
            loader.get_sic("2026-01-02", latitude=out_of_bounds_lat, longitude=0.0)

    def test_missing_and_flag_values_not_replaced_with_zero(self):
        ds = create_synthetic_nsidc_dataset(include_flags=True)
        loader = NSIDCLoader(ds, crs="EPSG:3412")

        # Directly verify flag cleaning helper
        assert np.isnan(loader._clean_value(254))  # Land mask
        assert np.isnan(loader._clean_value(255))  # Missing value
        assert np.isnan(loader._clean_value(np.nan))
        assert loader._clean_value(0.75) == 0.75

    def test_incompatible_dataset_detection(self):
        # Dataset with missing coordinates
        empty_ds = xr.Dataset({"dummy": (("dim1",), [1, 2, 3])})
        with pytest.raises(IncompatibleDatasetError):
            NSIDCLoader(empty_ds)


class TestERA5Loader:
    """Tests for ERA5 atmospheric reanalysis loader."""

    def test_load_and_query_era5(self):
        ds = create_synthetic_era5_dataset()
        loader = ERA5Loader(ds)

        assert loader.var_map["wind_u"] == "u10"
        assert loader.var_map["wind_v"] == "v10"
        assert loader.var_map["temperature"] == "t2m"
        assert loader.var_map["pressure"] == "msl"

        res = loader.get_forcing("2026-01-01 06:00:00", latitude=-65.0, longitude=25.0)
        assert "wind_u" in res
        assert "wind_v" in res
        assert "temperature" in res
        assert "pressure" in res
        assert pytest.approx(res["wind_v"]) == 2.0
        assert pytest.approx(res["pressure"]) == 98500.0

    def test_out_of_bounds_latitude(self):
        ds = create_synthetic_era5_dataset()
        loader = ERA5Loader(ds)

        with pytest.raises(CoordinateOutOfBoundsError):
            loader.get_forcing("2026-01-01", latitude=10.0, longitude=25.0)

    def test_temporal_resampling(self):
        ds = create_synthetic_era5_dataset(num_hours=48)
        loader = ERA5Loader(ds)
        daily_loader = loader.resample_temporal(freq="1D")

        assert len(daily_loader.ds[daily_loader.time_dim]) == 2


class TestCopernicusLoader:
    """Tests for Copernicus GLORYS ocean reanalysis loader."""

    def test_load_and_query_surface_currents(self):
        ds = create_synthetic_copernicus_dataset()
        loader = CopernicusLoader(ds)

        assert loader.depth_dim == "depth"
        res = loader.get_ocean_currents("2026-01-02", latitude=-65.0, longitude=25.0, depth_m=0.0)

        assert "ocean_u" in res
        assert "ocean_v" in res
        assert "sst" in res
        # Near surface uo is ~ 0.10 m/s * exp(-0.5/100) ~ 0.0995
        assert pytest.approx(res["ocean_u"], abs=0.01) == 0.10
        assert pytest.approx(res["ocean_v"], abs=0.01) == -0.04
        # SST should be standardized to Kelvin
        assert res["sst"] > 250.0

    def test_keel_draft_averaging(self):
        ds = create_synthetic_copernicus_dataset()
        loader = CopernicusLoader(ds)

        # Querying over 100m draft depth integrates upper layers
        res_draft = loader.get_ocean_currents(
            "2026-01-02", latitude=-65.0, longitude=25.0, draft_meters=100.0
        )
        assert res_draft["ocean_u"] > 0.0


class TestIcebergDatabaseLoader:
    """Tests for BYU/NIC iceberg trajectory loader."""

    def test_synthetic_iceberg_generation_and_loading(self):
        track_df = create_synthetic_iceberg_track(
            iceberg_id="A23a",
            num_observations=6,
            min_interval_days=1.5,
            max_interval_days=3.5,
        )
        loader = IcebergDatabaseLoader(track_df)

        ids = loader.get_iceberg_ids()
        assert ids == ["A23a"]

        traj = loader.get_trajectory("A23a")
        assert len(traj) == 6
        assert "time_delta_days" in traj.columns
        # Check that consecutive observations have irregular time differences > 1.0 day
        deltas = traj["time_delta_days"].dropna()
        assert len(deltas) == 5
        assert (deltas >= 1.4).all()
        assert (deltas <= 3.6).all()

    def test_trajectory_filtering(self):
        track_df = create_synthetic_iceberg_track(
            iceberg_id="B15a",
            start_time="2026-01-01",
            num_observations=10,
        )
        loader = IcebergDatabaseLoader(track_df)

        sub = loader.filter_by_time("2026-01-01", "2026-01-06")
        assert len(sub) > 0
        assert (sub["timestamp"] <= pd.to_datetime("2026-01-06", utc=True)).all()


class TestCompositeEnvironmentProvider:
    """Tests for Composite provider combining loaders."""

    def test_missing_component_raises_error(self):
        # Provider with missing ERA5 loader must raise MissingDataError, never return 0.0 silently
        era5_ds = create_synthetic_era5_dataset()
        era5_loader = ERA5Loader(era5_ds)

        composite = CompositeEnvironmentProvider(era5_loader=era5_loader)
        with pytest.raises(MissingDataError, match="No NSIDC loader"):
            composite.get_environment("2026-01-01 00:00:00", latitude=-65.0, longitude=25.0)
