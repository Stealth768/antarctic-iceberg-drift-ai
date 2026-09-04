"""
Tests for Stage 6B: Real Environmental Data Alignment.

Verifies:
1. HistoricalEnvironmentProvider variable extraction & EnvironmentState conformity.
2. Causal timestamp alignment (zero future observation leakage).
3. Bilinear spatial interpolation accuracy.
4. Longitude normalization and range wrapping ([0, 360) vs [-180, 180]).
5. Historical cutoff enforcement (HistoricalIntegrityViolationError on query > cutoff).
6. Rejection of unobserved past timestamps (no forward-filling from future data).
7. Full end-to-end compatibility with the Stage 3 iceberg drift physics solver.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.environment import (
    CoordinateOutOfBoundsError,
    EnvironmentState,
    HistoricalEnvironmentProvider,
    HistoricalIntegrityViolationError,
    MissingDataError,
)
from src.data.era5 import ERA5Loader
from src.data.copernicus import CopernicusLoader
from src.evaluation.historical_pairs import EvaluationPair
from src.evaluation.physics_evaluator import (
    IcebergPhysicsEvaluator,
    create_default_iceberg_properties,
)


@pytest.fixture
def alignment_datasets() -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Construct small synthetic NetCDF datasets representing ERA5 and Copernicus reanalyses
    covering 2020-01-01 to 2020-01-05 in the Weddell Sea / South Atlantic sector.
    """
    times_era5 = pd.date_range("2020-01-01 00:00:00", "2020-01-05 00:00:00", freq="6h")
    times_copernicus = pd.date_range("2020-01-01 00:00:00", "2020-01-05 00:00:00", freq="1D")

    lats = np.linspace(-70.0, -60.0, 11)  # 1-degree resolution
    lons_360 = np.linspace(300.0, 340.0, 9)  # 300E to 340E (-60W to -20W)
    lons_180 = np.linspace(-60.0, -20.0, 9)   # -60 to -20

    nt_era = len(times_era5)
    nt_cop = len(times_copernicus)
    nlat = len(lats)
    nlon = len(lons_360)

    # ERA5 with known linear spatial gradients
    # u10 has timestamp-dependent values: hour / 2.0
    u10_data = np.zeros((nt_era, nlat, nlon), dtype=np.float32)
    for i, t in enumerate(times_era5):
        # Step value increases every 6h: 1.0, 2.0, 3.0, ...
        u10_data[i, :, :] = float(i + 1)

    ds_era5 = xr.Dataset(
        data_vars={
            "u10": (("time", "latitude", "longitude"), u10_data),
            "v10": (("time", "latitude", "longitude"), np.ones((nt_era, nlat, nlon), dtype=np.float32) * 2.5),
            "t2m": (("time", "latitude", "longitude"), np.ones((nt_era, nlat, nlon), dtype=np.float32) * 265.0),
            "msl": (("time", "latitude", "longitude"), np.ones((nt_era, nlat, nlon), dtype=np.float32) * 98500.0),
        },
        coords={"time": times_era5, "latitude": lats, "longitude": lons_360},
    )

    # Copernicus ocean dataset
    depths = np.array([0.5, 50.0, 150.0, 250.0], dtype=np.float32)
    ndepth = len(depths)

    uo_data = np.ones((nt_cop, ndepth, nlat, nlon), dtype=np.float32) * 0.12
    vo_data = np.ones((nt_cop, ndepth, nlat, nlon), dtype=np.float32) * -0.04
    thetao_data = np.ones((nt_cop, ndepth, nlat, nlon), dtype=np.float32) * -1.8

    ds_copernicus = xr.Dataset(
        data_vars={
            "uo": (("time", "depth", "latitude", "longitude"), uo_data),
            "vo": (("time", "depth", "latitude", "longitude"), vo_data),
            "thetao": (("time", "depth", "latitude", "longitude"), thetao_data),
        },
        coords={"time": times_copernicus, "depth": depths, "latitude": lats, "longitude": lons_180},
    )

    return ds_era5, ds_copernicus


def test_historical_provider_variable_extraction(alignment_datasets):
    """
    Test 1: Verify all 8 environmental variables and EnvironmentState are produced correctly.
    """
    ds_era5, ds_copernicus = alignment_datasets
    provider = HistoricalEnvironmentProvider(
        era5_source=ds_era5,
        copernicus_source=ds_copernicus,
        default_sea_ice_concentration=0.10,
    )

    t = "2020-01-02 06:00:00"
    data = provider.get_environment(t, latitude=-65.0, longitude=-40.0)

    expected_keys = {
        "ocean_u", "ocean_v", "sst",
        "wind_u", "wind_v", "temperature", "pressure",
        "sea_ice_concentration",
    }
    assert set(data.keys()) == expected_keys
    assert data["ocean_u"] == pytest.approx(0.12, abs=1e-3)
    assert data["ocean_v"] == pytest.approx(-0.04, abs=1e-3)
    assert data["sst"] > 250.0  # Converted to Kelvin
    assert data["sea_ice_concentration"] == 0.10

    state = provider.get_state(t, latitude=-65.0, longitude=-40.0)
    assert isinstance(state, EnvironmentState)
    assert state.latitude == -65.0
    assert state.longitude == -40.0


def test_causal_timestamp_alignment(alignment_datasets):
    """
    Test 2: Causal alignment must never select observations after the query time.
    Query between 06:00 (i=1, u10=2.0) and 12:00 (i=2, u10=3.0).
    A nearest-neighbor lookup would pick 12:00 (future!).
    Causal indexing MUST select 06:00 (past/present).
    """
    ds_era5, ds_copernicus = alignment_datasets
    loader_era5 = ERA5Loader(ds_era5)

    # Query at 10:30 (closer to 12:00 than 06:00)
    t_query = "2020-01-01 10:30:00"
    res = loader_era5.get_forcing(t_query, latitude=-65.0, longitude=-40.0)

    # i=0 is 00:00 (u10=1.0), i=1 is 06:00 (u10=2.0), i=2 is 12:00 (u10=3.0)
    # Causal selection must return 2.0 (from 06:00), NOT 3.0 (from 12:00)
    assert res["wind_u"] == pytest.approx(2.0, abs=1e-4)


def test_spatial_interpolation_linear():
    """
    Test 3: Verify bilinear spatial interpolation in lat and lon.
    """
    times = pd.date_range("2020-01-01", periods=2, freq="1D")
    lats = np.array([-70.0, -60.0])
    lons = np.array([0.0, 10.0])

    # Corner values: (-70, 0)=10, (-70, 10)=20, (-60, 0)=30, (-60, 10)=40
    data = np.array([
        [[10.0, 20.0],
         [30.0, 40.0]],
        [[10.0, 20.0],
         [30.0, 40.0]],
    ], dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "u10": (("time", "latitude", "longitude"), data),
            "v10": (("time", "latitude", "longitude"), data),
            "t2m": (("time", "latitude", "longitude"), data + 200.0),
            "msl": (("time", "latitude", "longitude"), data + 90000.0),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )

    loader = ERA5Loader(ds)
    # Midpoint (-65.0, 5.0): expected is (10 + 20 + 30 + 40) / 4 = 25.0
    res = loader.get_forcing("2020-01-01 12:00:00", latitude=-65.0, longitude=5.0, method="linear")
    assert res["wind_u"] == pytest.approx(25.0, abs=1e-4)


def test_longitude_normalization_and_wrapping(alignment_datasets):
    """
    Test 4: ERA5 dataset has lons [300, 340]. Query at -40.0 W (equivalent to 320.0 E).
    Copernicus dataset has lons [-60, -20]. Query at 320.0 E (equivalent to -40.0 W).
    Both must extract data without throwing CoordinateOutOfBoundsError.
    """
    ds_era5, ds_copernicus = alignment_datasets
    provider = HistoricalEnvironmentProvider(
        era5_source=ds_era5,
        copernicus_source=ds_copernicus,
    )

    # Query with negative longitude (-40.0)
    data1 = provider.get_environment("2020-01-02 00:00:00", latitude=-65.0, longitude=-40.0)
    assert np.isfinite(data1["wind_u"])
    assert np.isfinite(data1["ocean_u"])

    # Query with equivalent 360-degree longitude (320.0)
    data2 = provider.get_environment("2020-01-02 00:00:00", latitude=-65.0, longitude=320.0)
    assert data1["wind_u"] == pytest.approx(data2["wind_u"], abs=1e-4)
    assert data1["ocean_u"] == pytest.approx(data2["ocean_u"], abs=1e-4)


def test_historical_cutoff_rejection(alignment_datasets):
    """
    Test 5: Queries after max_allowed_timestamp must raise HistoricalIntegrityViolationError.
    """
    ds_era5, ds_copernicus = alignment_datasets
    cutoff = "2020-01-02 12:00:00"
    provider = HistoricalEnvironmentProvider(
        era5_source=ds_era5,
        copernicus_source=ds_copernicus,
        max_allowed_timestamp=cutoff,
    )

    # Query at or before cutoff succeeds
    data_valid = provider.get_environment("2020-01-02 12:00:00", latitude=-65.0, longitude=-40.0)
    assert data_valid is not None

    # Query 1 second after cutoff must fail
    with pytest.raises(HistoricalIntegrityViolationError):
        provider.get_environment("2020-01-02 12:00:01", latitude=-65.0, longitude=-40.0)


def test_no_future_data_leakage(alignment_datasets):
    """
    Test 6: Query before the earliest record must raise MissingDataError,
    confirming the provider never forward-fills backwards from future data.
    """
    ds_era5, ds_copernicus = alignment_datasets
    provider = HistoricalEnvironmentProvider(
        era5_source=ds_era5,
        copernicus_source=ds_copernicus,
    )

    # Earliest record is 2020-01-01 00:00:00. Query at 2019-12-31 23:59:59.
    with pytest.raises(MissingDataError, match="No ERA5 observations available"):
        provider.get_environment("2019-12-31 23:59:59", latitude=-65.0, longitude=-40.0)


def test_compatibility_with_iceberg_physics_solver(alignment_datasets):
    """
    Test 7: Verify that the real HistoricalEnvironmentProvider seamlessly integrates
    with IcebergPhysicsEvaluator and the Stage 3 RK4 solver to evaluate an EvaluationPair.
    """
    ds_era5, ds_copernicus = alignment_datasets
    provider = HistoricalEnvironmentProvider(
        era5_source=ds_era5,
        copernicus_source=ds_copernicus,
        default_sea_ice_concentration=0.05,
        draft_meters=200.0,
        spatial_interpolation="linear",
    )

    evaluator = IcebergPhysicsEvaluator(
        environment_provider=provider,
        default_properties=create_default_iceberg_properties(),
        dt_seconds=1800.0,  # 30-minute timestep
    )

    t0 = pd.Timestamp("2020-01-01 00:00:00+00:00")
    pair = EvaluationPair(
        iceberg_id="REAL_ALIGN_01",
        prediction_time=t0,
        previous_observation_time=t0 - pd.Timedelta(days=1),
        target_time=t0 + pd.Timedelta(days=3),
        horizon_days=3,
        initial_latitude=-65.0,
        initial_longitude=-40.0,
        previous_latitude=-65.1,
        previous_longitude=-40.2,
        target_latitude=-64.8,
        target_longitude=-39.5,
        initial_position_source="ascat",
        previous_position_source="ascat",
        target_position_source="ascat",
        initial_is_raw=True,
        previous_is_raw=True,
        target_is_raw=True,
        initial_vx_mps=0.08,
        initial_vy_mps=0.02,
        initial_speed_mps=float(np.hypot(0.08, 0.02)),
        initial_bearing_deg=75.0,
    )

    result = evaluator.evaluate_pair(pair)

    assert result.horizon_days == 3
    assert np.isfinite(result.predicted_latitude)
    assert np.isfinite(result.predicted_longitude)
    assert np.isfinite(result.geodesic_error_km)
    assert result.geodesic_error_km >= 0.0


def test_copernicus_depth_orientation_ascending_vs_descending():
    """
    Test 8: Verify that Copernicus depth averaging is invariant to coordinate orientation
    (ascending 0 -> 200m vs descending 200m -> 0).
    """
    times = pd.date_range("2020-01-01", periods=2, freq="1D")
    lats = np.array([-65.0, -64.0])
    lons = np.array([-40.0, -39.0])

    depths_asc = np.array([0.5, 10.0, 50.0, 100.0, 200.0])
    # Velocity decreases with depth: 0.20 near surface down to 0.04 at 200m
    uo_profile = np.array([0.20, 0.18, 0.14, 0.08, 0.04], dtype=np.float32)

    # Ascending dataset
    uo_data_asc = np.broadcast_to(uo_profile[None, :, None, None], (2, 5, 2, 2)).copy()
    ds_asc = xr.Dataset(
        data_vars={
            "uo": (("time", "depth", "latitude", "longitude"), uo_data_asc),
            "vo": (("time", "depth", "latitude", "longitude"), np.zeros_like(uo_data_asc)),
            "thetao": (("time", "depth", "latitude", "longitude"), np.ones_like(uo_data_asc) * -1.5),
        },
        coords={"time": times, "depth": depths_asc, "latitude": lats, "longitude": lons},
    )

    # Descending dataset (reversed depth axis)
    depths_desc = depths_asc[::-1]
    uo_data_desc = uo_data_asc[:, ::-1, :, :].copy()
    ds_desc = xr.Dataset(
        data_vars={
            "uo": (("time", "depth", "latitude", "longitude"), uo_data_desc),
            "vo": (("time", "depth", "latitude", "longitude"), np.zeros_like(uo_data_desc)),
            "thetao": (("time", "depth", "latitude", "longitude"), np.ones_like(uo_data_desc) * -1.5),
        },
        coords={"time": times, "depth": depths_desc, "latitude": lats, "longitude": lons},
    )

    loader_asc = CopernicusLoader(ds_asc)
    loader_desc = CopernicusLoader(ds_desc)

    res_asc = loader_asc.get_ocean_currents("2020-01-01", latitude=-65.0, longitude=-40.0, draft_meters=200.0)
    res_desc = loader_desc.get_ocean_currents("2020-01-01", latitude=-65.0, longitude=-40.0, draft_meters=200.0)

    # Must be mathematically identical regardless of coordinate ordering
    assert res_asc["ocean_u"] == pytest.approx(res_desc["ocean_u"], rel=1e-5)
    assert res_asc["sst"] == pytest.approx(res_desc["sst"], rel=1e-5)


def test_copernicus_trapezoidal_weighting_vs_unweighted_mean():
    """
    Test 9: Verify depth integration uses layer-weighted trapezoidal integration.
    In a non-uniform grid with dense surface levels, unweighted mean heavily overweights
    the surface, while trapezoidal integration correctly weights thicker deeper layers.
    """
    times = pd.date_range("2020-01-01", periods=2, freq="1D")
    lats = np.array([-65.0])
    lons = np.array([-40.0])
    depths = np.array([0.5, 10.0, 50.0, 100.0, 200.0])
    uo_profile = np.array([0.20, 0.18, 0.14, 0.08, 0.04], dtype=np.float32)

    # Unweighted level mean is: (0.20 + 0.18 + 0.14 + 0.08 + 0.04) / 5 = 0.1280
    unweighted_mean = float(np.mean(uo_profile))

    # Analytical trapezoidal average:
    # integral = 0.5*(0.20+0.18)*9.5 + 0.5*(0.18+0.14)*40 + 0.5*(0.14+0.08)*50 + 0.5*(0.08+0.04)*100
    #          = 1.805 + 6.4 + 5.5 + 6.0 = 19.705
    # delta_z = 200.0 - 0.5 = 199.5
    # avg = 19.705 / 199.5 = 0.09877 m/s
    expected_trapz = 19.705 / 199.5

    uo_data = np.broadcast_to(uo_profile[None, :, None, None], (2, 5, 1, 1)).copy()
    ds = xr.Dataset(
        data_vars={
            "uo": (("time", "depth", "latitude", "longitude"), uo_data),
            "vo": (("time", "depth", "latitude", "longitude"), np.zeros_like(uo_data)),
            "thetao": (("time", "depth", "latitude", "longitude"), np.ones_like(uo_data)),
        },
        coords={"time": times, "depth": depths, "latitude": lats, "longitude": lons},
    )

    loader = CopernicusLoader(ds)
    res = loader.get_ocean_currents("2020-01-01", latitude=-65.0, longitude=-40.0, draft_meters=200.0)

    # CopernicusLoader must match the trapezoidal integral (~0.0988), NOT unweighted mean (0.1280)
    assert res["ocean_u"] == pytest.approx(expected_trapz, rel=1e-3)
    assert abs(res["ocean_u"] - unweighted_mean) > 0.02


def test_copernicus_interpolates_non_native_draft_boundary():
    """A 200 m draft uses a linearly interpolated 200 m endpoint when bracketed."""
    times = pd.date_range("2020-01-01", periods=1, freq="1D")
    depths = np.array([0.494025, 186.125595, 222.475204], dtype=np.float64)
    profile = depths.copy()
    data = np.broadcast_to(profile[None, :, None, None], (1, 3, 1, 1)).copy()
    ds = xr.Dataset(
        data_vars={
            "uo": (("time", "depth", "latitude", "longitude"), data),
            "vo": (("time", "depth", "latitude", "longitude"), np.zeros_like(data)),
            "thetao": (("time", "depth", "latitude", "longitude"), np.ones_like(data)),
        },
        coords={"time": times, "depth": depths, "latitude": [-65.0], "longitude": [-40.0]},
    )

    result = CopernicusLoader(ds).get_ocean_currents(
        "2020-01-01", latitude=-65.0, longitude=-40.0, draft_meters=200.0
    )

    # The profile is uo(depth)=depth, so its average over [0.494025, 200] is the midpoint.
    assert result["ocean_u"] == pytest.approx((0.494025 + 200.0) / 2.0, rel=1e-6)


def test_copernicus_rejects_unbracketed_non_native_draft_boundary():
    """A subset ending at 186.126 m cannot silently represent a 200 m draft."""
    times = pd.date_range("2020-01-01", periods=1, freq="1D")
    depths = np.array([0.494025, 186.125595], dtype=np.float64)
    data = np.ones((1, 2, 1, 1), dtype=np.float64)
    ds = xr.Dataset(
        data_vars={
            "uo": (("time", "depth", "latitude", "longitude"), data),
            "vo": (("time", "depth", "latitude", "longitude"), data),
            "thetao": (("time", "depth", "latitude", "longitude"), data),
        },
        coords={"time": times, "depth": depths, "latitude": [-65.0], "longitude": [-40.0]},
    )

    with pytest.raises(MissingDataError, match="exceeds maximum available ocean depth"):
        CopernicusLoader(ds).get_ocean_currents(
            "2020-01-01", latitude=-65.0, longitude=-40.0, draft_meters=200.0
        )


def test_copernicus_draft_exceeding_max_depth_raises_missing_data(alignment_datasets):
    """
    Test 10: Requesting an iceberg draft that exceeds available depth in the dataset
    must raise a clear MissingDataError rather than inventing unobserved values.
    """
    _, ds_copernicus = alignment_datasets
    # Maximum depth in fixture is 250.0m
    loader = CopernicusLoader(ds_copernicus)

    with pytest.raises(MissingDataError, match="exceeds maximum available ocean depth"):
        loader.get_ocean_currents("2020-01-02", latitude=-65.0, longitude=-40.0, draft_meters=600.0)


def test_copernicus_surface_behavior_when_no_draft(alignment_datasets):
    """
    Test 11: When draft_meters is None or 0, near-surface current is returned without averaging.
    """
    _, ds_copernicus = alignment_datasets
    loader = CopernicusLoader(ds_copernicus)

    res_surface = loader.get_ocean_currents("2020-01-02", latitude=-65.0, longitude=-40.0, draft_meters=None)
    assert res_surface["ocean_u"] == pytest.approx(0.12, abs=1e-3)
    # Surface temperature is standard SST
    assert res_surface["sst"] > 270.0

