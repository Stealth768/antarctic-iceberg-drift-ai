"""
Unit tests for BYU/NIC v8.0 Ingestion and Historical Evaluation Dataset (Stage 5A).

Tests all 17 required scenarios:
1. ZIP discovery
2. filename -> iceberg ID
3. YYYYJJJ date conversion
4. exceptional 2-digit-year handling
5. missing 0,0 coordinates
6. raw vs interpolated flag handling
7. sensor priority
8. NIC fallback
9. nautical-mile -> kilometre conversion
10. missing size remains missing
11. raw-only filtering
12. exact T+3 target
13. exact T+4 target
14. interpolated future target is rejected
15. future leakage is impossible
16. geodesic error
17. duplicate normalized timestamps are rejected/handled deterministically
"""

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.iceberg import BYUConsolidatedDatabaseLoader, parse_byu_date
from src.evaluation.historical_pairs import (
    EvaluationPair,
    build_evaluation_pairs,
    calculate_bearing_deg,
    calculate_geodesic_error_km,
    calculate_geodesic_errors_km,
    compute_evaluation_dataset_statistics,
    evaluation_pairs_to_dataframe,
)

ZIP_PATH = Path("data/raw/consolidated_database_v8.0.zip")


# ==============================================================================
# Ingestion Tests (Tests 1 to 11)
# ==============================================================================

class TestBYUIngestion:

    def test_01_zip_discovery(self):
        """Test 1: ZIP discovery discovers CSV files and ignores autosaves/non-CSVs."""
        if not ZIP_PATH.exists():
            pytest.skip(f"ZIP archive not found at {ZIP_PATH}")

        with BYUConsolidatedDatabaseLoader(ZIP_PATH) as loader:
            ids = loader.get_iceberg_ids()
            assert len(ids) == 647
            assert "A23A" in ids
            assert "B15A" in ids
            assert "C19A" in ids
            # Ensure autosave file #d15b.csv# was ignored and not made an ID
            assert "#D15B#" not in ids
            assert "D15B" in ids
            # Ensure README_consolidated.TXT was not made an ID
            assert "README_CONSOLIDATED" not in ids

    def test_02_filename_to_iceberg_id(self, tmp_path: Path):
        """Test 2: Filename correctly maps to normalized uppercase iceberg ID."""
        csv_file = tmp_path / "b15a.csv"
        csv_file.write_text("date,nic_1,nic_2,nic_3\n2000125,-78.0,-175.0,1\n")

        loader = BYUConsolidatedDatabaseLoader(csv_file)
        ids = loader.get_iceberg_ids()
        assert ids == ["B15A"]

        traj = loader.get_trajectory("b15a")
        assert len(traj) == 1
        assert traj.iloc[0]["iceberg_id"] == "B15A"

    def test_03_yyyyjjj_date_conversion(self):
        """Test 3: Standard 7-digit YYYYJJJ parsing into UTC pd.Timestamp."""
        # 2002045 = 2002, Day 45 = 2002-02-14
        t1 = parse_byu_date(2002045)
        assert t1 == pd.Timestamp("2002-02-14 00:00:00+00:00")

        # 1991314 = 1991, Day 314 = 1991-11-10
        t2 = parse_byu_date(1991314)
        assert t2 == pd.Timestamp("1991-11-10 00:00:00+00:00")

        # Leap year day 366: 2020366 = 2020-12-31
        t3 = parse_byu_date(2020366)
        assert t3 == pd.Timestamp("2020-12-31 00:00:00+00:00")

    def test_04_exceptional_2digit_year_handling(self):
        """Test 4: Exceptional 2-digit year format (e03.csv with 92226 -> 1992, Day 226)."""
        t = parse_byu_date(92226)
        assert t == pd.Timestamp("1992-08-13 00:00:00+00:00")

        # Year in 2000s with 5 digits (e.g., 05100 -> 2005, Day 100)
        t_2000 = parse_byu_date(5100)
        assert t_2000 == pd.Timestamp("2005-04-10 00:00:00+00:00")

    def test_05_missing_0_0_coordinates(self, tmp_path: Path):
        """Test 5: Coordinates 0.0, 0.0 are treated as missing and excluded from trajectory."""
        csv_file = tmp_path / "test_missing.csv"
        csv_content = (
            "date,nic_1,nic_2,nic_3\n"
            "2020001,0.0,0.0,0\n"
            "2020002,0.0,0.0,1\n"
            "2020003,-65.0,-50.0,1\n"
        )
        csv_file.write_text(csv_content)

        loader = BYUConsolidatedDatabaseLoader(csv_file)
        traj = loader.get_trajectory("TEST_MISSING")
        assert len(traj) == 1
        assert traj.iloc[0]["latitude"] == -65.0
        assert traj.iloc[0]["longitude"] == -50.0

    def test_06_raw_vs_interpolated_flag_handling(self, tmp_path: Path):
        """
        Test 6:
        - flag=1 with valid coords -> is_raw_observation=True, is_interpolated=False.
        - flag=0 with valid non-zero coords -> is_raw_observation=False, is_interpolated=True.
        - flag=0 with (0.0, 0.0) coords -> missing data, excluded, never marked interpolated.
        """
        csv_file = tmp_path / "test_flags.csv"
        csv_content = (
            "date,nic_1,nic_2,nic_3\n"
            "2020001,-65.0,-50.0,1\n"
            "2020002,-65.1,-50.2,0\n"
            "2020003,0.0,0.0,0\n"  # flag 0 with missing coords
        )
        csv_file.write_text(csv_content)

        loader = BYUConsolidatedDatabaseLoader(csv_file)
        traj = loader.get_trajectory("TEST_FLAGS")
        # Exactly 2 valid rows; the (0,0) record must be excluded and never marked interpolated
        assert len(traj) == 2

        # Row 1 is raw
        assert traj.iloc[0]["is_raw_observation"] is True or traj.iloc[0]["is_raw_observation"] == 1
        assert traj.iloc[0]["is_interpolated"] is False or traj.iloc[0]["is_interpolated"] == 0

        # Row 2 is interpolated (flag=0 with valid non-zero coordinates)
        assert traj.iloc[1]["is_raw_observation"] is False or traj.iloc[1]["is_raw_observation"] == 0
        assert traj.iloc[1]["is_interpolated"] is True or traj.iloc[1]["is_interpolated"] == 1

    def test_07_sensor_priority(self, tmp_path: Path):
        """Test 7: ascat takes priority over qscat and nic when both have raw observations."""
        csv_file = tmp_path / "test_prio.csv"
        csv_content = (
            "date,ascat_1,ascat_2,ascat_3,nic_1,nic_2,nic_3\n"
            "2020001,-66.0,-45.0,1,-66.5,-45.5,1\n"
        )
        csv_file.write_text(csv_content)

        loader = BYUConsolidatedDatabaseLoader(csv_file)
        traj = loader.get_trajectory("TEST_PRIO")
        assert len(traj) == 1
        assert traj.iloc[0]["position_source"] == "ascat"
        assert traj.iloc[0]["latitude"] == -66.0
        assert traj.iloc[0]["longitude"] == -45.0

    def test_08_nic_fallback(self, tmp_path: Path):
        """Test 8: NIC is retained whenever higher-priority sensors have no data."""
        csv_file = tmp_path / "test_fallback.csv"
        csv_content = (
            "date,ascat_1,ascat_2,ascat_3,nic_1,nic_2,nic_3\n"
            "2020001,0.0,0.0,0,-68.0,-55.0,1\n"
        )
        csv_file.write_text(csv_content)

        loader = BYUConsolidatedDatabaseLoader(csv_file)
        traj = loader.get_trajectory("TEST_FALLBACK")
        assert len(traj) == 1
        assert traj.iloc[0]["position_source"] == "nic"
        assert traj.iloc[0]["latitude"] == -68.0
        assert traj.iloc[0]["longitude"] == -55.0

    def test_09_nautical_mile_to_kilometre_conversion(self, tmp_path: Path):
        """Test 9: size_1 and size_2 (nM) are converted to km via factor 1.852."""
        csv_file = tmp_path / "test_size.csv"
        csv_content = (
            "date,nic_1,nic_2,nic_3,size_1,size_2\n"
            "2020001,-65.0,-50.0,1,10,20\n"
        )
        csv_file.write_text(csv_content)

        loader = BYUConsolidatedDatabaseLoader(csv_file)
        traj = loader.get_trajectory("TEST_SIZE")
        assert len(traj) == 1
        assert pytest.approx(traj.iloc[0]["length_km"], rel=1e-5) == 18.52  # 10 * 1.852
        assert pytest.approx(traj.iloc[0]["width_km"], rel=1e-5) == 37.04   # 20 * 1.852

    def test_10_missing_size_remains_missing(self, tmp_path: Path):
        """Test 10: Zero size in CSV is stored as NaN rather than 0.0 km."""
        csv_file = tmp_path / "test_nosize.csv"
        csv_content = (
            "date,nic_1,nic_2,nic_3,size_1,size_2\n"
            "2020001,-65.0,-50.0,1,0,0\n"
        )
        csv_file.write_text(csv_content)

        loader = BYUConsolidatedDatabaseLoader(csv_file)
        traj = loader.get_trajectory("TEST_NOSIZE")
        assert len(traj) == 1
        assert np.isnan(traj.iloc[0]["length_km"])
        assert np.isnan(traj.iloc[0]["width_km"])

    def test_11_raw_only_filtering(self, tmp_path: Path):
        """Test 11: only_observations=True filters out interpolated rows."""
        csv_file = tmp_path / "test_rawonly.csv"
        csv_content = (
            "date,nic_1,nic_2,nic_3\n"
            "2020001,-65.0,-50.0,1\n"
            "2020002,-65.1,-50.1,0\n"
            "2020003,-65.2,-50.2,1\n"
        )
        csv_file.write_text(csv_content)

        loader = BYUConsolidatedDatabaseLoader(csv_file)
        traj_all = loader.get_trajectory("TEST_RAWONLY", only_observations=False)
        assert len(traj_all) == 3

        traj_raw = loader.get_trajectory("TEST_RAWONLY", only_observations=True)
        assert len(traj_raw) == 2
        assert (traj_raw["is_raw_observation"] == True).all()
        # Ensure time_delta_days was recomputed correctly (2 days between day 1 and day 3)
        assert traj_raw.iloc[1]["time_delta_days"] == 2.0


# ==============================================================================
# Evaluation Pair Builder & Geodesic Tests (Tests 12 to 17)
# ==============================================================================

class TestEvaluationPairs:

    @pytest.fixture
    def sample_trajectory_df(self) -> pd.DataFrame:
        """
        Creates a synthetic daily trajectory spanning 10 days:
        Days 1-5: raw observations
        Day 6: interpolated position
        Days 7-10: raw observations
        """
        records = []
        base_time = pd.Timestamp("2020-01-01 00:00:00+00:00")
        for d in range(10):
            t = base_time + pd.Timedelta(days=d)
            is_raw = (d != 5)  # Day 6 (index 5) is interpolated
            records.append({
                "iceberg_id": "TEST_BERG",
                "timestamp": t,
                "latitude": -70.0 + d * 0.1,
                "longitude": -45.0 + d * 0.2,
                "length_km": 40.0,
                "width_km": 20.0,
                "position_source": "ascat" if is_raw else "nic",
                "is_raw_observation": is_raw,
                "is_interpolated": not is_raw,
                "time_delta_days": 1.0 if d > 0 else np.nan,
            })
        return pd.DataFrame(records)

    def test_12_exact_t_plus_3_target(self, sample_trajectory_df):
        """Test 12: Builds exact 3-day evaluation pairs when T+3d exists."""
        pairs = build_evaluation_pairs(sample_trajectory_df, horizons=[3])
        assert len(pairs) > 0

        for p in pairs:
            assert p.horizon_days == 3
            assert p.target_time == p.prediction_time + pd.Timedelta(days=3)
            assert p.target_is_raw is True
            assert p.initial_is_raw is True
            assert p.previous_is_raw is True

    def test_13_exact_t_plus_4_target(self, sample_trajectory_df):
        """Test 13: Builds exact 4-day evaluation pairs when T+4d exists."""
        pairs = build_evaluation_pairs(sample_trajectory_df, horizons=[4])
        assert len(pairs) > 0

        for p in pairs:
            assert p.horizon_days == 4
            assert p.target_time == p.prediction_time + pd.Timedelta(days=4)
            assert p.target_is_raw is True

    def test_14_interpolated_future_target_is_rejected(self, sample_trajectory_df):
        """
        Test 14: An interpolated position at T + horizon is strictly rejected as ground truth.
        Day 6 (2020-01-06) is interpolated.
        Therefore, origin T = Day 3 (2020-01-03) MUST NOT form a 3-day pair (since 3+3 = 6).
        Origin T = Day 2 (2020-01-02) MUST NOT form a 4-day pair (since 2+4 = 6).
        """
        pairs = build_evaluation_pairs(sample_trajectory_df, horizons=[3, 4])
        target_times = [p.target_time for p in pairs]
        bad_time = pd.Timestamp("2020-01-06 00:00:00+00:00")
        assert bad_time not in target_times

    def test_15_future_leakage_is_impossible(self, sample_trajectory_df):
        """
        Test 15: Modifying the future target coordinates at T+3/T+4 has ZERO effect
        on initial_vx_mps, initial_vy_mps, initial_speed_mps, or initial_bearing_deg.
        """
        pairs_orig = build_evaluation_pairs(sample_trajectory_df, horizons=[3])

        # Alter future target at Day 5 (index 4) drastically
        corrupted_df = sample_trajectory_df.copy()
        corrupted_df.loc[4, "latitude"] = -50.0  # huge leap in the future
        corrupted_df.loc[4, "longitude"] = 0.0

        pairs_corrupted = build_evaluation_pairs(corrupted_df, horizons=[3])

        # Prediction at Day 2 (index 1) targets Day 5 (index 4).
        # The initial velocity at Day 2 MUST BE IDENTICAL in both cases!
        orig_p2 = [p for p in pairs_orig if p.prediction_time == sample_trajectory_df.loc[1, "timestamp"]][0]
        corr_p2 = [p for p in pairs_corrupted if p.prediction_time == sample_trajectory_df.loc[1, "timestamp"]][0]

        assert orig_p2.initial_vx_mps == corr_p2.initial_vx_mps
        assert orig_p2.initial_vy_mps == corr_p2.initial_vy_mps
        assert orig_p2.initial_speed_mps == corr_p2.initial_speed_mps
        assert orig_p2.initial_bearing_deg == corr_p2.initial_bearing_deg

    def test_16_geodesic_error(self):
        """Test 16: Geodesic error returns accurate distance in km using WGS84 ellipsoid."""
        # 1. Zero distance for identical coordinates
        dist_zero = calculate_geodesic_error_km(-70.0, 0.0, -70.0, 0.0)
        assert dist_zero == 0.0

        # 2. Distance for 1 degree of latitude near -70S (~111.57 km)
        dist_1deg_lat = calculate_geodesic_error_km(-70.0, 0.0, -71.0, 0.0)
        assert 111.0 < dist_1deg_lat < 112.0

        # 3. Vectorized version matches scalar
        lats1 = np.array([-70.0, -65.0])
        lons1 = np.array([0.0, 10.0])
        lats2 = np.array([-71.0, -65.0])
        lons2 = np.array([0.0, 11.0])
        vec_dists = calculate_geodesic_errors_km(lats1, lons1, lats2, lons2)
        assert len(vec_dists) == 2
        assert pytest.approx(vec_dists[0], rel=1e-6) == dist_1deg_lat

        # 4. Compass bearing test (due South = 180 degrees)
        bearing_south = calculate_bearing_deg(-70.0, 0.0, -71.0, 0.0)
        assert pytest.approx(bearing_south, abs=1e-3) == 180.0

    def test_17_duplicate_normalized_timestamps_handled(self, tmp_path: Path):
        """Test 17: Duplicate timestamps in raw data are deduplicated deterministically."""
        csv_file = tmp_path / "test_dup.csv"
        csv_content = (
            "date,nic_1,nic_2,nic_3\n"
            "2020001,-65.0,-50.0,1\n"
            "2020001,-65.0,-50.0,1\n"  # duplicate date
            "2020002,-65.1,-50.1,1\n"
        )
        csv_file.write_text(csv_content)

        loader = BYUConsolidatedDatabaseLoader(csv_file)
        traj = loader.get_trajectory("TEST_DUP")
        assert len(traj) == 2
        assert traj["timestamp"].is_monotonic_increasing
        assert traj["timestamp"].nunique() == len(traj)


# ==============================================================================
# Live Dataset Integration Test (A23A and B15A from real ZIP)
# ==============================================================================

def test_live_zip_evaluation_statistics():
    """Verify evaluation pair generation and statistics on real v8.0 ZIP archive."""
    if not ZIP_PATH.exists():
        pytest.skip(f"ZIP archive not found at {ZIP_PATH}")

    with BYUConsolidatedDatabaseLoader(ZIP_PATH) as loader:
        stats = compute_evaluation_dataset_statistics(loader, iceberg_ids=["A23A", "B15A"])
        assert stats["total_icebergs"] == 2
        assert stats["valid_pairs_by_horizon"][3] > 0
        assert stats["valid_pairs_by_horizon"][4] > 0
        assert stats["valid_cases_both_horizons"] > 0

        # Check A23A specific counts
        a23a_stat = stats["per_iceberg_summary"]["A23A"]
        assert a23a_stat["total_rows"] > 10000
        assert a23a_stat["pairs_3d"] > 10000
        assert a23a_stat["pairs_4d"] > 10000

        # Check B15A specific counts
        b15a_stat = stats["per_iceberg_summary"]["B15A"]
        assert b15a_stat["pairs_3d"] > 2500
        assert b15a_stat["pairs_4d"] > 2500
