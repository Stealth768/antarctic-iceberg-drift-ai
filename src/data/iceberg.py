"""
BYU / National Ice Center (NIC) Antarctic Iceberg Tracking Database Loader.

Ingests, cleans, and standardizes historical iceberg observation tracks:
- iceberg_id (e.g. A23A, B15A, C19A)
- timestamp (UTC)
- latitude / longitude
- geometry: length, width, area
- position_source, is_raw_observation, is_interpolated

Supports both:
1. BYU Consolidated Database v8.0 ZIP archives / directories / files.
2. Standardized tabular tracking CSVs / DataFrames via IcebergDatabaseLoader.
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COLUMN_ALIASES = {
    "iceberg_id": ["iceberg_id", "iceberg", "id", "name", "berg_id", "berg"],
    "timestamp": ["timestamp", "datetime", "date", "time", "observation_time"],
    "latitude": ["latitude", "lat", "y"],
    "longitude": ["longitude", "lon", "long", "x"],
    "length": ["length_km", "length", "len", "major_axis_km", "l"],
    "width": ["width_km", "width", "wid", "minor_axis_km", "w"],
    "area": ["area_sqkm", "area", "size_sqkm", "size"],
}

# Deterministic satellite/sensor priority for BYU consolidated database
DEFAULT_SENSOR_PRIORITY: Tuple[str, ...] = (
    "ascat",
    "qscat",
    "oscat",
    "nscat",
    "seawinds",
    "ers",
    "sass",
    "nic",
)


def parse_byu_date(date_val: Union[int, str]) -> pd.Timestamp:
    """
    Parse BYU date format YYYYJJJ into UTC pd.Timestamp.

    Handles standard 7-digit YYYYJJJ (Year + Julian Day) and exceptional
    5/6-digit representations (such as e03.csv containing 2-digit years).
    """
    ival = int(date_val)
    if ival < 1000000:
        # 5 or 6 digit format: YYJJJ (e.g., 92226 -> 1992, Day 226)
        year_short = ival // 1000
        jday = ival % 1000
        year = 1900 + year_short if year_short >= 70 else 2000 + year_short
    else:
        year = ival // 1000
        jday = ival % 1000

    return pd.to_datetime(f"{year}-{jday:03d}", format="%Y-%j", utc=True)


class BYUConsolidatedDatabaseLoader:
    """
    Loader for the BYU/NIC Antarctic Iceberg Tracking Database v8.0.

    Supports reading directly from:
    1. ZIP archive (e.g., data/raw/consolidated_database_v8.0.zip)
    2. Extracted directory of CSV files
    3. An individual iceberg CSV file

    Extracts iceberg IDs from filenames, converts YYYYJJJ to UTC timestamps,
    resolves sensor positions according to deterministic priority, distinguishes
    raw observations from interpolated positions, and converts sizes to kilometres.
    """

    def __init__(
        self,
        source: Union[str, Path, zipfile.ZipFile],
        sensor_priority: Sequence[str] = DEFAULT_SENSOR_PRIORITY,
    ) -> None:
        self.sensor_priority = list(sensor_priority)
        self._zip_file: Optional[zipfile.ZipFile] = None
        self._dir_path: Optional[Path] = None
        self._single_file: Optional[Path] = None

        # Catalog mapping normalized iceberg ID -> internal file path / archive name
        self._catalog: Dict[str, str] = {}
        self._init_source(source)

    def _init_source(self, source: Union[str, Path, zipfile.ZipFile]) -> None:
        if isinstance(source, zipfile.ZipFile):
            self._zip_file = source
            self._scan_zip()
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Iceberg data source does not exist: {path}")

            if path.is_file():
                if zipfile.is_zipfile(path):
                    self._zip_file = zipfile.ZipFile(path, "r")
                    self._scan_zip()
                elif path.suffix.lower() == ".csv":
                    self._single_file = path
                    berg_id = path.stem.upper()
                    self._catalog[berg_id] = str(path)
                else:
                    raise ValueError(f"Unsupported file type: {path}")
            elif path.is_dir():
                self._dir_path = path
                self._scan_dir()
            else:
                raise ValueError(f"Invalid source path: {path}")

    def _is_valid_csv_filename(self, filename: str) -> bool:
        base = os.path.basename(filename)
        # Ignore editor autosave lockfiles like #d15b.csv# or hidden files
        if base.startswith("#") or base.endswith("#") or base.startswith("."):
            return False
        return base.lower().endswith(".csv")

    def _scan_zip(self) -> None:
        assert self._zip_file is not None
        for name in self._zip_file.namelist():
            if self._is_valid_csv_filename(name):
                base = os.path.basename(name)
                berg_id = os.path.splitext(base)[0].upper()
                self._catalog[berg_id] = name

    def _scan_dir(self) -> None:
        assert self._dir_path is not None
        for root, _, files in os.walk(self._dir_path):
            for file in files:
                if self._is_valid_csv_filename(file):
                    berg_id = os.path.splitext(file)[0].upper()
                    full_path = os.path.join(root, file)
                    self._catalog[berg_id] = full_path

    def get_iceberg_ids(self) -> List[str]:
        """Return sorted list of all available iceberg IDs in the dataset."""
        return sorted(list(self._catalog.keys()))

    def _read_raw_bytes(self, iceberg_id: str) -> bytes:
        norm_id = iceberg_id.strip().upper()
        if norm_id not in self._catalog:
            raise KeyError(f"Iceberg '{iceberg_id}' not found in database catalog.")

        path_or_name = self._catalog[norm_id]
        if self._zip_file is not None:
            return self._zip_file.read(path_or_name)
        else:
            with open(path_or_name, "rb") as f:
                return f.read()

    def parse_raw_dataframe(
        self,
        raw_df: pd.DataFrame,
        iceberg_id: str,
    ) -> pd.DataFrame:
        """
        Normalize and clean raw BYU CSV DataFrame.
        """
        if raw_df.empty or "date" not in raw_df.columns:
            return pd.DataFrame()

        # Parse date YYYYJJJ
        raw_dates = pd.to_numeric(raw_df["date"], errors="coerce").dropna().astype(int)
        if raw_dates.empty:
            return pd.DataFrame()

        years = np.where(
            raw_dates < 1000000,
            np.where((raw_dates // 1000) >= 70, 1900 + (raw_dates // 1000), 2000 + (raw_dates // 1000)),
            raw_dates // 1000,
        )
        jdays = raw_dates % 1000
        date_strs = [f"{y}-{j:03d}" for y, j in zip(years, jdays)]
        timestamps = pd.to_datetime(date_strs, format="%Y-%j", utc=True)

        n_rows = len(raw_df)
        available_sensors = [
            s for s in self.sensor_priority if f"{s}_1" in raw_df.columns and f"{s}_2" in raw_df.columns
        ]

        selected_lats = np.full(n_rows, np.nan, dtype=np.float64)
        selected_lons = np.full(n_rows, np.nan, dtype=np.float64)
        selected_sources = [None] * n_rows
        is_raw_obs = np.zeros(n_rows, dtype=bool)
        is_interpolated = np.zeros(n_rows, dtype=bool)

        # Priority Pass 1: Direct raw observations (flag == 1 with valid non-(0,0) coordinates)
        for s in available_sensors:
            lat_col = pd.to_numeric(raw_df[f"{s}_1"], errors="coerce").values
            lon_col = pd.to_numeric(raw_df[f"{s}_2"], errors="coerce").values
            flag_col = (
                pd.to_numeric(raw_df[f"{s}_3"], errors="coerce").fillna(0).values
                if f"{s}_3" in raw_df.columns
                else np.ones(n_rows, dtype=int)
            )

            # Direct observation: flag == 1 AND coordinates are non-zero
            valid_raw = (flag_col == 1) & ((lat_col != 0.0) | (lon_col != 0.0)) & np.isnan(selected_lats)
            idx = np.where(valid_raw)[0]
            if len(idx) > 0:
                selected_lats[idx] = lat_col[idx]
                selected_lons[idx] = lon_col[idx]
                is_raw_obs[idx] = True
                is_interpolated[idx] = False
                for i in idx:
                    selected_sources[i] = s

        # Priority Pass 2: Linearly interpolated positions (flag == 0 with valid non-(0,0) coordinates)
        # CRITICAL SEMANTICS:
        # flag == 0 can denote either an interpolated position OR missing/no data.
        # Coordinates (0.0, 0.0) indicate missing data and must NEVER be marked is_interpolated=True.
        # Only valid, non-zero coordinates with flag == 0 represent genuine linearly interpolated positions.
        for s in available_sensors:
            lat_col = pd.to_numeric(raw_df[f"{s}_1"], errors="coerce").values
            lon_col = pd.to_numeric(raw_df[f"{s}_2"], errors="coerce").values
            flag_col = (
                pd.to_numeric(raw_df[f"{s}_3"], errors="coerce").fillna(0).values
                if f"{s}_3" in raw_df.columns
                else np.zeros(n_rows, dtype=int)
            )

            valid_interp = (flag_col == 0) & ((lat_col != 0.0) | (lon_col != 0.0)) & np.isnan(selected_lats)
            idx = np.where(valid_interp)[0]
            if len(idx) > 0:
                selected_lats[idx] = lat_col[idx]
                selected_lons[idx] = lon_col[idx]
                is_raw_obs[idx] = False
                is_interpolated[idx] = True
                for i in idx:
                    selected_sources[i] = s

        # Sizes in nautical miles to kilometres (1 nM = 1.852 km)
        size1 = (
            pd.to_numeric(raw_df["size_1"], errors="coerce").fillna(0).values
            if "size_1" in raw_df.columns
            else np.zeros(n_rows)
        )
        size2 = (
            pd.to_numeric(raw_df["size_2"], errors="coerce").fillna(0).values
            if "size_2" in raw_df.columns
            else np.zeros(n_rows)
        )
        length_km = np.where(size1 > 0, size1 * 1.852, np.nan)
        width_km = np.where(size2 > 0, size2 * 1.852, np.nan)

        clean_df = pd.DataFrame({
            "iceberg_id": iceberg_id.upper(),
            "timestamp": timestamps,
            "latitude": selected_lats,
            "longitude": selected_lons,
            "length_km": length_km,
            "width_km": width_km,
            "position_source": selected_sources,
            "is_raw_observation": is_raw_obs,
            "is_interpolated": is_interpolated,
        })

        # Exclude missing coordinates (NaN or 0.0, 0.0)
        clean_df = clean_df.dropna(subset=["latitude", "longitude"])
        # Explicit guard: ensure (0.0, 0.0) is never treated as a valid location
        clean_df = clean_df[~((clean_df["latitude"] == 0.0) & (clean_df["longitude"] == 0.0))]
        # Ensure is_interpolated is strictly exclusive of is_raw_observation
        clean_df["is_interpolated"] = clean_df["is_interpolated"] & (~clean_df["is_raw_observation"])

        # Validate latitude [-90, 90]
        clean_df = clean_df[(clean_df["latitude"] >= -90.0) & (clean_df["latitude"] <= 90.0)]
        # Normalize longitude to [-180, 180]
        clean_df["longitude"] = ((clean_df["longitude"] + 180.0) % 360.0) - 180.0

        # Sort chronologically and drop duplicate timestamps
        clean_df = clean_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        # Compute time delta
        clean_df["time_delta_days"] = clean_df["timestamp"].diff().dt.total_seconds() / 86400.0

        return clean_df

    def get_trajectory(
        self,
        iceberg_id: str,
        only_observations: bool = False,
    ) -> pd.DataFrame:
        """
        Extract normalized trajectory for an iceberg.

        Args:
            iceberg_id: Iceberg identifier (case-insensitive).
            only_observations: If True, filters strictly to raw/direct observations
                (is_raw_observation == True), discarding interpolated days.

        Returns:
            DataFrame with columns:
            [iceberg_id, timestamp, latitude, longitude, length_km, width_km,
             position_source, is_raw_observation, is_interpolated, time_delta_days]
        """
        norm_id = iceberg_id.strip().upper()
        raw_bytes = self._read_raw_bytes(norm_id)
        raw_df = pd.read_csv(io.BytesIO(raw_bytes))

        clean_df = self.parse_raw_dataframe(raw_df, norm_id)

        if only_observations:
            clean_df = clean_df[clean_df["is_raw_observation"]].copy()
            clean_df["time_delta_days"] = clean_df["timestamp"].diff().dt.total_seconds() / 86400.0
            clean_df = clean_df.reset_index(drop=True)

        return clean_df

    def close(self) -> None:
        """Close internal zipfile if open."""
        if self._zip_file is not None:
            self._zip_file.close()
            self._zip_file = None

    def __enter__(self) -> BYUConsolidatedDatabaseLoader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ==============================================================================
# Existing General-Purpose Iceberg Loader (Retained for Backwards Compatibility)
# ==============================================================================

class IcebergDatabaseLoader:
    """
    Parser and trajectory query interface for Antarctic iceberg tracking records.
    """

    def __init__(self, data: Union[str, Path, pd.DataFrame]):
        """
        Args:
            data: Filepath to CSV tracking file, or pre-loaded pandas.DataFrame.
        """
        if isinstance(data, (str, Path)):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise TypeError(f"Expected path or DataFrame, got {type(data)}")

        self.df = self._standardize_and_clean(self.df)

    def _standardize_and_clean(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names, parse dates, enforce polar bounds, and compute delta_t."""
        df = raw_df.copy()
        col_map = {}

        # Map column names
        for canonical, candidates in COLUMN_ALIASES.items():
            for c in df.columns:
                if c.strip().lower() in candidates:
                    col_map[c] = canonical
                    break

        df = df.rename(columns=col_map)

        # Check mandatory columns
        mandatory = ["iceberg_id", "timestamp", "latitude", "longitude"]
        missing = [m for m in mandatory if m not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns {missing} in iceberg data. Available: {list(raw_df.columns)}"
            )

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Ensure numeric coordinates
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

        # Drop invalid lat/lon
        df = df.dropna(subset=["latitude", "longitude", "timestamp"])

        # Normalize longitude to [-180, 180]
        df["longitude"] = ((df["longitude"] + 180.0) % 360.0) - 180.0

        # Sort chronologically per iceberg
        df["iceberg_id"] = df["iceberg_id"].astype(str)
        df = df.sort_values(by=["iceberg_id", "timestamp"]).reset_index(drop=True)

        # Calculate time difference between consecutive observations for each iceberg
        df["time_delta_days"] = df.groupby("iceberg_id")["timestamp"].diff().dt.total_seconds() / 86400.0

        # Fill optional geometric columns if missing
        if "length" not in df.columns:
            df["length"] = np.nan
        if "width" not in df.columns:
            df["width"] = np.nan
        if "area" not in df.columns:
            df["area"] = np.nan

        return df

    def get_iceberg_ids(self) -> List[str]:
        """List unique iceberg identifiers present in the dataset."""
        return sorted(self.df["iceberg_id"].unique().tolist())

    def get_trajectory(self, iceberg_id: str) -> pd.DataFrame:
        """
        Extract the complete historical trajectory for a single iceberg.

        Returns:
            Chronologically sorted DataFrame with columns:
            [iceberg_id, timestamp, latitude, longitude, length, width, area, time_delta_days]
        """
        sub = self.df[self.df["iceberg_id"] == str(iceberg_id)].copy()
        if sub.empty:
            raise KeyError(f"Iceberg ID '{iceberg_id}' not found in database.")
        return sub.reset_index(drop=True)

    def filter_by_time(
        self,
        start_time: Union[str, pd.Timestamp],
        end_time: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        """Filter observations within a specific UTC time window."""
        t0 = pd.to_datetime(start_time, utc=True)
        t1 = pd.to_datetime(end_time, utc=True)
        return self.df[(self.df["timestamp"] >= t0) & (self.df["timestamp"] <= t1)].copy()

    def filter_by_region(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> pd.DataFrame:
        """Filter observations within a geographic bounding box."""
        return self.df[
            (self.df["latitude"] >= lat_min)
            & (self.df["latitude"] <= lat_max)
            & (self.df["longitude"] >= lon_min)
            & (self.df["longitude"] <= lon_max)
        ].copy()


def create_synthetic_iceberg_track(
    iceberg_id: str = "A23a_TEST",
    start_time: str = "2026-01-01 00:00:00",
    start_lat: float = -65.0,
    start_lon: float = -50.0,
    speed_km_per_day: float = 12.0,
    heading_deg: float = 45.0,
    num_observations: int = 10,
    min_interval_days: float = 1.0,
    max_interval_days: float = 4.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic BYU/NIC iceberg observations with realistic irregular 1-4 day intervals.
    """
    rng = np.random.default_rng(random_seed)
    current_time = pd.to_datetime(start_time, utc=True)
    current_lat = start_lat
    current_lon = start_lon

    rows = []
    for i in range(num_observations):
        rows.append({
            "iceberg_id": iceberg_id,
            "timestamp": current_time,
            "latitude": current_lat,
            "longitude": current_lon,
            "length": 40.0,
            "width": 25.0,
            "area": 950.0,
        })

        dt_days = float(rng.uniform(min_interval_days, max_interval_days))
        current_time += pd.Timedelta(days=dt_days)

        dist_km = speed_km_per_day * dt_days
        rad_heading = np.radians(heading_deg)
        dlat = (dist_km * np.cos(rad_heading)) / 111.139
        dlon = (dist_km * np.sin(rad_heading)) / (111.139 * np.cos(np.radians(current_lat)))

        current_lat += dlat
        current_lon += dlon

    return pd.DataFrame(rows)
