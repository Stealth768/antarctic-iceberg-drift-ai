from __future__ import annotations

from pathlib import Path
import pandas as pd


class IcebergObservationLoader:
    """Loads and normalizes real iceberg satellite/GPS telemetry observations."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)

    def load_track(
        self,
        time_col: str = "timestamp",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
    ) -> pd.DataFrame:
        """Reads observation file (CSV/Parquet) and returns normalized DataFrame."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Observation file not found: {self.filepath}")

        if self.filepath.suffix == ".csv":
            df = pd.read_csv(self.filepath)
        elif self.filepath.suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(self.filepath)
        else:
            raise ValueError(f"Unsupported file format: {self.filepath.suffix}")

        # Ensure required columns exist
        for col in [time_col, lat_col, lon_col]:
            if col not in df.columns:
                raise KeyError(f"Missing required column '{col}' in {self.filepath}")

        df = df.rename(columns={time_col: "timestamp", lat_col: "latitude", lon_col: "longitude"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        return df[["timestamp", "latitude", "longitude"]]
