"""
Historical Evaluation Dataset & Pair Builder.

Constructs scientifically defensible historical trajectory prediction cases:
    Observation at T (with velocity estimated strictly from t_prev < T)
    → Predict at T + 3 days → evaluate against RAW observed position at T + 3 days
    → Predict at T + 4 days → evaluate against RAW observed position at T + 4 days

Enforces:
1. Exact calendar-day future targets (no synthetic or interpolated target coordinates).
2. Strict historical cutoff / no future leakage: only observations <= T inform the initial state.
3. Geodesic distance error calculation on WGS84 ellipsoid via pyproj.Geod.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyproj

from src.data.iceberg import BYUConsolidatedDatabaseLoader
from src.models.baselines import estimate_velocity_projected
from src.models.iceberg_physics import CoordinateHandler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationPair:
    """
    Standardized evaluation pair for multi-day trajectory forecasting.
    """
    iceberg_id: str
    prediction_time: pd.Timestamp
    previous_observation_time: pd.Timestamp
    target_time: pd.Timestamp
    horizon_days: int

    initial_latitude: float
    initial_longitude: float

    previous_latitude: float
    previous_longitude: float

    target_latitude: float
    target_longitude: float

    initial_position_source: str
    previous_position_source: str
    target_position_source: str

    initial_is_raw: bool
    previous_is_raw: bool
    target_is_raw: bool

    initial_vx_mps: float
    initial_vy_mps: float
    initial_speed_mps: float
    initial_bearing_deg: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Geodesic Error and Bearing Calculations
# ==============================================================================

def calculate_geodesic_error_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Compute geodesic (shortest path on WGS84 ellipsoid) distance in kilometres.

    Args:
        lat1, lon1: First position in degrees.
        lat2, lon2: Second position in degrees.

    Returns:
        Distance in kilometres.
    """
    if not (np.isfinite(lat1) and np.isfinite(lon1) and np.isfinite(lat2) and np.isfinite(lon2)):
        raise ValueError(f"Non-finite coordinates encountered: ({lat1}, {lon1}), ({lat2}, {lon2})")

    geod = pyproj.Geod(ellps="WGS84")
    _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
    return float(dist_m / 1000.0)


def calculate_geodesic_errors_km(
    lats1: Union[Sequence[float], np.ndarray],
    lons1: Union[Sequence[float], np.ndarray],
    lats2: Union[Sequence[float], np.ndarray],
    lons2: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """
    Compute vectorized geodesic distance in kilometres on WGS84 ellipsoid.
    """
    geod = pyproj.Geod(ellps="WGS84")
    _, _, dist_m = geod.inv(lons1, lats1, lons2, lats2)
    return np.asarray(dist_m, dtype=np.float64) / 1000.0


def calculate_bearing_deg(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate initial forward azimuth (compass bearing) from point 1 to point 2.

    Returns:
        Bearing in degrees [0, 360).
    """
    geod = pyproj.Geod(ellps="WGS84")
    fwd_az, _, _ = geod.inv(lon1, lat1, lon2, lat2)
    return float((fwd_az + 360.0) % 360.0)


# ==============================================================================
# Evaluation Pair Builder
# ==============================================================================

def build_evaluation_pairs(
    trajectory_df: pd.DataFrame,
    horizons: Sequence[int] = (3, 4),
    max_prev_gap_days: Optional[float] = 14.0,
    crs: str = "EPSG:3412",
) -> List[EvaluationPair]:
    """
    Construct historical evaluation pairs from a normalized iceberg trajectory.

    For each prediction origin T:
    1. Initial velocity is estimated strictly from (t_prev, T) where both are
       raw satellite observations (is_raw_observation == True).
    2. For each horizon H in horizons:
       Target at T + H calendar days must exist as a direct raw observation.
       Interpolated future targets are strictly rejected.
    3. Future leakage is strictly forbidden: target observations do not enter
       the predictor initial state.

    Args:
        trajectory_df: Normalized trajectory DataFrame.
        horizons: Target horizons in calendar days (default: 3 and 4 days).
        max_prev_gap_days: Maximum allowable gap between t_prev and T to estimate
            initial velocity (default 14 days).
        crs: Coordinate reference system for velocity differencing.

    Returns:
        List of EvaluationPair instances.
    """
    if trajectory_df.empty:
        return []

    coord_handler = CoordinateHandler(crs=crs)
    pairs: List[EvaluationPair] = []

    # Process each iceberg independently if multiple icebergs are present in the DataFrame
    if "iceberg_id" in trajectory_df.columns:
        grouped = trajectory_df.groupby("iceberg_id", sort=False)
    else:
        grouped = [("UNKNOWN", trajectory_df)]

    for berg_id, berg_df in grouped:
        df_sorted = berg_df.sort_values("timestamp").reset_index(drop=True)
        raw_obs = df_sorted[df_sorted["is_raw_observation"]].copy().reset_index(drop=True)

        if len(raw_obs) < 2:
            continue

        # Detect duplicate raw observation timestamps
        dup_mask = raw_obs["timestamp"].duplicated(keep=False)
        if dup_mask.any():
            dup_timestamps = raw_obs.loc[dup_mask, "timestamp"].unique()
            dup_strs = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S") for t in dup_timestamps[:5]]
            raise ValueError(
                f"Duplicate raw observation timestamps found for iceberg '{berg_id}': {dup_strs}. "
                "Raw observation timestamps must be strictly unique to construct deterministic evaluation pairs."
            )

        # Map timestamps to raw observation rows for fast exact-horizon lookups
        raw_map: Dict[pd.Timestamp, pd.Series] = {row["timestamp"]: row for _, row in raw_obs.iterrows()}

        # Pre-project raw coordinates once for rapid velocity estimation
        proj_coords: Dict[pd.Timestamp, Tuple[float, float]] = {}
        for _, row in raw_obs.iterrows():
            t = row["timestamp"]
            x, y = coord_handler.to_projected(row["longitude"], row["latitude"])
            proj_coords[t] = (x, y)

        for i in range(1, len(raw_obs)):
            curr_row = raw_obs.iloc[i]
            prev_row = raw_obs.iloc[i - 1]

            t_curr = curr_row["timestamp"]
            t_prev = prev_row["timestamp"]

            dt_prev_sec = (t_curr - t_prev).total_seconds()
            dt_prev_days = dt_prev_sec / 86400.0

            if dt_prev_days <= 0:
                continue

            if max_prev_gap_days is not None and dt_prev_days > max_prev_gap_days:
                continue

            # Velocity strictly from (t_prev, t_curr)
            x_prev, y_prev = proj_coords[t_prev]
            x_curr, y_curr = proj_coords[t_curr]

            vx, vy, _ = estimate_velocity_projected(x_prev, y_prev, t_prev, x_curr, y_curr, t_curr)
            speed = float(np.hypot(vx, vy))
            bearing = calculate_bearing_deg(
                prev_row["latitude"], prev_row["longitude"],
                curr_row["latitude"], curr_row["longitude"],
            )

            for h in horizons:
                t_target = t_curr + pd.Timedelta(days=int(h))

                # Exact calendar day raw observation required
                target_row = raw_map.get(t_target)
                if target_row is None:
                    continue

                # Strict guard against non-raw targets
                if not target_row["is_raw_observation"]:
                    continue

                pair = EvaluationPair(
                    iceberg_id=str(curr_row["iceberg_id"]).upper(),
                    prediction_time=t_curr,
                    previous_observation_time=t_prev,
                    target_time=t_target,
                    horizon_days=int(h),
                    initial_latitude=float(curr_row["latitude"]),
                    initial_longitude=float(curr_row["longitude"]),
                    previous_latitude=float(prev_row["latitude"]),
                    previous_longitude=float(prev_row["longitude"]),
                    target_latitude=float(target_row["latitude"]),
                    target_longitude=float(target_row["longitude"]),
                    initial_position_source=str(curr_row["position_source"]),
                    previous_position_source=str(prev_row["position_source"]),
                    target_position_source=str(target_row["position_source"]),
                    initial_is_raw=True,
                    previous_is_raw=True,
                    target_is_raw=True,
                    initial_vx_mps=vx,
                    initial_vy_mps=vy,
                    initial_speed_mps=speed,
                    initial_bearing_deg=bearing,
                )
                pairs.append(pair)

    return pairs


def evaluation_pairs_to_dataframe(pairs: Sequence[EvaluationPair]) -> pd.DataFrame:
    """Convert sequence of EvaluationPair objects to a pandas DataFrame."""
    if not pairs:
        return pd.DataFrame()
    return pd.DataFrame([p.to_dict() for p in pairs])


# ==============================================================================
# Dataset Statistics Function
# ==============================================================================

def compute_evaluation_dataset_statistics(
    loader: BYUConsolidatedDatabaseLoader,
    iceberg_ids: Optional[Sequence[str]] = None,
    horizons: Sequence[int] = (3, 4),
) -> Dict[str, Any]:
    """
    Compute database-wide metrics and valid evaluation pair counts.

    Args:
        loader: Active BYUConsolidatedDatabaseLoader instance.
        iceberg_ids: Optional subset of iceberg IDs to process. If None, processes all.
        horizons: Prediction horizons to evaluate (default: 3 and 4 days).

    Returns:
        Dictionary of dataset metrics and pair counts.
    """
    all_ids = loader.get_iceberg_ids()
    target_ids = [i.upper() for i in iceberg_ids] if iceberg_ids is not None else all_ids

    total_rows = 0
    total_raw = 0
    min_date: Optional[pd.Timestamp] = None
    max_date: Optional[pd.Timestamp] = None

    pairs_by_horizon: Dict[int, int] = {h: 0 for h in horizons}
    both_horizons_count = 0
    per_berg_stats: Dict[str, Dict[str, Any]] = {}

    for berg_id in target_ids:
        try:
            df = loader.get_trajectory(berg_id, only_observations=False)
        except KeyError:
            continue

        if df.empty:
            continue

        n_rows = len(df)
        n_raw = int(df["is_raw_observation"].sum())
        total_rows += n_rows
        total_raw += n_raw

        t_min = df["timestamp"].min()
        t_max = df["timestamp"].max()
        if min_date is None or t_min < min_date:
            min_date = t_min
        if max_date is None or t_max > max_date:
            max_date = t_max

        # Build evaluation pairs for this berg
        pairs = build_evaluation_pairs(df, horizons=horizons)
        h3_times = {p.prediction_time for p in pairs if p.horizon_days == 3}
        h4_times = {p.prediction_time for p in pairs if p.horizon_days == 4}
        both_count = len(h3_times.intersection(h4_times))
        both_horizons_count += both_count

        for p in pairs:
            pairs_by_horizon[p.horizon_days] += 1

        per_berg_stats[berg_id] = {
            "total_rows": n_rows,
            "raw_observations": n_raw,
            "raw_pct": round(n_raw / n_rows * 100.0, 1) if n_rows > 0 else 0.0,
            "start_date": t_min.strftime("%Y-%m-%d"),
            "end_date": t_max.strftime("%Y-%m-%d"),
            "pairs_3d": len(h3_times),
            "pairs_4d": len(h4_times),
            "pairs_both": both_count,
        }

    raw_pct = round(total_raw / total_rows * 100.0, 1) if total_rows > 0 else 0.0

    return {
        "total_icebergs": len(target_ids),
        "total_normalized_rows": total_rows,
        "total_raw_observations": total_raw,
        "raw_observation_pct": raw_pct,
        "date_range": (
            min_date.strftime("%Y-%m-%d") if min_date else None,
            max_date.strftime("%Y-%m-%d") if max_date else None,
        ),
        "valid_pairs_by_horizon": pairs_by_horizon,
        "valid_cases_both_horizons": both_horizons_count,
        "per_iceberg_summary": per_berg_stats,
    }
