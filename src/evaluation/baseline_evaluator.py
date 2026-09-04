"""
Constant-Velocity Baseline Evaluator.

Evaluates the persistence reference model against historical ground truth
from the BYU/NIC Antarctic Iceberg Tracking Database v8.0.

For each evaluation pair:
1. Takes the iceberg position and velocity estimated at origin time T (from observations <= T).
2. Predicts future position at T + 3d or T + 4d assuming constant velocity in projected coordinates:
       x_pred = x_init + vx * dt
       y_pred = y_init + vy * dt
3. Transforms predicted coordinates back to geographic (WGS84).
4. Computes geodesic distance error (km) against the verified RAW satellite observation at T + H.
5. Produces aggregate statistics: N, mean, median, RMSE, P90, max, min error.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.data.iceberg import BYUConsolidatedDatabaseLoader
from src.evaluation.historical_pairs import (
    EvaluationPair,
    build_evaluation_pairs,
    calculate_geodesic_error_km,
)
from src.models.baselines import ConstantVelocityPredictor, IcebergState
from src.models.iceberg_physics import CoordinateHandler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselinePredictionResult:
    """Detailed evaluation result for a single historical prediction case."""
    iceberg_id: str
    prediction_time: pd.Timestamp
    target_time: pd.Timestamp
    horizon_days: int

    initial_latitude: float
    initial_longitude: float

    predicted_latitude: float
    predicted_longitude: float

    target_latitude: float
    target_longitude: float

    initial_vx_mps: float
    initial_vy_mps: float
    initial_speed_mps: float

    geodesic_error_km: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HorizonMetrics:
    """Aggregate accuracy metrics for a specific forecast horizon."""
    horizon_days: int
    num_cases: int
    mean_error_km: float
    median_error_km: float
    rmse_km: float
    p90_error_km: float
    max_error_km: float
    min_error_km: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_horizon_metrics(errors: Sequence[float], horizon_days: int) -> HorizonMetrics:
    """Compute aggregate accuracy metrics from an array of displacement errors."""
    if not errors:
        return HorizonMetrics(
            horizon_days=horizon_days,
            num_cases=0,
            mean_error_km=0.0,
            median_error_km=0.0,
            rmse_km=0.0,
            p90_error_km=0.0,
            max_error_km=0.0,
            min_error_km=0.0,
        )

    err_arr = np.asarray(errors, dtype=np.float64)
    return HorizonMetrics(
        horizon_days=horizon_days,
        num_cases=len(err_arr),
        mean_error_km=float(np.mean(err_arr)),
        median_error_km=float(np.median(err_arr)),
        rmse_km=float(np.sqrt(np.mean(err_arr ** 2))),
        p90_error_km=float(np.percentile(err_arr, 90)),
        max_error_km=float(np.max(err_arr)),
        min_error_km=float(np.min(err_arr)),
    )


@dataclass
class BaselineEvaluationReport:
    """Complete evaluation report covering overall and per-iceberg performance."""
    overall_metrics: Dict[int, HorizonMetrics]
    per_iceberg_metrics: Dict[str, Dict[int, HorizonMetrics]]
    predictions: List[BaselinePredictionResult]

    def predictions_dataframe(self) -> pd.DataFrame:
        """Return all prediction results as a pandas DataFrame."""
        if not self.predictions:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in self.predictions])

    def summary_table(self) -> pd.DataFrame:
        """Return a formatted DataFrame summarizing overall metrics by horizon."""
        rows = []
        for h in sorted(self.overall_metrics.keys()):
            m = self.overall_metrics[h]
            rows.append({
                "horizon_days": m.horizon_days,
                "num_cases": m.num_cases,
                "mean_error_km": round(m.mean_error_km, 2),
                "median_error_km": round(m.median_error_km, 2),
                "rmse_km": round(m.rmse_km, 2),
                "p90_error_km": round(m.p90_error_km, 2),
                "max_error_km": round(m.max_error_km, 2),
                "min_error_km": round(m.min_error_km, 2),
            })
        return pd.DataFrame(rows)


class ConstantVelocityBaselineEvaluator:
    """
    Evaluator for the Constant Velocity / Persistence trajectory baseline.
    """

    def __init__(self, crs: str = "EPSG:3412") -> None:
        self.crs = crs
        self.predictor = ConstantVelocityPredictor(crs=crs)
        self.coord_handler = CoordinateHandler(crs=crs)

    def evaluate_pair(self, pair: EvaluationPair) -> BaselinePredictionResult:
        """
        Evaluate constant velocity forecast on a single historical pair.
        """
        x_init, y_init = self.coord_handler.to_projected(
            longitude=pair.initial_longitude,
            latitude=pair.initial_latitude,
        )
        state_init = IcebergState(
            x_m=x_init,
            y_m=y_init,
            vx_mps=pair.initial_vx_mps,
            vy_mps=pair.initial_vy_mps,
        )

        dt_seconds = float(pair.horizon_days * 86400.0)
        state_pred = self.predictor.predict_state(state_init, dt_seconds)

        pred_lon, pred_lat = self.coord_handler.to_geographic(state_pred.x_m, state_pred.y_m)
        error_km = calculate_geodesic_error_km(
            pred_lat, pred_lon,
            pair.target_latitude, pair.target_longitude,
        )

        return BaselinePredictionResult(
            iceberg_id=pair.iceberg_id,
            prediction_time=pair.prediction_time,
            target_time=pair.target_time,
            horizon_days=pair.horizon_days,
            initial_latitude=pair.initial_latitude,
            initial_longitude=pair.initial_longitude,
            predicted_latitude=pred_lat,
            predicted_longitude=pred_lon,
            target_latitude=pair.target_latitude,
            target_longitude=pair.target_longitude,
            initial_vx_mps=pair.initial_vx_mps,
            initial_vy_mps=pair.initial_vy_mps,
            initial_speed_mps=pair.initial_speed_mps,
            geodesic_error_km=error_km,
        )

    def evaluate_pairs(
        self,
        pairs: Sequence[EvaluationPair],
    ) -> Tuple[List[BaselinePredictionResult], Dict[int, HorizonMetrics]]:
        """
        Evaluate a sequence of historical pairs and compute aggregate metrics.
        """
        results: List[BaselinePredictionResult] = []
        errors_by_horizon: Dict[int, List[float]] = {}

        for p in pairs:
            res = self.evaluate_pair(p)
            results.append(res)
            h = res.horizon_days
            if h not in errors_by_horizon:
                errors_by_horizon[h] = []
            errors_by_horizon[h].append(res.geodesic_error_km)

        metrics = {
            h: compute_horizon_metrics(errors_by_horizon[h], h)
            for h in sorted(errors_by_horizon.keys())
        }

        return results, metrics

    def evaluate_trajectory(
        self,
        trajectory_df: pd.DataFrame,
        horizons: Sequence[int] = (3, 4),
        max_prev_gap_days: Optional[float] = 14.0,
    ) -> BaselineEvaluationReport:
        """
        Extract evaluation pairs from a normalized trajectory and evaluate them.
        """
        pairs = build_evaluation_pairs(
            trajectory_df=trajectory_df,
            horizons=horizons,
            max_prev_gap_days=max_prev_gap_days,
            crs=self.crs,
        )

        results, overall_metrics = self.evaluate_pairs(pairs)

        # Group prediction results by iceberg_id to compute independent per-iceberg metrics
        per_berg: Dict[str, Dict[int, HorizonMetrics]] = {}
        berg_results_map: Dict[str, List[BaselinePredictionResult]] = {}
        for r in results:
            if r.iceberg_id not in berg_results_map:
                berg_results_map[r.iceberg_id] = []
            berg_results_map[r.iceberg_id].append(r)

        for b_id, b_results in berg_results_map.items():
            b_errors_by_horizon: Dict[int, List[float]] = {}
            for r in b_results:
                if r.horizon_days not in b_errors_by_horizon:
                    b_errors_by_horizon[r.horizon_days] = []
                b_errors_by_horizon[r.horizon_days].append(r.geodesic_error_km)

            per_berg[b_id] = {
                h: compute_horizon_metrics(b_errors_by_horizon[h], h)
                for h in sorted(b_errors_by_horizon.keys())
            }

        return BaselineEvaluationReport(
            overall_metrics=overall_metrics,
            per_iceberg_metrics=per_berg,
            predictions=results,
        )

    def evaluate_database(
        self,
        loader: BYUConsolidatedDatabaseLoader,
        iceberg_ids: Optional[Sequence[str]] = None,
        horizons: Sequence[int] = (3, 4),
        max_prev_gap_days: Optional[float] = 14.0,
    ) -> BaselineEvaluationReport:
        """
        Evaluate the constant-velocity baseline across multiple icebergs in a database.
        """
        all_ids = loader.get_iceberg_ids()
        target_ids = [i.upper() for i in iceberg_ids] if iceberg_ids is not None else all_ids

        all_results: List[BaselinePredictionResult] = []
        errors_overall: Dict[int, List[float]] = {int(h): [] for h in horizons}
        per_berg_metrics: Dict[str, Dict[int, HorizonMetrics]] = {}

        for berg_id in target_ids:
            try:
                df = loader.get_trajectory(berg_id, only_observations=False)
            except KeyError:
                continue

            if df.empty:
                continue

            pairs = build_evaluation_pairs(
                trajectory_df=df,
                horizons=horizons,
                max_prev_gap_days=max_prev_gap_days,
                crs=self.crs,
            )

            if not pairs:
                continue

            berg_results, berg_metrics = self.evaluate_pairs(pairs)
            all_results.extend(berg_results)
            per_berg_metrics[berg_id] = berg_metrics

            for res in berg_results:
                if res.horizon_days in errors_overall:
                    errors_overall[res.horizon_days].append(res.geodesic_error_km)

        overall_metrics = {
            h: compute_horizon_metrics(errors_overall[h], h)
            for h in sorted(errors_overall.keys())
        }

        return BaselineEvaluationReport(
            overall_metrics=overall_metrics,
            per_iceberg_metrics=per_berg_metrics,
            predictions=all_results,
        )
