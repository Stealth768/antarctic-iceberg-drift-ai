"""
Physics Historical Evaluation Harness.

Evaluates the Stage 3 momentum-conservation iceberg drift physics solver
against standardized historical evaluation pairs from Stage 5.

For each evaluation pair:
1. Constructs an initial IcebergState at origin time T using (x, y, vx, vy)
   from the EvaluationPair (derived strictly from observations <= T).
2. Uses dependency-injected EnvironmentProvider for environmental forcing
   (ocean currents, 10m winds, SST, SIC).
3. Enforces historical cutoff integrity (rejection of future temporal queries).
4. Runs RK4 numerical integration forward in time over horizon H (e.g. 3 or 4 days).
5. Computes WGS84 geodesic displacement error against the verified raw satellite target.
6. Aggregates accuracy metrics (N, mean, median, RMSE, P90, max, min) per horizon.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.data.environment import EnvironmentProvider, HistoricalIntegrityViolationError
from src.evaluation.baseline_evaluator import HorizonMetrics, compute_horizon_metrics
from src.evaluation.historical_pairs import (
    EvaluationPair,
    build_evaluation_pairs,
    calculate_geodesic_error_km,
)
from src.models.iceberg_physics import (
    CoordinateHandler,
    IcebergProperties,
    IcebergState,
    NumericalInstabilityError,
    simulate_iceberg,
)

logger = logging.getLogger(__name__)

# Default uncalibrated prototype physical properties for Antarctic tabular icebergs
# Note: These are research prototype reference values, not calibrated operational constants.
DEFAULT_PROTOTYPE_MASS_KG = 1.0e12     # ~1 billion metric tonnes (~1 Gt)
DEFAULT_PROTOTYPE_LENGTH_M = 5000.0    # 5 km characteristic length
DEFAULT_PROTOTYPE_WIDTH_M = 2500.0     # 2.5 km characteristic width
DEFAULT_PROTOTYPE_DRAFT_M = 200.0      # 200 m submerged keel draft


def create_default_iceberg_properties(
    mass_kg: float = DEFAULT_PROTOTYPE_MASS_KG,
    length_m: float = DEFAULT_PROTOTYPE_LENGTH_M,
    width_m: float = DEFAULT_PROTOTYPE_WIDTH_M,
    draft_m: float = DEFAULT_PROTOTYPE_DRAFT_M,
    air_drag_coefficient: float = 1.30,
    water_drag_coefficient: float = 0.90,
    damping_coefficient: float = 0.00,
    enable_coriolis: bool = True,
    **kwargs: Any,
) -> IcebergProperties:
    """
    Construct uncalibrated prototype IcebergProperties.

    Explicitly labeled as prototype reference parameters, not calibrated operational constants.
    """
    return IcebergProperties(
        mass_kg=mass_kg,
        length_m=length_m,
        width_m=width_m,
        draft_m=draft_m,
        air_drag_coefficient=air_drag_coefficient,
        water_drag_coefficient=water_drag_coefficient,
        damping_coefficient=damping_coefficient,
        enable_coriolis=enable_coriolis,
        **kwargs,
    )


@dataclass(frozen=True)
class PhysicsPredictionResult:
    """Detailed evaluation result for a single physics trajectory simulation."""
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

    predicted_vx_mps: float
    predicted_vy_mps: float
    predicted_speed_mps: float

    geodesic_error_km: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhysicsEvaluationReport:
    """Complete evaluation report for the physics drift solver."""
    overall_metrics: Dict[int, HorizonMetrics]
    per_iceberg_metrics: Dict[str, Dict[int, HorizonMetrics]]
    predictions: List[PhysicsPredictionResult]

    def predictions_dataframe(self) -> pd.DataFrame:
        """Return all prediction results as a pandas DataFrame."""
        if not self.predictions:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in self.predictions])

    def summary_table(self) -> pd.DataFrame:
        """Return formatted summary DataFrame matching baseline report schema."""
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


class IcebergPhysicsEvaluator:
    """
    Evaluation harness for Stage 3 iceberg drift physics solver.
    """

    def __init__(
        self,
        environment_provider: EnvironmentProvider,
        default_properties: Optional[IcebergProperties] = None,
        dt_seconds: float = 600.0,
        crs: str = "EPSG:3412",
    ) -> None:
        """
        Args:
            environment_provider: Provider supplying ocean currents, winds, sea ice.
            default_properties: Prototype physical properties (mass, dimensions, drag).
            dt_seconds: Numerical integration time-step in seconds (default 600s = 10 min).
            crs: Coordinate reference system for numerical integration (default EPSG:3412).
        """
        self.environment_provider = environment_provider
        self.default_properties = (
            default_properties
            if default_properties is not None
            else create_default_iceberg_properties()
        )
        self.dt_seconds = dt_seconds
        self.crs = crs
        self.coord_handler = CoordinateHandler(crs=crs)

    def evaluate_pair(
        self,
        pair: EvaluationPair,
        properties: Optional[IcebergProperties] = None,
    ) -> PhysicsPredictionResult:
        """
        Evaluate physics solver forward forecast on a single historical pair.

        Args:
            pair: Historical evaluation pair containing origin T and ground truth target T+H.
            properties: Optional override for iceberg physical properties.

        Returns:
            PhysicsPredictionResult containing predicted coordinates and geodesic error.
        """
        # Historical cutoff guard: reject evaluation if origin time exceeds provider cutoff
        if (
            self.environment_provider.max_allowed_timestamp is not None
            and pair.prediction_time > self.environment_provider.max_allowed_timestamp
        ):
            raise HistoricalIntegrityViolationError(
                f"Prediction origin time {pair.prediction_time} exceeds environment provider "
                f"historical cutoff {self.environment_provider.max_allowed_timestamp}."
            )

        props = properties if properties is not None else self.default_properties

        # Construct initial state in projected coordinates from pair
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

        duration_seconds = float(pair.horizon_days * 86400.0)

        # Run Stage 3 physics simulation
        sim_df = simulate_iceberg(
            initial_state=state_init,
            start_time=pair.prediction_time,
            duration_seconds=duration_seconds,
            dt_seconds=self.dt_seconds,
            environment_provider=self.environment_provider,
            iceberg_properties=props,
            crs=self.crs,
        )

        final_row = sim_df.iloc[-1]
        pred_lat = float(final_row["latitude"])
        pred_lon = float(final_row["longitude"])
        pred_vx = float(final_row["vx_mps"])
        pred_vy = float(final_row["vy_mps"])
        pred_speed = float(final_row["speed_mps"])

        # Compute geodesic distance error on WGS84 ellipsoid
        error_km = calculate_geodesic_error_km(
            pred_lat, pred_lon,
            pair.target_latitude, pair.target_longitude,
        )

        return PhysicsPredictionResult(
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
            predicted_vx_mps=pred_vx,
            predicted_vy_mps=pred_vy,
            predicted_speed_mps=pred_speed,
            geodesic_error_km=error_km,
        )

    def evaluate_pairs(
        self,
        pairs: Sequence[EvaluationPair],
        properties: Optional[IcebergProperties] = None,
    ) -> Tuple[List[PhysicsPredictionResult], Dict[int, HorizonMetrics]]:
        """
        Evaluate a sequence of historical pairs and compute aggregate metrics by horizon.
        """
        results: List[PhysicsPredictionResult] = []
        errors_by_horizon: Dict[int, List[float]] = {}

        for p in pairs:
            res = self.evaluate_pair(p, properties=properties)
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
        properties: Optional[IcebergProperties] = None,
    ) -> PhysicsEvaluationReport:
        """
        Extract evaluation pairs from a trajectory DataFrame and evaluate physics drift.
        Groups by iceberg_id to compute independent per-iceberg metrics.
        """
        pairs = build_evaluation_pairs(
            trajectory_df=trajectory_df,
            horizons=horizons,
            max_prev_gap_days=max_prev_gap_days,
            crs=self.crs,
        )

        results, overall_metrics = self.evaluate_pairs(pairs, properties=properties)

        per_berg: Dict[str, Dict[int, HorizonMetrics]] = {}
        berg_results_map: Dict[str, List[PhysicsPredictionResult]] = {}
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

        return PhysicsEvaluationReport(
            overall_metrics=overall_metrics,
            per_iceberg_metrics=per_berg,
            predictions=results,
        )
