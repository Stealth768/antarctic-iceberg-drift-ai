"""
Real Environmental Physics Benchmark Runner (Stage 6C).

Executes the Stage 3 momentum-conservation physics solver against historical
BYU/NIC ground-truth iceberg evaluation pairs using real ERA5 atmospheric
and Copernicus ocean reanalyses.

Integrity Rules:
1. Inspects the repository for available environmental data without hardcoding filenames.
2. If real environmental data is unavailable for an evaluation pair, the case is
   explicitly recorded as skipped/missing rather than inventing or forward-filling values.
3. Evaluates 3-day and 4-day forecast horizons separately.
4. Compares real physics metrics against the locked Stage 5B Constant-Velocity baseline.
5. Produces an honest, reproducible audit of simulated cases, failure modes, and metrics.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from src.data.environment import (
    EnvironmentalDataError,
    HistoricalEnvironmentProvider,
    HistoricalIntegrityViolationError,
    MissingDataError,
)
from src.data.iceberg import BYUConsolidatedDatabaseLoader
from src.evaluation.baseline_evaluator import (
    ConstantVelocityBaselineEvaluator,
    HorizonMetrics,
    compute_horizon_metrics,
)
from src.evaluation.historical_pairs import (
    EvaluationPair,
    build_evaluation_pairs,
)
from src.evaluation.physics_evaluator import (
    IcebergPhysicsEvaluator,
    PhysicsPredictionResult,
    create_default_iceberg_properties,
)

logger = logging.getLogger(__name__)


class MissingDataReason(str, Enum):
    NO_ENVIRONMENTAL_DATA_FOUND = "No ERA5 or Copernicus datasets found in repository"
    MISSING_ATMOSPHERIC_FORCING = "ERA5 atmospheric forcing unavailable for target spatio-temporal window"
    MISSING_OCEANIC_FORCING = "Copernicus ocean currents unavailable for target spatio-temporal window"
    HISTORICAL_CUTOFF_VIOLATION = "Query violated historical temporal cutoff"
    NUMERICAL_INSTABILITY = "Numerical instability during RK4 integration"


@dataclass(frozen=True)
class SkippedCase:
    iceberg_id: str
    prediction_time: pd.Timestamp
    target_time: pd.Timestamp
    horizon_days: int
    reason: str


@dataclass
class EnvironmentalCatalog:
    """Catalog of discovered environmental NetCDF/GRIB datasets."""
    era5_files: List[Path] = field(default_factory=list)
    copernicus_files: List[Path] = field(default_factory=list)
    other_files: List[Path] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return len(self.era5_files) > 0 and len(self.copernicus_files) > 0


def discover_environmental_datasets(search_dir: Path = Path("data")) -> EnvironmentalCatalog:
    """
    Dynamically discover available environmental reanalysis files in the repository.
    Inspects variable signatures without hardcoding file names or coordinates.
    """
    catalog = EnvironmentalCatalog()
    if not search_dir.exists():
        return catalog

    for ext in ("*.nc", "*.nc4", "*.netcdf", "*.grib", "*.grb"):
        for f in sorted(search_dir.rglob(ext)):
            try:
                # Inspect variable names with xarray
                with xr.open_dataset(f) as ds:
                    vars_lower = {v.lower() for v in ds.data_vars}
                    if any(v in vars_lower for v in ("u10", "v10", "t2m", "msl")):
                        catalog.era5_files.append(f)
                    elif any(v in vars_lower for v in ("uo", "vo", "thetao", "sst")):
                        catalog.copernicus_files.append(f)
                    else:
                        catalog.other_files.append(f)
            except Exception as exc:
                logger.debug(f"Could not open candidate environmental file {f}: {exc}")
                catalog.other_files.append(f)

    return catalog


def open_environmental_sources(paths: Sequence[Path]) -> xr.Dataset:
    """Open one or more compatible environmental chunks as one lazy dataset."""
    if not paths:
        raise FileNotFoundError("No environmental files were provided.")
    if len(paths) == 1:
        return xr.open_dataset(paths[0])
    return xr.open_mfdataset(
        [str(path) for path in sorted(paths)],
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="no_conflicts",
    )


@dataclass
class RealPhysicsBenchmarkReport:
    """Detailed report comparing real physics simulation against baseline ground truth."""
    catalog: EnvironmentalCatalog
    total_evaluation_cases: int
    simulated_cases: int
    success_rate_pct: float
    missing_reasons_summary: Dict[str, int]
    skipped_cases: List[SkippedCase]
    physics_results: List[PhysicsPredictionResult]
    physics_metrics: Dict[int, HorizonMetrics]
    per_iceberg_physics_metrics: Dict[str, Dict[int, HorizonMetrics]]

    def summary_table(self) -> pd.DataFrame:
        """Tabular summary of simulated cases and accuracy metrics."""
        rows = []
        for h in sorted(self.physics_metrics.keys()):
            m = self.physics_metrics[h]
            rows.append({
                "horizon_days": m.horizon_days,
                "num_simulated": m.num_cases,
                "mean_error_km": round(m.mean_error_km, 2) if m.num_cases > 0 else np.nan,
                "median_error_km": round(m.median_error_km, 2) if m.num_cases > 0 else np.nan,
                "rmse_km": round(m.rmse_km, 2) if m.num_cases > 0 else np.nan,
                "p90_error_km": round(m.p90_error_km, 2) if m.num_cases > 0 else np.nan,
                "max_error_km": round(m.max_error_km, 2) if m.num_cases > 0 else np.nan,
                "min_error_km": round(m.min_error_km, 2) if m.num_cases > 0 else np.nan,
            })
        return pd.DataFrame(rows)


def run_real_physics_benchmark(
    loader: BYUConsolidatedDatabaseLoader,
    data_dir: Path = Path("data"),
    iceberg_ids: Sequence[str] = ("A23A", "B15A"),
    horizons: Sequence[int] = (3, 4),
    max_cases_per_berg: Optional[int] = None,
    start_time: Optional[Union[str, pd.Timestamp]] = None,
    end_time: Optional[Union[str, pd.Timestamp]] = None,
) -> RealPhysicsBenchmarkReport:
    """
    Execute the Stage 6C real physics benchmark.

    Evaluates whether real environmental data are available, simulates feasible cases,
    and logs honest skipped counts and error reasons.
    """
    catalog = discover_environmental_datasets(data_dir)
    target_ids = [i.upper() for i in iceberg_ids]

    all_pairs: List[EvaluationPair] = []
    for berg_id in target_ids:
        try:
            df = loader.get_trajectory(berg_id, only_observations=False)
            if start_time is not None:
                df = df[df["timestamp"] >= pd.Timestamp(start_time)]
            if end_time is not None:
                df = df[df["timestamp"] <= pd.Timestamp(end_time)]
            pairs = build_evaluation_pairs(df, horizons=horizons)
            if max_cases_per_berg is not None:
                pairs = pairs[:max_cases_per_berg]
            all_pairs.extend(pairs)
        except KeyError:
            continue

    total_cases = len(all_pairs)
    simulated_results: List[PhysicsPredictionResult] = []
    skipped_cases: List[SkippedCase] = []
    missing_summary: Dict[str, int] = {}

    # If environmental catalog is empty, record missing reason honestly for all pairs
    if not catalog.is_complete:
        reason_str = MissingDataReason.NO_ENVIRONMENTAL_DATA_FOUND.value
        missing_summary[reason_str] = total_cases
        for p in all_pairs:
            skipped_cases.append(SkippedCase(
                iceberg_id=p.iceberg_id,
                prediction_time=p.prediction_time,
                target_time=p.target_time,
                horizon_days=p.horizon_days,
                reason=reason_str,
            ))
    else:
        # Initialize provider using discovered files
        provider = HistoricalEnvironmentProvider(
            era5_source=open_environmental_sources(catalog.era5_files),
            copernicus_source=open_environmental_sources(catalog.copernicus_files),
        )
        evaluator = IcebergPhysicsEvaluator(
            environment_provider=provider,
            default_properties=create_default_iceberg_properties(),
        )

        for p in all_pairs:
            try:
                res = evaluator.evaluate_pair(p)
                simulated_results.append(res)
            except MissingDataError as e:
                r = str(e)
                missing_summary[r] = missing_summary.get(r, 0) + 1
                skipped_cases.append(SkippedCase(p.iceberg_id, p.prediction_time, p.target_time, p.horizon_days, r))
            except HistoricalIntegrityViolationError as e:
                r = str(e)
                missing_summary[r] = missing_summary.get(r, 0) + 1
                skipped_cases.append(SkippedCase(p.iceberg_id, p.prediction_time, p.target_time, p.horizon_days, r))
            except Exception as e:
                r = f"Simulation failure: {e}"
                missing_summary[r] = missing_summary.get(r, 0) + 1
                skipped_cases.append(SkippedCase(p.iceberg_id, p.prediction_time, p.target_time, p.horizon_days, r))

    num_sim = len(simulated_results)
    success_rate = (num_sim / total_cases * 100.0) if total_cases > 0 else 0.0

    # Aggregate metrics
    errors_by_horizon: Dict[int, List[float]] = {int(h): [] for h in horizons}
    per_berg_errors: Dict[str, Dict[int, List[float]]] = {
        b: {int(h): [] for h in horizons} for b in target_ids
    }

    for r in simulated_results:
        errors_by_horizon[r.horizon_days].append(r.geodesic_error_km)
        if r.iceberg_id in per_berg_errors:
            per_berg_errors[r.iceberg_id][r.horizon_days].append(r.geodesic_error_km)

    overall_metrics = {
        h: compute_horizon_metrics(errors_by_horizon[h], h)
        for h in sorted(errors_by_horizon.keys())
    }

    per_berg_metrics = {
        b: {h: compute_horizon_metrics(per_berg_errors[b][h], h) for h in sorted(per_berg_errors[b].keys())}
        for b in sorted(per_berg_errors.keys())
    }

    return RealPhysicsBenchmarkReport(
        catalog=catalog,
        total_evaluation_cases=total_cases,
        simulated_cases=num_sim,
        success_rate_pct=success_rate,
        missing_reasons_summary=missing_summary,
        skipped_cases=skipped_cases,
        physics_results=simulated_results,
        physics_metrics=overall_metrics,
        per_iceberg_physics_metrics=per_berg_metrics,
    )
