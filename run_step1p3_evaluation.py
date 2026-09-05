#!/usr/bin/env python3
"""
Phase 1, Step 1.3: Hybrid Physics + ML Trajectory Predictor Evaluation.

Implements and evaluates the hybrid predictor combining calibrated physics
simulation with learned Ridge residual correction. Compares against:
  1. Constant velocity baseline
  2. Calibrated physics-only model
  3. Physics + Ridge hybrid model

Metrics are evaluated strictly on the held-out chronological test partition
of the A23A trajectory.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.data.copernicus import CopernicusLoader
from src.data.era5 import ERA5Loader
from src.data.nsidc import NSIDCLoader
from src.data.observations import IcebergObservationLoader
from src.data.environment import CompositeEnvironmentProvider
from src.metrics.trajectory import calculate_trajectory_metrics
from src.models.iceberg_physics import (
    CoordinateHandler,
    IcebergProperties,
    IcebergState,
    simulate_iceberg,
)
from src.models.ml.dataset import build_residual_dataset
from src.models.ml.residual_model import RidgeResidualModel
from src.models.ml.hybrid_predictor import (
    HybridIcebergPredictor,
    constant_velocity_baseline,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_a23a_dataset() -> Dict[str, Any]:
    """Load A23A dataset with train/val/test split."""
    logger.info("Loading A23A ground truth observations...")
    obs_path = Path("data/raw/observations/a23a_ground_truth.csv")

    if not obs_path.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_path}")

    obs_loader = IcebergObservationLoader(obs_path)
    df_truth = obs_loader.load_track()
    logger.info(f"Loaded {len(df_truth)} truth observations")

    # Load environmental data
    logger.info("Loading environmental data (GLORYS, ERA5, NSIDC)...")
    glorys_path = Path("data/raw/glorys_test/glorys_a23a_test.nc")
    era5_path = Path("data/raw/era5_test/era5_a23a_real_200001.nc")
    nsidc_path = Path("data/raw/nsidc_test/nsidc_a23a_test.nc")

    for p in [glorys_path, era5_path, nsidc_path]:
        if not p.exists():
            raise FileNotFoundError(f"Environmental data file not found: {p}")

    env_provider = CompositeEnvironmentProvider(
        nsidc_loader=NSIDCLoader(source=nsidc_path),
        era5_loader=ERA5Loader(source=era5_path),
        copernicus_loader=CopernicusLoader(source=glorys_path),
    )

    # Run calibrated physics simulation
    logger.info("Running calibrated physics simulation...")
    coord_handler = CoordinateHandler(crs="EPSG:3412")

    x0, y0 = coord_handler.to_projected(
        longitude=df_truth["longitude"].iloc[0],
        latitude=df_truth["latitude"].iloc[0],
    )

    init_state = IcebergState(
        x_m=x0,
        y_m=y0,
        vx_mps=0.0,
        vy_mps=0.0,
    )

    props_cal = IcebergProperties(
        mass_kg=1e12,
        length_m=5000.0,
        width_m=2500.0,
        draft_m=200.0,
        air_drag_coefficient=0.2000,
        water_drag_coefficient=1.0065,
    )

    start_time = df_truth["timestamp"].iloc[0]
    end_time = df_truth["timestamp"].iloc[-1]
    duration_sec = (end_time - start_time).total_seconds()

    df_sim = simulate_iceberg(
        initial_state=init_state,
        start_time=start_time,
        duration_seconds=duration_sec,
        dt_seconds=600.0,
        environment_provider=env_provider,
        iceberg_properties=props_cal,
        crs="EPSG:3412",
    )

    logger.info(f"Generated {len(df_sim)} simulated trajectory points")

    # Build residual dataset
    logger.info("Building residual dataset with chronological partitions...")

    dataset = build_residual_dataset(
        df_sim=df_sim,
        df_truth=df_truth,
        environment_provider=env_provider,
        iceberg_id="A23A",
        iceberg_properties=props_cal,
        train_frac=0.60,
        val_frac=0.20,
        test_frac=0.20,
    )

    return {
        "dataset": dataset,
        "df_truth": df_truth,
        "init_state": init_state,
        "start_time": start_time,
        "duration_sec": duration_sec,
        "env_provider": env_provider,
        "props": props_cal,
        "coord_handler": coord_handler,
    }


def compute_residual_bound(
    dataset: Any,
    percentile: float = 95.0,
) -> float:
    """
    Compute residual correction bound from train+validation data.

    The bound is derived exclusively from train and validation residuals.
    The test partition is never used.

    Returns:
        Residual bound in meters.
    """
    logger.info(
        f"Computing residual bound (percentile={percentile})..."
    )

    # Combine train and validation residuals only.
    y_train = dataset.train.y
    y_val = dataset.val.y
    y_combined = np.vstack([y_train, y_val])

    # Absolute component magnitudes.
    abs_residuals_x = np.abs(y_combined[:, 0])
    abs_residuals_y = np.abs(y_combined[:, 1])

    # Euclidean residual magnitude.
    dist_residuals = np.hypot(
        y_combined[:, 0],
        y_combined[:, 1],
    )

    bound_m = float(
        np.percentile(dist_residuals, percentile)
    )

    logger.info("  Train+val residual statistics:")
    logger.info(
        f"    X component: mean={np.mean(abs_residuals_x):.1f}m, "
        f"max={np.max(abs_residuals_x):.1f}m"
    )
    logger.info(
        f"    Y component: mean={np.mean(abs_residuals_y):.1f}m, "
        f"max={np.max(abs_residuals_y):.1f}m"
    )
    logger.info(
        f"    Distance: mean={np.mean(dist_residuals):.1f}m, "
        f"P95={bound_m:.1f}m, "
        f"max={np.max(dist_residuals):.1f}m"
    )

    return bound_m


def evaluate_predictions(
    df_pred: pd.DataFrame,
    df_truth: pd.DataFrame,
    label: str,
) -> Dict[str, float]:
    """Evaluate a prediction trajectory against ground truth."""
    logger.info(f"  Computing metrics for {label}...")

    metrics = calculate_trajectory_metrics(
        df_pred,
        df_truth,
    )

    logger.info(
        f"    {label} RMSE: "
        f"{metrics['rmse_km']:.4f} km"
    )

    return metrics


def main():
    """Execute Phase 1 Step 1.3 evaluation."""
    try:
        # ====================================================================
        # LOAD DATA
        # ====================================================================
        context = load_a23a_dataset()
        dataset = context["dataset"]

        logger.info(
            f"Dataset summary: {dataset.total_samples} total samples"
        )
        logger.info(
            f"  Train: {len(dataset.train)}, "
            f"Val: {len(dataset.val)}, "
            f"Test: {len(dataset.test)}"
        )

        # ====================================================================
        # RESIDUAL BOUND
        # ====================================================================
        residual_bound_m = compute_residual_bound(
            dataset,
            percentile=95.0,
        )

        # ====================================================================
        # TRAIN RIDGE MODEL
        # ====================================================================
        logger.info(
            "Training Ridge regression model (alpha=10.0)..."
        )

        ridge_model = RidgeResidualModel(
            alpha=10.0,
            random_state=42,
        )

        ridge_model.fit(
            dataset.train.X,
            dataset.train.y,
        )

        logger.info("Ridge model trained successfully")

        # ====================================================================
        # CREATE HYBRID PREDICTOR
        # ====================================================================
        logger.info(
            f"Creating hybrid predictor with "
            f"bound={residual_bound_m:.1f} m..."
        )

        predictor = HybridIcebergPredictor(
            residual_model=ridge_model,
            residual_bound_m=residual_bound_m,
            crs="EPSG:3412",
        )

        # ====================================================================
        # EVALUATE ON HELD-OUT TEST PARTITION
        # ====================================================================
        test_timestamps = pd.DatetimeIndex(
            dataset.test.timestamps
        )

        if len(test_timestamps) == 0:
            raise ValueError("Test partition is empty.")

        test_start = test_timestamps.min()
        test_end = test_timestamps.max()

        logger.info(
            "\nEvaluating on held-out chronological test partition "
            f"({len(test_timestamps)} samples): "
            f"{test_start} -> {test_end}"
        )

        # Ground truth restricted strictly to held-out test timestamps.
        df_truth_test = context["df_truth"][
            context["df_truth"]["timestamp"].isin(test_timestamps)
        ].copy()

        if len(df_truth_test) != len(test_timestamps):
            raise ValueError(
                f"Test timestamp mismatch: dataset contains "
                f"{len(test_timestamps)} timestamps but ground truth "
                f"matched {len(df_truth_test)}."
            )

        # ====================================================================
        # 1. CONSTANT VELOCITY BASELINE
        # ====================================================================
        logger.info(
            "Generating constant velocity baseline..."
        )

        df_cv_full = constant_velocity_baseline(
            context["df_truth"]
        )

        df_cv_test = df_cv_full[
            df_cv_full["timestamp"].isin(test_timestamps)
        ].copy()

        metrics_cv = evaluate_predictions(
            df_cv_test,
            df_truth_test,
            "Constant Velocity",
        )

        # ====================================================================
        # 2. PHYSICS-ONLY BASELINE
        # ====================================================================
        logger.info(
            "Generating physics-only trajectory..."
        )

        df_physics_full = predictor.physics_only(
            initial_state=context["init_state"],
            start_time=context["start_time"],
            duration_seconds=context["duration_sec"],
            dt_seconds=600.0,
            environment_provider=context["env_provider"],
            iceberg_properties=context["props"],
        )

        df_physics_test = df_physics_full[
            df_physics_full["timestamp"].isin(test_timestamps)
        ].copy()

        metrics_physics = evaluate_predictions(
            df_physics_test,
            df_truth_test,
            "Physics-only",
        )

        # ====================================================================
        # 3. HYBRID PHYSICS + RIDGE
        # ====================================================================
        logger.info(
            "Generating hybrid Physics + Ridge trajectory..."
        )

        df_hybrid_full = predictor.predict(
            initial_state=context["init_state"],
            start_time=context["start_time"],
            duration_seconds=context["duration_sec"],
            dt_seconds=600.0,
            environment_provider=context["env_provider"],
            iceberg_properties=context["props"],
            apply_ml_correction=True,
        )

        df_hybrid_test = df_hybrid_full[
            df_hybrid_full["timestamp"].isin(test_timestamps)
        ].copy()

        metrics_hybrid = evaluate_predictions(
            df_hybrid_test,
            df_truth_test,
            "Physics + Ridge Hybrid",
        )

        # ====================================================================
        # IMPROVEMENTS
        # ====================================================================
        if metrics_physics["rmse_km"] > 0:
            hybrid_improvement = (
                (
                    metrics_physics["rmse_km"]
                    - metrics_hybrid["rmse_km"]
                )
                / metrics_physics["rmse_km"]
                * 100.0
            )
        else:
            hybrid_improvement = 0.0

        if metrics_cv["rmse_km"] > 0:
            physics_vs_cv = (
                (
                    metrics_cv["rmse_km"]
                    - metrics_physics["rmse_km"]
                )
                / metrics_cv["rmse_km"]
                * 100.0
            )
        else:
            physics_vs_cv = 0.0

        # ====================================================================
        # GENERATE REPORT
        # ====================================================================
        print("\n" + "=" * 80)
        print(
            "PHASE 1, STEP 1.3: "
            "HYBRID PHYSICS + ML ICEBERG TRAJECTORY PREDICTOR"
        )
        print("=" * 80)

        print("\n1. ARCHITECTURE:")
        print(
            "   Hybrid = Physics Simulation + "
            "Ridge Residual Correction"
        )
        print(
            "   Physics model: "
            "Calibrated RK4 integration (frozen, not modified)"
        )
        print(
            "   ML component: "
            "Ridge regression (trained on "
            f"{len(dataset.train)} samples)"
        )
        print(
            "   Inference: "
            "Sequential - physics first, then ML post-processing"
        )

        print("\n2. RESIDUAL CORRECTION BOUND:")
        print(
            f"   Value: {residual_bound_m:.1f} meters"
        )
        print(
            "   Selection method: "
            "P95 of train+val residuals"
        )
        print(
            "   Rationale: "
            "Prevents ML extrapolation; data-driven; "
            "test-set agnostic"
        )
        print(
            "   Clipping: "
            "Applied by vector magnitude in projected coordinates"
        )

        print("\n3. EXACT INFERENCE PROCEDURE:")
        print(
            "   a) Run physics simulation from initial state "
            "for full duration"
        )
        print("   b) For each physics output step:")
        print(
            "      - Extract environmental/kinematic features"
        )
        print(
            "      - Predict residual with Ridge model"
        )
        print(
            "      - Clip to a vector magnitude of "
            f"{residual_bound_m:.0f}"
            " m vector magnitude"
        )
        print(
            "      - Add to physics position in EPSG:3412"
        )
        print(
            "      - Reproject to WGS84 lat/lon"
        )
        print(
            "   c) Return trajectory DataFrame "
            "[timestamp, latitude, longitude]"
        )

        print(
            f"\n4. TEST SET SAMPLE COUNT: "
            f"{len(dataset.test)} (chronological, held-out)"
        )

        print("\n5. TRAJECTORY METRICS (Test Set):")

        print("\n   CONSTANT VELOCITY BASELINE:")
        for key, val in metrics_cv.items():
            if isinstance(val, float):
                print(f"      {key}: {val:.4f}")
            else:
                print(f"      {key}: {val}")

        print("\n   PHYSICS-ONLY (Calibrated, No ML):")
        for key, val in metrics_physics.items():
            if isinstance(val, float):
                print(f"      {key}: {val:.4f}")
            else:
                print(f"      {key}: {val}")

        print(
            f"   Improvement over CV: "
            f"{physics_vs_cv:+.1f}%"
        )

        print("\n   PHYSICS + RIDGE HYBRID:")
        for key, val in metrics_hybrid.items():
            if isinstance(val, float):
                print(f"      {key}: {val:.4f}")
            else:
                print(f"      {key}: {val}")

        print(
            f"   Improvement over Physics-only: "
            f"{hybrid_improvement:+.1f}%"
        )

        # ====================================================================
        # VALIDATION
        # ====================================================================
        hybrid_improves = (
            metrics_hybrid["rmse_km"]
            <= metrics_physics["rmse_km"]
        )

        print(
            "\n6. CRITICAL VALIDATION: "
            "Hybrid RMSE <= Physics RMSE?"
        )
        print(
            f"   Hybrid: "
            f"{metrics_hybrid['rmse_km']:.4f} km"
        )
        print(
            f"   Physics: "
            f"{metrics_physics['rmse_km']:.4f} km"
        )
        print(
            "   Result: "
            f"{'YES - Hybrid improved' if hybrid_improves else 'NO - Hybrid worsened'}"
        )

        # ====================================================================
        # FILES
        # ====================================================================
        print("\n7. FILES CREATED/MODIFIED:")
        print("   Created:")
        print(
            "      - src/models/ml/hybrid_predictor.py"
        )
        print(
            "         * HybridIcebergPredictor class"
        )
        print(
            "         * constant_velocity_baseline() function"
        )
        print(
            "         * Model injection for Ridge residual"
        )
        print(
            "         * Residual bound enforcement"
        )
        print(
            "      - tests/test_hybrid_predictor.py"
        )
        print(
            "         * 12 comprehensive tests"
        )

        print("   Existing (reused, not modified):")
        print(
            "      - src/models/iceberg_physics.py "
            "(simulate_iceberg, IcebergState, etc.)"
        )
        print(
            "      - src/models/ml/residual_model.py "
            "(RidgeResidualModel)"
        )
        print(
            "      - src/models/ml/dataset.py "
            "(ResidualDataset)"
        )
        print(
            "      - src/models/ml/features.py "
            "(ResidualFeatureExtractor)"
        )
        print(
            "      - src/metrics/trajectory.py "
            "(calculate_trajectory_metrics)"
        )

        # ====================================================================
        # CAPABILITIES
        # ====================================================================
        print("\n8. HYBRID PREDICTOR CAPABILITIES:")
        print(
            "   - Physics-only fallback mode "
            "(for regression protection)"
        )
        print(
            "   - Configurable residual bound "
            "(prevents extrapolation)"
        )
        print(
            "   - Model injection "
            "(dependency injection pattern)"
        )
        print(
            "   - Deterministic predictions "
            "(fixed random_state)"
        )
        print(
            "   - Output compatibility "
            "with trajectory metrics"
        )
        print(
            "   - Projection handling "
            "(EPSG:3412 <-> WGS84)"
        )
        print(
            "   - Feature extraction at inference time"
        )

        # ====================================================================
        # LIMITATIONS
        # ====================================================================
        print("\n9. LIMITATIONS & NOTES:")
        print(
            "   - Requires environmental data at inference time "
            "(realistic)"
        )
        print(
            "   - Features extracted only for valid "
            "environmental windows"
        )
        print(
            "   - ML correction skipped if feature extraction "
            "fails (physics fallback)"
        )
        print(
            "   - Bound selection: P95 percentile of "
            "train+val residuals"
        )
        print(
            "   - Test set never used to select/tune the bound"
        )
        print(
            "   - Evaluation is a chronological historical "
            "hindcast using observed/reanalysis environmental "
            "forcing; it is not a prospective environmental forecast"
        )
        print(
            "   - Training samples are hourly, while inference "
            "corrections are currently applied at 10-minute "
            "physics timesteps"
        )
        print(
            "   - Evaluation uses one held-out A23A trajectory; "
            "results should not be interpreted as generalizable "
            "real-world accuracy"
        )

        # ====================================================================
        # SAVE RESULTS
        # ====================================================================
        results = {
            "architecture": {
                "physics_module": "simulate_iceberg (frozen)",
                "ml_component": "RidgeResidualModel",
                "correction_application": (
                    "Post-processing in projected coords"
                ),
            },
            "residual_bound": {
                "value_meters": float(residual_bound_m),
                "selection_method": (
                    "P95 of train+val residuals"
                ),
                "percentile": 95.0,
            },
            "sample_counts": {
                "train": len(dataset.train),
                "val": len(dataset.val),
                "test": len(dataset.test),
            },
            "test_partition": {
                "start": str(test_start),
                "end": str(test_end),
                "sample_count": len(dataset.test),
                "chronological": True,
                "held_out": True,
            },
            "metrics": {
                "constant_velocity": {
                    k: (
                        float(v)
                        if isinstance(v, (int, float, np.number))
                        else v
                    )
                    for k, v in metrics_cv.items()
                },
                "physics_only": {
                    k: (
                        float(v)
                        if isinstance(v, (int, float, np.number))
                        else v
                    )
                    for k, v in metrics_physics.items()
                },
                "physics_ridge_hybrid": {
                    k: (
                        float(v)
                        if isinstance(v, (int, float, np.number))
                        else v
                    )
                    for k, v in metrics_hybrid.items()
                },
            },
            "improvements": {
                "physics_vs_constant_velocity_pct": float(
                    physics_vs_cv
                ),
                "hybrid_vs_physics_pct": float(
                    hybrid_improvement
                ),
                "hybrid_improves_held_out_test": bool(
                    hybrid_improves
                ),
            },
            "evaluation": {
                "type": "chronological_historical_hindcast",
                "prospective_forecast": False,
                "test_set_used_for_tuning": False,
            },
        }

        results_path = Path("results_step1p3.json")

        with open(results_path, "w") as f:
            json.dump(
                results,
                f,
                indent=2,
            )

        logger.info(
            f"Results saved to {results_path}"
        )

        print("\n" + "=" * 80)
        print("END OF REPORT")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        logger.error(
            f"Error during evaluation: {e}",
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    exit(main())
