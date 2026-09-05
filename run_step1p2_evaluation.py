#!/usr/bin/env python3
"""
Phase 1, Step 1.2: Physics-Residual ML Model Training and Evaluation.

Trains Ridge and Tree models on the 73 A23A samples to predict residuals,
then evaluates against held-out test set and compares trajectory metrics.

Output: Comprehensive results report with metrics for Physics-only vs 
Physics+Ridge vs Physics+Tree approaches.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from src.data.copernicus import CopernicusLoader
from src.data.era5 import ERA5Loader
from src.data.nsidc import NSIDCLoader
from src.data.observations import IcebergObservationLoader
from src.data.environment import CompositeEnvironmentProvider
from src.models.iceberg_physics import (
    CoordinateHandler,
    IcebergProperties,
    IcebergState,
    simulate_iceberg,
)
from src.models.ml.dataset import build_residual_dataset
from src.models.ml.residual_model import (
    RidgeResidualModel,
    TreeResidualModel,
    evaluate_residual_corrections,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_a23a_dataset() -> Dict[str, Any]:
    """Load and prepare the A23A residual dataset (73 samples, 44/15/14 split)."""
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

    # Run physics simulation
    logger.info("Running calibrated physics simulation...")
    coord_handler = CoordinateHandler(crs="EPSG:3412")
    x0, y0 = coord_handler.to_projected(
        longitude=df_truth["longitude"].iloc[0],
        latitude=df_truth["latitude"].iloc[0],
    )
    init_state = IcebergState(x_m=x0, y_m=y0, vx_mps=0.0, vy_mps=0.0)

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
        "props": props_cal,
        "env_provider": env_provider,
    }


def train_ridge_model(X_train: np.ndarray, y_train: np.ndarray) -> RidgeResidualModel:
    """Train Ridge regression model on training data."""
    logger.info("Training Ridge regression model (alpha=10.0)...")
    model = RidgeResidualModel(alpha=10.0, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_tree_model(X_train: np.ndarray, y_train: np.ndarray) -> TreeResidualModel:
    """Train Tree-based model on training data."""
    logger.info("Training Tree-based model (RF, depth=3, n_est=50)...")
    model = TreeResidualModel(
        n_estimators=50,
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_models(
    ridge_model: RidgeResidualModel,
    tree_model: TreeResidualModel,
    dataset: Any,
) -> Dict[str, Any]:
    """
    Evaluate both models on train, validation, and test sets.
    Compare residual prediction errors and trajectory metrics.
    """
    logger.info("Evaluating models on train/val/test partitions...")
    
    results = {
        "models_trained": ["Ridge (alpha=10.0)", "Tree (depth=3, n_est=50)"],
        "sklearn_version": ">=1.3.0",
        "hyperparameters": {
            "ridge": {
                "alpha": 10.0,
                "random_state": 42,
            },
            "tree": {
                "n_estimators": 50,
                "max_depth": 3,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
        },
        "sample_counts": {
            "train": len(dataset.train),
            "validation": len(dataset.val),
            "test": len(dataset.test),
            "total": dataset.total_samples,
        },
    }

    # Residual RMSE metrics for each partition
    logger.info("Computing residual RMSE metrics...")
    results["residual_metrics"] = {}
    
    for split_name, partition in [
        ("train", dataset.train),
        ("validation", dataset.val),
        ("test", dataset.test),
    ]:
        results["residual_metrics"][split_name] = {
            "ridge": ridge_model.residual_rmse(partition.X, partition.y),
            "tree": tree_model.residual_rmse(partition.X, partition.y),
        }
        logger.info(f"  {split_name} RMSE (ridge): {results['residual_metrics'][split_name]['ridge']['rmse_dist_km']:.4f} km")
        logger.info(f"  {split_name} RMSE (tree):  {results['residual_metrics'][split_name]['tree']['rmse_dist_km']:.4f} km")

    # Trajectory metrics on TEST set only (held-out evaluation)
    logger.info("Computing trajectory metrics on TEST set...")
    models = {"ridge": ridge_model, "tree": tree_model}
    traj_results = evaluate_residual_corrections(dataset.test, models)
    results["trajectory_metrics"] = traj_results
    
    # Log trajectory improvements
    physics_rmse = traj_results.get("physics_only", {}).get("rmse_km")
    ridge_rmse = traj_results.get("ridge", {}).get("rmse_km")
    tree_rmse = traj_results.get("tree", {}).get("rmse_km")
    
    if physics_rmse and ridge_rmse:
        ridge_improvement = ((physics_rmse - ridge_rmse) / physics_rmse * 100) if physics_rmse != 0 else 0
        logger.info(f"  Physics-only RMSE: {physics_rmse:.4f} km")
        logger.info(f"  Physics+Ridge RMSE: {ridge_rmse:.4f} km ({ridge_improvement:+.1f}%)")
    
    if physics_rmse and tree_rmse:
        tree_improvement = ((physics_rmse - tree_rmse) / physics_rmse * 100) if physics_rmse != 0 else 0
        logger.info(f"  Physics+Tree RMSE: {tree_rmse:.4f} km ({tree_improvement:+.1f}%)")

    return results


def detect_overfitting(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze residual metrics to detect overfitting."""
    logger.info("Analyzing overfitting indicators...")
    analysis = {
        "ridge": {},
        "tree": {},
    }

    for model_name in ["ridge", "tree"]:
        train_rmse = results["residual_metrics"]["train"][model_name]["rmse_dist_km"]
        val_rmse = results["residual_metrics"]["validation"][model_name]["rmse_dist_km"]
        test_rmse = results["residual_metrics"]["test"][model_name]["rmse_dist_km"]
        
        analysis[model_name]["train_rmse_km"] = train_rmse
        analysis[model_name]["val_rmse_km"] = val_rmse
        analysis[model_name]["test_rmse_km"] = test_rmse
        
        # Overfitting indicators
        if train_rmse > 0:
            val_deterioration = (val_rmse - train_rmse) / train_rmse * 100
            test_deterioration = (test_rmse - train_rmse) / train_rmse * 100
            analysis[model_name]["val_deterioration_pct"] = val_deterioration
            analysis[model_name]["test_deterioration_pct"] = test_deterioration
            
            # Flag overfitting if val/test significantly worse than train
            if val_deterioration > 20:
                logger.warning(f"  {model_name}: Validation RMSE {val_deterioration:+.1f}% worse than train (potential overfitting)")
            if test_deterioration > 20:
                logger.warning(f"  {model_name}: Test RMSE {test_deterioration:+.1f}% worse than train (potential overfitting)")

    return analysis


def generate_report(results: Dict[str, Any], overfitting_analysis: Dict[str, Any]) -> None:
    """Print comprehensive results report."""
    print("\n" + "=" * 80)
    print("PHASE 1, STEP 1.2: PHYSICS-RESIDUAL ML MODEL EVALUATION")
    print("=" * 80)
    
    print("\n1. MODELS IMPLEMENTED:")
    for model in results["models_trained"]:
        print(f"   - {model}")
    
    print("\n2. SKLEARN VERSION:")
    print(f"   {results['sklearn_version']}")
    
    print("\n3. HYPERPARAMETERS:")
    for model_name, params in results["hyperparameters"].items():
        print(f"\n   {model_name.upper()}:")
        for key, val in params.items():
            print(f"      {key}: {val}")
    
    print("\n4. SAMPLE COUNTS:")
    for partition, count in results["sample_counts"].items():
        print(f"   {partition}: {count}")
    
    print("\n5. RESIDUAL PREDICTION ERROR METRICS (RMSE):")
    print("\n   Train Partition (44 samples):")
    for model in ["ridge", "tree"]:
        metrics = results["residual_metrics"]["train"][model]
        print(f"      {model.upper()}:")
        print(f"         RMSE_X: {metrics['rmse_x_m']:.2f} m")
        print(f"         RMSE_Y: {metrics['rmse_y_m']:.2f} m")
        print(f"         RMSE_DIST: {metrics['rmse_dist_km']:.4f} km")
    
    print("\n   Validation Partition (15 samples):")
    for model in ["ridge", "tree"]:
        metrics = results["residual_metrics"]["validation"][model]
        print(f"      {model.upper()}:")
        print(f"         RMSE_X: {metrics['rmse_x_m']:.2f} m")
        print(f"         RMSE_Y: {metrics['rmse_y_m']:.2f} m")
        print(f"         RMSE_DIST: {metrics['rmse_dist_km']:.4f} km")
    
    print("\n   Test Partition (14 samples - HELD-OUT EVALUATION):")
    for model in ["ridge", "tree"]:
        metrics = results["residual_metrics"]["test"][model]
        print(f"      {model.upper()}:")
        print(f"         RMSE_X: {metrics['rmse_x_m']:.2f} m")
        print(f"         RMSE_Y: {metrics['rmse_y_m']:.2f} m")
        print(f"         RMSE_DIST: {metrics['rmse_dist_km']:.4f} km")
    
    print("\n6. TRAJECTORY METRICS (Test Set - Final Evaluation):")
    traj_metrics = results["trajectory_metrics"]
    
    for approach in ["physics_only", "ridge", "tree"]:
        if approach in traj_metrics:
            metrics = traj_metrics[approach]
            label = "Physics-only Baseline" if approach == "physics_only" else f"Physics+{approach.upper()}"
            print(f"\n   {label}:")
            for key, val in metrics.items():
                if val is not None:
                    if "km" in key or "deg" in key or "error" in key or "distance" in key:
                        print(f"      {key}: {val:.4f}")
                    else:
                        print(f"      {key}: {val}")
    
    print("\n7. OVERFITTING ANALYSIS:")
    for model in ["ridge", "tree"]:
        analysis = overfitting_analysis[model]
        print(f"\n   {model.upper()}:")
        print(f"      Train RMSE:  {analysis['train_rmse_km']:.4f} km")
        print(f"      Val RMSE:    {analysis['val_rmse_km']:.4f} km ({analysis['val_deterioration_pct']:+.1f}%)")
        print(f"      Test RMSE:   {analysis['test_rmse_km']:.4f} km ({analysis['test_deterioration_pct']:+.1f}%)")
        
        if analysis['test_deterioration_pct'] <= 20:
            print(f"      Status: ✓ ACCEPTABLE (test only {analysis['test_deterioration_pct']:.1f}% worse than train)")
        else:
            print(f"      Status: ⚠ POTENTIAL OVERFITTING (test {analysis['test_deterioration_pct']:.1f}% worse than train)")
    
    print("\n8. MODEL IMPROVEMENT ON HELD-OUT TEST SET:")
    physics_metrics = traj_metrics.get("physics_only", {})
    physics_rmse = physics_metrics.get("rmse_km")
    
    if physics_rmse is not None:
        for model in ["ridge", "tree"]:
            if model in traj_metrics:
                model_rmse = traj_metrics[model].get("rmse_km")
                if model_rmse is not None:
                    improvement_pct = ((physics_rmse - model_rmse) / physics_rmse * 100) if physics_rmse != 0 else 0
                    status = "✓ IMPROVES" if improvement_pct > 0 else "✗ WORSENS"
                    print(f"   {model.upper()}: {status} trajectory RMSE by {improvement_pct:+.2f}%")
    
    print("\n9. FILES CREATED/MODIFIED:")
    print("   Created:")
    print("      - src/models/ml/residual_model.py (RidgeResidualModel, TreeResidualModel classes)")
    print("      - tests/test_residual_model.py (comprehensive unit & integration tests)")
    print("   Modified:")
    print("      - src/models/ml/dataset.py (from Step 1.1)")
    print("      - src/models/ml/features.py (from Step 1.1)")
    print("      - requirements.txt (scikit-learn>=1.3.0 - already present)")
    
    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80 + "\n")


def main():
    """Execute Phase 1 Step 1.2 evaluation."""
    try:
        # Load data
        context = load_a23a_dataset()
        dataset = context["dataset"]
        
        logger.info(f"Dataset summary: {dataset.total_samples} total samples")
        logger.info(f"  Train: {len(dataset.train)}, Val: {len(dataset.val)}, Test: {len(dataset.test)}")
        
        # Train models
        ridge_model = train_ridge_model(dataset.train.X, dataset.train.y)
        tree_model = train_tree_model(dataset.train.X, dataset.train.y)
        
        # Evaluate
        results = evaluate_models(ridge_model, tree_model, dataset)
        
        # Overfitting analysis
        overfitting_analysis = detect_overfitting(results)
        
        # Generate report
        generate_report(results, overfitting_analysis)
        
        # Save results to JSON
        results_path = Path("results_step1p2.json")
        with open(results_path, "w") as f:
            # Make results JSON-serializable
            results_serializable = {
                "models_trained": results["models_trained"],
                "sklearn_version": results["sklearn_version"],
                "hyperparameters": results["hyperparameters"],
                "sample_counts": results["sample_counts"],
                "residual_metrics": {
                    split: {
                        model: {k: float(v) for k, v in metrics.items()}
                        for model, metrics in split_metrics.items()
                    }
                    for split, split_metrics in results["residual_metrics"].items()
                },
                "trajectory_metrics": {
                    approach: {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                               for k, v in metrics.items()}
                    for approach, metrics in results["trajectory_metrics"].items()
                },
                "overfitting_analysis": overfitting_analysis,
            }
            json.dump(results_serializable, f, indent=2)
        logger.info(f"Results saved to {results_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
