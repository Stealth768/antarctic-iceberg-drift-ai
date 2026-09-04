"""
Evaluation package.
"""

from src.evaluation.baseline_evaluator import (
    BaselineEvaluationReport,
    BaselinePredictionResult,
    ConstantVelocityBaselineEvaluator,
    HorizonMetrics,
    compute_horizon_metrics,
)
from src.evaluation.historical_pairs import (
    EvaluationPair,
    build_evaluation_pairs,
    calculate_bearing_deg,
    calculate_geodesic_error_km,
    calculate_geodesic_errors_km,
    compute_evaluation_dataset_statistics,
    evaluation_pairs_to_dataframe,
)
from src.evaluation.physics_evaluator import (
    DEFAULT_PROTOTYPE_DRAFT_M,
    DEFAULT_PROTOTYPE_LENGTH_M,
    DEFAULT_PROTOTYPE_MASS_KG,
    DEFAULT_PROTOTYPE_WIDTH_M,
    IcebergPhysicsEvaluator,
    PhysicsEvaluationReport,
    PhysicsPredictionResult,
    create_default_iceberg_properties,
)
from src.evaluation.real_physics_benchmark import (
    EnvironmentalCatalog,
    RealPhysicsBenchmarkReport,
    SkippedCase,
    discover_environmental_datasets,
    run_real_physics_benchmark,
)

__all__ = [
    "EvaluationPair",
    "build_evaluation_pairs",
    "calculate_geodesic_error_km",
    "calculate_geodesic_errors_km",
    "calculate_bearing_deg",
    "evaluation_pairs_to_dataframe",
    "compute_evaluation_dataset_statistics",
    "BaselinePredictionResult",
    "HorizonMetrics",
    "BaselineEvaluationReport",
    "ConstantVelocityBaselineEvaluator",
    "compute_horizon_metrics",
    "PhysicsPredictionResult",
    "PhysicsEvaluationReport",
    "IcebergPhysicsEvaluator",
    "create_default_iceberg_properties",
    "DEFAULT_PROTOTYPE_MASS_KG",
    "DEFAULT_PROTOTYPE_LENGTH_M",
    "DEFAULT_PROTOTYPE_WIDTH_M",
    "DEFAULT_PROTOTYPE_DRAFT_M",
    "EnvironmentalCatalog",
    "RealPhysicsBenchmarkReport",
    "SkippedCase",
    "discover_environmental_datasets",
    "run_real_physics_benchmark",
]
