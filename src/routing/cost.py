"""
Antarctic Navigation Risk & Cost Model.

Calculates explicit physical and environmental costs for polar vessel navigation:
- Sea-ice concentration & vessel polar ice class compatibility
- Proximity to drifting icebergs
- High wind / adverse weather conditions
- Ocean current opposition/assistance
- Distance and ice-induced fuel consumption penalties
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# Polar ice class thresholds: maximum nominal safe sea ice concentration (SIC)
ICE_CLASS_SIC_TOLERANCE = {
    "PC1": 1.00,
    "PC2": 0.95,
    "PC3": 0.85,
    "PC4": 0.75,
    "PC5": 0.60,
    "PC6": 0.45,
    "PC7": 0.30,
    "1A-Super": 0.35,
    "1A": 0.25,
    "Open Water": 0.05,
}


@dataclass
class CostWeights:
    """Weights for the composite navigation cost function."""
    distance_weight: float = 1.0
    ice_risk_weight: float = 8.0
    iceberg_risk_weight: float = 10.0
    weather_risk_weight: float = 4.0
    current_weight: float = 2.0


@dataclass
class StepCostResult:
    """Detailed breakdown of costs incurred for a single navigation step."""
    step_distance_km: float
    total_step_cost: float
    sea_ice_risk: float        # 0.0 - 1.0
    iceberg_risk: float        # 0.0 - 1.0
    weather_risk: float        # 0.0 - 1.0
    current_penalty: float     # -1.0 (assist) to +1.0 (oppose)
    fuel_multiplier: float     # Factor >= 1.0 representing fuel burn amplification in ice


class NavigationCostModel:
    """
    Evaluates step costs and safety scores for polar maritime transit.
    """

    def __init__(
        self,
        polar_ice_class: str = "PC5",
        weights: Optional[CostWeights] = None,
    ) -> None:
        self.polar_ice_class = polar_ice_class
        self.max_tolerable_sic = ICE_CLASS_SIC_TOLERANCE.get(polar_ice_class, 0.30)
        self.weights = weights or CostWeights()

    def evaluate_step(
        self,
        step_distance_km: float,
        move_vector_dx: float,
        move_vector_dy: float,
        sea_ice_concentration: Optional[float] = 0.0,
        wind_u: Optional[float] = 0.0,
        wind_v: Optional[float] = 0.0,
        ocean_u: Optional[float] = 0.0,
        ocean_v: Optional[float] = 0.0,
        nearest_iceberg_dist_km: Optional[float] = None,
    ) -> StepCostResult:
        """
        Evaluate physical traversal cost between two adjacent positions.
        """
        dist_km = max(0.001, float(step_distance_km))

        # 1. Sea Ice Risk Assessment based on Polar Ice Class
        sic = max(0.0, min(1.0, float(sea_ice_concentration if sea_ice_concentration is not None else 0.0)))
        if sic <= 0.05:
            ice_risk = 0.0
        elif sic <= self.max_tolerable_sic:
            # Linear ramp inside vessel capability
            ice_risk = 0.5 * (sic / self.max_tolerable_sic)
        else:
            # Exponential penalty when exceeding vessel ice tolerance
            excess = (sic - self.max_tolerable_sic) / max(0.01, 1.0 - self.max_tolerable_sic)
            ice_risk = min(1.0, 0.5 + 0.5 * (excess ** 1.5))

        # 2. Iceberg Proximity Risk
        if nearest_iceberg_dist_km is not None and nearest_iceberg_dist_km >= 0.0:
            if nearest_iceberg_dist_km < 3.0:
                iceberg_risk = 1.0
            elif nearest_iceberg_dist_km < 20.0:
                iceberg_risk = (20.0 - nearest_iceberg_dist_km) / 17.0
            else:
                iceberg_risk = 0.0
        else:
            iceberg_risk = 0.0

        # 3. Weather / Wind Risk
        wu = float(wind_u if wind_u is not None else 0.0)
        wv = float(wind_v if wind_v is not None else 0.0)
        wind_speed = math.hypot(wu, wv)
        # Gale scale: >=17 m/s (34 kts) high risk, >=25 m/s (48 kts) severe
        if wind_speed < 10.0:
            weather_risk = 0.0
        elif wind_speed < 20.0:
            weather_risk = (wind_speed - 10.0) / 20.0
        else:
            weather_risk = min(1.0, 0.5 + (wind_speed - 20.0) / 20.0)

        # 4. Ocean Current Resistance / Assistance
        ou = float(ocean_u if ocean_u is not None else 0.0)
        ov = float(ocean_v if ocean_v is not None else 0.0)
        current_speed = math.hypot(ou, ov)
        move_len = math.hypot(move_vector_dx, move_vector_dy)
        if move_len > 1e-6 and current_speed > 1e-4:
            # Unit move direction
            ux = move_vector_dx / move_len
            uy = move_vector_dy / move_len
            # Dot product with current vector
            dot = (ux * ou + uy * ov) / current_speed
            # dot = +1 when current aligns with movement, -1 when opposing
            current_penalty = -float(dot) * min(1.0, current_speed / 1.5)
        else:
            current_penalty = 0.0

        # 5. Fuel Multiplier: pushing through ice significantly escalates fuel burn
        fuel_multiplier = 1.0 + 3.0 * (sic ** 2) + max(0.0, current_penalty) * 0.5

        # 6. Composite Step Cost
        step_cost = dist_km * (
            self.weights.distance_weight
            + self.weights.ice_risk_weight * ice_risk
            + self.weights.iceberg_risk_weight * iceberg_risk
            + self.weights.weather_risk_weight * weather_risk
            + self.weights.current_weight * max(0.0, current_penalty)
        )

        return StepCostResult(
            step_distance_km=dist_km,
            total_step_cost=step_cost,
            sea_ice_risk=ice_risk,
            iceberg_risk=iceberg_risk,
            weather_risk=weather_risk,
            current_penalty=current_penalty,
            fuel_multiplier=fuel_multiplier,
        )


def compute_route_safety_score(
    mean_ice_risk: float,
    max_ice_risk: float,
    mean_weather_risk: float,
    iceberg_hazard_count: int,
) -> float:
    """
    Calculate an objective 0 to 100 route safety score based on exposure.
    """
    deduction = (
        (mean_ice_risk * 45.0)
        + (max_ice_risk * 25.0)
        + (mean_weather_risk * 20.0)
        + min(20.0, iceberg_hazard_count * 5.0)
    )
    score = max(5.0, min(100.0, 100.0 - deduction))
    return round(score, 1)
