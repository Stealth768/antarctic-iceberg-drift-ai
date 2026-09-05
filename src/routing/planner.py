"""
Multi-Candidate Route Planner for Antarctic Maritime Operations.

Deterministically computes 3 candidate routes (Route A Direct, Route B Recommended, Route C Alternative)
using A* graph search over the PolarNavigationGrid evaluated by NavigationCostModel.
"""

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.routing.grid import PolarNavigationGrid
from src.routing.cost import CostWeights, NavigationCostModel, compute_route_safety_score


class RouteMetricSummary(BaseModel):
    """Metrics and properties of a single calculated candidate route."""
    route_id: str
    route_label: str
    risk_level: str  # "High Risk", "Recommended", "Moderate Risk"
    color_code: str  # "red", "green", "yellow"
    path_coordinates: List[List[float]] = Field(..., description="List of [lon, lat] coordinates (GeoJSON standard)")
    distance_km: float
    distance_nm: float
    eta_hours: float
    estimated_fuel_mt: float
    safety_score: float = Field(..., ge=0.0, le=100.0)
    sea_ice_risk: float = Field(..., ge=0.0, le=100.0)
    weather_risk: float = Field(..., ge=0.0, le=100.0)
    fuel_saving_percent: float


@dataclass
class EnvironmentSnapshot:
    """Snapshot or lookup function for environmental forcing over the grid."""
    sea_ice_concentration: Dict[Tuple[int, int], float]
    wind_u: Dict[Tuple[int, int], float]
    wind_v: Dict[Tuple[int, int], float]
    ocean_u: Dict[Tuple[int, int], float]
    ocean_v: Dict[Tuple[int, int], float]
    iceberg_locations: List[Tuple[float, float]]  # [(lat, lon)]


class MultiCandidateRoutePlanner:
    """
    Computes three distinct candidate routes between Antarctic waypoints.
    """

    def __init__(
        self,
        grid: PolarNavigationGrid,
        polar_ice_class: str = "PC5",
        cruising_speed_knots: float = 12.0,
        fuel_consumption_rate_mt_per_day: float = 18.0,
    ) -> None:
        self.grid = grid
        self.polar_ice_class = polar_ice_class
        self.cruising_speed_knots = max(1.0, float(cruising_speed_knots))
        self.fuel_rate_mt_per_day = max(0.1, float(fuel_consumption_rate_mt_per_day))

    def _find_nearest_iceberg_dist_km(
        self, cell_lat: float, cell_lon: float, icebergs: List[Tuple[float, float]]
    ) -> Optional[float]:
        if not icebergs:
            return None
        min_d = float("inf")
        for ib_lat, ib_lon in icebergs:
            # Approximate great-circle distance in km
            d_lat = math.radians(ib_lat - cell_lat)
            d_lon = math.radians(ib_lon - cell_lon)
            a = math.sin(d_lat / 2.0) ** 2 + math.cos(math.radians(cell_lat)) * math.cos(math.radians(ib_lat)) * math.sin(d_lon / 2.0) ** 2
            c = 2.0 * math.atan2(math.sqrt(max(0.0, a)), math.sqrt(max(0.0, 1.0 - a)))
            d_km = 6371.0 * c
            if d_km < min_d:
                min_d = d_km
        return min_d if min_d != float("inf") else None

    def _run_astar(
        self,
        start_cell: Tuple[int, int],
        goal_cell: Tuple[int, int],
        cost_model: NavigationCostModel,
        env: EnvironmentSnapshot,
        avoid_cells: Optional[Set[Tuple[int, int]]] = None,
        avoid_weight: float = 0.0,
    ) -> Optional[List[Tuple[int, int]]]:
        """Run A* search on the PolarNavigationGrid."""
        if not self.grid.is_traversable(*start_cell) or not self.grid.is_traversable(*goal_cell):
            return None

        avoid_set = avoid_cells or set()
        goal_x, goal_y = self.grid.grid_to_projected(*goal_cell)

        # Open set: (priority_f, g_cost, (i, j))
        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        start_x, start_y = self.grid.grid_to_projected(*start_cell)
        h0 = math.hypot(goal_x - start_x, goal_y - start_y) / 1000.0  # km
        heapq.heappush(open_set, (h0, 0.0, start_cell))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_scores: Dict[Tuple[int, int], float] = {start_cell: 0.0}

        visited = set()

        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == goal_cell:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            cur_x, cur_y = self.grid.grid_to_projected(*current)

            for ni, nj, step_km in self.grid.get_neighbors(*current):
                neighbor = (ni, nj)
                if neighbor in visited:
                    continue

                nxt_x, nxt_y = self.grid.grid_to_projected(ni, nj)
                dx = nxt_x - cur_x
                dy = nxt_y - cur_y

                # Query environmental conditions
                sic = env.sea_ice_concentration.get(neighbor, 0.0)
                wu = env.wind_u.get(neighbor, 0.0)
                wv = env.wind_v.get(neighbor, 0.0)
                ou = env.ocean_u.get(neighbor, 0.0)
                ov = env.ocean_v.get(neighbor, 0.0)

                n_lat, n_lon = self.grid.grid_to_geo(ni, nj)
                ib_dist = self._find_nearest_iceberg_dist_km(n_lat, n_lon, env.iceberg_locations)

                step_res = cost_model.evaluate_step(
                    step_distance_km=step_km,
                    move_vector_dx=dx,
                    move_vector_dy=dy,
                    sea_ice_concentration=sic,
                    wind_u=wu,
                    wind_v=wv,
                    ocean_u=ou,
                    ocean_v=ov,
                    nearest_iceberg_dist_km=ib_dist,
                )

                edge_cost = step_res.total_step_cost
                if neighbor in avoid_set:
                    edge_cost += avoid_weight * step_km

                tentative_g = current_g + edge_cost

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h = math.hypot(goal_x - nxt_x, goal_y - nxt_y) / 1000.0
                    f = tentative_g + h
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f, tentative_g, neighbor))

        return None

    def _evaluate_path(
        self,
        path: List[Tuple[int, int]],
        route_id: str,
        route_label: str,
        risk_level: str,
        color_code: str,
        cost_model: NavigationCostModel,
        env: EnvironmentSnapshot,
        baseline_fuel_mt: Optional[float] = None,
    ) -> RouteMetricSummary:
        """Calculate complete physical & operational metrics for a reconstructed path."""
        coords: List[List[float]] = []
        total_dist_km = 0.0
        ice_risks: List[float] = []
        weather_risks: List[float] = []
        fuel_multipliers: List[float] = []
        iceberg_hazards = 0

        for idx in range(len(path)):
            cell = path[idx]
            lat, lon = self.grid.grid_to_geo(*cell)
            coords.append([round(lon, 5), round(lat, 5)])

            if idx > 0:
                prev_cell = path[idx - 1]
                p_x, p_y = self.grid.grid_to_projected(*prev_cell)
                c_x, c_y = self.grid.grid_to_projected(*cell)
                dx = c_x - p_x
                dy = c_y - p_y
                step_km = math.hypot(dx, dy) / 1000.0
                total_dist_km += step_km

                sic = env.sea_ice_concentration.get(cell, 0.0)
                wu = env.wind_u.get(cell, 0.0)
                wv = env.wind_v.get(cell, 0.0)
                ou = env.ocean_u.get(cell, 0.0)
                ov = env.ocean_v.get(cell, 0.0)
                ib_dist = self._find_nearest_iceberg_dist_km(lat, lon, env.iceberg_locations)
                if ib_dist is not None and ib_dist < 10.0:
                    iceberg_hazards += 1

                step_res = cost_model.evaluate_step(
                    step_distance_km=step_km,
                    move_vector_dx=dx,
                    move_vector_dy=dy,
                    sea_ice_concentration=sic,
                    wind_u=wu,
                    wind_v=wv,
                    ocean_u=ou,
                    ocean_v=ov,
                    nearest_iceberg_dist_km=ib_dist,
                )
                ice_risks.append(step_res.sea_ice_risk)
                weather_risks.append(step_res.weather_risk)
                fuel_multipliers.append(step_res.fuel_multiplier)

        mean_ice_risk = float(sum(ice_risks) / len(ice_risks)) if ice_risks else 0.0
        max_ice_risk = float(max(ice_risks)) if ice_risks else 0.0
        mean_weather_risk = float(sum(weather_risks) / len(weather_risks)) if weather_risks else 0.0
        mean_fuel_mult = float(sum(fuel_multipliers) / len(fuel_multipliers)) if fuel_multipliers else 1.0

        dist_nm = total_dist_km * 0.539957

        # In dense ice, speed degrades up to 50%
        speed_degrade_factor = max(0.5, 1.0 - (mean_ice_risk * 0.5))
        effective_speed_knots = self.cruising_speed_knots * speed_degrade_factor
        eta_hours = dist_nm / max(1.0, effective_speed_knots)

        # Fuel consumption in metric tons (fuel_rate_mt_per_day / 24 * hours * fuel_multiplier)
        fuel_mt = (self.fuel_rate_mt_per_day / 24.0) * eta_hours * mean_fuel_mult

        # Safety score 0-100 calculated from physical risk exposure
        safety_score = compute_route_safety_score(
            mean_ice_risk=mean_ice_risk,
            max_ice_risk=max_ice_risk,
            mean_weather_risk=mean_weather_risk,
            iceberg_hazard_count=iceberg_hazards,
        )

        # Fuel saving percentage compared to baseline (Route A)
        if baseline_fuel_mt is not None and baseline_fuel_mt > 1e-3:
            fuel_saving_pct = ((baseline_fuel_mt - fuel_mt) / baseline_fuel_mt) * 100.0
        else:
            fuel_saving_pct = 0.0

        return RouteMetricSummary(
            route_id=route_id,
            route_label=route_label,
            risk_level=risk_level,
            color_code=color_code,
            path_coordinates=coords,
            distance_km=round(total_dist_km, 2),
            distance_nm=round(dist_nm, 2),
            eta_hours=round(eta_hours, 1),
            estimated_fuel_mt=round(fuel_mt, 2),
            safety_score=safety_score,
            sea_ice_risk=round(mean_ice_risk * 100.0, 1),
            weather_risk=round(mean_weather_risk * 100.0, 1),
            fuel_saving_percent=round(fuel_saving_pct, 1),
        )

    def plan_routes(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
        env: EnvironmentSnapshot,
    ) -> List[RouteMetricSummary]:
        """
        Compute three candidate routes:
        - Route A: Direct / shortest path (ignoring ice, incurring higher risks)
        - Route B: Recommended / optimal safety & fuel path
        - Route C: Alternative / moderate risk path
        """
        start_cell = self.grid.geo_to_grid(start_lat, start_lon)
        goal_cell = self.grid.geo_to_grid(goal_lat, goal_lon)

        # 1. Route A: Direct / Distance-dominated
        cost_model_direct = NavigationCostModel(
            polar_ice_class=self.polar_ice_class,
            weights=CostWeights(
                distance_weight=1.0,
                ice_risk_weight=0.1,
                iceberg_risk_weight=0.1,
                weather_risk_weight=0.1,
                current_weight=0.0,
            ),
        )
        path_a = self._run_astar(start_cell, goal_cell, cost_model_direct, env)
        if not path_a:
            path_a = [start_cell, goal_cell]

        # Standard assessment cost model for all evaluations
        standard_cost_model = NavigationCostModel(polar_ice_class=self.polar_ice_class)
        route_a = self._evaluate_path(
            path=path_a,
            route_id="route_a_direct",
            route_label="Route A: Direct Transit (High Risk)",
            risk_level="High Risk",
            color_code="red",
            cost_model=standard_cost_model,
            env=env,
            baseline_fuel_mt=None,
        )

        # 2. Route B: Recommended / Safety & Fuel Optimized
        cost_model_opt = NavigationCostModel(
            polar_ice_class=self.polar_ice_class,
            weights=CostWeights(
                distance_weight=1.0,
                ice_risk_weight=12.0,
                iceberg_risk_weight=14.0,
                weather_risk_weight=5.0,
                current_weight=2.5,
            ),
        )
        path_b = self._run_astar(start_cell, goal_cell, cost_model_opt, env)
        if not path_b:
            path_b = path_a

        route_b = self._evaluate_path(
            path=path_b,
            route_id="route_b_recommended",
            route_label="Route B: Recommended Passage (Safe & Efficient)",
            risk_level="Recommended",
            color_code="green",
            cost_model=standard_cost_model,
            env=env,
            baseline_fuel_mt=route_a.estimated_fuel_mt,
        )

        # 3. Route C: Alternative Path (Discourages cells used by Route B)
        cells_in_b = set(path_b[1:-1])
        cost_model_alt = NavigationCostModel(
            polar_ice_class=self.polar_ice_class,
            weights=CostWeights(
                distance_weight=1.0,
                ice_risk_weight=4.0,
                iceberg_risk_weight=6.0,
                weather_risk_weight=2.0,
                current_weight=1.0,
            ),
        )
        path_c = self._run_astar(
            start_cell, goal_cell, cost_model_alt, env, avoid_cells=cells_in_b, avoid_weight=15.0
        )
        if not path_c:
            path_c = path_a

        route_c = self._evaluate_path(
            path=path_c,
            route_id="route_c_alternative",
            route_label="Route C: Alternative Corridor (Moderate Risk)",
            risk_level="Moderate Risk",
            color_code="yellow",
            cost_model=standard_cost_model,
            env=env,
            baseline_fuel_mt=route_a.estimated_fuel_mt,
        )

        return [route_a, route_b, route_c]
