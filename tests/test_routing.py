"""
Unit and Integration Tests for Antarctic Navigation and Routing Engine.
"""

import pytest
from src.routing.vessel import get_all_vessels, get_vessel_by_id, VesselStatus
from src.routing.grid import PolarNavigationGrid
from src.routing.cost import NavigationCostModel, CostWeights, compute_route_safety_score
from src.routing.planner import MultiCandidateRoutePlanner, EnvironmentSnapshot
from src.routing.zones import generate_risk_zones_geojson


class TestVesselModels:
    """Validation of demonstration vessels and fleet roster."""

    def test_all_vessels_retrieval(self):
        vessels = get_all_vessels()
        assert len(vessels) == 8
        vsl_ids = {v.vessel_id for v in vessels}
        expected_ids = {"VSL-047", "VSL-118", "VSL-302", "VSL-409", "VSL-088", "VSL-221", "VSL-154", "VSL-021"}
        assert expected_ids.issubset(vsl_ids)

    def test_vessel_lookup_valid(self):
        vsl = get_vessel_by_id("VSL-047")
        assert vsl is not None
        assert vsl.vessel_id == "VSL-047"
        assert vsl.polar_ice_class == "PC3"
        assert vsl.is_demonstration is True

        # Case-insensitive lookup
        vsl_lower = get_vessel_by_id("vsl-047")
        assert vsl_lower is not None
        assert vsl_lower.vessel_id == "VSL-047"

    def test_vessel_lookup_invalid(self):
        assert get_vessel_by_id("INVALID_VSL") is None

    def test_vessel_statuses(self):
        valid_statuses = {VesselStatus.ACTIVE, VesselStatus.ANCHORED, VesselStatus.IN_TRANSIT}
        for v in get_all_vessels():
            assert v.status in valid_statuses
            assert -90.0 <= v.latitude <= -50.0
            assert -180.0 <= v.longitude <= 180.0
            assert v.speed >= 0.0


class TestPolarNavigationGrid:
    """Validation of Polar Stereographic spatial grid graph."""

    def test_grid_creation(self):
        grid = PolarNavigationGrid(
            min_lat=-65.0,
            max_lat=-62.0,
            min_lon=-58.0,
            max_lon=-54.0,
            resolution_km=25.0,
        )
        assert grid.nx >= 3
        assert grid.ny >= 3
        assert grid.is_traversable(0, 0)

    def test_coordinate_transformations_roundtrip(self):
        grid = PolarNavigationGrid(
            min_lat=-65.0,
            max_lat=-62.0,
            min_lon=-58.0,
            max_lon=-54.0,
            resolution_km=25.0,
        )
        test_lat, test_lon = -63.5, -56.0
        i, j = grid.geo_to_grid(test_lat, test_lon)
        assert grid.in_bounds(i, j)

        ret_lat, ret_lon = grid.grid_to_geo(i, j)
        assert abs(ret_lat - test_lat) < 0.5
        assert abs(ret_lon - test_lon) < 1.0

    def test_grid_neighbors_and_blocking(self):
        grid = PolarNavigationGrid(
            min_lat=-65.0,
            max_lat=-62.0,
            min_lon=-58.0,
            max_lon=-54.0,
            resolution_km=25.0,
        )
        i, j = 1, 1
        neighbors = grid.get_neighbors(i, j, allow_diagonal=True)
        assert len(neighbors) == 8

        # Block one neighbor
        ni, nj, _ = neighbors[0]
        grid.set_blocked(ni, nj, True)
        assert not grid.is_traversable(ni, nj)

        new_neighbors = grid.get_neighbors(i, j, allow_diagonal=True)
        assert len(new_neighbors) == 7
        assert (ni, nj) not in [(n[0], n[1]) for n in new_neighbors]


class TestNavigationCostModel:
    """Validation of maritime risk and step cost calculations."""

    def test_cost_calculation_open_water_vs_ice(self):
        cost_pc7 = NavigationCostModel(polar_ice_class="PC7")
        # In open water
        res_open = cost_pc7.evaluate_step(
            step_distance_km=20.0,
            move_vector_dx=20000.0,
            move_vector_dy=0.0,
            sea_ice_concentration=0.02,
        )
        assert res_open.sea_ice_risk == 0.0

        # In heavy sea ice (0.80) exceeding PC7 (0.30) tolerance
        res_heavy = cost_pc7.evaluate_step(
            step_distance_km=20.0,
            move_vector_dx=20000.0,
            move_vector_dy=0.0,
            sea_ice_concentration=0.80,
        )
        assert res_heavy.sea_ice_risk > 0.6
        assert res_heavy.total_step_cost > res_open.total_step_cost
        assert res_heavy.fuel_multiplier > res_open.fuel_multiplier

    def test_iceberg_proximity_cost(self):
        model = NavigationCostModel()
        res_close = model.evaluate_step(20.0, 20000.0, 0.0, nearest_iceberg_dist_km=2.0)
        assert res_close.iceberg_risk == 1.0

        res_far = model.evaluate_step(20.0, 20000.0, 0.0, nearest_iceberg_dist_km=50.0)
        assert res_far.iceberg_risk == 0.0

    def test_weather_and_current_cost(self):
        model = NavigationCostModel()
        # High wind (24 m/s storm)
        res_storm = model.evaluate_step(20.0, 20000.0, 0.0, wind_u=24.0, wind_v=0.0)
        assert res_storm.weather_risk > 0.5

        # Opposing current (move East, current West)
        res_opposing = model.evaluate_step(20.0, 20000.0, 0.0, ocean_u=-0.8, ocean_v=0.0)
        assert res_opposing.current_penalty > 0.0

    def test_safety_score_calculation(self):
        # Calm open water
        score_safe = compute_route_safety_score(0.0, 0.0, 0.0, 0)
        assert score_safe == 100.0

        # Hazardous route
        score_danger = compute_route_safety_score(0.8, 0.95, 0.7, 3)
        assert 5.0 <= score_danger < 50.0


class TestMultiCandidateRoutePlanner:
    """Validation of deterministic multi-route generation."""

    def test_route_generation_produces_three_routes(self):
        grid = PolarNavigationGrid(
            min_lat=-65.0,
            max_lat=-62.5,
            min_lon=-58.0,
            max_lon=-54.0,
            resolution_km=20.0,
        )

        # Populate synthetic environmental snapshot with ice concentrated in the middle
        sic_map = {}
        wu_map = {}
        wv_map = {}
        ou_map = {}
        ov_map = {}
        for i in range(grid.nx):
            for j in range(grid.ny):
                cell = (i, j)
                # Dense ice band in center
                sic_map[cell] = 0.85 if 2 <= i <= 5 and 2 <= j <= 5 else 0.05
                wu_map[cell] = -3.0
                wv_map[cell] = 5.0
                ou_map[cell] = 0.05
                ov_map[cell] = 0.02

        env = EnvironmentSnapshot(
            sea_ice_concentration=sic_map,
            wind_u=wu_map,
            wind_v=wv_map,
            ocean_u=ou_map,
            ocean_v=ov_map,
            iceberg_locations=[(-63.8, -56.0)],
        )

        planner = MultiCandidateRoutePlanner(grid, polar_ice_class="PC5")
        routes = planner.plan_routes(
            start_lat=-64.5,
            start_lon=-57.5,
            goal_lat=-63.0,
            goal_lon=-54.5,
            env=env,
        )

        assert len(routes) == 3
        route_a, route_b, route_c = routes

        assert route_a.route_id == "route_a_direct"
        assert route_b.route_id == "route_b_recommended"
        assert route_c.route_id == "route_c_alternative"

        # Check required fields
        for r in routes:
            assert len(r.path_coordinates) >= 2
            assert r.distance_km > 0.0
            assert r.distance_nm > 0.0
            assert r.eta_hours > 0.0
            assert r.estimated_fuel_mt > 0.0
            assert 0.0 <= r.safety_score <= 100.0
            assert 0.0 <= r.sea_ice_risk <= 100.0
            assert 0.0 <= r.weather_risk <= 100.0

        # Route B (Recommended) should prioritize safety over Route A (Direct)
        assert route_b.safety_score >= route_a.safety_score


class TestRiskZones:
    """Validation of GeoJSON risk zone generation."""

    def test_risk_zones_generation(self):
        grid = PolarNavigationGrid(
            min_lat=-65.0,
            max_lat=-63.0,
            min_lon=-58.0,
            max_lon=-54.0,
            resolution_km=25.0,
        )
        sic_map = {(i, j): 0.70 if i < 3 else 0.35 for i in range(grid.nx) for j in range(grid.ny)}
        env = EnvironmentSnapshot(
            sea_ice_concentration=sic_map,
            wind_u={},
            wind_v={},
            ocean_u={},
            ocean_v={},
            iceberg_locations=[(-64.0, -56.0)],
        )

        geojson = generate_risk_zones_geojson(grid, env)
        assert geojson["type"] == "FeatureCollection"
        assert "features" in geojson
        assert len(geojson["features"]) > 0

        zone_types = {f["properties"]["zone_type"] for f in geojson["features"]}
        assert "High Ice Risk" in zone_types or "Moderate Ice Risk" in zone_types
