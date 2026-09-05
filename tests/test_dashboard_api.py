"""
Integration and End-to-End API Tests for Antarctic Navigation Dashboard Backend.
"""

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "modules" in data


def test_get_vessels_list(client):
    response = client.get("/api/v1/vessels")
    assert response.status_code == 200
    vessels = response.json()
    assert len(vessels) == 8
    vsl_ids = [v["vessel_id"] for v in vessels]
    assert "VSL-047" in vsl_ids
    assert "VSL-118" in vsl_ids


def test_get_vessel_by_id_valid(client):
    response = client.get("/api/v1/vessels/VSL-047")
    assert response.status_code == 200
    vsl = response.json()
    assert vsl["vessel_id"] == "VSL-047"
    assert vsl["polar_ice_class"] == "PC3"
    assert vsl["is_demonstration"] is True


def test_get_vessel_by_id_invalid(client):
    response = client.get("/api/v1/vessels/UNKNOWN-999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_plan_routes_post(client):
    payload = {
        "start_latitude": -64.25,
        "start_longitude": -56.75,
        "goal_latitude": -63.80,
        "goal_longitude": -57.20,
        "vessel_id": "VSL-047",
        "grid_resolution_km": 20.0,
    }
    response = client.post("/api/v1/routes/plan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "routes" in data
    assert len(data["routes"]) == 3

    route_ids = [r["route_id"] for r in data["routes"]]
    assert "route_a_direct" in route_ids
    assert "route_b_recommended" in route_ids
    assert "route_c_alternative" in route_ids

    # Route B recommended should have safety score
    rec = data["recommendation_summary"]
    assert rec["selected_route_id"] == "route_b_recommended"
    assert 0.0 <= rec["safety_score"] <= 100.0


def test_get_recommended_routes(client):
    response = client.get("/api/v1/routes/recommend?vessel_id=VSL-047")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["routes"]) == 3


def test_get_recommended_routes_invalid_vessel(client):
    response = client.get("/api/v1/routes/recommend?vessel_id=INVALID_SHIP")
    assert response.status_code == 404


def test_get_risk_zones(client):
    response = client.get("/api/v1/risk-zones")
    assert response.status_code == 200
    geojson = response.json()
    assert geojson["type"] == "FeatureCollection"
    assert "features" in geojson
    assert geojson["metadata"]["is_demonstration"] is True


def test_get_live_environmental_conditions(client):
    response = client.get("/api/v1/environmental/live?latitude=-64.25&longitude=-56.75")
    assert response.status_code == 200
    env = response.json()
    assert env["latitude"] == -64.25
    assert env["longitude"] == -56.75
    # Transparently null values must be present and None
    assert env["wave_height_m"] is None
    assert env["rainfall_mm_hr"] is None
    assert env["visibility_km"] is None
    assert env["storm_probability_pct"] is None
    assert "data_source_status" in env


def test_get_alerts(client):
    response = client.get("/api/v1/alerts")
    assert response.status_code == 200
    alerts = response.json()
    assert isinstance(alerts, list)
    assert len(alerts) > 0
    assert "alert_id" in alerts[0]
    assert "severity" in alerts[0]


def test_get_system_status(client):
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    status = response.json()
    assert status["api_status"] == "healthy"
    assert "operational" in status["physics_engine_status"].lower()
    assert status["registered_vessels_count"] == 8


def test_get_spatial_layers_valid(client):
    for layer in ["sea-ice", "currents", "wind"]:
        response = client.get(f"/api/v1/layers/{layer}?resolution_km=40.0")
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) > 0


def test_get_spatial_layers_invalid(client):
    response = client.get("/api/v1/layers/radioactivity")
    assert response.status_code == 400
    assert "invalid layer_type" in response.json()["detail"].lower()


def test_existing_simulate_endpoint(client):
    payload = {
        "initial_latitude": -76.35,
        "initial_longitude": -43.10,
        "start_time_iso": "2000-01-01T00:00:00",
        "duration_hours": 2.0,
        "timestep_seconds": 600.0,
        "mass_kg": 1e11,
        "length_m": 2000.0,
        "width_m": 1000.0,
        "draft_m": 150.0,
        "air_drag_coefficient": 0.20,
        "water_drag_coefficient": 1.00,
    }
    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "geojson" in data
    assert data["geojson"]["type"] == "FeatureCollection"
    assert len(data["geojson"]["features"]) >= 2
