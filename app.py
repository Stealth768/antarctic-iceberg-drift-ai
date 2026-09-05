import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.copernicus import CopernicusLoader
from src.data.era5 import ERA5Loader
from src.data.nsidc import NSIDCLoader
from src.data.observations import IcebergObservationLoader
from src.data.environment import CompositeEnvironmentProvider
from src.models.iceberg_physics import simulate_iceberg, IcebergProperties, IcebergState, CoordinateHandler
from src.metrics.trajectory import calculate_trajectory_metrics

from src.routing.vessel import Vessel, get_all_vessels, get_vessel_by_id
from src.routing.grid import PolarNavigationGrid
from src.routing.planner import MultiCandidateRoutePlanner, RouteMetricSummary
from src.routing.zones import generate_risk_zones_geojson
from src.services.environment_service import EnvironmentalService, LiveEnvironmentalConditions
from src.services.alert_service import AlertService, MaritimeAlert, SystemStatusReport

app = FastAPI(
    title="Antarctic Navigation AI — Maritime Decision Support Engine",
    description="REST API delivering physics-based iceberg trajectory predictions, multi-candidate polar route planning, risk zones, environmental telemetry, and active alerts (SIH26059).",
    version="1.0.0"
)

# Enable CORS for Front-End integration (React, Vue, Deck.gl, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Environment Loaders and Coordinate Handler
glorys_path = Path("data/raw/glorys_test/glorys_a23a_test.nc")
era5_path = Path("data/raw/era5_test/era5_a23a_real_200001.nc")
nsidc_path = Path("data/raw/nsidc_test/nsidc_a23a_test.nc")

nsidc_loader = NSIDCLoader(source=nsidc_path) if nsidc_path.exists() else None
era5_loader = ERA5Loader(source=era5_path) if era5_path.exists() else None
copernicus_loader = CopernicusLoader(source=glorys_path) if glorys_path.exists() else None

env_provider = None
if nsidc_loader and era5_loader and copernicus_loader:
    try:
        env_provider = CompositeEnvironmentProvider(
            nsidc_loader=nsidc_loader,
            era5_loader=era5_loader,
            copernicus_loader=copernicus_loader,
        )
    except Exception:
        env_provider = None

coord_handler = CoordinateHandler(crs="EPSG:3412")

# Internal Services
env_service = EnvironmentalService(
    glorys_path=glorys_path,
    era5_path=era5_path,
    nsidc_path=nsidc_path,
)
alert_service = AlertService(env_service=env_service)


class SimulationRequest(BaseModel):
    initial_latitude: float = Field(-63.507, description="Starting latitude (degrees)")
    initial_longitude: float = Field(-55.679, description="Starting longitude (degrees)")
    start_time_iso: str = Field("2020-01-01T00:00:00", description="ISO format start timestamp")
    duration_hours: float = Field(72.0, ge=1.0, le=168.0, description="Simulation length in hours")
    timestep_seconds: float = Field(600.0, ge=60.0, le=3600.0, description="Physics integration step in seconds")
    
    # Physical Properties (Defaults set to A-23a scale + Calibrated Drag)
    mass_kg: float = Field(1e12, description="Iceberg mass in kg")
    length_m: float = Field(5000.0, description="Characteristic length in meters")
    width_m: float = Field(2500.0, description="Characteristic width in meters")
    draft_m: float = Field(200.0, description="Underwater keel draft in meters")
    air_drag_coefficient: float = Field(0.2000, description="Calibrated air drag coefficient (Ca)")
    water_drag_coefficient: float = Field(1.0065, description="Calibrated water drag coefficient (Cw)")


class RoutePlanRequest(BaseModel):
    start_latitude: float = Field(-64.25, ge=-90.0, le=-50.0, description="Start latitude")
    start_longitude: float = Field(-56.75, ge=-180.0, le=180.0, description="Start longitude")
    goal_latitude: float = Field(-63.80, ge=-90.0, le=-50.0, description="Destination latitude")
    goal_longitude: float = Field(-57.20, ge=-180.0, le=180.0, description="Destination longitude")
    vessel_id: Optional[str] = Field(None, description="Optional registered vessel identifier, e.g. VSL-047")
    polar_ice_class: Optional[str] = Field(None, description="Polar ice class (e.g. PC3, PC5, PC7, Open Water)")
    cruising_speed_knots: Optional[float] = Field(None, ge=1.0, le=40.0, description="Cruising speed in knots")
    fuel_consumption_rate_mt_per_day: Optional[float] = Field(None, ge=0.1, description="Nominal fuel burn rate")
    grid_resolution_km: float = Field(20.0, ge=5.0, le=50.0, description="Routing grid resolution in km")


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "environment": "Antarctic Coastal Current Engine v1.0",
        "modules": ["physics", "hybrid_ml", "routing", "risk_zones", "telemetry"]
    }


@app.post("/api/v1/simulate")
def run_simulation(req: SimulationRequest) -> Dict[str, Any]:
    """
    Simulate physics-based iceberg trajectory using RK4 numerical integration.
    """
    try:
        start_dt = datetime.fromisoformat(req.start_time_iso)
        duration_sec = req.duration_hours * 3600.0

        if env_provider is None:
            raise HTTPException(
                status_code=503,
                detail="CompositeEnvironmentProvider unavailable. NetCDF datasets missing or incomplete."
            )

        # Projected Initial State
        x0, y0 = coord_handler.to_projected(longitude=req.initial_longitude, latitude=req.initial_latitude)
        init_state = IcebergState(x_m=x0, y_m=y0, vx_mps=0.0, vy_mps=0.0)

        props = IcebergProperties(
            mass_kg=req.mass_kg,
            length_m=req.length_m,
            width_m=req.width_m,
            draft_m=req.draft_m,
            air_drag_coefficient=req.air_drag_coefficient,
            water_drag_coefficient=req.water_drag_coefficient
        )

        # Run Physics Engine
        df_sim = simulate_iceberg(
            initial_state=init_state,
            start_time=start_dt,
            duration_seconds=duration_sec,
            dt_seconds=req.timestep_seconds,
            environment_provider=env_provider,
            iceberg_properties=props,
            crs="EPSG:3412"
        )

        # Convert Trajectory DataFrame to GeoJSON FeatureCollection
        coordinates = []
        timestamps = []
        features = []

        for _, row in df_sim.iterrows():
            lon, lat = row["longitude"], row["latitude"]
            ts_str = row["timestamp"].isoformat()
            coordinates.append([lon, lat])
            timestamps.append(ts_str)

            # Point Feature per timestep
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "timestamp": ts_str,
                    "velocity_x_mps": float(row["vx_mps"]),
                    "velocity_y_mps": float(row["vy_mps"])
                }
            })

        # LineString Feature for full path visualization
        line_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            },
            "properties": {
                "trajectory_id": "simulated_drift_path",
                "start_time": timestamps[0],
                "end_time": timestamps[-1],
                "total_timesteps": len(coordinates)
            }
        }
        
        features.insert(0, line_feature)

        geojson_payload = {
            "type": "FeatureCollection",
            "features": features
        }

        # Optional Metric Validation if Ground Truth CSV exists
        metrics = None
        obs_path = Path("data/raw/observations/a23a_ground_truth.csv")
        if obs_path.exists():
            try:
                obs_loader = IcebergObservationLoader(obs_path)
                df_truth = obs_loader.load_track()
                metrics = calculate_trajectory_metrics(df_sim, df_truth)
            except Exception:
                metrics = None

        return {
            "status": "success",
            "parameters": req.model_dump(),
            "metrics": metrics,
            "geojson": geojson_payload
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# FLEET & VESSEL ENDPOINTS
# =====================================================================

@app.get("/api/v1/vessels", response_model=List[Vessel])
def list_vessels() -> List[Vessel]:
    """Retrieve roster of demonstration Antarctic vessels."""
    return get_all_vessels()


@app.get("/api/v1/vessels/{vessel_id}", response_model=Vessel)
def get_vessel(vessel_id: str) -> Vessel:
    """Retrieve details for a specific Antarctic vessel."""
    vsl = get_vessel_by_id(vessel_id)
    if not vsl:
        raise HTTPException(status_code=404, detail=f"Vessel '{vessel_id}' not found in fleet roster.")
    return vsl


# =====================================================================
# ROUTE PLANNING & RECOMMENDATIONS
# =====================================================================

@app.post("/api/v1/routes/plan")
def plan_routes(req: RoutePlanRequest) -> Dict[str, Any]:
    """
    Plan multi-candidate navigation routes (Route A Direct, Route B Recommended, Route C Alternative)
    evaluating physical sea-ice, iceberg proximity, weather, and fuel costs.
    """
    try:
        # Default vessel parameters or inherit from vessel_id
        ice_class = req.polar_ice_class or "PC5"
        speed = req.cruising_speed_knots or 12.0
        fuel_rate = req.fuel_consumption_rate_mt_per_day or 18.0

        if req.vessel_id:
            vsl = get_vessel_by_id(req.vessel_id)
            if vsl:
                ice_class = req.polar_ice_class or vsl.polar_ice_class
                speed = req.cruising_speed_knots or vsl.speed or 12.0
                fuel_rate = req.fuel_consumption_rate_mt_per_day or vsl.fuel_consumption_rate

        # Bounding box around waypoints
        min_lat = min(req.start_latitude, req.goal_latitude) - 2.0
        max_lat = max(req.start_latitude, req.goal_latitude) + 2.0
        min_lon = min(req.start_longitude, req.goal_longitude) - 4.0
        max_lon = max(req.start_longitude, req.goal_longitude) + 4.0

        grid = PolarNavigationGrid(
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            resolution_km=req.grid_resolution_km,
        )

        env_snapshot = env_service.create_grid_snapshot(grid)

        planner = MultiCandidateRoutePlanner(
            grid=grid,
            polar_ice_class=ice_class,
            cruising_speed_knots=speed,
            fuel_consumption_rate_mt_per_day=fuel_rate,
        )

        routes = planner.plan_routes(
            start_lat=req.start_latitude,
            start_lon=req.start_longitude,
            goal_lat=req.goal_latitude,
            goal_lon=req.goal_longitude,
            env=env_snapshot,
        )

        rec_route = routes[1] if len(routes) > 1 else routes[0]

        return {
            "status": "success",
            "request": req.model_dump(),
            "recommendation_summary": {
                "selected_route_id": rec_route.route_id,
                "safety_score": rec_route.safety_score,
                "fuel_saving_percent": rec_route.fuel_saving_percent,
                "distance_nm": rec_route.distance_nm,
                "eta_hours": rec_route.eta_hours,
            },
            "routes": [r.model_dump() for r in routes]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route planning failed: {e}")


@app.get("/api/v1/routes/recommend")
def get_recommended_routes(
    vessel_id: str = Query("VSL-047", description="Vessel identifier to plan recommendation for"),
    goal_latitude: Optional[float] = Query(None, description="Optional override goal latitude"),
    goal_longitude: Optional[float] = Query(None, description="Optional override goal longitude"),
) -> Dict[str, Any]:
    """
    Convenience endpoint returning multi-candidate route recommendations for an active vessel.
    """
    vsl = get_vessel_by_id(vessel_id)
    if not vsl:
        raise HTTPException(status_code=404, detail=f"Vessel '{vessel_id}' not found.")

    # Destination waypoint: default to Antarctic Peninsula / Weddell corridor if not overridden
    dest_lat = goal_latitude if goal_latitude is not None else -63.50
    dest_lon = goal_longitude if goal_longitude is not None else -55.70

    req = RoutePlanRequest(
        start_latitude=vsl.latitude,
        start_longitude=vsl.longitude,
        goal_latitude=dest_lat,
        goal_longitude=dest_lon,
        vessel_id=vsl.vessel_id,
        polar_ice_class=vsl.polar_ice_class,
        cruising_speed_knots=max(5.0, vsl.speed),
        fuel_consumption_rate_mt_per_day=vsl.fuel_consumption_rate,
        grid_resolution_km=20.0,
    )
    return plan_routes(req)


# =====================================================================
# RISK ZONES & SPATIAL LAYERS
# =====================================================================

@app.get("/api/v1/risk-zones")
def get_risk_zones() -> Dict[str, Any]:
    """Generate GeoJSON-compatible maritime risk zones (High Ice, Moderate Ice, Iceberg Drift Hazard)."""
    try:
        grid = PolarNavigationGrid(
            min_lat=-66.5,
            max_lat=-61.5,
            min_lon=-61.0,
            max_lon=-52.0,
            resolution_km=25.0,
        )
        env_snapshot = env_service.create_grid_snapshot(grid)
        return generate_risk_zones_geojson(grid, env_snapshot)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate risk zones: {e}")


@app.get("/api/v1/layers/{layer_type}")
def get_spatial_layer(
    layer_type: str,
    min_lat: float = Query(-66.5, ge=-90.0, le=-50.0),
    max_lat: float = Query(-61.5, ge=-90.0, le=-50.0),
    min_lon: float = Query(-61.0, ge=-180.0, le=180.0),
    max_lon: float = Query(-52.0, ge=-180.0, le=180.0),
    resolution_km: float = Query(35.0, ge=10.0, le=100.0),
) -> Dict[str, Any]:
    """
    Retrieve spatial raster/point layer for 'sea-ice', 'currents', or 'wind'.
    """
    valid_layers = {"sea-ice", "currents", "wind"}
    if layer_type not in valid_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer_type '{layer_type}'. Supported layers: {sorted(list(valid_layers))}"
        )

    grid = PolarNavigationGrid(
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        resolution_km=resolution_km,
    )
    snapshot = env_service.create_grid_snapshot(grid)

    features = []
    for i in range(grid.nx):
        for j in range(grid.ny):
            cell = (i, j)
            lat, lon = grid.grid_to_geo(i, j)

            if layer_type == "sea-ice":
                sic = snapshot.sea_ice_concentration.get(cell, 0.0)
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [round(lon, 4), round(lat, 4)]},
                    "properties": {
                        "sea_ice_concentration": round(sic, 3),
                        "sea_ice_pct": round(sic * 100.0, 1),
                    }
                })
            elif layer_type == "currents":
                ou = snapshot.ocean_u.get(cell, 0.0)
                ov = snapshot.ocean_v.get(cell, 0.0)
                spd_kts = math.hypot(ou, ov) * 1.94384
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [round(lon, 4), round(lat, 4)]},
                    "properties": {
                        "ocean_u_mps": round(ou, 3),
                        "ocean_v_mps": round(ov, 3),
                        "speed_knots": round(spd_kts, 2),
                        "direction_deg": round((math.degrees(math.atan2(ou, ov)) + 360.0) % 360.0, 1),
                    }
                })
            elif layer_type == "wind":
                wu = snapshot.wind_u.get(cell, 0.0)
                wv = snapshot.wind_v.get(cell, 0.0)
                spd_kts = math.hypot(wu, wv) * 1.94384
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [round(lon, 4), round(lat, 4)]},
                    "properties": {
                        "wind_u_mps": round(wu, 2),
                        "wind_v_mps": round(wv, 2),
                        "speed_knots": round(spd_kts, 1),
                        "direction_deg": round((math.degrees(math.atan2(-wu, -wv)) + 360.0) % 360.0, 1),
                    }
                })

    return {
        "type": "FeatureCollection",
        "metadata": {
            "layer_type": layer_type,
            "grid_resolution_km": resolution_km,
            "total_points": len(features),
        },
        "features": features,
    }


# =====================================================================
# LIVE ENVIRONMENTAL TELEMETRY
# =====================================================================

@app.get("/api/v1/environmental/live", response_model=LiveEnvironmentalConditions)
def get_live_environmental_conditions(
    latitude: float = Query(-64.25, ge=-90.0, le=-50.0, description="Query latitude"),
    longitude: float = Query(-56.75, ge=-180.0, le=180.0, description="Query longitude"),
) -> LiveEnvironmentalConditions:
    """
    Retrieve live environmental condition telemetry for a specific coordinate.
    Parameters not available in active reanalysis datasets are explicitly null.
    """
    return env_service.query_live_conditions(latitude=latitude, longitude=longitude)


# =====================================================================
# ACTIVE ALERTS & SYSTEM STATUS
# =====================================================================

@app.get("/api/v1/alerts", response_model=List[MaritimeAlert])
def get_active_alerts() -> List[MaritimeAlert]:
    """Retrieve active maritime navigation alerts and advisories."""
    return alert_service.generate_alerts()


@app.get("/api/v1/system/status", response_model=SystemStatusReport)
def get_system_status() -> SystemStatusReport:
    """Retrieve platform, engine, ML, and data pipeline operational status."""
    return alert_service.get_system_status()
