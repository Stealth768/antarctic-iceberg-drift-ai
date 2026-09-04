from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.copernicus import CopernicusLoader
from src.data.era5 import ERA5Loader
from src.data.nsidc import NSIDCLoader
from src.data.observations import IcebergObservationLoader
from src.data.environment import CompositeEnvironmentProvider
from src.models.iceberg_physics import simulate_iceberg, IcebergProperties, IcebergState, CoordinateHandler
from src.metrics.trajectory import calculate_trajectory_metrics

app = FastAPI(
    title="Antarctic Iceberg Drift Simulation Engine",
    description="REST API delivering physics-based ocean/atmosphere iceberg trajectory predictions in GeoJSON format.",
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

# Global Environment Loaders
glorys_path = Path('data/raw/glorys_test/glorys_a23a_test.nc')
era5_path = Path('data/raw/era5_test/era5_a23a_real_200001.nc')
nsidc_path = Path('data/raw/nsidc_test/nsidc_a23a_test.nc')

env_provider = CompositeEnvironmentProvider(
    nsidc_loader=NSIDCLoader(source=nsidc_path),
    era5_loader=ERA5Loader(source=era5_path),
    copernicus_loader=CopernicusLoader(source=glorys_path)
)

coord_handler = CoordinateHandler(crs='EPSG:3412')

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

@app.get("/health")
def health_check():
    return {"status": "online", "environment": "Antarctic Coastal Current Engine v1.0"}

@app.post("/api/v1/simulate")
def run_simulation(req: SimulationRequest) -> Dict[str, Any]:
    try:
        start_dt = datetime.fromisoformat(req.start_time_iso)
        duration_sec = req.duration_hours * 3600.0

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
            crs='EPSG:3412'
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

        # LineString Feature for full path path visualization
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
            obs_loader = IcebergObservationLoader(obs_path)
            df_truth = obs_loader.load_track()
            metrics = calculate_trajectory_metrics(df_sim, df_truth)

        return {
            "status": "success",
            "parameters": req.model_dump(),
            "metrics": metrics,
            "geojson": geojson_payload
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
