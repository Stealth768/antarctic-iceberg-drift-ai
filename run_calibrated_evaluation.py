from datetime import datetime
from pathlib import Path
import pandas as pd

from src.data.copernicus import CopernicusLoader
from src.data.era5 import ERA5Loader
from src.data.nsidc import NSIDCLoader
from src.data.observations import IcebergObservationLoader
from src.data.environment import CompositeEnvironmentProvider
from src.models.iceberg_physics import simulate_iceberg, IcebergProperties, IcebergState, CoordinateHandler
from src.metrics.trajectory import calculate_trajectory_metrics

# 1. Load Ground Truth Observation Track
obs_path = Path("data/raw/observations/a23a_ground_truth.csv")
obs_loader = IcebergObservationLoader(obs_path)
df_truth = obs_loader.load_track()

start_time = df_truth["timestamp"].iloc[0]
end_time = df_truth["timestamp"].iloc[-1]
duration_sec = (end_time - start_time).total_seconds()

init_lat = df_truth["latitude"].iloc[0]
init_lon = df_truth["longitude"].iloc[0]

# 2. Setup Environment Loaders
glorys_path = Path('data/raw/glorys_test/glorys_a23a_test.nc')
era5_path = Path('data/raw/era5_test/era5_a23a_real_200001.nc')
nsidc_path = Path('data/raw/nsidc_test/nsidc_a23a_test.nc')

env_provider = CompositeEnvironmentProvider(
    nsidc_loader=NSIDCLoader(source=nsidc_path),
    era5_loader=ERA5Loader(source=era5_path),
    copernicus_loader=CopernicusLoader(source=glorys_path)
)

coord_handler = CoordinateHandler(crs='EPSG:3412')
x0, y0 = coord_handler.to_projected(longitude=init_lon, latitude=init_lat)
init_state = IcebergState(x_m=x0, y_m=y0, vx_mps=0.0, vy_mps=0.0)

# Physical dimensions
mass = 1e12
length = 5000.0
width = 2500.0
draft = 200.0

# --- 1. UNCALIBRATED DEFAULT BASELINE (Ca=1.3, Cw=0.9) ---
props_default = IcebergProperties(
    mass_kg=mass,
    length_m=length,
    width_m=width,
    draft_m=draft,
    air_drag_coefficient=1.3,
    water_drag_coefficient=0.9
)
df_sim_default = simulate_iceberg(
    initial_state=init_state,
    start_time=start_time,
    duration_seconds=duration_sec,
    dt_seconds=600.0,
    environment_provider=env_provider,
    iceberg_properties=props_default,
    crs='EPSG:3412'
)
metrics_default = calculate_trajectory_metrics(df_sim_default, df_truth)

# --- 2. CALIBRATED RUN (Ca=0.2000, Cw=1.0065) ---
props_cal = IcebergProperties(
    mass_kg=mass,
    length_m=length,
    width_m=width,
    draft_m=draft,
    air_drag_coefficient=0.2000,
    water_drag_coefficient=1.0065
)
df_sim_cal = simulate_iceberg(
    initial_state=init_state,
    start_time=start_time,
    duration_seconds=duration_sec,
    dt_seconds=600.0,
    environment_provider=env_provider,
    iceberg_properties=props_cal,
    crs='EPSG:3412'
)
metrics_cal = calculate_trajectory_metrics(df_sim_cal, df_truth)

print('\n===================================================================')
print('        STAGE 7: EMPIRICAL CALIBRATION COMPARISON EVALUATION       ')
print('===================================================================')
print(f"{'Metric':<28} | {'Default (Ca=1.3, Cw=0.9)':<22} | {'Calibrated (Ca=0.2, Cw=1.01)':<15}")
print('-------------------------------------------------------------------')
for k in metrics_default.keys():
    v_def = metrics_default[k]
    v_cal = metrics_cal[k]
    print(f" {k:<27} | {v_def:<22.4f} | {v_cal:<15.4f}")
print('===================================================================\n')
