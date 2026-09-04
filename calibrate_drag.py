from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.data.copernicus import CopernicusLoader
from src.data.era5 import ERA5Loader
from src.data.nsidc import NSIDCLoader
from src.data.observations import IcebergObservationLoader
from src.data.environment import CompositeEnvironmentProvider
from src.models.iceberg_physics import simulate_iceberg, IcebergProperties, IcebergState, CoordinateHandler
from src.metrics.trajectory import calculate_trajectory_metrics

# 1. Load Ground Truth Track
obs_path = Path("data/raw/observations/a23a_ground_truth.csv")
obs_loader = IcebergObservationLoader(obs_path)
df_truth = obs_loader.load_track()

start_time = df_truth["timestamp"].iloc[0]
end_time = df_truth["timestamp"].iloc[-1]
duration_sec = (end_time - start_time).total_seconds()

init_lat = df_truth["latitude"].iloc[0]
init_lon = df_truth["longitude"].iloc[0]

# 2. Setup Environment Provider
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

# Base physical dimensions (A-23a approximate scale)
mass = 1e12
length = 5000.0
width = 2500.0
draft = 200.0

print("==================================================")
print("    DRAG COEFFICIENT (Ca, Cw) OPTIMIZATION      ")
print("==================================================")

iteration_count = 0

def objective_function(params):
    global iteration_count
    iteration_count += 1
    ca, cw = params

    props = IcebergProperties(
        mass_kg=mass,
        length_m=length,
        width_m=width,
        draft_m=draft,
        air_drag_coefficient=ca,
        water_drag_coefficient=cw
    )

    try:
        df_sim = simulate_iceberg(
            initial_state=init_state,
            start_time=start_time,
            duration_seconds=duration_sec,
            dt_seconds=600.0,
            environment_provider=env_provider,
            iceberg_properties=props,
            crs='EPSG:3412'
        )
        metrics = calculate_trajectory_metrics(df_sim, df_truth)
        rmse = metrics["rmse_km"]
        print(f" Iter {iteration_count:02d} | Ca: {ca:.4f}, Cw: {cw:.4f} --> RMSE: {rmse:.4f} km")
        return rmse
    except Exception as e:
        print(f" Iter {iteration_count:02d} | Failed with params ({ca:.4f}, {cw:.4f}): {e}")
        return 1e6

# Initial guess and physical parameter bounds
initial_params = [1.3, 0.9]  # Standard defaults
bounds = [(0.2, 3.0), (0.1, 2.5)]

res = minimize(
    objective_function,
    x0=initial_params,
    method='Nelder-Mead',
    bounds=bounds,
    options={'maxiter': 30, 'xatol': 0.01, 'fatol': 0.05}
)

best_ca, best_cw = res.x
best_rmse = res.fun

print("\n--------------------------------------------------")
print("               OPTIMIZATION RESULTS               ")
print("--------------------------------------------------")
print(f" Initial Default RMSE : 8.7222 km (Ca=1.3000, Cw=0.9000)")
print(f" Calibrated Best Ca   : {best_ca:.4f}")
print(f" Calibrated Best Cw   : {best_cw:.4f}")
print(f" Calibrated Best RMSE : {best_rmse:.4f} km")
print(f" Total Simulations    : {iteration_count}")
print("==================================================\n")
