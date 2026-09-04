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

# 1. Load Real Observations
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

# 3. Setup Physics Initial State matching initial observation coordinate
coord_handler = CoordinateHandler(crs='EPSG:3412')
x0, y0 = coord_handler.to_projected(longitude=init_lon, latitude=init_lat)

init_state = IcebergState(x_m=x0, y_m=y0, vx_mps=0.0, vy_mps=0.0)
props = IcebergProperties(mass_kg=1e12, length_m=5000.0, width_m=2500.0, draft_m=200.0)

# 4. Simulate Iceberg Trajectory
df_sim = simulate_iceberg(
    initial_state=init_state,
    start_time=start_time,
    duration_seconds=duration_sec,
    dt_seconds=600.0,
    environment_provider=env_provider,
    iceberg_properties=props,
    crs='EPSG:3412'
)

# 5. Evaluate Metrics against Ground Truth
metrics = calculate_trajectory_metrics(df_sim, df_truth)

print('\n==================================================')
print('   STAGE 7: EMPIRICAL OBSERVATIONAL VALIDATION    ')
print('==================================================')
print(f' Observation Track Start : {start_time}')
print(f' Observation Track End   : {end_time}')
print(f' Initial Position        : ({init_lat:.4f} N, {init_lon:.4f} E)')
print('--------------------------------------------------')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')
print('==================================================\n')
