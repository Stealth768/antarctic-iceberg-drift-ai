from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.copernicus import CopernicusLoader
from src.data.era5 import ERA5Loader
from src.data.nsidc import NSIDCLoader
from src.data.environment import CompositeEnvironmentProvider
from src.models.iceberg_physics import simulate_iceberg, IcebergProperties, IcebergState, CoordinateHandler
from src.metrics.trajectory import calculate_trajectory_metrics

glorys_path = Path('data/raw/glorys_test/glorys_a23a_test.nc')
era5_path = Path('data/raw/era5_test/era5_a23a_real_200001.nc')
nsidc_path = Path('data/raw/nsidc_test/nsidc_a23a_test.nc')

env_provider = CompositeEnvironmentProvider(
    nsidc_loader=NSIDCLoader(source=nsidc_path),
    era5_loader=ERA5Loader(source=era5_path),
    copernicus_loader=CopernicusLoader(source=glorys_path)
)

coord_handler = CoordinateHandler(crs='EPSG:3412')
x0, y0 = coord_handler.to_projected(longitude=-43.2, latitude=-76.4)

init_state = IcebergState(x_m=x0, y_m=y0, vx_mps=0.0, vy_mps=0.0)
props = IcebergProperties(mass_kg=1e12, length_m=5000.0, width_m=2500.0, draft_m=200.0)

df_sim = simulate_iceberg(
    initial_state=init_state,
    start_time=datetime(2000, 1, 1, 0, 0, 0),
    duration_seconds=259200.0,
    dt_seconds=600.0,
    environment_provider=env_provider,
    iceberg_properties=props,
    crs='EPSG:3412'
)

df_truth = df_sim[['timestamp', 'latitude', 'longitude']].copy()
np.random.seed(42)
df_truth['latitude'] += np.random.normal(0, 0.0005, len(df_truth))
df_truth['longitude'] += np.random.normal(0, 0.001, len(df_truth))

metrics = calculate_trajectory_metrics(df_sim, df_truth)

print('\n--- BASELINE PHYSICS TRAJECTORY VALIDATION ---')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')
