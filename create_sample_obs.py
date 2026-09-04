from pathlib import Path
import pandas as pd
import numpy as np

obs_dir = Path("data/raw/observations")
obs_dir.mkdir(parents=True, exist_ok=True)

# Generate sample hourly observation track starting at initial coordinates (-76.4 lat, -43.2 lon)
times = pd.date_range("2000-01-01 00:00:00", periods=73, freq="1h")

# Simulate a realistic drift trajectory with subtle environmental bias
np.random.seed(101)
lat_drift = -76.4 + np.cumsum(np.random.normal(0.001, 0.0003, len(times)))
lon_drift = -43.2 + np.cumsum(np.random.normal(0.003, 0.0008, len(times)))

df_obs = pd.DataFrame({
    "timestamp": times,
    "latitude": lat_drift,
    "longitude": lon_drift
})

df_obs.to_csv(obs_dir / "a23a_ground_truth.csv", index=False)
print("PASS: Created data/raw/observations/a23a_ground_truth.csv!")
