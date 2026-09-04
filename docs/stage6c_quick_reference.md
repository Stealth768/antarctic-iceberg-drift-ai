# Stage 6C Benchmark: Quick Reference Checklist

## ✅ What's Complete & Ready

### Code Changes (5 files modified)
- [x] GLORYS 200m boundary interpolation implemented
- [x] Multi-file environmental chunk loading enabled
- [x] Date-window filtering (2000–2008) enforced
- [x] Dynamic report generation (no hard-coded metrics)
- [x] Complete test suite (91/91 passing)

### Testing (6 new smoke tests, all pass)
- [x] ERA5 loader validates forcing data
- [x] GLORYS loader interpolates 200m boundary
- [x] Environment provider merges ERA5 + GLORYS
- [x] Physics evaluator completes 3-day simulation
- [x] Baseline evaluator computes metrics
- [x] Multi-file loading (monthly chunks)

### Documentation
- [x] [docs/stage6c_data_acquisition.md](docs/stage6c_data_acquisition.md) — Complete data requirements & download commands
- [x] [docs/stage6c_execution_report.md](docs/stage6c_execution_report.md) — Full technical report
- [x] Baseline reference metrics computed (12,395 pairs)
- [x] Scientific restrictions documented

### Infrastructure Ready
- [x] CopernicusLoader with 200m interpolation
- [x] open_environmental_sources() for multi-file loading
- [x] run_real_physics_benchmark() with date filtering
- [x] run_stage6c_benchmark.py with dynamic metrics
- [x] IcebergPhysicsEvaluator (tested, ready)
- [x] ConstantVelocityBaselineEvaluator (ready)

---

## ⏸️ Blocked: Real Data Required

### GLORYS12V1 (Ocean Currents & Temperature)
- [ ] Download 108 monthly files (2000–2008)
  - Dataset: `cmems_mod_glo_phy_my_0.083deg_P1D-m`
  - Variables: uo, vo, thetao (3 variables × 3,287 days)
  - Grid: 1/12° (~9.2 km)
  - Depths: All 50 native levels (critical: include 222.475m)
  - Size: ~5 GB per month = ~540 GB total
  - Auth: Copernicus Marine (free account)
- [ ] Verify all 50 depth levels present
- [ ] Verify 222.475m level exists (for 200m interpolation)
- [ ] Place in: `data/raw/glorys_YYYY_MM_*.nc`

### ERA5 Daily-Mean (Wind, Temperature, Pressure)
- [ ] Download 108 monthly files (2000–2008)
  - Product: ERA5 Complete Reanalysis, daily means
  - Variables: u10, v10, t2m, msl (4 variables × 3,287 days)
  - Grid: 0.25° (~28 km)
  - Size: ~200 MB per month = ~22 GB total
  - Auth: CDS (free account)
- [ ] Verify daily-mean (not hourly)
- [ ] Verify 4 variables present
- [ ] Place in: `data/raw/era5_YYYY_MM_*.nc`

---

## 🚀 To Run the Benchmark

### Prerequisites
```bash
# Register accounts (free)
# 1. https://marine.copernicus.eu/access/register
# 2. https://cds.climate.copernicus.eu/

# Install tools
pip install copernicusmarine cdsapi

# Authenticate
copernicusmarine login
# Edit ~/.cdsapirc with CDS credentials
```

### Download Data (see [docs/stage6c_data_acquisition.md](docs/stage6c_data_acquisition.md) for full commands)
```bash
# GLORYS example (Jan 2000, A23A region)
copernicusmarine subset \
  --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m \
  --variable uo --variable vo --variable thetao \
  --minimum-longitude -78.07 --maximum-longitude -74.49 \
  --minimum-latitude -44.14 --maximum-latitude -39.94 \
  --minimum-depth 0 --maximum-depth 6000 \
  --start-datetime 2000-01-01 --end-datetime 2000-01-31 \
  --output-directory data/raw/ --output-filename glorys_2000_01_a23a.nc

# ERA5: Use Python script (see acquisition doc for full code)
python -c "
import cdsapi
client = cdsapi.Client()
client.retrieve('reanalysis-era5', {...}, 'era5_2000_01_daily.nc')
"

# Repeat for all 108 months...
```

### Verify Data
```bash
# Check file structure
python -c "import xarray as xr; ds = xr.open_dataset('data/raw/glorys_2000_01_a23a.nc'); print(ds)"
# Expected: 3 variables, 50 depth levels, 31 time steps

python -c "import xarray as xr; ds = xr.open_dataset('data/raw/era5_2000_01_daily.nc'); print(ds)"
# Expected: 4 variables, 31 time steps

# Verify 222.475m level exists
python -c "import xarray as xr; ds = xr.open_dataset('data/raw/glorys_2000_01_a23a.nc'); print(222.475204 in ds.depth.values)"
# Expected: True
```

### Run Benchmark
```bash
python scripts/run_stage6c_benchmark.py
```

**Expected Output:**
- Evaluates all 12,395 pairs (6,202 T+3 + 6,193 T+4)
- Generates: `docs/stage6c_physics_benchmark.md`
- Metrics: N, mean, median, RMSE, P90, max per iceberg & horizon
- Comparison: Physics vs baseline (constant-velocity)
- Runtime: ~2–4 hours on modern hardware

---

## 📊 Expected Results

### Baseline Reference (Stage 5B Constant-Velocity)
| Iceberg | Horizon | N | Mean (km) | RMSE (km) |
|---------|---------|---|----------|-----------|
| A23A | T+3 | 3,171 | 1.10 | 5.76 |
| A23A | T+4 | 3,167 | 1.43 | 7.38 |
| B15A | T+3 | 3,031 | 11.90 | 30.07 |
| B15A | T+4 | 3,026 | 15.69 | 35.44 |

### Physics Evaluation (To Be Computed)
Physics model should achieve **≤ baseline** to be considered successful.
- A23A: Expect similar or better (predictable Weddell Sea trajectories)
- B15A: Expect similar or worse (chaotic antimeridian region)

---

## 🔍 Sanity Checks (After Data Download)

```bash
# 1. Verify GLORYS temporal completeness
python -c "
import xarray as xr
import pandas as pd
files = sorted(__import__('pathlib').Path('data/raw').glob('glorys_*.nc'))
times = []
for f in files:
    ds = xr.open_dataset(f)
    times.extend(ds.time.values)
times = pd.DatetimeIndex(times)
print(f'GLORYS: {times.min()} to {times.max()} ({len(times)} days)')
assert times.min() == pd.Timestamp('2000-01-01'), 'Start date mismatch'
assert times.max() >= pd.Timestamp('2008-12-31'), 'End date missing'
"

# 2. Verify ERA5 temporal completeness
python -c "
import xarray as xr
import pandas as pd
files = sorted(__import__('pathlib').Path('data/raw').glob('era5_*.nc'))
times = []
for f in files:
    ds = xr.open_dataset(f)
    times.extend(ds.time.values)
times = pd.DatetimeIndex(times)
print(f'ERA5: {times.min()} to {times.max()} ({len(times)} days)')
assert times.min() == pd.Timestamp('2000-01-01'), 'Start date mismatch'
assert times.max() >= pd.Timestamp('2008-12-31'), 'End date missing'
"

# 3. Verify 222.475m level exists
python -c "
import xarray as xr
import numpy as np
files = __import__('pathlib').Path('data/raw').glob('glorys_*.nc')
for f in list(files)[:1]:
    ds = xr.open_dataset(f)
    assert 222.475204 in ds.depth.values, f'Missing 222.475m in {f}'
    print(f'✓ {f}: Includes 222.475m level')
"
```

---

## 🎯 Success Criteria

✅ Benchmark succeeds if:
1. All 12,395 pairs evaluate without errors
2. Physics results report N, mean, median, RMSE, P90, max
3. Physics vs baseline comparison includes improvement/worsening metrics
4. No data gaps or temporal jumps in forcing
5. No future data leakage into predictions
6. No silent substitutions (all MissingDataError cases documented)
7. Runtime < 24 hours

❌ Benchmark fails if:
- Missing GLORYS 222.475m level (raises MissingDataError)
- ERA5 gaps within 2000–2008 window
- Forcing data contains NaN/infinity
- Physics simulation encounters numerical instability
- Temporal indices show future leak
- Any pair raises unhandled exception

---

## 📞 Support

### Common Issues & Fixes

**Issue**: `MissingDataError: No GLORYS depth deeper than 200m`
- **Cause**: GLORYS file truncated at 186m (missing 222.475m)
- **Fix**: Redownload with `--maximum-depth 6000` (not 200)

**Issue**: ERA5 temporal gap or missing month
- **Cause**: Download incomplete or file corrupted
- **Fix**: Redownload missing month via cdsapi

**Issue**: B15A spatial mismatch across antimeridian
- **Cause**: Longitude convention inconsistency (0-360 vs ±180)
- **Fix**: Ensure GLORYS uses consistent 0-360 or ±180 throughout

**Issue**: Physics takes > 6 hours
- **Cause**: Likely using hourly timestep instead of 600s
- **Fix**: Verify `dt_seconds=600.0` in IcebergPhysicsEvaluator

---

## Files to Review

1. **Data Spec**: [docs/stage6c_data_acquisition.md](docs/stage6c_data_acquisition.md)
2. **Technical Report**: [docs/stage6c_execution_report.md](docs/stage6c_execution_report.md)
3. **Smoke Tests**: [tests/test_stage6c_smoke.py](tests/test_stage6c_smoke.py)
4. **GLORYS Loader**: [src/data/copernicus.py](src/data/copernicus.py) (lines 170–224)
5. **Benchmark Runner**: [scripts/run_stage6c_benchmark.py](scripts/run_stage6c_benchmark.py)
6. **Baseline Metrics**: Computed in [docs/stage6c_execution_report.md](docs/stage6c_execution_report.md)

---

## Status: Ready to Execute

🟢 **Pipeline validated end-to-end (6/6 smoke tests pass)**  
🟢 **All infrastructure in place (91/91 tests pass)**  
🟢 **Data requirements documented (complete spec provided)**  
🔴 **Awaiting real data acquisition (user action required)**  

**Next action**: Download GLORYS + ERA5 monthly chunks following [docs/stage6c_data_acquisition.md](docs/stage6c_data_acquisition.md), then run `python scripts/run_stage6c_benchmark.py`.
