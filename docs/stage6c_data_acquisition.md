# Stage 6C Real-Data Benchmark: Acquisition Requirements

## Overview
This document specifies the **exact datasets, formats, and regions** required to run the first scientifically defensible real-data physics benchmark for Antarctic icebergs A23A and B15A (2000–2008).

## Benchmark Specifications

| Aspect | Specification |
|--------|---------------|
| **Icebergs** | A23A, B15A (raw/direct observations only) |
| **Prediction origins** | 2000-01-01 through 2008-12-31 |
| **Horizons** | T+3 and T+4 days |
| **Physics** | Stage 3 iceberg momentum-conservation RK4 integrator (600s timestep) |
| **Baseline** | Stage 5B constant-velocity model |
| **Expected pairs** | 6,202 T+3 + 6,193 T+4 = **12,395 total** |

---

## PART 1: ERA5 Daily-Mean Forcing

### Dataset Identifier
- **Product**: ERA5 Complete Reanalysis
- **Temporal Frequency**: **Daily-mean** (NOT hourly; not instantaneous fields)
- **Native Spatial Grid**: 0.25° × 0.25° latitude–longitude
- **Temporal Coverage**: 2000-01-01 through 2008-12-31 inclusive (9 years)

### Required Variables
| Variable | Short Name | Standard Name | Units |
|----------|-----------|---------------|-------|
| 10m U wind component | u10 | eastward_wind | m/s |
| 10m V wind component | v10 | northward_wind | m/s |
| 2m temperature | t2m | air_temperature | K |
| Mean sea-level pressure | msl | air_pressure_at_sea_level | Pa |

### Spatial Coverage

#### A23A Region (Weddell Sea)
- **Latitude range**: [-77.57, -74.99] (stable, no antimeridian)
- **Longitude range**: [-43.64, -40.44]
- **Buffer**: Add 1.0° margin on all sides for interpolation
- **Request bounds**: [-78.57, -73.99] × [-44.64, -39.44]

#### B15A Region (Antimeridian-Crossing)
- **Latitude range**: [-79.03, -51.63]
- **Longitude range (0-360 convention)**: [148.09, 360] ∪ [0, 210.09]
- **Longitude range (±180 convention)**: [-211.91, -180] ∪ [-180, -149.91]
- **Buffer**: Add 1.0° margin on all sides
- **Request bounds (±180 convention)**: [-213.0, -148.9] × [-80.0, -50.6]

### Data Organization
To avoid single-file bloat and memory constraints:
1. **Chunking Strategy**: Monthly or 3-month seasonal chunks
   - 9 years × 12 months = 108 monthly files (ideal)
   - OR 9 years × 4 seasons = 36 seasonal files (acceptable)
   - Example filename: `era5_2000_01_daily.nc` or `era5_2000_Q1_daily.nc`

2. **File format**: NetCDF4 with standard CF conventions
   - Coordinates: time (days), latitude, longitude
   - Dimensions: [time × latitude × longitude]
   - **No data gaps** within specified temporal window

3. **Causality**: All timestamps must be from the **past or present** relative to prediction origin. No future forcing.

### Delivery Checklist
- [ ] All 108 monthly files present (or 36 seasonal)
- [ ] Files cover 2000-01-01 through 2008-12-31 without gaps
- [ ] All 4 required variables present in every file
- [ ] Spatial coverage includes A23A [-78.57, -73.99] × [-44.64, -39.44]
- [ ] Spatial coverage includes B15A [-213.0, -148.9] × [-80.0, -50.6]
- [ ] No longitude folding/wrapping issues
- [ ] Temporal coordinates are explicitly labeled (UTC)
- [ ] Daily-mean values (not instantaneous)

---

## PART 2: GLORYS12V1 Daily Ocean Reanalysis

### Dataset Identifier
- **Product**: GLORYS12V1 Global Ocean Physics Reanalysis
- **CMEMS Dataset ID**: `cmems_mod_glo_phy_my_0.083deg_P1D-m`
- **Temporal Frequency**: Daily-mean (1 value per calendar day)
- **Native Spatial Grid**: 1/12° (~9.2 km at equator)
- **Temporal Coverage**: 2000-01-01 through 2008-12-31 inclusive
- **Depth Levels**: All 50 native levels (as-is, no interpolation at source)

### Required Variables
| Variable | Short Name | Standard Name | Units |
|----------|-----------|---------------|-------|
| Zonal ocean velocity | uo | eastward_sea_water_velocity | m/s |
| Meridional ocean velocity | vo | northward_sea_water_velocity | m/s |
| Potential temperature | thetao | sea_water_potential_temperature | K |

### Native Depth Levels (50 total)
All levels from 0.494 m to deepest. **Critical for 200 m interpolation:**
- Levels ≤ 200 m (first 26 levels):
  ```
  0.494025, 1.541375, 2.645669, 3.819495, 5.078224, 6.440614,
  7.929560, 9.572998, 11.405000, 13.467140, 15.810070, 18.495560,
  21.598820, 25.211410, 29.444731, 34.434151, 40.344051, 47.373692,
  55.764290, 65.807266, 77.853851, 92.326073, 109.729302, 130.666000,
  155.850693, 186.125595
  ```
- Levels > 200 m (remaining 24 levels, must include for interpolation):
  - **222.475204 m** ← **CRITICAL**: Required for 200 m interpolation
  - [22 additional deeper levels through ~5700 m]

### Spatial Coverage

#### A23A Region
- **Latitude**: [-77.57, -74.99]
- **Longitude**: [-43.64, -40.44]
- **Buffer**: Add 0.5° margin
- **Request bounds**: [-78.07, -74.49] × [-44.14, -39.94]

#### B15A Region (Antimeridian-Aware)
- **Latitude**: [-79.03, -51.63]
- **Longitude (0-360 convention)**: [148.09, 360] ∪ [0, 210.09]
- **Buffer**: Add 0.5° margin
- **Request strategy**: Download as two separate regions:
  1. Eastern hemisphere: [147.59, 360] × [-79.53, -51.13]
  2. Western hemisphere: [0, 210.59] × [-79.53, -51.13]
  - OR use CMEMS 0-360° convention consistently

### 200 m Draft Integration
The loader interpolates 200 m when:
1. Request `draft_meters=200.0`
2. Subset contains level 186.125595 m (bracketing from above)
3. Subset contains level 222.475204 m (bracketing from below)
4. Linear interpolation performed at 200 m boundary
5. Trapezoidal integration over [0.494, 200] m

**If 222.475204 m is missing**, benchmark **cannot run** (MissingDataError raised, no silent fallback).

### Data Organization
To manage file sizes (~3–5 GB per month per variable):
1. **Chunking Strategy**: Monthly files (similar to ERA5)
   - 108 monthly files (9 years × 12 months)
   - Example: `glorys_2000_01_daily.nc` or `glorys_2000_01_uo_vo_thetao.nc`

2. **Multi-file support**: Benchmark runner uses `xr.open_mfdataset(..., combine="by_coords")` to lazily load and concatenate monthly chunks

3. **File format**: NetCDF4 with CF conventions
   - Coordinates: time (days), depth (50 levels), latitude, longitude
   - Dimensions: [time × depth × latitude × longitude]
   - **Depth levels must be native GLORYS** (no interpolation before download)
   - **No data gaps** within 2000–2008 window

### Delivery Checklist
- [ ] All 108 monthly files present for 2000-01-01 through 2008-12-31
- [ ] All 50 native depth levels present in every file
- [ ] All 3 required variables (uo, vo, thetao) present
- [ ] **Level 222.475204 m present** (not truncated at 200 m)
- [ ] Spatial coverage: A23A [-78.07, -74.49] × [-44.14, -39.94]
- [ ] Spatial coverage: B15A [-79.53, -51.13] × [147.59, 360] ∪ [0, 210.59]
- [ ] No longitude wrapping issues (use 0-360 or ±180 consistently)
- [ ] Temporal coordinates labeled UTC
- [ ] Daily-mean (not instantaneous)

---

## PART 3: Iceberg Trajectories (BYU Consolidated Database)

### Dataset
- **Source**: Consolidated_Database_v8.0.zip (already available locally)
- **Subset**: Raw/direct observations only
- **Temporal Window**: 2000-01-01 through 2008-12-31

### Evaluation Pair Counts (Verified via Previous Session)
| Iceberg | Horizon | Count |
|---------|---------|-------|
| A23A | T+3 | 3,171 |
| A23A | T+4 | 3,167 |
| B15A | T+3 | 3,031 |
| B15A | T+4 | 3,026 |
| **TOTAL** | | **12,395** |

**Notes:**
- All pairs use only observations at or before prediction time (causal)
- Target positions are evaluation-only (not used during prediction)
- Baseline (constant-velocity) reference metrics already computed:
  - A23A T+3: mean 1.10 km, RMSE 5.76 km
  - A23A T+4: mean 1.43 km, RMSE 7.38 km
  - B15A T+3: mean 11.90 km, RMSE 30.07 km
  - B15A T+4: mean 15.69 km, RMSE 35.44 km

---

## Acquisition Instructions

### Method 1: Copernicus Marine (Recommended for GLORYS)

#### Authentication
```bash
# Download and install
pip install copernicusmarine

# Set credentials (interactive or environment variable)
copernicusmarine login  # Interactive prompt
# OR
export COPERNICUSMARINE_USERNAME="your_username"
export COPERNICUSMARINE_PASSWORD="your_password"
```

#### GLORYS Monthly Download (Example: Jan 2000)
```bash
copernicusmarine subset \
  --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m \
  --variable uo \
  --variable vo \
  --variable thetao \
  --minimum-longitude -78.07 \
  --maximum-longitude -74.49 \
  --minimum-latitude -44.14 \
  --maximum-latitude -39.94 \
  --minimum-depth 0 \
  --maximum-depth 6000 \
  --start-datetime 2000-01-01 \
  --end-datetime 2000-01-31 \
  --output-directory data/raw/ \
  --output-filename glorys_2000_01_a23a.nc
```

**For B15A (antimeridian handling)**: Request twice
```bash
# Eastern: lon [147.59, 180]
copernicusmarine subset --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m ... \
  --minimum-longitude 147.59 --maximum-longitude 180.0 ... \
  --output-filename glorys_2000_01_b15a_east.nc

# Western: lon [0, 210.59]  
copernicusmarine subset --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m ... \
  --minimum-longitude 0 --maximum-longitude 210.59 ... \
  --output-filename glorys_2000_01_b15a_west.nc
```

### Method 2: ERA5 via CDS or cdsapi (Recommended)

#### Authentication
Register at: https://cds.climate.copernicus.eu/

#### Sample ERA5 Download (Python, Example: Jan 2000)
```python
import cdsapi

client = cdsapi.Client()

client.retrieve(
    'reanalysis-era5',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind',
                     '2m_temperature', 'mean_sea_level_pressure'],
        'date': '2000-01-01/2000-01-31',
        'time': '00:00',  # Daily mean (00:00 request gives daily avg)
        'area': [-74.49, -44.14, -78.07, -39.94],  # [N, W, S, E]
    },
    'era5_2000_01_daily.nc')
```

---

## Data Validation Checklist

### Before Benchmark Execution
- [ ] **Temporal continuity**: No gaps in daily timestamps for 2000–2008
- [ ] **Spatial bounds verified**: All required regions present
- [ ] **Variable completeness**: All 4 ERA5 + 3 GLORYS variables in place
- [ ] **Depth consistency**: GLORYS includes level 222.475 m
- [ ] **Coordinate conventions**: Consistent (±180 or 0-360 for longitude)
- [ ] **Missing value handling**: No NaN silently replaced with zero
- [ ] **Causality enforced**: No future timestamps in forcing data

### After Successful Smoke Test (Synthetic Data)
Run: `pytest tests/test_stage6c_smoke.py -v`
- [x] ERA5 loader integration
- [x] GLORYS 200 m interpolation
- [x] Environment provider integration
- [x] Physics evaluator (full 3-day simulation)
- [x] Baseline evaluator
- [x] Multi-file loading

---

## Expected Benchmark Runtime

**Estimate for 12,395 pairs:**
- Single-threaded physics integration: ~1–3 hours (600s × 108,000 steps ≈ 18M seconds of simulation)
- Baseline evaluation: ~1 minute
- I/O (reading 3 years of monthly files): ~5–10 minutes
- **Total wall-clock**: ~2–4 hours on modern hardware

---

## Next Steps (User Action Required)

1. **Obtain Copernicus Marine account** (free research account available)
2. **Obtain CDS account** (free, linked to Copernicus)
3. **Download monthly GLORYS chunks** (108 files, ~5 GB each ≈ 540 GB total)
4. **Download monthly ERA5 chunks** (108 files, ~200 MB each ≈ 22 GB total)
5. **Place in** `data/raw/` directory
6. **Run benchmark**: `python scripts/run_stage6c_benchmark.py`

---

## Scientific Restrictions (Enforced by Code)

- ❌ No parameter tuning on evaluation cases
- ❌ No invention of missing environmental values
- ❌ No silent fallback to future data
- ❌ No replacement of missing data with zero
- ❌ No use of target positions during prediction
- ❌ **No silent equating of 186.125595 m with 200 m** (raises MissingDataError instead)
- ✅ Prototype iceberg properties clearly labeled as reference/experimental

---

## References

- [GLORYS12V1 Documentation](https://resources.marine.copernicus.eu/documents/PUM/CMEMS_PUM_GLOB_PHY_L4_001_025.pdf)
- [ERA5 Complete Reanalysis](https://cds.climate.copernicus.eu/cdsapp)
- [CMEMS Toolbox](https://github.com/mercatorocean/copernicusmarine)
