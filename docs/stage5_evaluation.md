# Stage 5A: BYU/NIC v8.0 Ingestion and Historical Evaluation Dataset

**Project:** SIH26059 — AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System  
**Modules:** `src/data/iceberg.py`, `src/evaluation/historical_pairs.py`  
**Dataset:** BYU/NIC Antarctic Iceberg Tracking Database v8.0 (`data/raw/consolidated_database_v8.0.zip`)  

---

> **Core Scientific Evaluation Principle:**  
> Historical trajectory forecasting models are evaluated strictly against **direct, raw satellite observations** (`is_raw_observation == True`) at exact multi-day calendar horizons ($T+3$ days and $T+4$ days).  
> Future targets are **never** synthetically generated, linearly interpolated, or forward-filled.

---

## 1. BYU/NIC v8.0 Database Structure

The BYU MERS Consolidated Antarctic Iceberg Database v8.0 consolidates observations from satellite scatterometers and the US National Ice Center (NIC).
* **Archive Format:** Single ZIP archive (`data/raw/consolidated_database_v8.0.zip`, 3.88 MB compressed, 23.53 MB uncompressed).
* **Directory Structure:** Flat archive containing 647 individual iceberg tracking files under `updated7_consol/<iceberg_id>.csv`.
* **Identification:** The iceberg identifier is derived directly from the filename (e.g. `a23a.csv` $\to$ `A23A`, `b15a.csv` $\to$ `B15A`) and normalized to uppercase. Editor lockfiles (such as `#d15b.csv#`) and text documentation (`README_consolidated.TXT`) are automatically ignored.

---

## 2. Sensor Fields & Data Encoding

Each CSV file records daily positions across the satellite sensors operational during that iceberg's lifetime:
* `ascat`: MetOp Advanced Scatterometer (2007–2026)
* `qscat`: QuikSCAT SeaWinds (1999–2009)
* `oscat`: Oceansat-2 OSCAT (2009–2014)
* `nscat`: ADEOS-1 NSCAT (1996–1997)
* `seawinds`: ADEOS-2 SeaWinds (2002–2003)
* `ers`: ERS-1/2 AMI Scatterometer (1992–2001)
* `sass`: Seasat SASS (1978)
* `nic`: US National Ice Center Analyst Charts (1978–2026)

### Triplets per Sensor:
* `<sensor>_1`: Latitude in degrees North (negative in Southern Hemisphere). `0.0` denotes missing/no-data.
* `<sensor>_2`: Longitude in degrees East ($-180^\circ$ to $+180^\circ$). `0.0` denotes missing/no-data.
* `<sensor>_3`: Binary quality/interpolation flag:
  * `1`: **Direct/raw sensor observation** (valid non-zero coordinates).
  * `0`: **Linearly interpolated position** (when coordinates are non-zero) **OR missing data** (when coordinates are `0.0, 0.0`). Missing coordinates are excluded and never marked `is_interpolated=True`.

### Temporal Representation:
* Format: Integer `YYYYJJJ` (4-digit calendar year + 3-digit Julian day of year, e.g. `2002045` $\to$ `2002-02-14`).
* Exceptional representations: One historical file (`e03.csv`) contains 5-digit `YYJJJ` dates (`92226` $\to$ `1992-08-13`), which the parser converts unambiguously.

### Dimensions:
* `size_1`: Major axis length in **nautical miles (nM)**.
* `size_2`: Minor axis length in **nautical miles (nM)**.
* **Conversion:** Multiplied by $1.852$ to convert to kilometres ($1\,\text{nM} = 1.852\,\text{km}$).
* **Missingness:** Value of $0$ indicates no dimension was reported on that date and is ingested as `NaN` (not $0.0\,\text{km}$).

---

## 3. Raw vs. Interpolated Semantics & Sensor Priority

When multiple sensors report on the same calendar day, the ingestion pipeline selects a single definitive position per date following a deterministic hierarchy:

### Sensor Priority:
$$\text{ascat} \succ \text{qscat} \succ \text{oscat} \succ \text{nscat} \succ \text{seawinds} \succ \text{ers} \succ \text{sass} \succ \text{nic}$$

1. **Pass 1 (Raw Observations):** The highest-priority sensor with a direct raw observation (`flag == 1` and valid non-zero coordinates) is selected.
   * `is_raw_observation = True`
   * `is_interpolated = False`
   * `position_source = <sensor>`
2. **Pass 2 (Interpolated Positions):** If no raw observation exists, the highest-priority sensor with an interpolated position (`flag == 0` and valid non-zero coordinates) is selected.
   * `is_raw_observation = False`
   * `is_interpolated = True`
   * `position_source = <sensor>`
3. **Missing Coordinates:** Days where all sensors record `0.0, 0.0` represent missing/unobserved positions. They are excluded from the normalized trajectory, and are **never** marked `is_interpolated = True`. Coordinates are never fabricated.

---

## 4. Exact-Horizon Historical Evaluation Methodology

To evaluate trajectory predictors (e.g. constant-velocity baseline and future physics/ML models), `src/evaluation/historical_pairs.py` constructs standardized evaluation cases:

### Initial State at Prediction Origin $T$:
1. $T$ must be a verified raw satellite observation (`is_raw_observation == True`).
2. There must exist a prior raw observation at $t_{\text{prev}} < T$ within a maximum allowable gap ($\Delta t \le 14$ days).
3. Initial velocity $(v_x, v_y)$ in projected Cartesian meters per second is estimated strictly using $(t_{\text{prev}}, T)$ via forward finite differencing in `EPSG:3412`.

### Future Target Horizons ($T+3$ days and $T+4$ days):
* Target at $T + 3$ calendar days must exist as a **direct raw observation** (`is_raw_observation == True`).
* Target at $T + 4$ calendar days must exist as a **direct raw observation** (`is_raw_observation == True`).
* If an interpolated position (`is_raw_observation == False`) or no observation exists at $T+H$, the case is discarded.

### Strict Prohibition of Future Data Leakage:
* Only observations $t \le T$ enter the predictor state.
* The target position at $T+H$ is kept strictly segregated as ground truth for metric evaluation after the forecast is generated.

---

## 5. Geodesic Distance Error Metric

Displacement errors between predicted positions $(\hat{\phi}, \hat{\lambda})$ and observed ground truth positions $(\phi^*, \lambda^*)$ are computed using the geodesic distance on the WGS84 reference ellipsoid via `pyproj.Geod`:

$$e_{\text{geo}} = \text{Geod}_{\text{WGS84}}\left((\phi^*, \lambda^*), (\hat{\phi}, \hat{\lambda})\right) \quad [\text{km}]$$

This avoids planar stereographic map distortion errors and degree-scaling inaccuracies in polar latitudes.

---

## 6. Verified Ground Truth Statistics on Key Antarctic Icebergs

Evaluating the 6 major Antarctic benchmark icebergs yields:

| Iceberg ID | Trajectory Span | Daily Rows | Raw Observations | Valid $T \to T+3\text{d}$ Cases | Valid $T \to T+4\text{d}$ Cases | Cases with Both Horizons |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **A23A** | 1991–2026 | 11,864 | 11,356 (95.7%) | 10,961 | 10,947 | 10,668 |
| **B15A** | 2000–2009 | 3,205 | 3,125 (97.5%) | 3,056 | 3,050 | 3,011 |
| **C19A** | 2003–2009 | 2,153 | 2,062 (95.8%) | 2,008 | 2,003 | 1,971 |
| **A68A** | 2017–2021 | 1,324 | 795 (60.0%) | 549 | 550 | 453 |
| **B09B** | 1989–2026 | 12,452 | 11,551 (92.8%) | 11,135 | 11,120 | 10,915 |
| **A22A** | 1994–2007 | 5,000 | 4,939 (98.8%) | 4,882 | 4,880 | 4,844 |
| **Total (Sample)** | — | **35,998** | **33,828 (94.0%)** | **32,591** | **32,550** | **31,862** |

---

## 7. Known Limitations

1. **Scatterometer Centering vs. Visual Centering:** The center of backscatter in Ku/C-band scatterometer imagery may differ slightly ($\approx 1\text{–}3\,\text{km}$) from optical/SAR center-of-mass estimates.
2. **Dimension Reporting Gaps:** Dimensions are updated intermittently (typically weekly) via NIC analyst charts; intermediate days require forward-filling if physical dimensions are needed by downstream solvers.
3. **Summer Melt Gaps:** Scatterometer contrast degrades during peak austral summer (December–January) when surface meltwater absorbs microwave radar signals, occasionally producing brief gaps in raw observations.
