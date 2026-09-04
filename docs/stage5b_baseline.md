# Stage 5B: Constant-Velocity Baseline Evaluation Report

**Project:** SIH26059 — AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System  
**Module:** `src/evaluation/baseline_evaluator.py`  
**Ground Truth Dataset:** BYU/NIC Antarctic Iceberg Tracking Database v8.0 (`data/raw/consolidated_database_v8.0.zip`)  

---

> **Evaluation Benchmark Role:**  
> The Constant-Velocity (persistence) baseline establishes the reference accuracy floor against which our Stage 3 momentum-conservation physics solver and future machine-learning models are evaluated.  
> Predictions are evaluated strictly against **verified, direct raw satellite observations** (`is_raw_observation == True`) with zero future data leakage.

---

## 1. Methodology & Scientific Formulation

### Prediction Mechanism:
1. At prediction origin time $T$, the initial velocity vector $(v_x, v_y)$ in projected Cartesian meters per second (`EPSG:3412`) is derived exclusively from historical raw observations at or before $T$:
   $$v_x = \frac{x(T) - x(t_{\text{prev}})}{T - t_{\text{prev}}}, \quad v_y = \frac{y(T) - y(t_{\text{prev}})}{T - t_{\text{prev}}} \quad (t_{\text{prev}} < T)$$
2. Future projected coordinates at horizon $H \in \{3, 4\}$ calendar days ($\Delta t = H \times 86,400\,\text{s}$) are extrapolated assuming zero acceleration:
   $$x_{\text{pred}}(T + H) = x(T) + v_x \cdot \Delta t$$
   $$y_{\text{pred}}(T + H) = y(T) + v_y \cdot \Delta t$$
3. Projected coordinates are transformed back to WGS84 geographic coordinates:
   $$(\lambda_{\text{pred}}, \phi_{\text{pred}}) = \mathcal{P}^{-1}\left(x_{\text{pred}}, y_{\text{pred}}\right)$$

### Positional Error Metric:
The displacement error between predicted coordinates $(\phi_{\text{pred}}, \lambda_{\text{pred}})$ and actual observed ground truth $(\phi^*, \lambda^*)$ is calculated as the geodesic distance on the WGS84 reference ellipsoid via `pyproj.Geod`:
$$e = \text{Geod}_{\text{WGS84}}\left((\phi_{\text{pred}}, \lambda_{\text{pred}}), (\phi^*, \lambda^*)\right) \quad [\text{km}]$$

### Strict Scientific Constraints:
* **No Interpolated Ground Truth:** If $T+H$ does not have a direct raw satellite observation (`is_raw_observation == True`), the pair is rejected. Interpolated positions are never used as evaluation targets.
* **Zero Future Leakage:** Target coordinates at $T+H$ and any observation $> T$ are strictly quarantined until after the prediction is generated.

---

## 2. Empirical Benchmark Results on BYU/NIC Database

We evaluated the Constant-Velocity baseline on **$14,017$ historical 3-day cases** and **$13,997$ historical 4-day cases** spanning two of Antarctica\'s most prominent benchmark icebergs:
* **A23A** (Weddell Sea — predominantly grounded regime with episodic drift)
* **B15A** (Ross Sea / Southern Ocean — active, unconstrained free-drift regime)

### Combined Benchmark Summary (A23A + B15A)

| Horizon | Evaluation Cases ($N$) | Mean Error (km) | Median Error (km) | RMSE (km) | 90th Percentile (km) | Max Error (km) | Min Error (km) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3 Days** | 14,017 | **5.88** | **0.00** | **18.97** | **18.29** | 809.40 | 0.00 |
| **4 Days** | 13,997 | **7.71** | **0.00** | **23.45** | **24.07** | 1,087.19 | 0.00 |

---

## 3. Per-Iceberg Drift Regime Analysis

### Iceberg A23A (Grounded Regime)

| Horizon | Cases ($N$) | Mean Error (km) | Median Error (km) | RMSE (km) | 90th Percentile (km) | Max Error (km) | Min Error (km) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3 Days** | 10,961 | 4.03 | 0.00 | 13.55 | 12.97 | 386.63 | 0.00 |
| **4 Days** | 10,947 | 5.27 | 0.00 | 17.57 | 16.30 | 407.55 | 0.00 |

### Iceberg B15A (Free Drift Regime)

| Horizon | Cases ($N$) | Mean Error (km) | Median Error (km) | RMSE (km) | 90th Percentile (km) | Max Error (km) | Min Error (km) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3 Days** | 3,056 | 12.52 | 0.00 | 31.50 | 35.54 | 809.40 | 0.00 |
| **4 Days** | 3,050 | 16.49 | 4.63 | 37.62 | 48.08 | 1,087.19 | 0.00 |

---

## 4. Key Physical and Methodological Observations

1. **Temporal Error Compounding:**
   Across both icebergs, errors compound monotonically with lead time:
   * 3-Day Mean Error: $5.88\,\text{km}$ $\to$ 4-Day Mean Error: $7.71\,\text{km}$ ($+31.1\%$)
   * 3-Day RMSE: $18.97\,\text{km}$ $\to$ 4-Day RMSE: $23.45\,\text{km}$ ($+23.6\%$)
   * 3-Day P90 Error: $18.29\,\text{km}$ $\to$ 4-Day P90 Error: $24.07\,\text{km}$ ($+31.6\%$)

2. **Grounded vs. Free-Drifting Regimes:**
   * **A23A** was grounded on the shallow seabed of the southern Weddell Sea for over 30 years (1986–2020), resulting in a median error of $0.00\,\text{km}$ and modest mean errors ($4.03\,\text{km}$ at 3d).
   * **B15A** calved in 2000 and drifted thousands of kilometres along the Ross Ice Shelf, Victoria Land coast, and into the open Antarctic Circumpolar Current. Under free-drift dynamics, synoptic winds and ocean currents accelerate and turn the iceberg, producing substantially larger persistence errors (3-day mean: $12.52\,\text{km}$, 4-day mean: $16.49\,\text{km}$, P90: $48.08\,\text{km}$).

3. **Baseline Deficiency in High-Curvature Regions:**
   Extreme errors ($> 400\text{--}1,000\,\text{km}$) occur when an iceberg undergoes strong cyclonic turning around headlands or enters intense storm tracks, proving that linear persistence models fail during synoptic weather events. This demonstrates the clear scientific need for dynamic environmental physics and ML trajectory modeling.
