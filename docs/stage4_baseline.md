# Stage 4: Constant-Velocity / Persistence Iceberg Trajectory Baseline

**Project:** SIH26059 — AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System  
**Module:** `src/models/baselines.py`  
**Purpose:** Establish a transparent, reproducible persistence benchmark model for iceberg trajectory forecasting  

---

> This baseline assumes constant velocity and serves as a reference model for evaluating more complex physics and machine-learning approaches.

---

## 1. Mathematical Formulation

The constant-velocity trajectory model assumes that the instantaneous velocity vector estimated at the forecast initialization time $T$ persists indefinitely without acceleration:
$$\vec{a}(t) = \frac{d\vec{v}}{dt} = 0, \quad \forall t \ge T$$
$$\vec{v}(t) = \vec{v}(T) = \text{constant}$$

In projected Cartesian coordinates $(x, y)$:
$$x(T + \Delta t) = x(T) + v_x(T) \cdot \Delta t$$
$$y(T + \Delta t) = y(T) + v_y(T) \cdot \Delta t$$

where:
* $x(T), y(T)$ are the projected coordinates at time $T$ in metres ($\text{m}$).
* $v_x(T), v_y(T)$ are the horizontal and vertical velocity components in metres per second ($\text{m/s}$).
* $\Delta t$ is the elapsed forecast duration in seconds ($\text{s}$).

Geographic coordinates $(\lambda, \phi)$ (longitude and latitude) are obtained by applying the inverse projection:
$$(\lambda, \phi) = \mathcal{P}^{-1}(x, y)$$

---

## 2. Velocity Estimation from Historical Observations

Given two consecutive historical observations of an iceberg:
* Observation 1: $(x_1, y_1)$ at timestamp $t_1$
* Observation 2: $(x_2, y_2)$ at timestamp $t_2$, with $t_2 > t_1$

The elapsed observation interval is:
$$\Delta t_{\text{obs}} = t_2 - t_1 \quad [\text{seconds}]$$

The velocity components are estimated by central/forward finite differencing in projected metres:
$$v_x = \frac{x_2 - x_1}{\Delta t_{\text{obs}}}\quad [\text{m/s}]$$
$$v_y = \frac{y_2 - y_1}{\Delta t_{\text{obs}}}\quad [\text{m/s}]$$

### Geographic Observations
When observations are provided in geographic coordinates $(\lambda_1, \phi_1)$ and $(\lambda_2, \phi_2)$, they are first projected to $(x_1, y_1)$ and $(x_2, y_2)$ using the project's coordinate reference system (`EPSG:3412` or `EPSG:3031`).  
**Velocity is never computed by subtracting latitude and longitude degrees directly.**

### Error Guards
The estimator strictly validates:
* $\Delta t_{\text{obs}} > 0$: Zero and negative time intervals ($t_2 \le t_1$) raise `ValueError`.
* Finite values: Coordinates containing NaN or $\pm\infty$ raise `ValueError`.
* Zero velocity is **never** silently substituted for missing or invalid data.

---

## 3. Coordinate System & Projection Conventions

* **Projected CRS:** Uses the project-standard `CoordinateHandler` defaulting to `EPSG:3412` (NSIDC Sea Ice Polar Stereographic South, WGS 84), with full compatibility for `EPSG:3031` (Antarctic Polar Stereographic).
* **Axis Interpretation:**
  * $x$: Eastings in metres ($+X$ aligned with the central meridian $\lambda_0 = 0^\circ$).
  * $y$: Northings in metres.
  * $v_x$: Grid-X rate of change ($dx/dt$ in $\text{m/s}$).
  * $v_y$: Grid-Y rate of change ($dy/dt$ in $\text{m/s}$).
* **Transformation Rigor:** Bidirectional conversions use `pyproj.Transformer` with `always_xy=True` ensuring consistent coordinate ordering:
  * WGS84 (`EPSG:4326`): `(longitude, latitude)`
  * Projected (`EPSG:3412`): `(x_m, y_m)`

---

## 4. Physical Quantities and Units

| Quantity | Code Variable | SI Unit | Dimension | Description |
| :--- | :--- | :--- | :--- | :--- |
| Projected X | `x_m` | $\text{m}$ | $[L]$ | Eastings in projected plane |
| Projected Y | `y_m` | $\text{m}$ | $[L]$ | Northings in projected plane |
| Velocity X | `vx_mps` | $\text{m/s}$ | $[L T^{-1}]$ | Constant rate of change in X |
| Velocity Y | `vy_mps` | $\text{m/s}$ | $[L T^{-1}]$ | Constant rate of change in Y |
| Scalar Speed | `speed_mps` | $\text{m/s}$ | $[L T^{-1}]$ | $\sqrt{v_x^2 + v_y^2}$ |
| Forecast Horizon | `forecast_seconds`, `dt_seconds` | $\text{s}$ | $[T]$ | Lead time from initialization |
| Observation Interval | `dt_sec` | $\text{s}$ | $[T]$ | Time difference $t_2 - t_1$ |

---

## 5. Multi-Day Forecast Horizons

The baseline natively evaluates multi-day operational prediction benchmarks:
* **3-Day Forecast Horizon:**
  $$\Delta t_{\text{3d}} = 3 \times 24 \times 3600 = 259,200\,\text{s}$$
  $$x(T + 3\text{d}) = x(T) + 259,200 \cdot v_x$$
  $$y(T + 3\text{d}) = y(T) + 259,200 \cdot v_y$$

* **4-Day Forecast Horizon:**
  $$\Delta t_{\text{4d}} = 4 \times 24 \times 3600 = 345,600\,\text{s}$$
  $$x(T + 4\text{d}) = x(T) + 345,600 \cdot v_x$$
  $$y(T + 4\text{d}) = y(T) + 345,600 \cdot v_y$$

---

## 6. Environmental Independence & Historical Integrity

* **Zero Environmental Coupling:** The baseline does not import, query, or depend upon `EnvironmentProvider`, NSIDC sea ice, ERA5 winds, or Copernicus ocean currents.
* **Leakage Immunity:** Because it does not ingest external reanalysis fields, it is naturally immune to environmental data leakage.
* **Input Observation Integrity:** When deployed in historical validation pipelines, the two input observations $(t_1, t_2)$ used to estimate velocity must both satisfy $t_1 < t_2 \le T_{\text{cutoff}}$ to guarantee that future positional ground truth is never leaked into the initial velocity state.

---

## 7. Model Assumptions and Limitations

1. **Zero Acceleration Assumption:** Assumes oceanic and atmospheric forces remain in exact balance with drag, ignoring changes in synoptic wind fields, coastal currents, or bathymetric steering.
2. **Trajectory Linearity in Projected Space:** Extrapolates along a straight line in the projected Cartesian plane. Over thousands of kilometers, map distortion factors cause straight lines in projection space to deviate slightly from great-circle geodesics.
3. **No Grounding or Collision:** Assumes the iceberg encounters no obstacles, shallow bathymetry, or coastline barriers.
4. **No Rotational Dynamics:** Ignores changes in iceberg orientation and mass degradation.
