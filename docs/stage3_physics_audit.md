# Stage 3 Physics & Code Implementation Audit Report

**Project:** SIH26059 — AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System  
**Audit Scope:** Stage 3 Iceberg Drift Physics Solver and Numerical Integrator  
**Audit Mode:** READ-ONLY Architectural & Physical Inspection  
**Date of Audit:** 2026-09-03  
**Repository Path:** `C:\Users\steal\.gemini\antigravity\scratch\antarctic-nav-ai`  

---

## 1. Files Inspected

* `src/models/iceberg_physics.py` (Core physics solver, RK4 integrator, coordinate handler)
* `src/models/__init__.py` (Package exports)
* `tests/test_physics.py` (9 unit and numerical integration test suites)
* `src/data/environment.py` (Unified data abstraction, `EnvironmentProvider`, historical cutoff logic)
* `src/data/synthetic.py` (Synthetic environmental provider and vortex formulation)

---

## 2. Exact Physics Equations

The simulator formulates horizontal motion in a projected Cartesian plane using Newton's second law:
$$m \frac{d\vec{v}}{dt} = \vec{F}_{\text{total}} = \vec{F}_{\text{water}} + \vec{F}_{\text{air}} + \vec{F}_{\text{coriolis}} - \vec{F}_{\text{damping}}$$
$$\vec{a} = \frac{\vec{F}_{\text{total}}}{m}$$

### A. Water Drag Force ($\vec{F}_{\text{water}}$)
From `src/models/iceberg_physics.py` lines 295–302:
$$\vec{v}_{\text{rel, water}} = \begin{bmatrix} u_{\text{ocean, grid}} - v_x \\ v_{\text{ocean, grid}} - v_y \end{bmatrix}$$
$$s_{\text{rel, water}} = |\vec{v}_{\text{rel, water}}| = \sqrt{(u_{\text{ocean, grid}} - v_x)^2 + (v_{\text{ocean, grid}} - v_y)^2}$$
$$C_{\text{drag, water}} = 0.5 \cdot \rho_{\text{water}} \cdot C_{d,\text{water}} \cdot A_{\text{underwater}}$$
$$\vec{F}_{\text{water}} = C_{\text{drag, water}} \cdot s_{\text{rel, water}} \cdot \vec{v}_{\text{rel, water}}$$

* **Water density ($\rho_{\text{water}}$):** `props.water_density_kg_per_m3` = $1025.0\,\text{kg/m}^3$ (configurable).
* **Water drag coefficient ($C_{d,\text{water}}$):** `props.water_drag_coefficient` = $0.90$ (configurable prototype value).
* **Underwater cross-sectional area ($A_{\text{underwater}}$):**
  $$A_{\text{underwater}} = w_{\text{eff}} \cdot H_{\text{draft}} = \sqrt{L \cdot W} \cdot H_{\text{draft}}\quad [\text{m}^2]$$
  where $L = \text{length\_m}$, $W = \text{width\_m}$, $H_{\text{draft}} = \text{draft\_m}$.
* **Relative velocity definition:** Fluid minus iceberg: $\vec{v}_{\text{ocean}} - \vec{v}_{\text{iceberg}}$.
* **Force direction:** Acts in the direction of relative water flow. If the ocean current is faster than the iceberg, the force accelerates the iceberg along the current; if the iceberg is faster than the current, the force opposes iceberg velocity.
* **Mass normalization:** The force is in Newtons ($\text{N}$); it is summed into $\vec{F}_{\text{total}}$ and divided by $m$ (`props.mass_kg`) in line 334 to yield acceleration in $\text{m/s}^2$.

### B. Air / Wind Drag Force ($\vec{F}_{\text{air}}$)
From lines 304–312:
$$\vec{v}_{\text{rel, air}} = \begin{bmatrix} u_{\text{wind, grid}} - v_x \\ v_{\text{wind, grid}} - v_y \end{bmatrix}$$
$$s_{\text{rel, air}} = |\vec{v}_{\text{rel, air}}| = \sqrt{(u_{\text{wind, grid}} - v_x)^2 + (v_{\text{wind, grid}} - v_y)^2}$$
$$C_{\text{drag, air}} = 0.5 \cdot \rho_{\text{air}} \cdot C_{d,\text{air}} \cdot A_{\text{above\_water}}$$
$$\vec{F}_{\text{air}} = C_{\text{drag, air}} \cdot s_{\text{rel, air}} \cdot \vec{v}_{\text{rel, air}}$$

* **Air density ($\rho_{\text{air}}$):** `props.air_density_kg_per_m3` = $1.25\,\text{kg/m}^3$ (configurable).
* **Air drag coefficient ($C_{d,\text{air}}$):** `props.air_drag_coefficient` = $1.30$ (configurable prototype value).
* **Exposed cross-sectional sail area ($A_{\text{above\_water}}$):**
  $$A_{\text{above\_water}} = w_{\text{eff}} \cdot H_{\text{freeboard}} = \sqrt{L \cdot W} \cdot H_{\text{freeboard}}\quad [\text{m}^2]$$
  If $H_{\text{freeboard}}$ is not explicitly supplied, it is derived via isostatic balance:
  $$H_{\text{freeboard}} = H_{\text{draft}} \cdot \frac{\rho_{\text{water}} - \rho_{\text{ice}}}{\rho_{\text{ice}}} = 150 \cdot \frac{1025 - 900}{900} \approx 20.833\,\text{m}$$
* **Relative air velocity definition:** Wind minus iceberg: $\vec{v}_{\text{wind}} - \vec{v}_{\text{iceberg}}$.
* **Force direction:** Downwind relative to the iceberg.
* **Mass normalization:** In Newtons ($\text{N}$); divided by `props.mass_kg` when computing total acceleration.

### C. Coriolis Acceleration & Force ($\vec{F}_{\text{coriolis}}$)
From lines 314–324:
$$f = 2 \cdot \Omega \cdot \sin(\phi)$$
where:
* $\Omega$ = `EARTH_OMEGA_RAD_PER_S` = $7.292115 \times 10^{-5}\,\text{rad/s}$ (derived from Earth sidereal rotation period $86164.0905\,\text{s}$).
* $\phi$ = geodetic latitude in radians (`np.radians(lat_deg)`).
* Coordinate axis interpretation: $x$ and $y$ are projected Cartesian axes (Easting and Northing in meters).
* Exact equations:
  $$\vec{F}_{\text{coriolis}} = \begin{bmatrix} m \cdot f \cdot v_y \\ -m \cdot f \cdot v_x \end{bmatrix}\quad [\text{N}]$$
  $$\vec{a}_{\text{coriolis}} = \begin{bmatrix} f \cdot v_y \\ -f \cdot v_x \end{bmatrix}\quad [\text{m/s}^2]$$
* **Southern Hemisphere Behavior:** In the Southern Hemisphere, $\phi < 0 \implies \sin(\phi) < 0 \implies f < 0$.
  For an iceberg moving Eastward along the positive X-axis ($v_x > 0, v_y = 0$):
  $$a_{\text{coriolis}, x} = f \cdot 0 = 0$$
  $$a_{\text{coriolis}, y} = -f \cdot v_x = -(-|f|) v_x = +|f| v_x > 0$$
  Because $a_y > 0$, the deflection is toward $+Y$ (grid North / Left of the velocity vector). This is verified by `TestIcebergPhysicsDynamics::test_southern_hemisphere_coriolis_deflection_direction`.

### D. Linear Damping Force ($\vec{F}_{\text{damping}}$)
From lines 326–328:
$$\vec{F}_{\text{damping}} = c \cdot \begin{bmatrix} v_x \\ v_y \end{bmatrix}\quad [\text{N}]$$
* **Damping coefficient ($c$):** `props.damping_coefficient` (default $0.0\,\text{N}\cdot\text{s/m}$).
* **Character:** Calculated as a force in Newtons.
* **Direction / Sign:** Subtracted in the momentum equation ($\vec{F}_{\text{total}} = \dots - \vec{F}_{\text{damping}}$), explicitly opposing the instantaneous velocity vector $\vec{v}$.
* **Units:** $c$ is in $\text{N}\cdot\text{s/m}$ (or $\text{kg/s}$).

### E. Total Dynamics Equation
From lines 330–342:
$$\frac{dx}{dt} = v_x$$
$$\frac{dy}{dt} = v_y$$
$$\frac{dv_x}{dt} = \frac{F_{\text{water}, x} + F_{\text{air}, x} + F_{\text{coriolis}, x} - F_{\text{damping}, x}}{m}$$
$$\frac{dv_y}{dt} = \frac{F_{\text{water}, y} + F_{\text{air}, y} + F_{\text{coriolis}, y} - F_{\text{damping}, y}}{m}$$

---

## 3. Relative Velocity Audit

### Water Relative Velocity
* **Exact code expression:**
  ```python
  # Line 296-297
  v_rel_water_x = u_ocean_grid - vx_mps
  v_rel_water_y = v_ocean_grid - vy_mps
  ```
* **Convention:** Fluid velocity minus body velocity ($\vec{v}_{\text{ocean}} - \vec{v}_{\text{iceberg}}$).
* **Force application:**
  ```python
  # Line 301-302
  F_water_x = drag_water_coeff * speed_rel_water * v_rel_water_x
  F_water_y = drag_water_coeff * speed_rel_water * v_rel_water_y
  ```
  With a positive scalar coefficient, $\vec{F}_{\text{water}}$ acts in the direction of $(\vec{v}_{\text{ocean}} - \vec{v}_{\text{iceberg}})$.
  * When stationary in a current ($\vec{v}_{\text{iceberg}} = 0, \vec{v}_{\text{ocean}} > 0$): $\vec{v}_{\text{rel}} > 0 \implies \vec{F} > 0$ (accelerates with current).
  * When moving through still water ($\vec{v}_{\text{iceberg}} > 0, \vec{v}_{\text{ocean}} = 0$): $\vec{v}_{\text{rel}} < 0 \implies \vec{F} < 0$ (opposes motion).
  * The force correctly opposes relative motion between fluid and body.

### Air Relative Velocity
* **Exact code expression:**
  ```python
  # Line 306-307
  v_rel_air_x = u_wind_grid - vx_mps
  v_rel_air_y = v_wind_grid - vy_mps
  ```
* **Convention:** Fluid velocity minus body velocity ($\vec{v}_{\text{wind}} - \vec{v}_{\text{iceberg}}$).
* **Force application:**
  ```python
  # Line 311-312
  F_air_x = drag_air_coeff * speed_rel_air * v_rel_air_x
  F_air_y = drag_air_coeff * speed_rel_air * v_rel_air_y
  ```
  The force acts in the direction of $(\vec{v}_{\text{wind}} - \vec{v}_{\text{iceberg}})$, pushing the iceberg downwind.

---

## 4. Mass and Geometry Audit

| Item | Status in Code | Details |
| :--- | :--- | :--- |
| **1. Is mass directly supplied?** | **YES** | `mass_kg` is an explicit input parameter to `IcebergProperties`. |
| **2. Is mass calculated from geometry?** | **NO** | The code does not calculate mass from volume and density. |
| **3. Is volume calculated?** | **NO** | Submerged or total volume is not calculated or stored. |
| **4. Is an iceberg density assumption made?** | **YES** | `ice_density_kg_per_m3 = 900.0` is present, used solely for freeboard ratio estimation. |
| **5. Is draft used to estimate submerged volume?** | **NO** | Draft is used only for underwater cross-sectional area and freeboard estimation. |
| **6. Is length × width × draft used anywhere?** | **NO** | The code never computes $L \times W \times H_{\text{draft}}$. |
| **7. Is underwater drag area physically consistent?** | **PARTIALLY** | Approximated as $A = \sqrt{L \cdot W} \cdot H_{\text{draft}}$. Assumes isotropic geometric-mean frontal width. |
| **8. Are dimensions all in metres?** | **YES** | `length_m`, `width_m`, `draft_m`, `freeboard_m` are all in metres. |
| **9. Are any values arbitrary prototype defaults?** | **YES** | $C_{d,\text{water}} = 0.90$, $C_{d,\text{air}} = 1.30$, and $c = 0.0$ are uncalibrated prototype configurations. |

**Audit Finding:** The code does **NOT** enforce a physical relationship between mass and dimensions (e.g. $m \approx \rho_{\text{ice}} \cdot L \cdot W \cdot (H_{\text{draft}} + H_{\text{freeboard}})$). A user could pass an arbitrary mass with mismatched dimensions.

---

## 5. Coordinate System & CRS Audit

* **Input Coordinate System:** Geographic WGS84 coordinates ($\text{lon}, \text{lat}$) are converted to projected Cartesian meters for the initial state.
* **Internal Numerical State:** Projected Cartesian $[x, y]$ in metres, $[v_x, v_y]$ in $\text{m/s}$.
* **Output Coordinates:** Both projected $[x, y]$ and geographic $[\text{lon}, \text{lat}]$ are recorded at each step.
* **CRS Used:** Configurable parameter `crs` in `CoordinateHandler` and `simulate_iceberg`, defaulting to `EPSG:3412` (NSIDC Sea Ice Polar Stereographic South, WGS 84). Supports `EPSG:3031` (Antarctic Polar Stereographic, WGS 84).
* **Pyproj Invocations:**
  ```python
  # src/models/iceberg_physics.py Line 198-200
  self.proj = pyproj.Proj(self.crs_str)
  self.geo_to_proj = pyproj.Transformer.from_crs("EPSG:4326", self.crs_str, always_xy=True)
  self.proj_to_geo = pyproj.Transformer.from_crs(self.crs_str, "EPSG:4326", always_xy=True)
  ```
* **Use of `always_xy=True`:** Both forward and reverse transformers explicitly set `always_xy=True`.
* **Coordinate Mapping:**
  * In WGS84: First coordinate is longitude (Easting), second is latitude (Northing).
  * In EPSG:3412: $x$ is Easting (metres), $y$ is Northing (metres).
* **Vector Rotation (Meridian Convergence):**
  Lines 214–243 use `self.proj.get_factors(lon, lat).meridian_convergence` ($\gamma$) to rotate geographic $(u_{\text{east}}, v_{\text{north}})$ to grid $(v_x, v_y)$:
  $$v_x = u_{\text{east}} \cos\gamma - v_{\text{north}} \sin\gamma$$
  $$v_y = u_{\text{east}} \sin\gamma + v_{\text{north}} \cos\gamma$$
* **Proof that Integration is in Projected Metres:**
  Lines 414–425 update `current_state` using `rk4_step`, which updates $[x, y]$ in metres and $[v_x, v_y]$ in $\text{m/s}$. Geodetic coordinates $(\text{lon}, \text{lat})$ are derived post-step via `coord_handler.to_geographic(x, y)` only for storage and environmental query lookups.
* **Coordinate Round-Trip Test Verification:**
  `TestCoordinatesAndInstrumentedCalls::test_coordinate_round_trip` tests 4 distinct Antarctic points across different quadrants ($0^\circ$, $45^\circ\text{E}$, $-120^\circ\text{W}$, $90^\circ\text{E}$) and asserts $|\Delta\text{lat}| < 10^{-6}$ and $|\Delta\text{lon}| < 10^{-6}$ degrees. This is a non-trivial geometric round trip.

---

## 6. Runge-Kutta 4 (RK4) Implementation Audit

From `src/models/iceberg_physics.py` lines 162–185 and 402–417:
```python
def rk4_step(state, t, dt, derivative_function):
    k1 = derivative_function(t, state)
    k2 = derivative_function(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = derivative_function(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = derivative_function(t + dt, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
```

### Stage-by-Stage Trace

| Stage | Evaluation Time ($t$) | State Vector Passed ($\vec{S}$) | Position Used | Velocity Used | Environment Query Call |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$k_1$** | $t_0$ | $\vec{S}_0$ | $x_0, y_0$ | $v_{x0}, v_{y0}$ | `get_environment(t_0, lat_0, lon_0)` |
| **$k_2$** | $t_0 + \tfrac{1}{2}\Delta t$ | $\vec{S}_0 + \tfrac{1}{2}\Delta t \cdot k_1$ | $x_0 + \tfrac{1}{2}\Delta t k_{1,x}$ | $v_{x0} + \tfrac{1}{2}\Delta t k_{1,vx}$ | `get_environment(t_0 + 0.5*dt, lat_k1, lon_k1)` |
| **$k_3$** | $t_0 + \tfrac{1}{2}\Delta t$ | $\vec{S}_0 + \tfrac{1}{2}\Delta t \cdot k_2$ | $x_0 + \tfrac{1}{2}\Delta t k_{2,x}$ | $v_{x0} + \tfrac{1}{2}\Delta t k_{2,vx}$ | `get_environment(t_0 + 0.5*dt, lat_k2, lon_k2)` |
| **$k_4$** | $t_0 + \Delta t$ | $\vec{S}_0 + \Delta t \cdot k_3$ | $x_0 + \Delta t k_{3,x}$ | $v_{x0} + \Delta t k_{3,vx}$ | `get_environment(t_0 + dt, lat_k3, lon_k3)` |

**Verification of Separate Environmental Queries:**
In `iceberg_derivative` (lines 270–278), `coord_handler.to_geographic(x_m, y_m)` and `environment_provider.get_environment(current_time, lat_deg, lon_deg)` are executed **inside** the derivative callback for every call.  
Unit test `TestCoordinatesAndInstrumentedCalls::test_rk4_intermediate_environment_calls` explicitly tests this with an instrumented mock provider and confirms that **exactly 4 independent calls** occur for a single timestep at $t_0$, $t_0 + 0.5\Delta t$, $t_0 + 0.5\Delta t$, and $t_0 + \Delta t$.

---

## 7. Units and Dimensional Consistency Audit

| Quantity | Code Variable | Expected / Actual Unit | Where Defined | Where Used | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Projected X position | `x_m` | $\text{m}$ | `IcebergState` line 81 | `deriv`, `rk4_step` line 269 | Consistent |
| Projected Y position | `y_m` | $\text{m}$ | `IcebergState` line 82 | `deriv`, `rk4_step` line 269 | Consistent |
| Geodetic Latitude | `lat_deg` | $\text{degrees}$ | `CoordinateHandler` line 209 | `deriv` line 272, 319 | Consistent |
| Geodetic Longitude | `lon_deg` | $\text{degrees}$ | `CoordinateHandler` line 209 | `deriv` line 272, 291 | Consistent |
| Velocity Components | `vx_mps`, `vy_mps` | $\text{m/s}$ | `IcebergState` line 83-84 | `deriv` line 296, 306 | Consistent |
| Acceleration Components | `ax_mps2`, `ay_mps2` | $\text{m/s}^2$ | `deriv` line 334-335 | Return vector line 337 | Consistent |
| Time | `t_seconds`, `t` | $\text{s}$ | `simulate_iceberg` line 383 | Time addition line 275 | Consistent |
| Integration Timestep | `dt_seconds` | $\text{s}$ | `simulate_iceberg` line 350 | `rk4_step` line 166 | Consistent |
| Iceberg Mass | `mass_kg` | $\text{kg}$ | `IcebergProperties` line 116 | `deriv` line 320, 334 | Consistent |
| Waterline Length | `length_m` | $\text{m}$ | `IcebergProperties` line 117 | `effective_width` line 153 | Consistent |
| Waterline Width | `width_m` | $\text{m}$ | `IcebergProperties` line 118 | `effective_width` line 153 | Consistent |
| Keel Draft | `draft_m` | $\text{m}$ | `IcebergProperties` line 119 | `underwater_area` line 154 | Consistent |
| Freeboard Height | `freeboard_m` | $\text{m}$ | `IcebergProperties` line 126 | `above_water_area` line 155 | Consistent |
| Keel Drag Area | `underwater_area_m2` | $\text{m}^2$ | `IcebergProperties` line 154 | `drag_water_coeff` line 300 | Consistent |
| Sail Drag Area | `above_water_area_m2` | $\text{m}^2$ | `IcebergProperties` line 155 | `drag_air_coeff` line 310 | Consistent |
| Seawater Density | `water_density_kg_per_m3` | $\text{kg/m}^3$ | `IcebergProperties` line 123 | `drag_water_coeff` line 300 | Consistent |
| Air Density | `air_density_kg_per_m3` | $\text{kg/m}^3$ | `IcebergProperties` line 124 | `drag_air_coeff` line 310 | Consistent |
| Glacial Ice Density | `ice_density_kg_per_m3` | $\text{kg/m}^3$ | `IcebergProperties` line 125 | `freeboard` line 150 | Consistent |
| Water Drag Coefficient | `water_drag_coefficient` | dimensionless | `IcebergProperties` line 121 | `drag_water_coeff` line 300 | Consistent |
| Air Drag Coefficient | `air_drag_coefficient` | dimensionless | `IcebergProperties` line 120 | `drag_air_coeff` line 310 | Consistent |
| Damping Coefficient | `damping_coefficient` | $\text{N}\cdot\text{s/m}$ | `IcebergProperties` line 122 | `F_damping` line 327 | Consistent |
| Ocean Velocity | `ocean_u`, `ocean_v` | $\text{m/s}$ | `EnvironmentProvider` | `deriv` line 279-280 | Consistent |
| Wind Velocity | `wind_u`, `wind_v` | $\text{m/s}$ | `EnvironmentProvider` | `deriv` line 281-282 | Consistent |
| Earth Angular Velocity | `EARTH_OMEGA_RAD_PER_S` | $\text{rad/s}$ | Module constant line 57 | `f_coriolis` line 319 | Consistent |
| Coriolis Parameter $f$ | `f_coriolis` | $\text{s}^{-1}$ | `deriv` line 319 | `F_coriolis` line 320 | Consistent |
| Forces ($F_{\text{water}}, F_{\text{air}}, F_{\text{coriolis}}, F_{\text{damping}}$) | `F_*` | $\text{N}$ ($\text{kg}\cdot\text{m/s}^2$) | `deriv` lines 301, 311, 320, 327 | Line 331, 334 | Consistent |

**Dimensional Check of Drag Force:**
$$[F] = [\rho] \cdot [C_d] \cdot [A] \cdot [v] \cdot [v] = \frac{\text{kg}}{\text{m}^3} \cdot 1 \cdot \text{m}^2 \cdot \frac{\text{m}}{\text{s}} \cdot \frac{\text{m}}{\text{s}} = \frac{\text{kg}\cdot\text{m}}{\text{s}^2} = \text{Newtons (N)}$$
$$[a] = \frac{[F]}{[m]} = \frac{\text{N}}{\text{kg}} = \frac{\text{m}}{\text{s}^2}\quad \text{Dimensionally verified.}$$

**Dimensional Check of Coriolis Force:**
$$[F] = [m] \cdot [f] \cdot [v] = \text{kg} \cdot \frac{1}{\text{s}} \cdot \frac{\text{m}}{\text{s}} = \frac{\text{kg}\cdot\text{m}}{\text{s}^2} = \text{Newtons (N)}$$
$$[a] = \frac{[F]}{[m]} = \frac{\text{m}}{\text{s}^2}\quad \text{Dimensionally verified.}$$

---

## 8. Historical Integrity & Future-Data Audit

1. **Where the historical cutoff is stored:**
   Stored as `self.max_allowed_timestamp: Optional[pd.Timestamp]` on the `EnvironmentProvider` base class (`src/data/environment.py` line 108).
2. **Where timestamps are checked:**
   Checked in `_check_temporal_integrity(timestamp)` called at the entry point of `get_environment()` in `CompositeEnvironmentProvider` (line 191) and `SyntheticEnvironment` (line 94).
3. **What happens when a query requests data after the cutoff:**
   Raises `HistoricalIntegrityViolationError` immediately:
   `"Temporal leak detected: Query timestamp {timestamp} exceeds forecast initialization cutoff {self.max_allowed_timestamp}."`
4. **Whether interpolation/resampling can accidentally use observations after the cutoff:**
   In `CompositeEnvironmentProvider`, the timestamp check occurs **before** any child loader (`NSIDCLoader`, `ERA5Loader`, `CopernicusLoader`) is invoked. If the requested query timestamp exceeds the cutoff, it is blocked prior to any xarray interpolation.
5. **Whether RK4 intermediate timestamps can cross the historical cutoff:**
   **YES (Edge Risk):** If a simulation starts at $T - \Delta t$ with cutoff $T$, intermediate stage $k_4$ queries at $t_0 + \Delta t = T$. If duration extends past $T$, stage $k_2, k_3, k_4$ will attempt to query beyond $T$ and be blocked by `HistoricalIntegrityViolationError`.
6. **Whether environment queries during historical replay can access future data:**
   Guarded by `max_allowed_timestamp`. However, note that if a user directly queries `ERA5Loader.get_forcing()` without wrapping it in an `EnvironmentProvider`, the standalone loader does not hold a cutoff timestamp.
7. **Whether the simulator itself performs cutoff enforcement:**
   **NO:** `simulate_iceberg()` does not inspect `environment_provider.max_allowed_timestamp`. It relies entirely on the provider raising `HistoricalIntegrityViolationError` when queried.

---

## 9. Numerical Stability Audit

* **Values considered invalid:** `NaN` and positive/negative infinity (`np.inf`, `-np.inf`).
* **Detection mechanisms:**
  1. `if not np.isfinite(current_state).all(): raise NumericalInstabilityError` at simulator entry (line 380).
  2. `if not np.isfinite(state_arr).all(): raise NumericalInstabilityError` at derivative entry (line 263).
  3. `if not np.isfinite([u_ocean, v_ocean, u_wind, v_wind]).all(): raise NumericalInstabilityError` after environmental query (line 284).
  4. `if not np.isfinite(out_deriv).all(): raise NumericalInstabilityError` before returning from derivative (line 338).
  5. `if not np.isfinite(current_state).all(): raise NumericalInstabilityError` after each RK4 step (line 419).
* **Exception raised:** `src.models.iceberg_physics.NumericalInstabilityError` (subclass of `PhysicsSimulationError`).
* **Can RK4 silently propagate NaN/Inf?**
  **NO:** With checks at both the entrance and exit of `iceberg_derivative` and after `rk4_step`, any non-finite state triggers an immediate exception before subsequent steps can execute.
* **Velocity ceiling check:**
  The code does **NOT** enforce a physical maximum velocity bound (e.g. $v < 100\,\text{m/s}$). While NaNs and Infs are caught, an exponential blowup that stays finite (e.g. $10^{20}\,\text{m/s}$) would not be intercepted until numerical overflow produces Inf.

---

## 10. Test-to-Implementation Mapping

| Test Function | What It Verifies | What It DOES NOT Verify |
| :--- | :--- | :--- |
| `test_rk4_exponential_decay_and_growth` | Mathematical convergence and order of the RK4 integrator on an exact analytical ODE ($dx/dt = x$). | Physical forces, units, or iceberg dynamics. |
| `test_zero_force_motion` | Pure inertial motion: $\vec{a} = 0 \implies \vec{x}(t) = \vec{x}_0 + \vec{v}_0 t$ in projected Cartesian coordinates. | Force balance with active wind/ocean drag or Coriolis deflection. |
| `test_constant_ocean_current_acceleration` | Water drag accelerates an initially stationary iceberg toward the ocean current velocity ($v_x \to u_{\text{ocean}}$). | Calibrated acceleration rates, turbulent boundary layer effects, or wave radiation. |
| `test_wind_forcing_acceleration` | Aerodynamic sail drag accelerates the iceberg in the downwind direction. | Wind shear profile with height, gustiness, or sail shape factors. |
| `test_southern_hemisphere_coriolis_deflection_direction` | Exact sign of Coriolis acceleration in S. Hemisphere ($f < 0 \implies a_y = -f v_x > 0$, deflection to the LEFT). | Beta-plane variations across broad latitudes or rotation of the iceberg body. |
| `test_coordinate_round_trip` | Bidirectional projection precision between WGS84 and EPSG:3412 is $< 10^{-6}$ degrees across 4 polar quadrants. | Grid distortion factors at extreme sub-polar latitudes. |
| `test_rk4_intermediate_environment_calls` | RK4 executes exactly 4 distinct environment queries per timestep at $t, t + \Delta t/2, t + \Delta t/2, t + \Delta t$. | Accuracy of environmental interpolation between hourly/daily satellite grids. |
| `test_invalid_parameters_raise_exceptions` | Input validation rejects non-positive mass, length, width, and negative draft/drag coefficients. | Whether the accepted positive parameters correspond to physically realistic icebergs. |
| `test_numerical_instability_detection` | Injection of `np.nan` into forcing cleanly raises `NumericalInstabilityError`. | Soft divergence (e.g. unphysically high speeds that remain finite). |

---

## 11. Parameter Magnitude Audit

| Parameter | Value in Code | Unit | Classification | Rationale / Documentation |
| :--- | ---: | :--- | :--- | :--- |
| `EARTH_OMEGA_RAD_PER_S` | $7.292115 \times 10^{-5}$ | $\text{rad/s}$ | `CODE/PHYSICS CONSTANT` | Earth rotation rate based on 1 sidereal day ($86164.0905\,\text{s}$). Standard GFD value. |
| `water_density_kg_per_m3` | $1025.0$ | $\text{kg/m}^3$ | `CODE/PHYSICS CONSTANT` | Standard oceanographic density for polar seawater near surface. |
| `air_density_kg_per_m3` | $1.25$ | $\text{kg/m}^3$ | `CODE/PHYSICS CONSTANT` | Standard atmospheric surface density at sub-zero polar temperatures. |
| `ice_density_kg_per_m3` | $900.0$ | $\text{kg/m}^3$ | `CODE/PHYSICS CONSTANT` | Standard density of compacted Antarctic glacial shelf ice. |
| `water_drag_coefficient` | $0.90$ | dimensionless | `CONFIGURABLE ASSUMPTION` | Prototype drag coefficient for bluff/tabular keel. Uncalibrated against BYU/NIC tracks. |
| `air_drag_coefficient` | $1.30$ | dimensionless | `CONFIGURABLE ASSUMPTION` | Prototype drag coefficient for bluff vertical sail. Uncalibrated against BYU/NIC tracks. |
| `damping_coefficient` | $0.00$ | $\text{N}\cdot\text{s/m}$ | `CONFIGURABLE ASSUMPTION` | Linear damping set to zero by default; available as a prototype dissipation term. |
| Test Mass (`standard_properties`) | $7.5 \times 10^{10}$ | $\text{kg}$ | `SYNTHETIC TEST VALUE` | Plausible synthetic mass for a $1\,\text{km} \times 0.5\,\text{km} \times 150\,\text{m}$ tabular iceberg. |
| Test Length (`standard_properties`) | $1000.0$ | $\text{m}$ | `SYNTHETIC TEST VALUE` | Synthetic test geometry. |
| Test Width (`standard_properties`) | $500.0$ | $\text{m}$ | `SYNTHETIC TEST VALUE` | Synthetic test geometry. |
| Test Draft (`standard_properties`) | $150.0$ | $\text{m}$ | `SYNTHETIC TEST VALUE` | Synthetic test geometry. |

---

## 12. Final Audit Summary

### PASS
1. **Dimensional Consistency:** All forces resolve strictly to Newtons ($\text{N} = \text{kg}\cdot\text{m/s}^2$) and accelerations to $\text{m/s}^2$. No incompatible unit additions exist.
2. **Relative Velocity Directionality:** Both water drag and air drag correctly employ fluid velocity minus body velocity ($\vec{v}_{\text{fluid}} - \vec{v}_{\text{iceberg}}$) with a positive force coefficient, ensuring the fluid accelerates the body down-flow and resists motion when the body exceeds fluid velocity.
3. **Southern Hemisphere Coriolis Direction:** The Coriolis formulation $a_x = f v_y, a_y = -f v_x$ with $f = 2\Omega\sin\phi < 0$ correctly deflects motion to the **LEFT** of the velocity vector in the Southern Hemisphere.
4. **Intermediate RK4 Evaluation:** RK4 evaluates environmental forcing separately at all 4 intermediate stages ($k_1, k_2, k_3, k_4$) using the projected intermediate coordinates and intermediate timestamps.
5. **Coordinate Separation:** Internal numerical integration is strictly performed in projected metres ($x, y, v_x, v_y$) on `EPSG:3412` rather than in degrees latitude/longitude.
6. **Vector Orientation:** Geographic wind and ocean velocities are rotated into the projected grid coordinate orientation using the local meridian convergence angle $\gamma$.
7. **Numerical Guards:** Active checks on input state, forcing returns, and output derivative prevent silent propagation of NaNs or Infs.

### CONCERNS
1. **Decoupling of Mass and Geometry:** `mass_kg` is an independent input parameter in `IcebergProperties`. The code does not check whether the provided mass is physically consistent with $\rho_{\text{ice}} \times \text{Length} \times \text{Width} \times (\text{Draft} + \text{Freeboard})$. An inconsistent mass will distort acceleration.
2. **Absence of a Physical Velocity Ceiling:** While non-finite states (NaN/Inf) raise `NumericalInstabilityError`, unphysically large finite velocities (e.g. $500\,\text{m/s}$ due to extreme wind inputs or poorly chosen $\Delta t$) are not explicitly intercepted.
3. **Isotropic Frontal Area Approximation:** The cross-sectional area is approximated as $\sqrt{L \cdot W} \cdot H_{\text{draft}}$, assuming the iceberg presents an average frontal width regardless of heading. For elongated tabular icebergs ($L \gg W$), the actual frontal area depends strongly on orientation relative to the current.
4. **Standalone Loader Cutoff Bypass:** `_check_temporal_integrity` is enforced by `EnvironmentProvider` and `CompositeEnvironmentProvider`, but individual loaders (`ERA5Loader`, `NSIDCLoader`) do not hold a cutoff timestamp if queried directly outside the provider abstraction.

### SCIENTIFIC LIMITATIONS
The current Stage 3 implementation is a research baseline prototype. It explicitly excludes:
1. **Iceberg Rotation / Yaw Dynamics:** Yaw torque caused by asymmetric drag, moment of inertia, and angular velocity are not modeled.
2. **Wave Radiation / Wave Drift Force:** Radiation stress from ocean swell pushing against the exposed ice cliff is excluded.
3. **Internal Sea-Ice Pack Pressure:** Sea ice is currently treated only via environmental concentration lookups; mechanical resistance, crushing force, or drifting with pack ice is not yet represented in the momentum equation.
4. **Vertical Stratification & Ekman Shear:** Ocean current is evaluated as a single near-surface or draft-averaged vector; vertical current shear across the 150m keel depth is not dynamically resolved.
5. **Thermodynamic Ablation & Calving:** Mass, draft, length, and width remain constant over time; wave erosion, basal melting, and mechanical calving are not simulated.
6. **Bathymetric Grounding:** Iceberg draft is not compared against seafloor bathymetry to simulate grounding or scouring.

### NOT YET VALIDATED
1. **No Real-World Trajectory Validation:** Passing the 25 unit tests proves mathematical correctness of the solver, sign correctness, and numerical stability. It **does NOT** demonstrate that the model can accurately predict real Antarctic iceberg trajectories.
2. **Uncalibrated Drag Coefficients:** Values $C_{d,\text{water}} = 0.90$ and $C_{d,\text{air}} = 1.30$ are uncalibrated prototype estimates. Calibration against BYU/NIC Antarctic tracking database trajectories is required in subsequent evaluation stages.
