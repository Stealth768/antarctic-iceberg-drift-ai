# Stage 3 Follow-Up Audit: Coriolis Coordinate Basis & Drag Area Dimensionality

**Project:** SIH26059 — AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System  
**Audit Scope:** Deep inspection of Coriolis coordinate representation and drag area dimensional formulation  
**Mode:** READ-ONLY Mathematical & Physical Audit  
**Date:** 2026-09-03  

---

## 1. Coriolis Coordinate Basis

### Implementation Inspection
In `src/models/iceberg_physics.py`:
```python
# Lines 269-272: Extraction of state and conversion to geographic coordinates
x_m, y_m, vx_mps, vy_mps = state_arr
lon_deg, lat_deg = coord_handler.to_geographic(x_m, y_m)

# Lines 291-292: Environmental forcing rotated to projected grid orientation
u_ocean_grid, v_ocean_grid = coord_handler.rotate_vector_to_grid(u_ocean, v_ocean, lon_deg, lat_deg)
u_wind_grid, v_wind_grid = coord_handler.rotate_vector_to_grid(u_wind, v_wind, lon_deg, lat_deg)

# Lines 318-324: Coriolis acceleration and force evaluation
if props.enable_coriolis:
    f_coriolis = 2.0 * EARTH_OMEGA_RAD_PER_S * np.sin(np.radians(lat_deg))
    F_coriolis_x = props.mass_kg * (f_coriolis * vy_mps)
    F_coriolis_y = props.mass_kg * (-f_coriolis * vx_mps)
else:
    F_coriolis_x = 0.0
    F_coriolis_y = 0.0

# Lines 331-335: Force summation and acceleration calculation
F_total_x = F_water_x + F_air_x + F_coriolis_x - F_damping_x
F_total_y = F_water_y + F_air_y + F_coriolis_y - F_damping_y

ax_mps2 = F_total_x / props.mass_kg
ay_mps2 = F_total_y / props.mass_kg
```

### Trace & Answers to Specific Questions

1. **What physical direction does `vx` represent at the point where Coriolis is calculated?**
   `vx_mps` represents the velocity component along the **projected grid X-axis** ($dx/dt$, Eastings rate of change in projected metres per second). It is NOT True East (except along the central meridian $\lambda_0 = 0^\circ$).
2. **What physical direction does `vy` represent?**
   `vy_mps` represents the velocity component along the **projected grid Y-axis** ($dy/dt$, Northings rate of change in projected metres per second). It is NOT True North (except along $\lambda_0 = 0^\circ$).
3. **Are they true East/North components or projected grid-X/grid-Y components?**
   They are strictly **projected grid-X / grid-Y components**.
4. **Where does the meridian-convergence rotation occur?**
   Lines 291–292: Environmental velocities (`ocean_u`, `ocean_v`, `wind_u`, `wind_v`), which are supplied by `EnvironmentProvider` in True East / True North coordinates, are rotated into grid-X / grid-Y orientations using the meridian convergence angle $\gamma$.
5. **Is the velocity rotated before the Coriolis calculation?**
   **No.** `vx_mps` and `vy_mps` are already in projected grid-X / grid-Y coordinates and are used directly in the Coriolis calculation.
6. **Is the Coriolis acceleration rotated back afterward?**
   **No.** $\vec{F}_{\text{coriolis}}$ is computed directly in grid-X / grid-Y components and summed with grid-drag forces $\vec{F}_{\text{water}}$ and $\vec{F}_{\text{air}}$ to integrate the state $[x, y, v_x, v_y]$.
7. **Is $f = 2\Omega\sin(\text{latitude})$ calculated using geographic latitude?**
   **Yes.** `lat_deg` is obtained by transforming $(x, y)$ back to geographic coordinates via WGS84 ellipsoidal transformation (`coord_handler.to_geographic(x_m, y_m)`).
8. **Does the implementation account for the fact that projected grid axes are not necessarily locally aligned with East/North?**
   **Yes, through mathematical invariance.** In 2D geophysical fluid dynamics, the horizontal Coriolis acceleration is defined by the 3D cross product:
   $$\vec{a}_{\text{coriolis}} = -2 (\vec{\Omega} \cdot \hat{k}) (\hat{k} \times \vec{v}) = -f (\hat{k} \times \vec{v})$$
   where $\hat{k}$ is the local unit normal to the horizontal tangent plane.
   In any right-handed orthogonal 2D Cartesian basis $(\hat{i}, \hat{j})$ spanning the plane where $\hat{i} \times \hat{j} = \hat{k}$:
   $$\hat{k} \times \vec{v} = \hat{k} \times (v_x \hat{i} + v_y \hat{j}) = -v_y \hat{i} + v_x \hat{j}$$
   $$\vec{a}_{\text{coriolis}} = -f (-v_y \hat{i} + v_x \hat{j}) = f v_y \hat{i} - f v_x \hat{j} = \begin{bmatrix} f v_y \\ -f v_x \end{bmatrix}$$
   Because rotating the orthonormal coordinate axes by an angle $\gamma$ in the plane preserves $\hat{i} \times \hat{j} = \hat{k}$, the operator $\vec{v} \mapsto (f v_y, -f v_x)$ commutes with any 2D planar rotation matrix $R(\gamma)$:
   $$R(\gamma) \begin{bmatrix} 0 & f \\ -f & 0 \end{bmatrix} = \begin{bmatrix} 0 & f \\ -f & 0 \end{bmatrix} R(\gamma)$$
   Therefore, evaluating $(f v_y, -f v_x)$ directly on projected grid velocities $(v_x, v_y)$ produces the exact, uncorrupted physical Coriolis acceleration in grid coordinates.

---

## 2. Coriolis Trace Example

Consider an iceberg at representative Antarctic coordinates:
* **Position:** $\phi = -70.0^\circ\text{S}$, $\lambda = +45.0^\circ\text{E}$ (Cosmonaut Sea off East Antarctica).
* **True Geographic Motion:** Purely Eastward velocity $\vec{v}_{\text{geo}} = (u_{\text{east}} = 1.0\,\text{m/s},\; v_{\text{north}} = 0.0\,\text{m/s})$.

### Step 1: Meridian Convergence at $(45^\circ\text{E}, -70^\circ\text{S})$ in EPSG:3412
In polar stereographic projection with standard central meridian $\lambda_0 = 0^\circ$:
$$\gamma = \lambda - \lambda_0 = -45.0^\circ = -\frac{\pi}{4}\,\text{rad}$$

### Step 2: Grid-X and Grid-Y Velocity Components
Using `rotate_vector_to_grid`:
$$v_x = u_{\text{east}} \cos\gamma - v_{\text{north}} \sin\gamma = 1.0 \cdot \cos(-45^\circ) - 0 = \frac{\sqrt{2}}{2} \approx +0.7071\,\text{m/s}$$
$$v_y = u_{\text{east}} \sin\gamma + v_{\text{north}} \cos\gamma = 1.0 \cdot \sin(-45^\circ) + 0 = -\frac{\sqrt{2}}{2} \approx -0.7071\,\text{m/s}$$

### Step 3: Coriolis Parameter $f$
$$\Omega = 7.292115 \times 10^{-5}\,\text{rad/s}$$
$$f = 2 \Omega \sin(-70^\circ) = 2(7.292115 \times 10^{-5})(-0.93969) \approx -1.3705 \times 10^{-4}\,\text{s}^{-1}$$

### Step 4: Coriolis Acceleration in Projected Grid Coordinates
$$a_{\text{grid}, x} = f \cdot v_y = (-1.3705 \times 10^{-4}) \cdot (-0.7071) \approx +9.6907 \times 10^{-5}\,\text{m/s}^2$$
$$a_{\text{grid}, y} = -f \cdot v_x = -(-1.3705 \times 10^{-4}) \cdot (+0.7071) \approx +9.6907 \times 10^{-5}\,\text{m/s}^2$$

### Step 5: Transformation Back to True Geographic East/North
Using `rotate_grid_to_vector`:
$$a_{\text{east}} = a_{\text{grid}, x} \cos\gamma + a_{\text{grid}, y} \sin\gamma = (9.6907 \times 10^{-5})\left(\frac{\sqrt{2}}{2}\right) + (9.6907 \times 10^{-5})\left(-\frac{\sqrt{2}}{2}\right) = 0.0000\,\text{m/s}^2$$
$$a_{\text{north}} = -a_{\text{grid}, x} \sin\gamma + a_{\text{grid}, y} \cos\gamma = -(9.6907 \times 10^{-5})\left(-\frac{\sqrt{2}}{2}\right) + (9.6907 \times 10^{-5})\left(\frac{\sqrt{2}}{2}\right) = +1.3705 \times 10^{-4}\,\text{m/s}^2$$

### Conclusion of Trace
* **Eastward Acceleration ($a_{\text{east}}$):** $0.0\,\text{m/s}^2$
* **Northward Acceleration ($a_{\text{north}}$):** $+1.3705 \times 10^{-4}\,\text{m/s}^2$
* For Eastward motion in the Southern Hemisphere, deflection is strictly **Northward** (to the **LEFT** of the velocity vector).
* The projected grid formulation produces the mathematically exact physical Coriolis acceleration at all polar longitudes.

---

## 3. Drag Area Implementation

### Code Excerpt
From `src/models/iceberg_physics.py` lines 145–156:
```python
# Compute isostatic freeboard height if not explicitly supplied:
# Archimedes principle: Total_height = Draft * (rho_water / rho_ice)
# Freeboard = Total_height - Draft = Draft * (rho_water - rho_ice) / rho_ice
if self.freeboard_m is None:
    delta_rho = max(0.0, self.water_density_kg_per_m3 - self.ice_density_kg_per_m3)
    self.freeboard_m = float(self.draft_m * (delta_rho / self.ice_density_kg_per_m3))

# Approximate isotropic projected cross-sectional area using geometric mean width
self.effective_width_m = float(np.sqrt(self.length_m * self.width_m))
self.underwater_area_m2 = float(self.effective_width_m * self.draft_m)
self.above_water_area_m2 = float(self.effective_width_m * self.freeboard_m)
```

And in `iceberg_derivative` lines 300–310:
```python
drag_water_coeff = 0.5 * props.water_density_kg_per_m3 * props.water_drag_coefficient * props.underwater_area_m2
F_water_x = drag_water_coeff * speed_rel_water * v_rel_water_x
F_water_y = drag_water_coeff * speed_rel_water * v_rel_water_y

drag_air_coeff = 0.5 * props.air_density_kg_per_m3 * props.air_drag_coefficient * props.above_water_area_m2
F_air_x = drag_air_coeff * speed_rel_air * v_rel_air_x
F_air_y = drag_air_coeff * speed_rel_air * v_rel_air_y
```

---

## 4. Dimensional Analysis

### Dimensional Trace of Every Dimension

| Quantity | Variable | Mathematical Definition | SI Unit | Dimensional Analysis |
| :--- | :--- | :--- | :--- | :--- |
| Waterline Length | `length_m` | Supplied input | $\text{m}$ | $[L]$ |
| Waterline Width | `width_m` | Supplied input | $\text{m}$ | $[L]$ |
| Keel Draft | `draft_m` | Supplied input | $\text{m}$ | $[L]$ |
| Water Density | `water_density_kg_per_m3` | Constant ($1025.0$) | $\text{kg/m}^3$ | $[M L^{-3}]$ |
| Glacial Ice Density | `ice_density_kg_per_m3` | Constant ($900.0$) | $\text{kg/m}^3$ | $[M L^{-3}]$ |
| Sail Freeboard | `freeboard_m` | $H_{\text{draft}} \cdot \frac{\rho_{\text{water}} - \rho_{\text{ice}}}{\rho_{\text{ice}}}$ | $\text{m}$ | $[L] \cdot \frac{[M L^{-3}]}{[M L^{-3}]} = [L]$ |
| Effective Width | `effective_width_m` | $\sqrt{\text{length\_m} \cdot \text{width\_m}}$ | $\text{m}$ | $\sqrt{[L] \cdot [L]} = \sqrt{[L^2]} = [L]$ |
| Keel Drag Area | `underwater_area_m2` | $\text{effective\_width\_m} \cdot \text{draft\_m}$ | $\text{m}^2$ | $[L] \cdot [L] = [L^2]$ |
| Sail Drag Area | `above_water_area_m2` | $\text{effective\_width\_m} \cdot \text{freeboard\_m}$ | $\text{m}^2$ | $[L] \cdot [L] = [L^2]$ |

### Dimensional Trace of Drag Force
$$F_{\text{water}} = 0.5 \cdot \rho_{\text{water}} \cdot C_{d,\text{water}} \cdot A_{\text{underwater}} \cdot |v_{\text{rel}}| \cdot v_{\text{rel}}$$
$$[F_{\text{water}}] = \left[\frac{\text{kg}}{\text{m}^3}\right] \cdot [1] \cdot [\text{m}^2] \cdot \left[\frac{\text{m}}{\text{s}}\right] \cdot \left[\frac{\text{m}}{\text{s}}\right] = \frac{\text{kg}\cdot\text{m}}{\text{s}^2} = \text{Newtons (N)}$$

$$[a_{\text{water}}] = \frac{[F_{\text{water}}]}{[m]} = \frac{\text{N}}{\text{kg}} = \frac{\text{m}}{\text{s}^2}$$

### Verification of Volume vs. Area
* **Is the code computing a volume ($L \times W \times H \to \text{m}^3$)?**  
  **NO.** The code computes the geometric mean $\sqrt{L \cdot W}$ (which has units of **metres**, $\text{m}$), and multiplies it by $H_{\text{draft}}$ (in **metres**, $\text{m}$).
* The result is $[L] \times [L] = [L^2]$ ($\text{m}^2$), which is strictly an **area**.
* There is **no volume calculation** ($\text{m}^3$) mistakenly passed into the drag force equation.

---

## 5. Verdict

### Issue 1: Coriolis Coordinate Basis
* **Classification:** `CORRECT`
* **Rationale:** The horizontal Coriolis operator $\vec{a} = -f (\hat{k} \times \vec{v}) = (f v_y, -f v_x)$ is mathematically invariant under 2D rotations of an orthonormal basis in the tangent plane. Because EPSG:3412 is a conformal, right-handed projected Cartesian coordinate system, applying $(f v_y, -f v_x)$ directly to grid velocities $(v_x, v_y)$ produces the exact physical Coriolis acceleration in grid coordinates without requiring pre-rotation or post-rotation. An Eastward velocity at $45^\circ\text{E}$ produces pure Northward (leftward) acceleration of exact magnitude $1.3705 \times 10^{-4}\,\text{m/s}^2$.

### Issue 2: Drag Area Dimensionality
* **Classification:** `CORRECT`
* **Rationale:** Both `underwater_area_m2` and `above_water_area_m2` are rigorously $[L^2]$ ($\text{m}^2$). The code uses the geometric mean $\sqrt{L \cdot W}\ [\text{m}] \times H\ [\text{m}]$ to model an isotropic characteristic cross-sectional frontal area. The resulting drag forces resolve strictly to Newtons ($\text{kg}\cdot\text{m/s}^2$) and accelerations to $\text{m/s}^2$. No volume term ($\text{m}^3$) is present in the force calculation.
