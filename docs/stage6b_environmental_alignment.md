# Stage 6B: Real Environmental Data Alignment & Historical Integrity Report

**Project:** SIH26059 — AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System  
**Module:** `src/data/environment.py`, `src/data/era5.py`, `src/data/copernicus.py`  
**Purpose:** Multi-source environmental data alignment layer connecting ERA5 and Copernicus reanalyses to the RK4 iceberg drift physics solver and Stage 5 historical evaluation harness.

---

## 1. Environmental Forcing Integration Architecture

The `HistoricalEnvironmentProvider` provides a standardized interface conforming to `EnvironmentProvider`:

```
                ┌──────────────────────────────┐
                │ HistoricalEnvironmentProvider│
                └──────────────┬───────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│  ERA5Loader  │        │CopernicusLdr │        │ NSIDCLoader  │
│(Atmosphere)  │        │   (Ocean)    │        │  (Sea Ice)   │
└──────┬───────┘        └──────┬───────┘        └──────┬───────┘
       │                       │                       │
       ▼                       ▼                       ▼
• 10m Wind (u10, v10)   • Currents (uo, vo)     • Concentration
• 2m Temp (t2m)         • SST (thetao)            (cdr_seaice_conc)
• Pressure (msl)        • Draft integration     • Open-water fallback
```

---

## 2. Causal Temporal Alignment & Leakage Safeguards

In historical reanalysis datasets, time grids are discrete (e.g. 1-hourly for ERA5, 24-hourly for Copernicus). When the RK4 solver steps forward in continuous time ($t = t_0 + \Delta t$), query timestamps fall between grid points.

### The Nearest-Neighbor Leakage Vulnerability:
Standard `xarray.Dataset.sel(time=t, method="nearest")` selects the closest temporal grid point. For example, for a daily dataset sampled at 12:00 UTC, a query at 08:00 UTC with `method="nearest"` selects 12:00 UTC (4 hours in the future). This introduces unphysical future information into the model.

### Causal Indexing Rule:
The alignment layer enforces strictly causal temporal indexing:
$$\mathcal{T}_{\text{valid}} = \{t \in \mathcal{T}_{\text{dataset}} \mid t \le t_{\text{query}}\}$$
$$t_{\text{effective}} = \max(\mathcal{T}_{\text{valid}})$$

* **Zero Future Observation Leakage:** $t_{\text{effective}}$ is guaranteed to satisfy $t_{\text{effective}} \le t_{\text{query}}$.
* **No Unobserved Extrapolation:** If $t_{\text{query}} < \min(\mathcal{T}_{\text{dataset}})$, the provider raises `MissingDataError` rather than forward-filling backward from future observations.
* **Forecast Cutoff Enforcement:** If $t_{\text{query}} > \text{max\_allowed\_timestamp}$, the provider immediately raises `HistoricalIntegrityViolationError`.

---

## 3. Spatial Interpolation & Coordinate Normalization

### Bilinear Spatial Interpolation:
To avoid step-function discontinuities in wind and current vectors during numerical integration, the loaders support bilinear spatial interpolation (`method="linear"`):
$$u(\phi, \lambda) = (1 - \alpha)(1 - \beta) u_{00} + \alpha(1 - \beta) u_{10} + (1 - \alpha)\beta u_{01} + \alpha\beta u_{11}$$
where $\alpha = \frac{\phi - \phi_0}{\Delta\phi}$ and $\beta = \frac{\lambda - \lambda_0}{\Delta\lambda}$.

### Longitude Normalization:
* ERA5 native grids typically use $[0, 360^\circ)$.
* Copernicus GLORYS native grids typically use $[-180^\circ, +180^\circ]$.
The alignment layer normalizes incoming query longitudes automatically to match the target dataset range, ensuring queries across the Greenwich Meridian and International Date Line do not fail.

### Submerged Keel Depth Averaging & Orientation Handling:
For tabular Antarctic icebergs with deep submerged drafts ($H_{\text{draft}} \sim 200\text{--}300\,\text{m}$), surface currents alone do not represent hydrodynamic forcing. `CopernicusLoader` integrates and averages ocean currents over $[0, H_{\text{draft}}]$:
$$\bar{\mathbf{u}}_w = \frac{1}{\Delta z} \int_{z_0}^{z_{\text{max}}} \mathbf{u}_w(z)\,dz$$
* **Trapezoidal Layer Weighting:** Rather than an unweighted arithmetic average of model levels (which overweights dense near-surface levels), trapezoidal integration correctly weights layers by vertical interval thickness ($dz$).
* **Coordinate Orientation Invariance:** Handled independently of whether depth is indexed ascending ($0 \to 500\,\text{m}$) or descending ($500 \to 0\,\text{m}$).
* **Depth Exceedance Protection:** If requested draft exceeds available bathymetry/grid depth ($H_{\text{draft}} > z_{\text{max}}$), the loader raises `MissingDataError` rather than silently extrapolating unobserved depths.
* **Near-Surface Default:** When `draft_meters` is `None` or $\le 0$, the nearest surface level ($\sim 0\,\text{m}$) is returned.

For GLORYS12V1, 200 m is not a native level: the deepest native level at or below 200 m is 186.125595 m and the next level is 222.475204 m. When both levels are present, the loader linearly interpolates the field at 200 m before trapezoidal integration. If the deeper bracketing level is absent from the subset, it raises `MissingDataError`; it does not treat 186.125595 m as 200 m.

### Temperature Naming & Physical Semantics:
* **Near-Surface SST:** When no draft averaging is requested, `thetao` near the surface represents standard Sea Surface Temperature (SST).
* **Depth-Averaged Temperature:** When `draft_meters > 0`, the returned temperature is the depth-averaged potential temperature across the submerged keel depth. It is returned under the `'sst'` key for interface and API backward compatibility.

---

## 4. Test Verification Summary

All 11 integration and regression tests in `tests/test_environmental_alignment.py` pass:
1. `test_historical_provider_variable_extraction`: Full 8-variable extraction and `EnvironmentState` validation.
2. `test_causal_timestamp_alignment`: Verification that intermediate queries select past records, rejecting nearest-future records.
3. `test_spatial_interpolation_linear`: Bilinear spatial interpolation matching analytical gradients.
4. `test_longitude_normalization_and_wrapping`: Seamless wrapping between $[0, 360^\circ)$ and $[-180^\circ, 180^\circ]$.
5. `test_historical_cutoff_rejection`: Rejection of queries beyond `max_allowed_timestamp`.
6. `test_no_future_data_leakage`: Rejection of queries before dataset start (`MissingDataError`).
7. `test_compatibility_with_iceberg_physics_solver`: Successful end-to-end 3-day RK4 simulation using `HistoricalEnvironmentProvider` within `IcebergPhysicsEvaluator`.
8. `test_copernicus_depth_orientation_ascending_vs_descending`: Mathematical equivalence between ascending and descending depth coordinates.
9. `test_copernicus_trapezoidal_weighting_vs_unweighted_mean`: Proof that layer-weighted trapezoidal integration is used rather than unweighted level averaging.
10. `test_copernicus_draft_exceeding_max_depth_raises_missing_data`: Strict `MissingDataError` on draft exceeding available ocean depth.
11. `test_copernicus_surface_behavior_when_no_draft`: Surface level extraction when draft is omitted or zero.
12. `test_copernicus_interpolates_non_native_draft_boundary`: Linear interpolation to a bracketed 200 m boundary.
13. `test_copernicus_rejects_unbracketed_non_native_draft_boundary`: Rejection when the deeper boundary level is absent.
