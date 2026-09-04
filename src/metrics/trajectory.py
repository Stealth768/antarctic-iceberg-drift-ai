import numpy as np
import pandas as pd
import pyproj


def haversine_distance(lat1, lon1, lat2, lon2):
    geod = pyproj.Geod(ellps="WGS84")
    _, _, dist_meters = geod.inv(lon1, lat1, lon2, lat2)
    return dist_meters / 1000.0


def calculate_trajectory_metrics(
    df_sim,
    df_truth,
    time_col="timestamp",
    lat_col="latitude",
    lon_col="longitude",
    heading_window=6,
):
    merged = pd.merge(
        df_sim[[time_col, lat_col, lon_col]],
        df_truth[[time_col, lat_col, lon_col]],
        on=time_col,
        suffixes=("_sim", "_truth"),
        how="inner",
    )

    if merged.empty:
        raise ValueError("No matching timestamps found.")

    lat_sim = merged[f"{lat_col}_sim"].values
    lon_sim = merged[f"{lon_col}_sim"].values
    lat_gt = merged[f"{lat_col}_truth"].values
    lon_gt = merged[f"{lon_col}_truth"].values

    # Position errors
    step_errors = haversine_distance(
        lat_sim, lon_sim, lat_gt, lon_gt
    )

    fpe_km = float(step_errors[-1])
    mae_km = float(np.mean(step_errors))
    rmse_km = float(np.sqrt(np.mean(step_errors**2)))
    max_err_km = float(np.max(step_errors))

    # Heading error over a longer window.
    # This avoids making heading excessively sensitive to
    # point-to-point positional noise.
    if len(merged) > heading_window:
        geod = pyproj.Geod(ellps="WGS84")

        az_sim, _, _ = geod.inv(
            lon_sim[:-heading_window],
            lat_sim[:-heading_window],
            lon_sim[heading_window:],
            lat_sim[heading_window:],
        )

        az_gt, _, _ = geod.inv(
            lon_gt[:-heading_window],
            lat_gt[:-heading_window],
            lon_gt[heading_window:],
            lat_gt[heading_window:],
        )

        angle_diff = np.abs((az_sim - az_gt + 180) % 360 - 180)
        mean_angle_err_deg = float(np.mean(angle_diff))
    else:
        mean_angle_err_deg = 0.0

    return {
        "final_position_error_km": fpe_km,
        "mae_km": mae_km,
        "rmse_km": rmse_km,
        "max_error_km": max_err_km,
        "mean_heading_error_deg": mean_angle_err_deg,
        "matched_timestamps": len(merged),
    }