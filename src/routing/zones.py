"""
Antarctic Maritime Risk Zones Generator.

Generates GeoJSON FeatureCollections representing distinct risk zones:
- High Ice Risk (concentrated pack ice)
- Moderate Ice Risk (marginal ice zone)
- Iceberg Drift Hazard (vicinity of tracked icebergs)

Note: Generated zones are derived from numerical model layers and demonstration snapshots
for decision support; they must not be interpreted as certified nautical charts.
"""

from typing import Any, Dict, List, Optional, Tuple
from src.routing.grid import PolarNavigationGrid
from src.routing.planner import EnvironmentSnapshot


def _cell_to_geojson_polygon(grid: PolarNavigationGrid, i: int, j: int) -> List[List[float]]:
    """Convert grid cell (i, j) to a GeoJSON polygon ring [lon, lat]."""
    x, y = grid.grid_to_projected(i, j)
    h = grid.step_m / 2.0
    corners_xy = [
        (x - h, y - h),
        (x + h, y - h),
        (x + h, y + h),
        (x - h, y + h),
        (x - h, y - h),  # Closed loop
    ]
    ring = []
    for px, py in corners_xy:
        lon, lat = grid.coord_handler.to_geographic(px, py)
        ring.append([round(lon, 5), round(lat, 5)])
    return ring


def generate_risk_zones_geojson(
    grid: PolarNavigationGrid,
    env: EnvironmentSnapshot,
) -> Dict[str, Any]:
    """
    Generate GeoJSON FeatureCollection of spatial risk zones.
    """
    features: List[Dict[str, Any]] = []

    high_ice_cells: List[Tuple[int, int]] = []
    mod_ice_cells: List[Tuple[int, int]] = []
    iceberg_cells: List[Tuple[int, int]] = []

    for i in range(grid.nx):
        for j in range(grid.ny):
            cell = (i, j)
            sic = env.sea_ice_concentration.get(cell, 0.0)

            # Iceberg proximity
            lat, lon = grid.grid_to_geo(i, j)
            near_ib = False
            for ib_lat, ib_lon in env.iceberg_locations:
                # Fast Euclidean approximation
                if abs(lat - ib_lat) < 0.25 and abs(lon - ib_lon) < 0.5:
                    near_ib = True
                    break

            if near_ib:
                iceberg_cells.append(cell)
            elif sic >= 0.60:
                high_ice_cells.append(cell)
            elif sic >= 0.25:
                mod_ice_cells.append(cell)

    # Convert sampled cells into polygon features
    # High Ice Risk
    for i, j in high_ice_cells[::2]:  # Subsample for lightweight GeoJSON payload
        ring = _cell_to_geojson_polygon(grid, i, j)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
            "properties": {
                "zone_type": "High Ice Risk",
                "severity": "critical",
                "color": "#ef4444",
                "fill_color": "rgba(239, 68, 68, 0.35)",
                "risk_score": 85.0,
                "description": "Dense sea ice pack (>60%) exceeding standard navigational clearance.",
            }
        })

    # Moderate Ice Risk
    for i, j in mod_ice_cells[::2]:
        ring = _cell_to_geojson_polygon(grid, i, j)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
            "properties": {
                "zone_type": "Moderate Ice Risk",
                "severity": "warning",
                "color": "#eab308",
                "fill_color": "rgba(234, 179, 8, 0.25)",
                "risk_score": 50.0,
                "description": "Marginal sea ice concentration (25-60%). Navigation caution advised.",
            }
        })

    # Iceberg Drift Hazard
    for i, j in iceberg_cells:
        ring = _cell_to_geojson_polygon(grid, i, j)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
            "properties": {
                "zone_type": "Iceberg Drift Hazard",
                "severity": "danger",
                "color": "#f97316",
                "fill_color": "rgba(249, 115, 22, 0.40)",
                "risk_score": 90.0,
                "description": "Active drifting tabular iceberg corridor and radar clutter zone.",
            }
        })

    return {
        "type": "FeatureCollection",
        "metadata": {
            "source": "Antarctic Navigation Decision Support System",
            "total_risk_zones": len(features),
            "is_demonstration": True,
            "disclaimer": "Simulated spatial risk classification for decision support testing.",
        },
        "features": features,
    }
