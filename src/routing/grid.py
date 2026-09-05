"""
Antarctic Polar Stereographic Navigation Grid.

Provides a 2D discrete spatial graph for polar maritime routing
using the project's native CoordinateHandler (EPSG:3412 / EPSG:3031).
"""

from typing import Dict, List, Optional, Set, Tuple
import math
import numpy as np

from src.models.iceberg_physics import CoordinateHandler


class PolarNavigationGrid:
    """
    Planar discrete graph representation of an Antarctic sea area
    projected in Polar Stereographic space.
    """

    def __init__(
        self,
        min_lat: float = -75.0,
        max_lat: float = -60.0,
        min_lon: float = -65.0,
        max_lon: float = -45.0,
        resolution_km: float = 20.0,
        crs: str = "EPSG:3412",
    ) -> None:
        """
        Initialize the navigation grid.

        Args:
            min_lat: Southernmost latitude bound.
            max_lat: Northernmost latitude bound.
            min_lon: Westernmost longitude bound.
            max_lon: Easternmost longitude bound.
            resolution_km: Grid step spacing in kilometers.
            crs: Coordinate Reference System string.
        """
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.resolution_km = float(resolution_km)
        self.step_m = self.resolution_km * 1000.0
        self.crs = crs
        self.coord_handler = CoordinateHandler(crs=self.crs)

        # Compute projected bounding box
        corners = [
            (min_lon, min_lat),
            (min_lon, max_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            ((min_lon + max_lon) / 2.0, min_lat),
            ((min_lon + max_lon) / 2.0, max_lat),
        ]
        xs = []
        ys = []
        for lon, lat in corners:
            x, y = self.coord_handler.to_projected(lon, lat)
            xs.append(x)
            ys.append(y)

        self.x_min = min(xs)
        self.x_max = max(xs)
        self.y_min = min(ys)
        self.y_max = max(ys)

        self.nx = max(3, int(math.ceil((self.x_max - self.x_min) / self.step_m)) + 1)
        self.ny = max(3, int(math.ceil((self.y_max - self.y_min) / self.step_m)) + 1)

        # Non-traversable cell set
        self.blocked_cells: Set[Tuple[int, int]] = set()

    def in_bounds(self, i: int, j: int) -> bool:
        """Check if grid index (i, j) is inside grid boundaries."""
        return 0 <= i < self.nx and 0 <= j < self.ny

    def is_traversable(self, i: int, j: int) -> bool:
        """Check if cell is inside bounds and not marked as obstacle."""
        if not self.in_bounds(i, j):
            return False
        return (i, j) not in self.blocked_cells

    def set_blocked(self, i: int, j: int, blocked: bool = True) -> None:
        """Mark a cell as blocked or traversable."""
        if self.in_bounds(i, j):
            if blocked:
                self.blocked_cells.add((i, j))
            else:
                self.blocked_cells.discard((i, j))

    def grid_to_projected(self, i: int, j: int) -> Tuple[float, float]:
        """Convert grid index (i, j) to projected coordinates (x_m, y_m)."""
        x = self.x_min + i * self.step_m
        y = self.y_min + j * self.step_m
        return float(x), float(y)

    def projected_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert projected coordinates (x_m, y_m) to nearest grid index (i, j)."""
        i = int(round((x - self.x_min) / self.step_m))
        j = int(round((y - self.y_min) / self.step_m))
        i = max(0, min(self.nx - 1, i))
        j = max(0, min(self.ny - 1, j))
        return i, j

    def grid_to_geo(self, i: int, j: int) -> Tuple[float, float]:
        """Convert grid index (i, j) to geographic coordinates (latitude, longitude)."""
        x, y = self.grid_to_projected(i, j)
        lon, lat = self.coord_handler.to_geographic(x, y)
        return float(lat), float(lon)

    def geo_to_grid(self, latitude: float, longitude: float) -> Tuple[int, int]:
        """Convert geographic coordinates (latitude, longitude) to nearest grid index (i, j)."""
        x, y = self.coord_handler.to_projected(longitude, latitude)
        return self.projected_to_grid(x, y)

    def get_neighbors(
        self, i: int, j: int, allow_diagonal: bool = True
    ) -> List[Tuple[int, int, float]]:
        """
        Get valid, traversable neighbor cells and step distances.

        Returns:
            List of tuples (neighbor_i, neighbor_j, step_distance_km)
        """
        neighbors: List[Tuple[int, int, float]] = []
        sqrt2 = math.sqrt(2.0)

        # 4-connected (cardinal)
        cardinals = [
            (i + 1, j, self.resolution_km),
            (i - 1, j, self.resolution_km),
            (i, j + 1, self.resolution_km),
            (i, j - 1, self.resolution_km),
        ]
        for ni, nj, dist in cardinals:
            if self.is_traversable(ni, nj):
                neighbors.append((ni, nj, dist))

        # 8-connected (diagonal)
        if allow_diagonal:
            diagonals = [
                (i + 1, j + 1, self.resolution_km * sqrt2),
                (i + 1, j - 1, self.resolution_km * sqrt2),
                (i - 1, j + 1, self.resolution_km * sqrt2),
                (i - 1, j - 1, self.resolution_km * sqrt2),
            ]
            for ni, nj, dist in diagonals:
                if self.is_traversable(ni, nj):
                    neighbors.append((ni, nj, dist))

        return neighbors
