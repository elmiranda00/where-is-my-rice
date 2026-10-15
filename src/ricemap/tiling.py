"""
Turn the Area of Interest into a list of tile squares (with a Tile ID and X/Y bounds) in UTM meters
"""

import math
from dataclasses import dataclass

from pyproj import Transformer

from ricemap.config import AOI, Tiling


@dataclass(frozen=True)
class Tile:
    tile_id: str
    minx: int
    miny: int
    maxx: int
    maxy: int

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return (self.minx, self.miny, self.maxx, self.maxy)


def aoi_bounds_utm(aoi: AOI) -> tuple[float, float, float, float]:
    """Project the lon/lat bbox into UTM CRS and return its UTM envelope."""
    to_utm = Transformer.from_crs("EPSG:4326", aoi.crs, always_xy=True)

    # Add 21 points along each edge before projecting
    # (singe a lon/lat rectangle is not a rectangle in UTM) - avoids bulging edges
    return to_utm.transform_bounds(*aoi.bbox, densify_pts=21)


def make_tiles(aoi: AOI, tiling: Tiling) -> list[Tile]:
    """Cover the AOI with a regular grid of square tiles, sorted by (row, col)."""
    size = tiling.tile_size_m
    if size % tiling.resolution_m:
        raise ValueError(
            f"tile_size_m ({size}) must be a multiple of resolution_m ({tiling.resolution_m})"
        )

    minx, miny, maxx, maxy = aoi_bounds_utm(aoi)
    col0, col1 = math.floor(minx / size), math.ceil(maxx / size)
    row0, row1 = math.floor(miny / size), math.ceil(maxy / size)

    return [
        Tile(
            tile_id=f"x{col * size}_y{row * size}",
            minx=col * size,
            miny=row * size,
            maxx=(col + 1) * size,
            maxy=(row + 1) * size,
        )
        for row in range(row0, row1)
        for col in range(col0, col1)
    ]
