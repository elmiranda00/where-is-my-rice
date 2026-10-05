import pytest

from ricemap.config import AOI, Tiling
from ricemap.tiling import aoi_bounds_utm, make_tiles

AOI_SMALL = AOI(bbox=(105.0, 10.0, 105.2, 10.2), crs="EPSG:32648")
TILING = Tiling(tile_size_m=10_000, resolution_m=10)


def test_tiles_are_square_and_grid_aligned():
    for t in make_tiles(AOI_SMALL, TILING):
        assert t.maxx - t.minx == 10_000
        assert t.maxy - t.miny == 10_000
        assert t.minx % 10_000 == 0
        assert t.miny % 10_000 == 0


def test_tiles_cover_the_aoi():
    tiles = make_tiles(AOI_SMALL, TILING)
    minx, miny, maxx, maxy = aoi_bounds_utm(AOI_SMALL)
    assert min(t.minx for t in tiles) <= minx
    assert min(t.miny for t in tiles) <= miny
    assert max(t.maxx for t in tiles) >= maxx
    assert max(t.maxy for t in tiles) >= maxy


def test_ids_are_unique_and_ordered_by_row_then_column():
    tiles = make_tiles(AOI_SMALL, TILING)
    ids = [t.tile_id for t in tiles]
    assert len(set(ids)) == len(ids)
    keys = [(t.miny, t.minx) for t in tiles]
    assert keys == sorted(keys)


def test_resolution_must_divide_tile_size():
    with pytest.raises(ValueError):
        make_tiles(AOI_SMALL, Tiling(tile_size_m=10_000, resolution_m=7))
