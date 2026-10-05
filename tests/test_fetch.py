import pytest
from pyproj import Transformer

from ricemap.fetch import parse_latlong, point_window

LATLONG = "(10.323727047081501, 105.2516346045924)"

def test_parse_latlong():
    assert parse_latlong(LATLONG) == (10.323727047081501, 105.2516346045924)


@pytest.mark.parametrize("bad", ["(10.3)", "(a, b)", "(95, 105)", "(10, 200)"])
def test_parse_latlong_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_latlong(bad)


def test_point_window_is_a_pixel_aligned_square_around_the_point():
    lat, lon = parse_latlong(LATLONG)
    minx, miny, maxx, maxy = point_window(lat, lon, "EPSG:32648", resolution=10, window_size=9)
    assert maxx - minx == 90
    assert maxy - miny == 90
    assert minx % 10 == 0
    assert miny % 10 == 0

    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True).transform(lon, lat)
    assert minx + 40 <= x < minx + 50  # the point sits in the centre (5th) pixel
    assert miny + 40 <= y < miny + 50


def test_point_window_needs_an_odd_size():
    with pytest.raises(ValueError):
        point_window(10.3, 105.2, "EPSG:32648", resolution=10, window_size=8)
