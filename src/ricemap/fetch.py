
"""Search and Load Planetary Computer's STAC catalog for Sentinel-1 RTC scenes"""

import math

import planetary_computer
import pystac_client
import xarray as xr
import odc.stac
from odc.geo.geobox import GeoBox
from pyproj import Transformer

from ricemap.config import Config, Stac, TimeRange


def parse_latlong(latlong: str) -> tuple[float, float]:
    """Parse a "(latitude, longitude)" string into floats, with range checks."""
    parts = [p.strip() for p in latlong.strip().strip("()").split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected (lat, long) format, got {latlong}")
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError as e:
        raise ValueError(f"Non-numeric characters within {latlong}") from e
    if not -90 <= lat <= 90:
        raise ValueError(f"Latitude out of range [-90, 90]: {lat}")
    if not -180 <= lon <= 180:
        raise ValueError(f"Longitude out of range [-180, 180]: {lon}")
    return lat, lon


def point_window(
    lat: float, lon: float, crs: str, resolution: int, window_size: int
) -> tuple[int, int, int, int]:
    """Finds the pixel containing a point and returns its UTM bounds (minx, miny, maxx, maxy)
    with dimensions window_size x window_size, centred on the pixel that contains the point"""
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = to_utm.transform(lon, lat)
    x0 = math.floor(x / resolution) * resolution  # west edge of the point's pixel
    y0 = math.floor(y / resolution) * resolution  # south edge of the point's pixel
    half = window_size // 2
    return (
        x0 - half * resolution,
        y0 - half * resolution,
        x0 + (half + 1) * resolution,
        y0 + (half + 1) * resolution,
    )


def search_items(stac: Stac, geometry: dict, time: TimeRange):
    """Find ascending Sentinel-1 RTC scenes that intersect a GeoJSON geometry."""
    catalog = pystac_client.Client.open(stac.url, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[stac.collection],
        intersects=geometry,
        datetime=f"{time.start}/{time.end}",
        query={"sat:orbit_state": {"eq": "ascending"}},
    )
    return search.item_collection()


def load_stack(
    items, bounds: tuple[int, int, int, int], crs: str, resolution: int
) -> xr.DataArray:
    """Returns an array (time, band, y, x) of VH and VV, clipped to bounds, in the given CRS."""

    # GeoBox is the grid definition (90m x 90m window)
    geobox = GeoBox.from_bbox(bounds, crs, resolution=resolution)
    # Take the grid above and pin it to the pixels. chunk{} => keeps load lazy
    ds = odc.stac.load(items, bands=["vh", "vv"], geobox=geobox, chunks={}, dtype="float32")
    
    return ds.to_dataarray("band").transpose("time", "band", "y", "x")


def get_sentinel_data(latlong: str, cfg: Config, window_size: int | None = None) -> xr.DataArray:
    """Main fetch function.
    Gets Sentinel-1 VH/VV patch around a point, one slice per scene in cfg.time.
    Nothing is downloaded until .compute() is called on the result.
    """
    lat, lon = parse_latlong(latlong)
    size = cfg.speckle_window if window_size is None else window_size
    bounds = point_window(lat, lon, cfg.aoi.crs, cfg.tiling.resolution_m, size)

    point = {"type": "Point", "coordinates": [lon, lat]}
    items = search_items(cfg.stac, point, cfg.time)
    if len(items) == 0:
        raise LookupError(
            f"No Sentinel-1 scenes found for {latlong} in {cfg.time.start}..{cfg.time.end}"
            )
    
    return load_stack(items, bounds, cfg.aoi.crs, cfg.tiling.resolution_m)
