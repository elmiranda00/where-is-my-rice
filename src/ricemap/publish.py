"""
Join the tile COGs into one mosaic, make web tiles and province statistics.
"""

import os
import posixpath
import shutil
import tempfile
import numpy as np

import rasterio
import rasterio.shutil
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from pmtiles.convert import mbtiles_to_pmtiles

from ricemap import storage
from ricemap.predict import NODATA
from ricemap.config import Config

MOSAIC_PATH = ("mosaic", "rice_mosaic.tif")
PMTILES_PATH = ("web", "rice.pmtiles")
WEB_CRS = "EPSG:3857"
TILE_SIZE = 256  # pixels on each side of a web tile


def download_tiles(tmp: str) -> list[str]:
    """Copy every tile COG from storage into a local folder. Returns the local paths."""
    paths = storage.list_files(storage.uri("tiles", "*.tif"))
    if not paths:
        raise FileNotFoundError("No tiles found in storage; run predict-tile first")
    local = []
    for p in paths:
        # GDAL reads local files faster and more reliably than remote ones.
        dst = os.path.join(tmp, posixpath.basename(p))
        with storage.open_file(p, "rb") as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out)
        local.append(dst)
    return local


def build_mosaic() -> str:
    """Join all tiles into one COG in the AOI UTM CRS. Returns the output path."""
    dest = storage.uri(*MOSAIC_PATH)
    with tempfile.TemporaryDirectory() as tmp:
        tiles = download_tiles(tmp)
        # Join the tiles. Pixels that no tile covers get NODATA.
        data, transform = merge(tiles, nodata=NODATA)
        with rasterio.open(tiles[0]) as first:
            profile = first.profile
        profile.update(
            driver="GTiff", height=data.shape[1], width=data.shape[2], transform=transform,
            nodata=NODATA, compress="DEFLATE", tiled=True, blockxsize=512, blockysize=512,
        )
        plain = os.path.join(tmp, "mosaic_plain.tif")
        with rasterio.open(plain, "w", **profile) as out:
            out.write(data)
        # GDAL makes a COG only by copying from an existing file.
        cog = os.path.join(tmp, "mosaic.tif")
        rasterio.shutil.copy(plain, cog, driver="COG", compress="DEFLATE")
        storage.upload(cog, dest)
    return dest


def to_web_mercator(src_path: str, dst_path: str) -> None:
    """Reproject the class map to Web Mercator. Nearest keeps the values 0, 1 and NODATA."""
    with rasterio.open(src_path) as src:
        with WarpedVRT(src, crs=WEB_CRS, resampling=Resampling.nearest, nodata=NODATA) as vrt:
            rasterio.shutil.copy(vrt, dst_path, driver="GTiff", tiled=True, compress="DEFLATE")


def colour_rgba(src_path: str, dst_path: str, colour: tuple[int, int, int]) -> None:
    """Class map -> 4-band RGBA image: rice in `colour`, all other pixels transparent."""
    with rasterio.open(src_path) as src:
        profile = src.profile
        profile.update(count=4, nodata=None, photometric="RGB", alpha="YES")
        with rasterio.open(dst_path, "w", **profile) as dst:
            # Work block by block, so the full Delta does not have to fit in memory.
            for _, win in src.block_windows(1):
                cls = src.read(1, window=win)
                out = np.zeros((4, *cls.shape), dtype="uint8")
                rice = cls == 1
                for band, value in enumerate((*colour, 255)):
                    out[band][rice] = value
                dst.write(out, window=win)


def build_pmtiles(cfg: Config) -> str:
    """Make the web map tiles from the mosaic, as one PMTiles file. Returns the output path."""
    dest = storage.uri(*PMTILES_PATH)
    with tempfile.TemporaryDirectory() as tmp:
        mosaic = os.path.join(tmp, "mosaic.tif")
        with storage.open_file(storage.uri(*MOSAIC_PATH), "rb") as src, open(mosaic, "wb") as out:
            shutil.copyfileobj(src, out)
        merc = os.path.join(tmp, "merc.tif")
        rgba = os.path.join(tmp, "rgba.tif")
        mbt = os.path.join(tmp, "rice.mbtiles")
        pmt = os.path.join(tmp, "rice.pmtiles")

        to_web_mercator(mosaic, merc)
        colour_rgba(merc, rgba, cfg.publish.rice_colour)
        # GDAL writes the most detailed zoom level (14 for 10 m pixels) by copying.
        rasterio.shutil.copy(rgba, mbt, driver="MBTiles", TILE_FORMAT="PNG", RESAMPLING="NEAREST")
        # Add the less detailed zoom levels, until the whole map fits in about one tile.
        with rasterio.open(mbt, "r+") as m:
            factors = [2**k for k in range(1, 20) if max(m.width, m.height) / 2**k >= TILE_SIZE]
            m.build_overviews(factors, Resampling.nearest)
        # 0 = no maximum zoom limit (my test kept zoom 9 to 14).
        mbtiles_to_pmtiles(mbt, pmt, 0)
        storage.upload(pmt, dest)
    return dest
