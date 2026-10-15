"""
Join the tile COGs into one mosaic, make web tiles and province statistics.
"""

import os
import posixpath
import shutil
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.shutil
from pmtiles.convert import mbtiles_to_pmtiles
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.mask import mask as clip_to_shapes
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT

from ricemap import storage
from ricemap.config import Config
from ricemap.predict import NODATA

MOSAIC_PATH = ("mosaic", "rice_mosaic.tif")
PMTILES_PATH = ("web", "rice.pmtiles")
STATS_PATH = ("stats", "provinces.csv")
WEB_CRS = "EPSG:3857"
TILE_SIZE = 256  # pixels on each side of a web tile
RICE_COLUMNS = [
    "single_rice_ha",
    "double_rice_ha",
    "triple_rice_ha",
]  # from the .csv in  reference/


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


def name_key(name: str) -> str:
    """Province name without spaces: 'Long An' (reference CSV) matches 'LongAn' (GADM)."""
    return name.replace(" ", "")


def load_reference(cfg: Config) -> pd.DataFrame:
    """Reference rice land per province, with a `key` column for joining."""
    # utf-8-sig: the CSV starts with a byte-order mark (BOM).
    ref = pd.read_csv(cfg.publish.reference, encoding="utf-8-sig")
    ref["key"] = ref["province"].map(name_key)
    return ref


def load_provinces(cfg: Config) -> gpd.GeoDataFrame:
    """Boundaries of the provinces in the reference CSV, in the AOI UTM CRS."""
    keys = set(load_reference(cfg)["key"])
    gdf = gpd.read_file(cfg.publish.provinces)
    gdf = gdf[gdf["NAME_1"].isin(keys)]
    missing = keys - set(gdf["NAME_1"])
    if missing:
        raise ValueError(f"Provinces not found in the boundary file: {sorted(missing)}")
    return gdf.rename(columns={"NAME_1": "key"})[["key", "geometry"]].to_crs(cfg.aoi.crs)


def build_mosaic(cfg: Config) -> str:
    """Join all tiles into one COG in the AOI UTM CRS. Returns the output path."""
    dest = storage.uri(*MOSAIC_PATH)
    with tempfile.TemporaryDirectory() as tmp:
        tiles = download_tiles(tmp)
        # Join the tiles. Pixels that no tile covers get NODATA.
        data, transform = merge(tiles, nodata=NODATA)

        # Set pixels outsie the study area to NODATA
        outside = geometry_mask(
            load_provinces(cfg).geometry, out_shape=data.shape[1:], transform=transform
        )
        data[0][outside] = NODATA

        with rasterio.open(tiles[0]) as first:
            profile = first.profile
        profile.update(
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
            nodata=NODATA,
            compress="DEFLATE",
            tiled=True,
            blockxsize=512,
            blockysize=512,
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


def province_stats(cfg: Config) -> str:
    """Rice area per province from the UTM mosaic, next to the reference figures.
    Writes a CSV to storage and returns its path."""
    dest = storage.uri(*STATS_PATH)
    pixel_ha = cfg.tiling.resolution_m**2 / 10_000
    provinces = load_provinces(cfg)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        mosaic = os.path.join(tmp, "mosaic.tif")
        with storage.open_file(storage.uri(*MOSAIC_PATH), "rb") as src, open(mosaic, "wb") as out:
            shutil.copyfileobj(src, out)
        with rasterio.open(mosaic) as src:
            for key, geom in zip(provinces["key"], provinces.geometry, strict=True):
                # Read only the pixels inside the province. Pixels outside it become NODATA.
                data, _ = clip_to_shapes(src, [geom], crop=True, nodata=NODATA)
                band = data[0]
                rows.append(
                    {
                        "key": key,
                        "province_ha": geom.area / 10_000,
                        "mapped_ha": int(((band == 0) | (band == 1)).sum()) * pixel_ha,
                        "rice_map_ha": int((band == 1).sum()) * pixel_ha,
                    }
                )

    ref = load_reference(cfg)
    # Rice land counts each field once: single + double + triple. Empty cells count as 0.
    # Update: Rice-shrimp land is left out of the comparison.
    ref["ref_rice_ha"] = ref[RICE_COLUMNS].sum(axis=1)
    ref = ref.drop(columns="rice_shrimp_ha")
    out = ref.merge(pd.DataFrame(rows), on="key").drop(columns="key")

    total = out.sum(numeric_only=True)
    total["province"] = "Mekong Delta (total)"
    out = pd.concat([out, total.to_frame().T], ignore_index=True)

    # Calculate Share of the province that the map covers
    # (will be <1 where islands are outside the grid)
    out["coverage"] = out["mapped_ha"] / out["province_ha"]

    # out["map_vs_ref"] = out["rice_map_ha"] / out["ref_rice_ha"]
    # out["map_vs_ref_with_shrimp"] = out["rice_map_ha"] / out["ref_rice_with_shrimp_ha"]

    # Calculate Map vs reference in percent: negative means the map finds less rice than the report.
    out["diff_pct"] = (out["rice_map_ha"] / out["ref_rice_ha"] - 1) * 100

    with storage.open_file(dest, "wb") as f:
        out.round(3).to_csv(f, index=False, encoding="utf-8")
    return dest
