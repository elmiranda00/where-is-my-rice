"""
Classify every pixel of one tile as rice / non-rice using a saved model.
Write the result as a Cloud-optimized GeoTIFF (COG)
"""

import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from ricemap import storage
from ricemap.config import Config
from ricemap.features import compute_features
from ricemap.fetch import get_sentinel_area
from ricemap.tiling import Tile
from ricemap.train import MODEL_PATH

NODATA = 255


def load_model() -> dict:
    """Load the saved model: {"model": fitted pipeline, "features": feature names in order}."""
    with storage.open_file(storage.uri(*MODEL_PATH), "rb") as f:
        return joblib.load(f)


def tile_output(tile: Tile) -> str:
    return storage.uri("tiles", f"{tile.tile_id}.tif")


def classify(feats: xr.DataArray, bundle: dict) -> xr.DataArray:
    """(feature, y, x) layers -> (y, x) uint8 map: 1 rice, 0 non-rice, NODATA where any
    feature is missing."""

    # Put the features in the same order that the model used during training.
    names = bundle["features"]
    cube = feats.sel(feature=names).transpose("y", "x", "feature").values
    # Make a table: one row for each pixel within the tile, one column for each feature.
    flat = cube.reshape(-1, len(names))
    # Find the pixels that have all features. Do not predict the other pixels.
    valid = np.isfinite(flat).all(axis=1)
    out = np.full(len(flat), NODATA, dtype="uint8")
    if valid.any():
        out[valid] = bundle["model"].predict(pd.DataFrame(flat[valid], columns=names))
    # Change the table back to a map of the tile.
    return xr.DataArray(
        out.reshape(cube.shape[:2]), coords={"y": feats.y, "x": feats.x}, dims=("y", "x")
    )


def write_cog(da: xr.DataArray, crs: str, dest: str) -> None:
    """Write a single-band map to storage as a deflate-compressed COG with overviews."""
    # Add the coordinate system and the nodata value to the map.
    da = da.rio.write_crs(crs).rio.write_nodata(NODATA)
    with tempfile.TemporaryDirectory() as tmp:
        # GDAL can only make a COG in a local file.
        local = os.path.join(tmp, "tile.tif")
        da.rio.to_raster(local, driver="COG", compress="DEFLATE")
        # Copy the file to the storage location.
        storage.upload(local, dest)


def empty_tile(tile: Tile, cfg: Config) -> xr.DataArray:
    """To handle tiles with no data {(y, x) uint8 map of the tile with every pixel NODATA}"""
    res = cfg.tiling.resolution_m
    # Pixel centres, from the top-left corner (y goes down, as in the classified tiles).
    xs = np.arange(tile.minx + res / 2, tile.maxx, res)
    ys = np.arange(tile.maxy - res / 2, tile.miny, -res)
    data = np.full((len(ys), len(xs)), NODATA, dtype="uint8")
    return xr.DataArray(data, coords={"y": ys, "x": xs}, dims=("y", "x"))


def predict_tile(tile: Tile, cfg: Config, bundle: dict) -> str | None:
    """Classify one tile and write its COG. Returns the output path, or None if the output
    already existed (tile jobs are idempotent)."""
    # Stop if the output file is in storage already.
    dest = tile_output(tile)
    if storage.exists(dest):
        return None

    # Make the tile area larger on each side
    # (for the speckle filter which needs the neighbor pixels)
    pad = cfg.speckle_window // 2
    margin = pad * cfg.tiling.resolution_m
    buffered = (tile.minx - margin, tile.miny - margin, tile.maxx + margin, tile.maxy + margin)

    # Load the Sentinel-1 data and calculate the features.
    try:
        stack = get_sentinel_area(buffered, cfg)
    except LookupError:
        # No Sentinel-1 scenes (for example open sea): write an all-nodata tile.
        write_cog(empty_tile(tile, cfg), cfg.aoi.crs, dest)
        return dest

    feats = compute_features(stack, cfg)

    # Remove the extra pixels from each side to get the tile area again.
    feats = feats.isel(y=slice(pad, feats.sizes["y"] - pad), x=slice(pad, feats.sizes["x"] - pad))

    # Classify each pixel and save the map.
    write_cog(classify(feats, bundle), cfg.aoi.crs, dest)
    return dest
