"""
Process Sentinel-1 data : Appply despeckling to deal with backscatter,
derive the VH/VV ratio and RVI used to separate rice from non-rice
"""

import numpy as np
import xarray as xr
from scipy.ndimage import median_filter

from ricemap.config import Config

BANDS = ["vv", "vh", "vh_vv_ratio", "rvi"]
STATS = ["mean", "std", "min", "max", "range"]
FEATURES = [f"{band}_{stat}" for band in BANDS for stat in STATS]

def speckle_filter(da: xr.DataArray, window: int) -> xr.DataArray:
    """Median-filter every scene and band over a window x window pixel neighbourhood.

    Expects a (time, band, y, x) array. Loads it into memory if it is lazy.
    """
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    filtered = median_filter(da.values, size=(1, 1, window, window), mode="nearest")
    return da.copy(data=filtered)


def scene_features(da: xr.DataArray) -> xr.DataArray:
    """Add the VH/VV ratio and RVI to a (time, band, y, x) VH/VV array.

    Returns a (time, band, y, x) array whose bands are BANDS, in that order.
    """
    vv = da.sel(band="vv", drop=True)
    vh = da.sel(band="vh", drop=True)
    ratio = vh / vv.where(vv > 0)
    rvi = 4 * vh / (vv + vh)
    out = xr.concat([vv, vh, ratio, rvi], dim="band")
    return out.assign_coords(band=BANDS).transpose("time", "band", "y", "x")


def composite(da: xr.DataArray, days: int, start: str) -> xr.DataArray:
    """Median of all scenes within each window (of duration defined by `days`, counted from `start`)

    Merges scenes from overlapping orbits or frames; windows with no data are dropped.
    The result include a "time" dimension, labelled by window number (0, 1, 2, ...).
    """
    window = (da.time - np.datetime64(start)) // np.timedelta64(days, "D")
    composites = da.groupby(window.rename("window")).median().rename(window="time")
    return composites.dropna("time", how="all")


def temporal_stats(da: xr.DataArray) -> xr.DataArray:
    """Collapse the time axis of a (time, band, y, x) array into per-pixel statistics.

    Returns a (feature, y, x) array whose features are FEATURES, e.g. "rvi_std".
    """
    layers = []
    for band in BANDS:
        series = da.sel(band=band, drop=True)
        low, high = series.min("time"), series.max("time")
        layers += [series.mean("time"), series.std("time", ddof=1), low, high, high - low]
    return xr.concat(layers, dim="feature").assign_coords(feature=FEATURES)


def compute_features(stack: xr.DataArray, cfg: Config) -> xr.DataArray:
    """
    Main feature engineering pipeline :
        Raw (time, band, y, x) VH/VV -> composites -> speckle filter -> ratio + RVI -> temporal
    Returns : (feature, y, x).
    """
    da = composite(stack.compute(), cfg.time.composite_days, cfg.time.start)
    da = speckle_filter(da, cfg.speckle_window)
    return temporal_stats(scene_features(da))