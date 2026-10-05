"""
Process Sentinel-1 data : Appply despeckling to deal with backscatter,
derive the VH/VV ratio and RVI used to separate rice from non-rice
"""

import xarray as xr
from scipy.ndimage import median_filter

BANDS = ["vv", "vh", "vh_vv_ratio", "rvi"]

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
