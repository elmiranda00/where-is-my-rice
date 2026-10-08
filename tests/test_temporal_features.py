from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ricemap.config import load_config
from ricemap.features import FEATURES, composite, compute_features, temporal_stats

CFG = load_config(Path(__file__).parents[1] / "config" / "mekong.yaml")


def make_series(times, ny=21, nx=21, seed=0):
    """Random (time, band, y, x) VH/VV stack with real timestamps."""
    rng = np.random.default_rng(seed)
    data = rng.gamma(shape=2.0, scale=0.05, size=(len(times), 2, ny, nx)).astype("float32")
    return xr.DataArray(
        data,
        dims=("time", "band", "y", "x"),
        coords={"time": pd.to_datetime(times), "band": ["vh", "vv"]},
    )


def test_composite_takes_the_median_within_each_window_and_drops_empty_ones():
    # Two scenes in the first 12-day window, none in the second, one in the third.
    da = make_series(["2020-01-02", "2020-01-05", "2020-01-30"])
    out = composite(da, days=12, start="2020-01-01")
    assert out.sizes["time"] == 2
    expected_first = da.isel(time=[0, 1]).median("time")
    np.testing.assert_allclose(out.isel(time=0).values, expected_first.values)


def test_composite_windows_are_counted_from_the_config_start_date():
    # Day 9 and day 13 after the start fall in windows 0 and 1, so they must not be merged,
    # even though they are only 4 days apart.
    da = make_series(["2020-01-10", "2020-01-14"])
    assert composite(da, days=12, start="2020-01-01").sizes["time"] == 2


def test_temporal_stats_names_and_values():
    times = ["2020-01-01", "2020-01-13", "2020-01-25"]
    da = make_series(times, ny=1, nx=1)
    da.loc[dict(band="vv")] = np.array([0.1, 0.2, 0.6]).reshape(3, 1, 1)
    full = da.reindex(band=["vv", "vh", "vh_vv_ratio", "rvi"], fill_value=0.5)
    out = temporal_stats(full)

    assert list(out.feature.values) == FEATURES
    assert out.dims == ("feature", "y", "x")
    vv = out.sel(feature=["vv_mean", "vv_min", "vv_max", "vv_range"]).values.ravel()
    np.testing.assert_allclose(vv, [0.3, 0.1, 0.6, 0.5], rtol=1e-5)
    np.testing.assert_allclose(out.sel(feature="vv_std").item(), np.std([0.1, 0.2, 0.6], ddof=1))


def test_compute_features_gives_a_point_patch_the_same_values_as_a_tile_pixel():
    # The training table reads pixels from small windows; prediction reads them from buffered
    # tiles. Any pixel with a full speckle neighbourhood must get identical features either way.
    times = pd.date_range("2020-01-01", periods=8, freq="12D")
    big = make_series(times)
    patch = big.isel(y=slice(6, 15), x=slice(6, 15))
    from_patch = compute_features(patch, CFG).isel(y=4, x=4)
    from_big = compute_features(big, CFG).isel(y=10, x=10)
    np.testing.assert_allclose(from_patch.values, from_big.values, rtol=1e-6)
