import numpy as np
import pytest
import xarray as xr

from ricemap.features import BANDS, scene_features, speckle_filter


def make_stack(ny=21, nx=21, nt=3, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.gamma(shape=2.0, scale=0.05, size=(nt, 2, ny, nx)).astype("float32")
    return xr.DataArray(
        data,
        dims=("time", "band", "y", "x"),
        coords={"band": ["vh", "vv"]},
    )


def test_speckle_filter_keeps_shape_and_a_flat_image():
    flat = make_stack().copy(data=np.full((3, 2, 21, 21), 0.1, dtype="float32"))
    out = speckle_filter(flat, 9)
    assert out.shape == flat.shape
    assert np.allclose(out.values, 0.1)


def test_speckle_filter_reduces_noise():
    da = make_stack()
    out = speckle_filter(da, 9)
    assert out.values.std() < da.values.std()


def test_speckle_filter_centre_pixel_matches_a_larger_array():
    # A 9x9 patch around a point must give the same centre value as the same pixel
    # inside a bigger tile. This is what lets training points and tiles share features.
    big = make_stack()
    patch = big.isel(y=slice(6, 15), x=slice(6, 15))
    from_patch = speckle_filter(patch, 9).isel(y=4, x=4)
    from_big = speckle_filter(big, 9).isel(y=10, x=10)
    np.testing.assert_array_equal(from_patch.values, from_big.values)


@pytest.mark.parametrize("window", [0, 8, -3])
def test_speckle_filter_needs_an_odd_window(window):
    with pytest.raises(ValueError):
        speckle_filter(make_stack(), window)


def test_scene_features_bands_and_values():
    da = make_stack().copy(data=np.zeros((3, 2, 21, 21), dtype="float32"))
    da.loc[dict(band="vh")] = 0.03
    da.loc[dict(band="vv")] = 0.09
    out = scene_features(da)
    assert list(out.band.values) == BANDS
    assert out.dims == ("time", "band", "y", "x")
    np.testing.assert_allclose(out.sel(band="vh_vv_ratio").values, 0.03 / 0.09, rtol=1e-5)
    np.testing.assert_allclose(out.sel(band="rvi").values, 4 * 0.03 / 0.12, rtol=1e-5)


def test_scene_features_uses_band_names_not_positions():
    # The same data stored in the opposite band order must give the same answer.
    da = make_stack()  # bands stored as [vh, vv]
    reordered = da.sel(band=["vv", "vh"])  # same values, stored as [vv, vh]
    a = scene_features(da).sel(band="rvi")
    b = scene_features(reordered).sel(band="rvi")
    np.testing.assert_allclose(a.values, b.values)


def test_ratio_is_nan_not_inf_when_vv_is_zero():
    da = make_stack()
    da.loc[dict(band="vv")] = 0.0
    out = scene_features(da)
    assert not np.isinf(out.sel(band="vh_vv_ratio").values).any()
