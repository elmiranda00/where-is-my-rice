from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr

from ricemap import predict
from ricemap.config import load_config
from ricemap.tiling import Tile

CFG = load_config(Path(__file__).parents[1] / "config" / "mekong.yaml")


class ThresholdModel:
    """Stand-in for the SVM: rice where feature "a" > 0."""

    def predict(self, X):
        return (X["a"].to_numpy() > 0).astype(int)


def make_feats():
    data = np.array([[[1.0, -1.0], [np.nan, 2.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype="float32")
    coords = {"feature": ["a", "b"], "y": [1_100_015.0, 1_100_005.0], "x": [500_005.0, 500_015.0]}
    return xr.DataArray(data, dims=("feature", "y", "x"), coords=coords)


def test_classify_marks_rice_non_rice_and_nodata():
    out = predict.classify(make_feats(), {"model": ThresholdModel(), "features": ["a", "b"]})
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out.values, [[1, 0], [predict.NODATA, 1]])


def test_write_cog_sets_crs_nodata_and_compression(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_URI", str(tmp_path))
    out = predict.classify(make_feats(), {"model": ThresholdModel(), "features": ["a", "b"]})
    dest = str(tmp_path / "tiles" / "t.tif")
    predict.write_cog(out, "EPSG:32648", dest)
    with rasterio.open(dest) as src:
        assert src.crs.to_epsg() == 32648
        assert src.nodata == predict.NODATA
        assert src.profile["compress"] == "deflate"
        assert src.read(1).tolist() == [[1, 0], [predict.NODATA, 1]]


def test_predict_tile_skips_existing_output_without_fetching(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_URI", str(tmp_path))
    tile = Tile("x500000_y1100000", 500_000, 1_100_000, 510_000, 1_110_000)
    existing = tmp_path / "tiles" / f"{tile.tile_id}.tif"
    existing.parent.mkdir()
    existing.write_bytes(b"done")

    def must_not_fetch(*args, **kwargs):
        pytest.fail("predict_tile fetched data for a tile that was already done")

    monkeypatch.setattr(predict, "get_sentinel_area", must_not_fetch)
    assert predict.predict_tile(tile, CFG, bundle={}) is None
