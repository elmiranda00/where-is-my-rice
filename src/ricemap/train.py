"""
build the SVM pipeline with spatial cross-validation
""" 

import json
import math

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from ricemap import storage
from ricemap.config import Config
from ricemap.features import FEATURES, compute_features
from ricemap.fetch import get_sentinel_area, parse_latlong

TABLE_PATH = ("training", "training_table.csv")
MODEL_PATH = ("model", "model.joblib")
METRICS_PATH = ("model", "cv_metrics.json")


# Build Feature Table
# -------------------------------------------------------------------

def load_labels(cfg: Config) -> pd.DataFrame:
    """Read the labelled points: lat/lon, UTM x/y, and label (1 = rice, 0 = non-rice)."""
    
    df = pd.read_csv(cfg.labels)
    latlon = df["Latitude and Longitude"].map(parse_latlong)
    
    out = pd.DataFrame({"lat": [p[0] for p in latlon], "lon": [p[1] for p in latlon]})
    to_utm = Transformer.from_crs("EPSG:4326", cfg.aoi.crs, always_xy=True)
    out["x"], out["y"] = to_utm.transform(out["lon"].to_numpy(), out["lat"].to_numpy())
    out["label"] = (df["Class of Land"] == "Rice").astype(int)
    
    return out # returns a DataFrame with [lat, lon, x, y , label] where label = 1 / 0


def build_training_table(cfg: Config) -> pd.DataFrame:
    """
    Build the feature row for each labelled point. Points are grouped by tile. 
    Each group loads one window around its points (plus the speckle buffer), 
    runs the same compute_features as prediction, and reads the pixel under each point.
    """
    # 1. Put each point in a tile. One download then serves all the points in that tile.
    labels = load_labels(cfg)
    res, size = cfg.tiling.resolution_m, cfg.tiling.tile_size_m
    pad = (cfg.speckle_window // 2) * res
    labels["tile_id"] = [
        f"x{int(x // size) * size}_y{int(y // size) * size}"
        for x, y in zip(labels["x"], labels["y"], strict=True)
    ]

    parts = []
    for tile_id, pts in labels.groupby("tile_id"):
        # 2. Load one area that contains all the points of the tile, plus the speckle buffer.
        #    Calculate the features with the same function that the prediction uses.
        bounds = (
            math.floor(pts["x"].min() / res) * res - pad,
            math.floor(pts["y"].min() / res) * res - pad,
            math.floor(pts["x"].max() / res) * res + res + pad,
            math.floor(pts["y"].max() / res) * res + res + pad,
        )
        print(f"{tile_id}: {len(pts)} points", flush=True)
        feats = compute_features(get_sentinel_area(bounds, cfg), cfg) # returns (feature, y, x)
        # 3. Read the features of the pixel under each point. Add them to the point rows.
        at_points = feats.sel(
            x=xr.DataArray(pts["x"].to_numpy(), dims="point"),
            y=xr.DataArray(pts["y"].to_numpy(), dims="point"),
            method="nearest",
        )
        values = pd.DataFrame(at_points.transpose("point", "feature").values, columns=FEATURES)
        parts.append(pd.concat([pts.reset_index(drop=True), values], axis=1))
    
    return pd.concat(parts, ignore_index=True)


# Spatial CV and Training
# ---------------------------------------------------------------

def site_groups(x: np.ndarray, y: np.ndarray, gap_m: float) -> np.ndarray:
    """
    Assign Site ID per (x,y) point: points are grouped to the same site if gap between them is less then gap_m

    Returns:
        np.ndarray of ints, one per input point (same order as x, y). Points sharing a value
        belong to the same site; IDs are arbitrary labels starting at 1, not distances.
    """
    return fcluster(linkage(np.c_[x, y], method="single"), t=gap_m, criterion="distance")


def block_groups(x: np.ndarray, y: np.ndarray, block_m: float) -> np.ndarray:
    """
    Assign Block ID per (x,y) point: which block_m x block_m square of the UTM grid it falls in.
    (note: Ideally in the config, block_m x block_m -> 2x2 km block within the UTM grid)

    Returns:
        np.ndarray of strings, one per input point (same order as x, y), formatted
        "<column>_<row>" e.g. "260_575". Points sharing a value fall in the same grid square.
    """
    return np.array([f"{int(a // block_m)}_{int(b // block_m)}" for a, b in zip(x, y, strict=True)])


def make_search(cfg: Config) -> GridSearchCV:
    """Scaled RBF SVM, tuned over the YAML param_grid with block-grouped CV."""
    # Classification pipeline: scale features with RobustScaler(), then an RBF-kernel SVM
    pipe = Pipeline([("scaler", RobustScaler()),
                     ("classifier", SVC(kernel="rbf"))])
    # Prefix each YAML param_grid key with "classifier__" so it targets the SVC step
    grid = {f"classifier__{name}": values for name, values in cfg.model.param_grid.items()}

    # Tune using 2 km-block-grouped folds, so nearby points never split across train/test
    cv = StratifiedGroupKFold(n_splits=cfg.model.cv_folds)

    return GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1)


def cross_validate_spatial(table: pd.DataFrame, cfg: Config) -> dict:
    """Score the whole tune-and-fit procedure on sites it never saw (nested, site-grouped CV)."""
    X, y = table[FEATURES], table["label"].to_numpy()
    x_m, y_m = table["x"].to_numpy(), table["y"].to_numpy()
    # Sites = natural geographic clusters (~3 km gaps).
    sites = site_groups(x_m, y_m, cfg.model.cv_site_gap_m)
    # Blocks = a uniform 2 km grid.
    blocks = block_groups(x_m, y_m, cfg.model.cv_block_size_m)
    outer = GroupKFold(n_splits=min(cfg.model.cv_folds, len(np.unique(sites))))

    # Every point gets exactly one out-of-site prediction by the end of the loop below.
    pred = np.zeros(len(y), dtype=int)
    score = np.zeros(len(y))

    # Outer CV: hold out whole sites, so the test fold is genuinely unseen terrain
    for train_idx, test_idx in outer.split(X, y, groups=sites):
        if len(np.unique(y[train_idx])) < 2:
            raise ValueError("A training fold has only one class; add sites or lower cv_folds")
        # Inner CV : re-tune C/gamma from scratch using only this fold's training sites, grouped by block so nearby points don't leak across train/test
        search = make_search(cfg).fit(X.iloc[train_idx], y[train_idx], groups=blocks[train_idx])
        pred[test_idx] = search.predict(X.iloc[test_idx])
        # decision_function -> gives a continuous score (signed distance from the boundary),
        # needed for roc_auc
        score[test_idx] = search.decision_function(X.iloc[test_idx])

    return {
        "n_points": int(len(y)),
        "n_sites": int(len(np.unique(sites))),
        "outer_folds": outer.n_splits,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, score),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "per_site": {
            f"site_{s}": {
                "class": "rice" if y[sites == s][0] == 1 else "non-rice",
                "n": int((sites == s).sum()),
                "accuracy": float((pred[sites == s] == y[sites == s]).mean()),
            }
            for s in np.unique(sites)
        },
    }


def fit_final(table: pd.DataFrame, cfg: Config) -> tuple[Pipeline, dict]:
    """Tune on all points; return the best pipeline and its parameters."""
    # This is the model that gets shipped, so it trains on every labelled point at once.
    # No outer/site split here: nothing is being scored, only fit, so no points are held out.
    blocks = block_groups(table["x"].to_numpy(), table["y"].to_numpy(), cfg.model.cv_block_size_m)
    search = make_search(cfg).fit(table[FEATURES], table["label"], groups=blocks)
    return search.best_estimator_, search.best_params_


# Storage
# -----------------------------------------------------------------------------------

def save_table(table: pd.DataFrame) -> None:
    with storage.open_file(storage.uri(*TABLE_PATH), "wb") as f:
        f.write(table.to_csv(index=False).encode())

def load_table() -> pd.DataFrame:
    with storage.open_file(storage.uri(*TABLE_PATH), "rb") as f:
        return pd.read_csv(f)

def save_model(model: Pipeline, metrics: dict) -> None:
    with storage.open_file(storage.uri(*MODEL_PATH), "wb") as f:
        joblib.dump({"model": model, "features": FEATURES}, f)
    with storage.open_file(storage.uri(*METRICS_PATH), "wb") as f:
        f.write(json.dumps(metrics, indent=2).encode())

def run_training(cfg: Config, rebuild_table: bool = False) -> dict:
    """Build (or reuse) the training table, cross-validate, fit the final model and save it."""
    if rebuild_table or not storage.exists(storage.uri(*TABLE_PATH)):
        save_table(build_training_table(cfg))
    table = load_table()
    n_missing = int(table[FEATURES].isna().any(axis=1).sum())
    table = table.dropna(subset=FEATURES)

    metrics = cross_validate_spatial(table, cfg)
    model, best_params = fit_final(table, cfg)
    metrics.update(dropped_points_with_missing_features=n_missing, best_params=best_params)
    save_model(model, metrics)
    return metrics
