from pathlib import Path

import numpy as np
import pandas as pd

from ricemap.config import load_config
from ricemap.features import FEATURES
from ricemap.train import block_groups, cross_validate_spatial, fit_final, site_groups

CFG = load_config(Path(__file__).parents[1] / "config" / "mekong.yaml")


def make_table(seed=0):
    """Four sites 50 km apart (two rice, two non-rice), 40 points each spread over ~4 km.

    Rice rows get shifted VH and RVI statistics so the classes are learnable.
    """
    rng = np.random.default_rng(seed)
    signal = [f for f in FEATURES if f.startswith(("vh_", "rvi_"))]
    rows = []
    for i, (cx, label) in enumerate([(500_000, 1), (550_000, 1), (600_000, 0), (650_000, 0)]):
        n = 40
        site = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
        site[signal] += 3 * label
        site["x"] = cx + rng.uniform(0, 4000, n)
        site["y"] = 1_100_000 + rng.uniform(0, 4000, n)
        site["label"] = label
        site["true_site"] = i
        rows.append(site)
    return pd.concat(rows, ignore_index=True)


def test_site_groups_finds_the_four_sites():
    t = make_table()
    sites = site_groups(t["x"].to_numpy(), t["y"].to_numpy(), gap_m=3000)
    assert len(np.unique(sites)) == 4
    # every true site maps to exactly one group
    assert (pd.crosstab(t["true_site"], sites) > 0).sum(axis=1).eq(1).all()


def test_block_groups_splits_a_site_into_several_blocks():
    t = make_table()
    blocks = block_groups(t["x"].to_numpy(), t["y"].to_numpy(), block_m=2000)
    assert len(np.unique(blocks[t["true_site"] == 0])) > 1


def test_cross_validate_spatial_reports_out_of_site_scores():
    metrics = cross_validate_spatial(make_table(), CFG)
    assert metrics["n_sites"] == 4
    assert metrics["outer_folds"] == 4
    assert metrics["accuracy"] > 0.9
    assert len(metrics["per_site"]) == 4


def test_fit_final_returns_a_model_that_predicts_both_classes():
    t = make_table()
    model, params = fit_final(t, CFG)
    assert set(model.predict(t[FEATURES])) == {0, 1}
    assert set(params) == {"classifier__C", "classifier__gamma"}
