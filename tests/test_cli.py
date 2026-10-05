from pathlib import Path

from typer.testing import CliRunner

from ricemap.cli import app
from ricemap.config import load_config
from ricemap.tiling import make_tiles

CONFIG = Path(__file__).parents[1] / "config" / "mekong.yaml"


def test_list_tiles_prints_one_line_per_tile():
    result = CliRunner().invoke(app, ["list-tiles", "--config", str(CONFIG)])
    assert result.exit_code == 0

    lines = result.stdout.strip().splitlines()
    cfg = load_config(CONFIG)
    assert len(lines) == len(make_tiles(cfg.aoi, cfg.tiling))
    assert lines[0].split("\t")[0] == "0"
