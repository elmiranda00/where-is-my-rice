"""
Entry point for CLI commands
"""

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from ricemap import predict
from ricemap.config import load_config
from ricemap.tiling import make_tiles
from ricemap.train import run_training

app = typer.Typer(no_args_is_help=True)

DEFAULT_CONFIG = Path("config/mekong.yaml")
ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Run config YAML.")]


@app.callback()
def main() -> None:
    """Rice vs non-rice mapping for the Mekong Delta from Sentinel-1."""


@app.command()
def list_tiles(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Print every tile in the AOI grid: task index, tile ID and UTM bounds."""
    cfg = load_config(config)
    tiles = make_tiles(cfg.aoi, cfg.tiling)
    for i, t in enumerate(tiles):
        typer.echo(f"{i}\t{t.tile_id}\t{t.minx}\t{t.miny}\t{t.maxx}\t{t.maxy}")
    typer.echo(f"{len(tiles)} tiles of {cfg.tiling.tile_size_m} m in {cfg.aoi.crs}", err=True)

@app.command()
def train(
    config: ConfigOption = DEFAULT_CONFIG,
    rebuild_table: Annotated[
        bool, typer.Option(help="Re-fetch features for the labelled points even if cached.")
    ] = False,
) -> None:
    """Build the training table, run spatial CV, fit the SVM, save model and metrics."""
    metrics = run_training(load_config(config), rebuild_table=rebuild_table)
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def predict_tile(
    config: ConfigOption = DEFAULT_CONFIG,
    tile_id: Annotated[
        str | None, typer.Option(help="Tile to classify. Default: CLOUD_RUN_TASK_INDEX.")
    ] = None,
) -> None:
    """Classify one tile and write it as a COG to STORAGE_URI/tiles/."""
    cfg = load_config(config)
    tiles = make_tiles(cfg.aoi, cfg.tiling)
    if tile_id is None:
        index = os.environ.get("CLOUD_RUN_TASK_INDEX")
        if index is None:
            raise typer.BadParameter("Pass --tile-id or set CLOUD_RUN_TASK_INDEX")
        tile = tiles[int(index)]
    else:
        matches = [t for t in tiles if t.tile_id == tile_id]
        if not matches:
            raise typer.BadParameter(f"Unknown tile {tile_id}; see `ricemap list-tiles`")
        tile = matches[0]

    dest = predict.predict_tile(tile, cfg, predict.load_model())
    typer.echo(f"{tile.tile_id}: " + (f"wrote {dest}" if dest else "already exists, skipped"))