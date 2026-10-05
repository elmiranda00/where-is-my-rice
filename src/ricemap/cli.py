"""
Entry point for CLI commands
"""

from pathlib import Path
from typing import Annotated

import typer

from ricemap.config import load_config
from ricemap.tiling import make_tiles

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
