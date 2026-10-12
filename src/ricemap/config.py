"""Load config/mekong.yaml into frozen, typed dataclasses for downstream Python modules"""


from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AOI:
    bbox: tuple[float, float, float, float]
    crs: str

@dataclass(frozen=True)
class TimeRange:
    start: str
    end: str
    composite_days: int

@dataclass(frozen=True)
class Tiling:
    tile_size_m: int
    resolution_m: int

@dataclass(frozen=True)
class Model:
    param_grid: dict
    cv_folds: int
    cv_block_size_m: int
    cv_site_gap_m: int

@dataclass(frozen=True)
class Stac:
    url: str
    collection: str

@dataclass(frozen=True)
class Publish:
    rice_colour: tuple[int, int, int]
    provinces: Path
    reference: Path

@dataclass(frozen=True)
class Config:
    name: str
    aoi: AOI
    time: TimeRange
    tiling: Tiling
    speckle_window: int
    model: Model
    stac: Stac
    labels: Path
    publish: Publish

def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        name=raw["name"],
        aoi=AOI(bbox=tuple(raw["aoi"]["bbox"]), crs=raw["aoi"]["crs"]),
        time=TimeRange(**raw["time"]),
        tiling=Tiling(**raw["tiling"]),
        speckle_window=raw["features"]["speckle_window"],
        model=Model(**raw["model"]),
        stac=Stac(**raw["stac"]),
        labels=Path(raw["labels"]),
        publish=Publish(
            rice_colour=tuple(raw["publish"]["rice_colour"]),
            provinces=Path(raw["publish"]["provinces"]),
            reference=Path(raw["publish"]["reference"]))

    )

