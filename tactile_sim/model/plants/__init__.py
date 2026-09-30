"""Task plants: the tool target (nail for Task A; drill/screw slot for Task B)."""

from tactile_sim.model.plants.base import Plant
from tactile_sim.model.plants.drill import DrillPlant
from tactile_sim.model.plants.nail import NailPlant


def make_plant(cfg) -> Plant:
    if cfg.plant.kind == "nail":
        return NailPlant(cfg)
    if cfg.plant.kind == "drill":
        return DrillPlant(cfg)
    raise ValueError(f"unknown plant kind {cfg.plant.kind!r}")


__all__ = ["Plant", "NailPlant", "DrillPlant", "make_plant"]
