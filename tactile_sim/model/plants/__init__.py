"""Task plants: the tool's target (nail for the hammer, a board to crosscut for the saw, a screw or a hole for the
driver)."""

from tactile_sim.model.plants.base import Plant
from tactile_sim.model.plants.drill import DrillPlant
from tactile_sim.model.plants.nail import NailPlant
from tactile_sim.model.plants.saw import SawPlant


def make_plant(cfg) -> Plant:
    if cfg.plant.kind == "nail":
        return NailPlant(cfg)
    if cfg.plant.kind == "saw":
        return SawPlant(cfg)
    if cfg.plant.kind == "drill":
        return DrillPlant(cfg)
    raise ValueError(f"unknown plant kind {cfg.plant.kind!r}")


__all__ = ["Plant", "NailPlant", "SawPlant", "DrillPlant", "make_plant"]
