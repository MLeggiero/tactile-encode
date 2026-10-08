"""Task plants: the tool's target (nail for the hammer, a board to crosscut for the saw, a screw or a hole for the
driver)."""

from tactile_sim.model.plants.base import Plant
from tactile_sim.model.plants.drill import DrillPlant
from tactile_sim.model.plants.drill_v2 import DrillPlantV2
from tactile_sim.model.plants.nail import NailPlant
from tactile_sim.model.plants.saw import SawPlant


def make_plant(cfg) -> Plant:
    if cfg.plant.kind == "nail":
        return NailPlant(cfg)
    if cfg.plant.kind == "saw":
        return SawPlant(cfg)
    if cfg.plant.kind == "drill":
        if cfg.drill.model == "v2":
            return DrillPlantV2(cfg)
        if cfg.drill.model != "12v":
            raise ValueError(f"unknown drill model {cfg.drill.model!r} (expected '12v' or 'v2')")
        return DrillPlant(cfg)
    raise ValueError(f"unknown plant kind {cfg.plant.kind!r}")


__all__ = ["Plant", "NailPlant", "SawPlant", "DrillPlant", "DrillPlantV2", "make_plant"]
