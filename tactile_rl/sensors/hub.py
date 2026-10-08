"""A post-substep hook for mjlab: a scene sensor whose `update` mjlab calls right after every physics substep
(ManagerBasedRlEnv.step: sim.step, then scene.update), before its per-env-step forward(), so what it reads is the
step's own data, as the testbed's sensors read MjData after mj_step. The tactile runtime (tactile_rl.mdp.runtime)
registers its post-substep and reset callbacks here."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from mjlab.sensor import Sensor, SensorCfg

HUB = "tactile_hub"


@dataclass
class TactileHubCfg(SensorCfg):
    name: str = HUB

    def build(self) -> TactileHub:
        return TactileHub(self)


class TactileHub(Sensor[torch.Tensor]):
    def __init__(self, cfg: TactileHubCfg):
        super().__init__()
        self.cfg = cfg
        self.post: Callable[[], None] | None = None
        self.on_reset: Callable[[torch.Tensor | slice | None], None] | None = None
        self._empty = torch.zeros(0)

    def edit_spec(self, scene_spec, entities) -> None:
        del scene_spec, entities

    def initialize(self, mj_model, model, data, device: str) -> None:
        self._empty = torch.zeros(0, device=device)

    def _compute_data(self) -> torch.Tensor:
        return self._empty

    def update(self, dt: float) -> None:
        super().update(dt)
        if self.post is not None:
            self.post()

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if self.on_reset is not None:
            self.on_reset(env_ids)
