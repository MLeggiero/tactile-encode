"""The curriculum's mixed base (stage C3): hammer episodes (the base library's swings) side by side with load-generator
episodes, one kind per env and episode.

`MixedBaseCommand` drives two engines over every env, a BaseTrajectoryCommand (the hammer: library entry, its start
state and randomization, the perturbed arm stream, the hand base at the episode's hold level) and a LoadGenCommand
(the hold under synthetic loads), and hands the residual, the observations, the rewards and the terminations the one
each env runs: at reset an env plays the hammer with probability `hammer_frac` (the curriculum's: 0 before C3, 0.7
in it) and the load generator otherwise. The hammer engine builds the runtime with the nail plant; a load-generator
env keeps the nail idle, a hammer env gets no loads. The student's task block says which (obs_spec TASKS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from tactile_rl.mdp.commands import BaseTrajectoryCommand, BaseTrajectoryCommandCfg
from tactile_rl.mdp.loadgen_command import LoadGenCommand, LoadGenCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MixedBaseCommandCfg(CommandTermCfg):
    scene_key: str
    library: str
    hammer_frac: float = 0.0
    perturb: bool = True
    hold_level: tuple[float, float] = (1.0, 1.0)
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: ManagerBasedRlEnv) -> MixedBaseCommand:
        return MixedBaseCommand(self, env)


class _ForecastFanOut:
    """randomize_forecast's handle: the error model goes to both engines' forecasters."""

    def __init__(self, *models):
        self.models = models

    def set_params(self, env_ids=None, **values) -> None:
        for m in self.models:
            m.set_params(env_ids, **values)


class MixedBaseCommand(CommandTerm):
    cfg: MixedBaseCommandCfg
    tactile_base = True

    def __init__(self, cfg: MixedBaseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # the hammer engine first: it creates the runtime with the nail plant
        self.ham = BaseTrajectoryCommand(BaseTrajectoryCommandCfg(scene_key=cfg.scene_key, library=cfg.library,
                                                                  perturb=cfg.perturb, task="hammer",
                                                                  hold_level=cfg.hold_level), env)
        self.lg = LoadGenCommand(LoadGenCommandCfg(scene_key=cfg.scene_key, hold_level=cfg.hold_level), env)
        self.rt = self.ham.rt
        N, dev = self.num_envs, self.device
        self.is_hammer = torch.zeros(N, dtype=torch.bool, device=dev)
        self.hammer_frac = float(cfg.hammer_frac)
        self.forecaster = _ForecastFanOut(self.ham.forecaster, self.lg.forecaster)
        self.depth0 = torch.zeros(N, dtype=torch.float64, device=dev)
        for k in ("hammer", "hammer_advance_mm", *self.lg.metrics):
            self.metrics[k] = torch.zeros(N, device=dev)
        self.trigger = torch.zeros(N, dtype=torch.bool, device=dev)

    # ------------------------------------------------------------------ the curriculum's knobs
    @property
    def hold_level(self) -> tuple[float, float]:
        return self.lg.hold_level

    @hold_level.setter
    def hold_level(self, v) -> None:
        self.lg.hold_level = tuple(v)
        self.ham.hold_level = tuple(v)

    @property
    def enabled(self) -> bool:
        return self.lg.enabled

    @enabled.setter
    def enabled(self, v: bool) -> None:
        self.lg.enabled = bool(v)

    @property
    def amp_scale(self) -> float:
        return self.lg.amp_scale

    @amp_scale.setter
    def amp_scale(self, v: float) -> None:
        self.lg.amp_scale = float(v)

    # ------------------------------------------------------------------ what the terms read, per env
    def _pick(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m = self.is_hammer.reshape(-1, *([1] * (a.ndim - 1)))
        return torch.where(m, a, b)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.lookahead, self.forecast], dim=1)

    @property
    def s_now(self) -> torch.Tensor:
        return self._pick(self.ham.s_now, self.lg.s_now)

    @property
    def base(self) -> torch.Tensor:
        return self._pick(self.ham.base, self.lg.base)

    @property
    def lookahead(self) -> torch.Tensor:
        return self._pick(self.ham.lookahead, self.lg.lookahead)

    @property
    def forecast(self) -> torch.Tensor:
        return self._pick(self.ham.forecast, self.lg.forecast)

    @property
    def truth(self) -> dict[str, torch.Tensor]:
        return {k: self._pick(self.ham.truth[k], self.lg.truth[k]) for k in self.lg.truth}

    @property
    def ref_rel(self) -> torch.Tensor:
        return self._pick(self.ham.ref_rel, self.lg.ref_rel)

    @property
    def f_now(self) -> torch.Tensor:
        return self.lg.f_now

    @property
    def tq_now(self) -> torch.Tensor:
        return self.lg.tq_now

    @property
    def depth_prev(self) -> torch.Tensor:  # the nail-advance reward's state (the hammer engine's)
        return self.ham.depth_prev

    @depth_prev.setter
    def depth_prev(self, v: torch.Tensor) -> None:
        self.ham.depth_prev = v

    def arm_target(self, tau: torch.Tensor) -> torch.Tensor:
        return self._pick(self.ham.arm_target(tau), self.lg.arm_target(tau))

    def set_arm_gains(self, env_ids: torch.Tensor, kp: torch.Tensor, kd: torch.Tensor) -> None:
        self.lg.set_arm_gains(env_ids, kp, kd)

    def done(self) -> torch.Tensor:
        return self.is_hammer & self.ham.done()

    def param_vector(self) -> torch.Tensor:
        return self.lg.param_vector()  # the events' state, with the library's draws written over on hammer envs

    def take_totals(self) -> torch.Tensor:
        return self.lg.take_totals()

    # ------------------------------------------------------------------ per policy tick
    def _update_metrics(self) -> None:
        self.lg._update_metrics()
        m = self.metrics
        for k, v in self.lg.metrics.items():
            m[k][:] = torch.where(self.is_hammer, torch.zeros_like(v), v)
        h = self.is_hammer.float()
        m["hammer"][:] = h  # (the hammer's net slip per strike is a metrics term: it needs the step's terminations)
        if self.rt.nail is not None:
            m["hammer_advance_mm"][:] = h * ((self.rt.nail.depth().double() - self.depth0) * 1e3).float()

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        self.ham._update_command(env_ids)
        self.lg._update_command(env_ids)

    # ------------------------------------------------------------------ resets
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        ham = torch.rand(n, device=self.device) < self.hammer_frac
        self.is_hammer[env_ids] = ham
        h_ids, l_ids = env_ids[ham], env_ids[~ham]  # (one host sync per reset step)
        # the load generator first, for every resetting env (its hold level, references); then the hammer
        # overwrites its envs' state, randomization and references
        self.lg._resample_command(env_ids)
        if len(h_ids):
            self.lg.gen.valid[h_ids] = False  # no loads on a hammer episode
            self.lg.ref_set[h_ids] = True  # nor the load generator's drift reference
            self.ham._resample_command(h_ids)
            self.rt.obs.set_task(h_ids, "hammer")
            if self.rt.nail is not None:
                self.depth0[h_ids] = self.rt.nail.depth()[h_ids].double()
        if len(l_ids):
            self.rt.obs.set_task(l_ids, "loadgen")
            if self.rt.nail is not None:
                self.depth0[l_ids] = self.rt.nail.depth()[l_ids].double()
