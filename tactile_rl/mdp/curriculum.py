"""The training curriculum's stages C0-C2 (the plan's "Curriculum"; C3 and C4, the plants and the mixed stage, come
with the hammer and drill teachers).

| Stage | Loads | Randomization | Forecast |
|---|---|---|---|
| C0 | none: the hold | near nominal (scale 0.1) | clean |
| C1 | load generator at a quarter of the amplitudes | scale 0.3 | clean (exact, timely, never masked) |
| C2 | the full band | the full table | noisy (the plan's ranges), whole episodes masked 25 % |

A stage advances when, over the episodes that ended since it began (at least `min_episodes`), drops stay below
`drop_max` and, once there are loads, at least `ok_min` of the events (at least `min_events`) kept the tool's slip
within the load generator's target. Stage settings apply to episodes that start after the change (the load
generator's amplitude and whether it runs, the DR scale, the forecast's mode).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tactile_rl.mdp.actions import base_command_term

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

STAGES = (
    {"name": "C0", "loads": False, "amp_scale": 0.0, "dr_scale": 0.1, "forecast": "clean", "mask": False},
    {"name": "C1", "loads": True, "amp_scale": 0.25, "dr_scale": 0.3, "forecast": "clean", "mask": False},
    {"name": "C2", "loads": True, "amp_scale": 1.0, "dr_scale": 1.0, "forecast": "noisy", "mask": True},
)


class TactileCurriculum:
    """A class-based mjlab curriculum term (called with the resetting envs before they reset)."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        p = cfg.params
        self.env = env
        self.scene_key = p["scene_key"]
        self.stage = int(p.get("start_stage", 0))
        self.max_stage = int(p.get("max_stage", len(STAGES) - 1))
        self.min_episodes = int(p.get("min_episodes", 2000))
        self.min_events = int(p.get("min_events", 2000))
        self.drop_max = float(p.get("drop_max", 0.01))
        self.ok_min = float(p.get("ok_min", 0.8))
        self.eval_every = int(p.get("eval_every", 100))
        self.drop_term = p.get("drop_term", "tool_dropped")
        dev = env.device
        self.win = torch.zeros(4, dtype=torch.float64, device=dev)  # episodes, drops, events, events ok
        self.calls = 0
        self.last = {"drop_rate": float("nan"), "events_ok_frac": float("nan"), "episodes": 0.0, "events": 0.0}
        self._applied = None

    def apply(self) -> None:
        """The stage's settings onto the load generator, the DR state and the forecast."""
        from tactile_rl.mdp.randomization import tactile_dr

        s = STAGES[self.stage]
        cmd = base_command_term(self.env)
        if cmd is not None and hasattr(cmd, "enabled"):
            cmd.enabled = bool(s["loads"])
            cmd.amp_scale = float(s["amp_scale"])
        dr = tactile_dr(self.env, self.scene_key)
        dr.scale = float(s["dr_scale"])
        dr.forecast_mode = s["forecast"]
        dr.forecast_masking = bool(s["mask"])
        self._applied = self.stage

    def __call__(self, env: ManagerBasedRlEnv, env_ids, **params) -> dict[str, float]:
        del params
        if self._applied != self.stage:
            self.apply()
        if isinstance(env_ids, torch.Tensor) and env.common_step_counter > 0:
            cmd = base_command_term(env)
            drops = env.termination_manager.get_term(self.drop_term)[env_ids].double().sum()
            tot = cmd.take_totals() if hasattr(cmd, "take_totals") else torch.zeros(4, dtype=torch.float64,
                                                                                    device=env.device)
            self.win += torch.stack([torch.as_tensor(float(len(env_ids)), dtype=torch.float64, device=env.device),
                                     drops, tot[0], tot[1]])
            self.calls += 1
            if self.calls % self.eval_every == 0:
                self._evaluate()
        return {"stage": float(self.stage), **self.last}

    def _evaluate(self) -> None:
        n_ep, drops, ev, ok = (float(x) for x in self.win.cpu())
        drop_rate = drops / max(n_ep, 1.0)
        ok_frac = ok / ev if ev > 0 else float("nan")
        self.last = {"drop_rate": drop_rate, "events_ok_frac": ok_frac, "episodes": n_ep, "events": ev}
        if n_ep < self.min_episodes or self.stage >= self.max_stage:
            return
        loads = STAGES[self.stage]["loads"]
        if drop_rate < self.drop_max and (not loads or (ev >= self.min_events and ok_frac >= self.ok_min)):
            self.stage += 1
            self.apply()
            self.win.zero_()

    def reset(self, env_ids=None) -> None:
        del env_ids
