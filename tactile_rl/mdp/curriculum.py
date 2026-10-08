"""The training curriculum's stages C0-C2 (the plan's "Curriculum"; C3 and C4, the plants and the mixed stage, come
with the hammer and drill teachers).

| Stage | Loads | Base grip | Randomization | Forecast |
|---|---|---|---|---|
| C0 | none: the hold | the settled hold | near nominal (scale 0.1) | clean |
| C1 | load generator at a quarter of the amplitudes | 0.5-1.0 of it | scale 0.3 | clean (exact, timely, never masked) |
| C2 | the full band | 0.3-0.8 of it (the plan's hold level) | the full table | noisy, 25 % of episodes masked |
| C3 | 70 % hammer episodes (library swings), 30 % the full band | 0.3-0.8 | the full table | noisy, masked |

The base grip below the settled hold is what leaves the residual a job: at the hold the WUJI's closing joints already
run near their current ceiling (the SDK effort limit is the rated current), and a residual that grips harder and
stiffens cannot change the slip; at half the hold it recovers the hold's slip (tactile_rl/README.md).

A stage advances when, over the episodes that began in it and ended recently (at least `min_episodes`; each evaluation
that does not pass halves the window's counts, so older episodes fade), drops stay below `drop_max` and, once there are
loads, at least `ok_min` of their events (at least `min_events`) kept the tool's slip within the load generator's
target. Stage settings apply to episodes that start after the change (the load generator's amplitude and whether it
runs, the DR scale, the forecast's mode), so the episodes under way at a change count for neither stage.

`stage_cap_s` (simulated seconds summed over the envs; None: the gates alone) bounds the time a stage may take: past
it the stage advances at its next evaluation whether or not it passed, and `capped` counts the stages that did. A
training budget then reaches the last stage in time (C3 keeps C2's load-generator episodes as its 30 %).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tactile_rl.mdp.actions import base_command_term

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

STAGES = (
    {"name": "C0", "loads": False, "amp_scale": 0.0, "hold": (1.0, 1.0), "dr_scale": 0.1, "forecast": "clean",
     "mask": False},
    {"name": "C1", "loads": True, "amp_scale": 0.25, "hold": (0.5, 1.0), "dr_scale": 0.3, "forecast": "clean",
     "mask": False},
    {"name": "C2", "loads": True, "amp_scale": 1.0, "hold": (0.3, 0.8), "dr_scale": 1.0, "forecast": "noisy",
     "mask": True},
    {"name": "C3", "loads": True, "amp_scale": 1.0, "hold": (0.3, 0.8), "dr_scale": 1.0, "forecast": "noisy",
     "mask": True, "hammer": 0.7},
)


class TactileCurriculum:
    """A class-based mjlab curriculum term (called with the resetting envs before they reset)."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        p = cfg.params
        self.env = env
        self.scene_key = p["scene_key"]
        self.stage = int(p.get("start_stage", 0))
        self.max_stage = int(p.get("max_stage", 2))
        self.min_episodes = int(p.get("min_episodes", 2000))
        self.min_events = int(p.get("min_events", 2000))
        self.drop_max = float(p.get("drop_max", 0.01))
        self.ok_min = float(p.get("ok_min", 0.8))
        self.eval_every = int(p.get("eval_every", 100))
        self.drop_term = p.get("drop_term", "tool_dropped")
        cap = p.get("stage_cap_s")
        self.stage_cap_s = None if cap is None else float(cap)
        dev = env.device
        self.win = torch.zeros(4, dtype=torch.float64, device=dev)  # episodes, drops, events, events ok
        self.ep_stage = torch.full((env.num_envs,), self.stage, dtype=torch.long, device=dev)  # each episode's stage
        self.calls = 0
        self.capped = 0
        self.stage_t0 = 0  # env.common_step_counter when the stage began
        self.history: list[dict] = []  # per advance: the stage left, simulated s at the change, passed or capped
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
            cmd.hold_level = tuple(s["hold"])
        if cmd is not None and hasattr(cmd, "hammer_frac"):  # the C3 mix (MixedBaseCommand)
            cmd.hammer_frac = float(s.get("hammer", 0.0))
        dr = tactile_dr(self.env, self.scene_key)
        dr.scale = float(s["dr_scale"])
        dr.forecast_mode = s["forecast"]
        dr.forecast_masking = bool(s["mask"])
        self._applied = self.stage

    def __call__(self, env: ManagerBasedRlEnv, env_ids, **params) -> dict[str, float]:
        del params
        if self._applied != self.stage:
            self.apply()
        if isinstance(env_ids, torch.Tensor):
            if env.common_step_counter > 0:
                cmd = base_command_term(env)
                # a stage is judged on the episodes that began in it (those under way when it began ran the last one's)
                same = (self.ep_stage[env_ids] == self.stage).double()
                drops = (env.termination_manager.get_term(self.drop_term)[env_ids].double() * same).sum()
                if hasattr(cmd, "episode_events"):
                    ev = cmd.episode_events(env_ids) * same[:, None]
                    n_ev, n_ok = ev[:, 0].sum(), ev[:, 1].sum()
                else:
                    n_ev = n_ok = torch.zeros((), dtype=torch.float64, device=env.device)
                self.win += torch.stack([same.sum(), drops, n_ev, n_ok])
                self.calls += 1
                if self.calls % self.eval_every == 0:
                    self._evaluate()
            self.ep_stage[env_ids] = self.stage  # their next episodes begin in the stage as it is now
        return {"stage": float(self.stage), "capped": float(self.capped), **self.last}

    def sim_seconds(self, since: int = 0) -> float:
        """Simulated seconds summed over the envs since env step `since`."""
        return (self.env.common_step_counter - since) * self.env.step_dt * self.env.num_envs

    def _evaluate(self) -> None:
        n_ep, drops, ev, ok = (float(x) for x in self.win.cpu())
        drop_rate = drops / max(n_ep, 1.0)
        ok_frac = ok / ev if ev > 0 else float("nan")
        self.last = {"drop_rate": drop_rate, "events_ok_frac": ok_frac, "episodes": n_ep, "events": ev}
        if self.stage < self.max_stage:
            loads = STAGES[self.stage]["loads"]
            passed = (n_ep >= self.min_episodes and drop_rate < self.drop_max
                      and (not loads or (ev >= self.min_events and ok_frac >= self.ok_min)))
            capped = (not passed and self.stage_cap_s is not None
                      and self.sim_seconds(self.stage_t0) >= self.stage_cap_s)
            if passed or capped:
                self.history.append({"stage": STAGES[self.stage]["name"], "sim_s": self.sim_seconds(),
                                     "how": "passed" if passed else "capped", **self.last})
                self.capped += int(capped)
                self.stage += 1
                self.stage_t0 = self.env.common_step_counter
                self.apply()
                self.win.zero_()
                return
        if n_ep >= self.min_episodes:
            self.win *= 0.5  # a full window that did not pass fades: the stage is judged on recent episodes

    def reset(self, env_ids=None) -> None:
        del env_ids
