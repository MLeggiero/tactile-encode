"""The load generator's base on the GPU: the hold, with synthetic loads at the tool's working point.

`LoadGenCommand` is the load-generator task's base term (base_command_term): the arm holds the hover pose on its
servos (the targets carry each env's gravity droop for its servo gains), the hand base is the grip synergy at the
settled hold's scalar from the joint angles the host last read, the lookahead is zero (the base does not move), and
the loads come from a TorchLoadGen schedule drawn at reset (from `t_start`, when the reset state has settled, to the
episode's end; `amp_scale` and `enabled` are the curriculum's). Every substep before MuJoCo steps (the runtime's
pre-hook) the schedule's wrench goes on the tool through xfrc_applied. The forecast is the schedule's next event
(forecast_truth) through the torch ForecastModel.

Events are scored for the curriculum and the metrics: the slip path the tool travels in the hand while an event is
under way and `settle` after it (the runtime's integrals per policy step) is compared with `slip_target` (mm, degrees)
when the event's window closes. The drift reference (the runtime's `ref_rel`) is the tool's pose in the hand at
`t_start`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from tactile_rl.mdp.action_map import TorchSynergy
from tactile_rl.mdp.forecast import TorchForecastModel
from tactile_rl.mdp.runtime import PREFIX, tactile_runtime
from tactile_rl.plants.loadgen import LoadApplicator, TorchLoadGen, tool_frame
from tactile_sim import names
from tactile_sim.config import ForecastCfg, LoadGenCfg
from tactile_sim.control.obs_spec import LOOKAHEAD, N_ARM, N_JOINTS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class LoadGenCommandCfg(CommandTermCfg):
    scene_key: str
    loadgen: LoadGenCfg = field(default_factory=LoadGenCfg)
    forecast: ForecastCfg = field(default_factory=ForecastCfg)
    task: str = "loadgen"
    mask_blocks: tuple[str, ...] = ()
    t_start: float = 0.2  # s: the first event no earlier (the reset state settles)
    settle: float = 0.05  # s after an event during which its slip still counts
    slip_target: tuple[float, float] = (0.7, 1.4)  # mm, degrees per event
    slots: int = 16
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # one schedule per episode

    def build(self, env: ManagerBasedRlEnv) -> LoadGenCommand:
        return LoadGenCommand(self, env)


class LoadGenCommand(CommandTerm):
    cfg: LoadGenCommandCfg
    tactile_base = True

    def __init__(self, cfg: LoadGenCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.rt = rt = tactile_runtime(env, cfg.scene_key, task=cfg.task, mask_blocks=cfg.mask_blocks, plant=False)
        parts = rt.parts
        N, dev = self.num_envs, self.device
        self.gen = TorchLoadGen(cfg.loadgen, N, dev, slots=cfg.slots)
        m = env.sim.mj_model
        body = m.body(PREFIX + parts.hammer_body).id
        face = m.site(PREFIX + names.HAMMER_FACE_SITE).id
        head = m.site(PREFIX + names.HAMMER_IMU_SITE).id
        self.app = LoadApplicator(env.sim, body, face, tool_frame(m, body, face, head), dev)
        rt.pre_hooks.append(self._apply)
        self.synergy = TorchSynergy(parts.synergy, dev)
        self.forecaster = TorchForecastModel(N, cfg.forecast, dev)
        f32, f64 = dict(dtype=torch.float32, device=dev), dict(dtype=torch.float64, device=dev)
        self.s_now = torch.full((N,), parts.s_hold, **f32)
        self.base = torch.zeros(N, 5, N_JOINTS, **f32)
        self.lookahead = torch.zeros(N, (N_JOINTS + N_ARM) * len(LOOKAHEAD), **f32)
        self.forecast = torch.zeros(N, 13, **f32)
        z64 = torch.zeros(N, **f64)
        self.truth = {"valid": torch.zeros(N, dtype=torch.bool, device=dev), "t_event": z64, "f": z64, "amp": z64,
                      "regime": torch.zeros(N, dtype=torch.long, device=dev),
                      "phase": torch.zeros(N, dtype=torch.long, device=dev), "t_c_pred": z64}
        self.trigger = torch.zeros(N, dtype=torch.bool, device=dev)
        self.ref_rel = torch.zeros(N, 6, **f32)
        self.ref_set = torch.zeros(N, dtype=torch.bool, device=dev)
        self.q_hover = torch.as_tensor(parts.q_hover, **f64)
        self.droop0 = torch.as_tensor(parts.arm_target - parts.q_hover, **f64)
        self.kp0 = torch.as_tensor(parts.arm_kp, **f64)
        self.droop = self.droop0.expand(N, -1).clone()
        self.f_now = torch.zeros(N, 3, **f32)  # the wrench on the tool now, tool frame (truth)
        self.tq_now = torch.zeros(N, 3, **f32)
        # the curriculum's knobs
        self.enabled = True
        self.amp_scale = 1.0
        # events
        K = cfg.slots
        self.ev_slip_mm = torch.zeros(N, K, **f32)
        self.ev_slip_deg = torch.zeros(N, K, **f32)
        self.ev_done = torch.zeros(N, K, dtype=torch.bool, device=dev)
        self.ep_events = torch.zeros(N, **f32)
        self.ep_ok = torch.zeros(N, **f32)
        self.ep_ev_mm = torch.zeros(N, **f32)
        self.ep_ev_deg = torch.zeros(N, **f32)
        # totals since the curriculum last read them (device tensors: no host sync per step)
        self.tot = torch.zeros(4, dtype=torch.float64, device=dev)  # events, events within target, mm, deg
        for k in ("events", "events_ok_frac", "slip_per_event_mm", "slip_per_event_deg"):
            self.metrics[k] = torch.zeros(N, **f32)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.lookahead, self.forecast], dim=1)

    # ------------------------------------------------------------------ per substep
    def _apply(self, t: torch.Tensor) -> None:
        f, tq = self.gen.wrench(t)
        self.f_now, self.tq_now = f.float(), tq.float()
        self.app.apply(f, tq)

    def arm_target(self, tau: torch.Tensor) -> torch.Tensor:
        del tau
        return (self.q_hover + self.droop).float()

    def set_arm_gains(self, env_ids: torch.Tensor, kp: torch.Tensor, kd: torch.Tensor) -> None:
        """The droop of the hold for these envs' servo gains (World._hold_arm: g(q) / kp)."""
        del kd
        self.droop[env_ids] = self.droop0[None] * (self.kp0[None] / kp.double())

    def done(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ per policy tick
    def _update_metrics(self) -> None:
        n = self.ep_events.clamp(min=1.0)
        self.metrics["events"][:] = self.ep_events
        self.metrics["events_ok_frac"][:] = torch.where(self.ep_events > 0, self.ep_ok / n,
                                                        torch.ones_like(self.ep_ok))
        self.metrics["slip_per_event_mm"][:] = self.ep_ev_mm / n
        self.metrics["slip_per_event_deg"][:] = self.ep_ev_deg / n

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids
        rt, c = self.rt, self.cfg
        t = rt.t
        self.base = self.synergy.mit_command(self.s_now, rt.rb_q.value.float())
        tr = self.gen.truth(t)
        z = torch.zeros_like(tr["t_event"])
        self.truth = {"valid": tr["valid"], "t_event": tr["t_event"], "regime": tr["regime"], "f": tr["f"],
                      "amp": tr["amp"], "phase": torch.zeros_like(tr["regime"]),
                      "t_c_pred": torch.where(tr["valid"], tr["t_event"], z)}
        self.forecast = self.forecaster(t, tr["valid"], tr["t_event"], tr["regime"], tr["f"], tr["amp"]).float()
        # the drift reference: the pose in the hand once the reset has settled
        newly = ~self.ref_set & (t >= c.t_start)
        hih = rt.hammer_in_hand()
        self.ref_rel = torch.where(newly[:, None], hih, self.ref_rel)
        rt.ref_rel = torch.where(newly[:, None], hih, rt.ref_rel)
        self.ref_set = self.ref_set | newly
        # events: this step's slip goes to the events under way; windows that closed are scored
        g = self.gen
        act = g.active(t, c.settle).float()
        self.ev_slip_mm += act * rt.acc["slip_mm"][:, None]
        self.ev_slip_deg += act * rt.acc["slip_deg"][:, None]
        closed = g.valid & ~self.ev_done & (g.end() + c.settle <= t.double()[:, None])
        ok = closed & (self.ev_slip_mm <= c.slip_target[0]) & (self.ev_slip_deg <= c.slip_target[1])
        cf = closed.float()
        n_c, n_ok = cf.sum(1), ok.float().sum(1)
        mm, deg = (self.ev_slip_mm * cf).sum(1), (self.ev_slip_deg * cf).sum(1)
        self.ep_events += n_c
        self.ep_ok += n_ok
        self.ep_ev_mm += mm
        self.ep_ev_deg += deg
        self.ev_done = self.ev_done | closed
        self.tot += torch.stack([n_c.sum(), n_ok.sum(), mm.sum(), deg.sum()]).double()

    # ------------------------------------------------------------------ resets
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        c = self.cfg
        if self.enabled:
            self.gen.sample(env_ids, self._env.max_episode_length_s, c.t_start, self.amp_scale)
        else:
            self.gen.valid[env_ids] = False
        self.forecaster.reset(env_ids)
        self.ev_slip_mm[env_ids] = 0.0
        self.ev_slip_deg[env_ids] = 0.0
        self.ev_done[env_ids] = False
        self.ep_events[env_ids] = 0.0
        self.ep_ok[env_ids] = 0.0
        self.ep_ev_mm[env_ids] = 0.0
        self.ep_ev_deg[env_ids] = 0.0
        self.ref_set[env_ids] = False
        from tactile_rl.mdp.events import tactile_state

        ref = tactile_state(self._env).ref_rel[env_ids]
        self.ref_rel[env_ids] = ref
        self.rt.ref_rel[env_ids] = ref
        self.f_now[env_ids] = 0.0
        self.tq_now[env_ids] = 0.0

    def param_vector(self) -> torch.Tensor:
        from tactile_rl.mdp.randomization import PARAM_DIM

        dr = getattr(self._env, "_tactile_dr", None)
        if dr is None:
            return torch.zeros(self.num_envs, PARAM_DIM, device=self.device)
        return dr.param_vector()

    def take_totals(self) -> torch.Tensor:
        """(4,) events, events within the slip target, their slip (mm, degrees) since the last call; restarts."""
        out = self.tot.clone()
        self.tot.zero_()
        return out
