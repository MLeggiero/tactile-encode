"""The load generator's base on the GPU: the hold, with synthetic loads at the tool's working point.

`LoadGenCommand` is the load-generator task's base term (base_command_term): the arm holds the hover pose on its
servos (the targets carry each env's gravity droop for its servo gains), the hand base is the grip synergy at a
fraction of the settled hold's scalar (`hold_level`, drawn per episode from the range the curriculum sets: the plan's
"hold level" of the base, which leaves the residual room to grip harder) from the joint angles the host last read, the
lookahead is zero (the base does not move; in a share `vla_frac` of the episodes the hand base is VLA-style:
action_map.TorchHandTable), and the loads come from a TorchLoadGen schedule drawn at reset (from
`t_start`, when the reset state has settled, to the episode's end; `amp_scale` and `enabled` are the curriculum's).
Every substep before MuJoCo steps (the runtime's pre-hook) the schedule's wrench goes on the tool through
xfrc_applied. The forecast is the schedule's next event (forecast_truth) through the torch ForecastModel.

Events are scored for the curriculum and the metrics over their window (under way, and `settle` after): the tool's
net slip in the hand (its pose at the window's end against its pose when the window began: the slip that stays) and
its slip path (what the reward integrates, elastic excursions included). An event meets the target when its net slip
is within `slip_target` (mm, degrees). The drift reference (the runtime's `ref_rel`) is the tool's pose in the hand at
`t_start`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from tactile_rl.mdp.action_map import TorchSynergy, vla_table
from tactile_rl.mdp.forecast import TorchForecastModel
from tactile_rl.mdp.runtime import PREFIX, tactile_runtime
from tactile_rl.plants.loadgen import LoadApplicator, TorchLoadGen, tool_frame
from tactile_sim import names
from tactile_sim.config import ForecastCfg, LoadGenCfg
from tactile_sim.control.obs_spec import LOOKAHEAD, N_ARM, N_JOINTS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# the residual's training loads (2026-10-13): impulses up to 60 degrees off the strike axis, binds and torque impulses
# about every tool axis (impulse periodic step bind push torque_impulse)
WIDE_P_KIND = (0.30, 0.21, 0.13, 0.13, 0.08, 0.15)
WIDE_CONE_DEG = 60.0
# the amplitude ranges a band file sets (tactile_rl.bench.loadgen_calibration --choose writes one)
BAND_KEYS = ("impulse_n", "periodic_n", "periodic_nm", "step_n", "bind_nm", "push_n", "torque_impulse_nm")


def wide_loads(band: str | dict | None = None) -> LoadGenCfg:
    """The residual's training loads: LoadGenCfg with impulses within WIDE_CONE_DEG of the strike axis, binds about
    every tool axis, torque impulses (WIDE_P_KIND), and the amplitude band of a calibration: `band` a band file (JSON:
    {"band": {key: [lo, hi]}} or the mapping itself) or its contents; None reads $TACTILE_LOADGEN_BAND (unset or
    empty: LoadGenCfg's ranges)."""
    cfg = LoadGenCfg(p_kind=WIDE_P_KIND, cone_deg=WIDE_CONE_DEG, bind_axes="xyz")
    if band is None:
        band = os.environ.get("TACTILE_LOADGEN_BAND") or None
    if band is None:
        return cfg
    if isinstance(band, str):
        with open(band) as fh:
            band = json.load(fh)
    ranges = band.get("band", band)
    unknown = set(ranges) - set(BAND_KEYS)
    if unknown:
        raise ValueError(f"unknown band keys {sorted(unknown)} (expected {BAND_KEYS})")
    over = {}
    for k, v in ranges.items():
        lo, hi = (float(x) for x in v)
        if not 0.0 < lo <= hi:
            raise ValueError(f"band {k}: [{lo}, {hi}]")
        over[k] = (lo, hi)
    return replace(cfg, **over)


@dataclass(kw_only=True)
class LoadGenCommandCfg(CommandTermCfg):
    scene_key: str
    loadgen: LoadGenCfg = field(default_factory=LoadGenCfg)
    forecast: ForecastCfg = field(default_factory=ForecastCfg)
    task: str = "loadgen"
    mask_blocks: tuple[str, ...] = ()
    t_start: float = 0.2  # s: the first event no earlier (the reset state settles)
    settle: float = 0.05  # s after an event during which its slip still counts
    slip_target: tuple[float, float] = (0.7, 1.4)  # mm, degrees of net slip per event
    hold_level: tuple[float, float] = (1.0, 1.0)  # the base's grip, fraction of the settled hold (the curriculum's)
    slots: int = 16
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # one schedule per episode
    vla_frac: float = 0.0  # the share of VLA-style episodes (TorchHandTable)
    vla_hz: float = 30.0

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
        self.s_hold = float(parts.s_hold)
        self.level = torch.ones(N, **f32)
        self.s_now = torch.full((N,), self.s_hold, **f32)
        self.base = torch.zeros(N, 5, N_JOINTS, **f32)
        self.lookahead = torch.zeros(N, N_ARM * len(LOOKAHEAD), **f32)
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
        self.hold_level = tuple(cfg.hold_level)
        # events: per slot the path over its window, the pose when the window began, whether it began / was scored
        K = cfg.slots
        self.ev_path = torch.zeros(N, K, 2, **f32)  # mm, degrees
        self.ev_p0 = torch.zeros(N, K, 6, **f32)
        self.ev_started = torch.zeros(N, K, dtype=torch.bool, device=dev)
        self.ev_done = torch.zeros(N, K, dtype=torch.bool, device=dev)
        self.ep = torch.zeros(N, 6, **f32)  # events, within target, net mm, net deg, path mm, path deg
        # totals since the curriculum last read them (device tensors: no host sync per step)
        self.tot = torch.zeros(4, dtype=torch.float64, device=dev)  # events, events within target, net mm, net deg
        for k in ("events", "events_ok_frac", "slip_per_event_mm", "slip_per_event_deg", "path_per_event_mm",
                  "path_per_event_deg", "hold_level"):
            self.metrics[k] = torch.zeros(N, **f32)
        # VLA-style episodes: drawn and applied by the term that drives the env (MixedBaseCommand clears `vla_top`)
        self.vla = vla_table(rt, cfg.vla_hz) if cfg.vla_frac > 0 else None
        self.vla_top = True

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
        e = self.ep
        n = e[:, 0].clamp(min=1.0)
        m = self.metrics
        m["events"][:] = e[:, 0]
        m["events_ok_frac"][:] = torch.where(e[:, 0] > 0, e[:, 1] / n, torch.ones_like(n))
        m["slip_per_event_mm"][:] = e[:, 2] / n
        m["slip_per_event_deg"][:] = e[:, 3] / n
        m["path_per_event_mm"][:] = e[:, 4] / n
        m["path_per_event_deg"][:] = e[:, 5] / n
        m["hold_level"][:] = self.level

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids
        rt, c = self.rt, self.cfg
        t = rt.t
        self.base = self.synergy.mit_command(self.s_now, rt.rb_q.value.float())
        if self.vla is not None and self.vla_top:
            self.base = self.vla.apply(self.base, rt.rb_q.value, t, rt.rb_q.seq >= 0)
        tr = self.gen.truth(t)
        z = torch.zeros_like(tr["t_event"])
        self.truth = {"valid": tr["valid"], "t_event": tr["t_event"], "regime": tr["regime"], "f": tr["f"],
                      "amp": tr["amp"], "phase": torch.zeros_like(tr["regime"]),
                      "t_c_pred": torch.where(tr["valid"], tr["t_event"], z)}
        self.forecast = self.forecaster(t, tr["valid"], tr["t_event"], tr["regime"], tr["f"], tr["amp"]).float()
        hih = rt.hammer_in_hand()
        # the drift reference: the pose in the hand once the reset has settled
        newly = ~self.ref_set & (t >= c.t_start)
        self.ref_rel = torch.where(newly[:, None], hih, self.ref_rel)
        rt.ref_rel = torch.where(newly[:, None], hih, rt.ref_rel)
        self.ref_set = self.ref_set | newly
        # events: a window that began now keeps the pose; this step's path goes to the windows under way; windows
        # that closed are scored on their net slip
        g = self.gen
        act = g.active(t, c.settle)
        begin = act & ~self.ev_started
        self.ev_p0 = torch.where(begin[..., None], hih[:, None, :].expand_as(self.ev_p0), self.ev_p0)
        self.ev_started = self.ev_started | act
        step = torch.stack([rt.acc["slip_mm"], rt.acc["slip_deg"]], dim=-1)  # (N, 2)
        self.ev_path += act.float()[..., None] * step[:, None, :]
        closed = g.valid & self.ev_started & ~self.ev_done & (g.end() + c.settle <= t.double()[:, None])
        d = hih[:, None, :] - self.ev_p0
        net_mm = torch.linalg.norm(d[..., :3], dim=-1) * 1e3
        net_deg = torch.rad2deg(torch.linalg.norm(d[..., 3:], dim=-1))
        ok = closed & (net_mm <= c.slip_target[0]) & (net_deg <= c.slip_target[1])
        cf = closed.float()
        add = torch.stack([cf.sum(1), ok.float().sum(1), (net_mm * cf).sum(1), (net_deg * cf).sum(1),
                           (self.ev_path[..., 0] * cf).sum(1), (self.ev_path[..., 1] * cf).sum(1)], dim=1)
        self.ep += add
        self.ev_done = self.ev_done | closed
        self.tot += add[:, :4].sum(0).double()

    # ------------------------------------------------------------------ resets
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        c = self.cfg
        n = len(env_ids)
        if self.enabled:
            self.gen.sample(env_ids, self._env.max_episode_length_s, c.t_start, self.amp_scale)
        else:
            self.gen.valid[env_ids] = False
        lo, hi = self.hold_level
        level = lo + (hi - lo) * torch.rand(n, device=self.device)
        self.level[env_ids] = level
        self.s_now[env_ids] = level * self.s_hold
        dr = getattr(self._env, "_tactile_dr", None)
        if dr is not None:
            dr.v["hold_level"][env_ids] = level.double()
        if self.vla is not None and self.vla_top:
            self.vla.reset(env_ids, c.vla_frac)
            if dr is not None:
                dr.v["vla"][env_ids] = self.vla.active[env_ids].double()
        self.forecaster.reset(env_ids)
        self.ev_path[env_ids] = 0.0
        self.ev_started[env_ids] = False
        self.ev_done[env_ids] = False
        self.ep[env_ids] = 0.0
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
            out = torch.zeros(self.num_envs, PARAM_DIM, device=self.device)
            out[:, -1] = self.level
            return out
        return dr.param_vector()

    def episode_events(self, env_ids: torch.Tensor) -> torch.Tensor:
        """(n, 2) float64: each env's events so far this episode and those within the slip target (scored windows)."""
        return self.ep[env_ids, :2].double()

    def take_totals(self) -> torch.Tensor:
        """(4,) events, events within the slip target, their net slip (mm, degrees) since the last call; restarts."""
        out = self.tot.clone()
        self.tot.zero_()
        return out
