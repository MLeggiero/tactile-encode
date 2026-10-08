"""The base under the residual on the GPU: scripted hammer swings from the CPU's base library
(tactile_sim.behaviors.base_library), one per env and episode.

`BaseTrajectoryCommand` samples an entry per env at reset and starts the env in the CPU's state at that entry's start
(qpos, qvel, the solver's warm start, the WUJI drives' full state, nail depth included) with the entry episode's
randomization (the hammer's inertial, the skin's friction and contact softness, the arm servo gains, the nail, the
drives, the TaxelScan readout; `apply_params`). It then plays the entry, optionally perturbed (`perturb`):

- time warp w (playback rate) and amplitude a (the arm's displacement from the entry's first target, scaled), and a
  start delay d during which the base holds its first sample: library time u = w max(tau - d, 0);
- the arm stream: 100 Hz command ticks in env time, phased so that unperturbed they fall on the CPU's ticks; at each
  tick the target q, its velocity feedforward and the droop are interpolated between the CPU's ticks at u, and the
  servo target is q0 + a (q - q0) + droop + kd / kp a w qd (World.set_arm_position), held until the next tick;
- the hand base: the grip synergy at the library's scalar s(u) from the joint angles the host last read
  (WrapSynergy.mit_command), recomputed at every policy tick;
- the lookahead (arm part; the hand's targets do not move ahead under this base), refreshed on the arm's ticks: the
  library's displacements at the LOOKAHEAD horizons, re-sampled at the warped horizons and scaled by a;
- the forecast: the library's truth (event time mapped to env time, amplitude scaled by a w) through the torch
  ForecastModel.

Unperturbed (w = a = 1, d = 0) an env replays the entry exactly as base_library.replay_entry does on the CPU.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from tactile_rl.mdp.action_map import TorchSynergy
from tactile_rl.mdp.forecast import TorchForecastModel
from tactile_rl.mdp.runtime import tactile_runtime
from tactile_sim.behaviors.base_library import DRIVE_STATE, arm_tick_phase_ms, base_signature, load_library
from tactile_sim.config import ForecastCfg
from tactile_sim.control.obs_spec import LOOKAHEAD, N_ARM, N_JOINTS
from tactile_sim.sim.training_parts import residual_training_config
from tactile_sim.sim.wuji_drive import JOINT_GROUPS, joint_group

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

TICK = 0.01  # s, the arm's command period (ArmCfg.command_rate 100 Hz)


class BaseLibrary:
    """The library's entries on the device: streams padded to the longest entry (holding the last sample), the arm's
    command ticks per entry, start states, episode randomization and the CPU's outcomes."""

    def __init__(self, path: str, device: str | torch.device, entries: list[int] | None = None):
        lib = load_library(path)
        self.path = str(path)
        self.meta = json.loads(str(lib["meta"]))
        want = base_signature(residual_training_config())
        if self.meta.get("base") != want:
            warnings.warn(f"{path}: the library's base {self.meta.get('base')} is not residual_config's {want}; "
                          "regenerate it (python -m tactile_sim.behaviors.base_library export)", stacklevel=2)
        dev = self.dev = torch.device(device)
        idx = np.arange(len(lib["e_t0"])) if entries is None else np.asarray(entries)
        self.index = idx
        self.E = E = len(idx)
        lens = lib["e_len"][idx].astype(int)
        L = int(lens.max())
        starts = lib["e_start"][idx].astype(int)
        rows = np.minimum(np.arange(L)[None], lens[:, None] - 1) + starts[:, None]  # (E, L): edge-padded
        f32, f64 = dict(dtype=torch.float32, device=dev), dict(dtype=torch.float64, device=dev)

        def stream(k, **kw):
            return torch.as_tensor(lib[f"s_{k}"][rows], **kw)

        self.len = torch.as_tensor(lens, dtype=torch.long, device=dev)
        self.arm_q, self.arm_qd, self.arm_droop = (stream(k, **f64) for k in ("arm_q", "arm_qd", "arm_droop"))
        self.s = stream("s", **f32)
        self.look = stream("look", **f32).reshape(E, L, len(LOOKAHEAD), N_ARM)
        self.fc_valid = stream("fc_valid", dtype=torch.bool, device=dev)
        self.fc_t_event = stream("fc_t_event", **f64)
        self.fc_regime = stream("fc_regime", dtype=torch.long, device=dev)
        self.fc_f, self.fc_amp = stream("fc_f", **f64), stream("fc_amp", **f64)
        self.phase = stream("phase", dtype=torch.long, device=dev)
        self.t0 = torch.as_tensor(lib["e_t0"][idx], **f64)
        self.tick_ms = torch.as_tensor([arm_tick_phase_ms(float(t)) for t in lib["e_t0"][idx]], dtype=torch.long,
                                       device=dev)
        # the CPU's tick values per entry (ticks at phase + 10 k ms), edge-padded
        nT = (L + 9) // 10 + 1
        k = np.arange(nT)[None] * 10 + self.tick_ms.cpu().numpy()[:, None]
        trow = np.minimum(k, lens[:, None] - 1)
        e_ar = np.arange(E)[:, None]
        self.q_tick = self.arm_q[e_ar, trow]
        self.qd_tick = self.arm_qd[e_ar, trow]
        self.droop_tick = self.arm_droop[e_ar, trow]
        self.nT = nT
        self.qpos = torch.as_tensor(lib["e_qpos"][idx], **f32)
        self.qvel = torch.as_tensor(lib["e_qvel"][idx], **f32)
        self.qacc_ws = torch.as_tensor(lib["e_qacc_ws"][idx], **f32)
        self.drive = {k: torch.as_tensor(lib[f"e_drive_{k}"][idx], device=dev) for k in DRIVE_STATE}
        self.hih0 = torch.as_tensor(lib["e_hih0"][idx], **f32)
        self.t_swing = torch.as_tensor(lib["e_t_swing"][idx], **f64)
        self.t_c_pred = torch.as_tensor(lib["e_t_c_pred"][idx], **f64)
        self.outcome = {k: lib[f"e_{k}"][idx] for k in ("hit", "drop", "advance", "peak", "width", "impulse",
                                                       "v_face", "t_contact", "slip_trans", "slip_rot")}
        ep = lib["e_episode"][idx].astype(int)
        self.episode = ep
        meta_keys = [k[3:] for k in lib if k.startswith("ep_") and k not in ("ep_cfg", "ep_seed", "ep_wall_s")]
        self.params = {k: torch.as_tensor(lib[f"ep_{k}"][ep], **f64) for k in meta_keys}

    def entry_cfg(self, e: int) -> str:
        """The SimConfig key of entry e's episode."""
        return str(load_library(self.path)["ep_cfg"][self.episode[e]])


@dataclass(kw_only=True)
class BaseTrajectoryCommandCfg(CommandTermCfg):
    scene_key: str
    library: str
    entries: tuple[int, ...] | None = None  # a subset of the library (None: all)
    perturb: bool = True
    time_warp: tuple[float, float] = (0.8, 1.2)
    amplitude: tuple[float, float] = (0.9, 1.1)
    start_delay: tuple[float, float] = (0.0, 0.1)  # s
    apply_params: bool = True  # the entries' randomization
    assign: str = "random"  # "random" entries, or "sequential": env i plays entry i mod E (evaluation, parity)
    forecast: ForecastCfg = field(default_factory=ForecastCfg)
    task: str = "hammer"
    mask_blocks: tuple[str, ...] = ()
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # one entry per episode

    def build(self, env: ManagerBasedRlEnv) -> BaseTrajectoryCommand:
        return BaseTrajectoryCommand(self, env)


class BaseTrajectoryCommand(CommandTerm):
    cfg: BaseTrajectoryCommandCfg

    def __init__(self, cfg: BaseTrajectoryCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.rt = rt = tactile_runtime(env, cfg.scene_key, task=cfg.task, mask_blocks=cfg.mask_blocks)
        parts = rt.parts
        dev, N = self.device, self.num_envs
        self.lib = BaseLibrary(cfg.library, dev, None if cfg.entries is None else list(cfg.entries))
        m = env.sim.mj_model
        self.arm_act = torch.as_tensor([m.actuator(f"robot/{a}").id for a in parts.arm_actuators], device=dev)
        b = m.body(f"robot/{parts.hammer_body}").id
        self.hammer_bid = b
        self.hammer_dofs = torch.as_tensor(range(int(m.jnt_dofadr[m.joint(f"robot/{parts.hammer_joint}").id]),
                                                 int(m.jnt_dofadr[m.joint(f"robot/{parts.hammer_joint}").id]) + 6),
                                           device=dev)
        self.skin_pairs = torch.as_tensor(parts.skin_pairs, dtype=torch.long, device=dev)
        self.soft_pairs = torch.as_tensor(parts.skin_pairs + parts.self_pairs, dtype=torch.long, device=dev)
        self.group = torch.as_tensor([JOINT_GROUPS.index(joint_group(j)) for j in parts.hand_joints], device=dev)
        self.synergy = TorchSynergy(parts.synergy, dev)
        self.forecaster = TorchForecastModel(N, cfg.forecast, dev)
        f64 = dict(dtype=torch.float64, device=dev)
        self.entry = torch.zeros(N, dtype=torch.long, device=dev)
        self.warp = torch.ones(N, **f64)
        self.amp = torch.ones(N, **f64)
        self.delay = torch.zeros(N, **f64)
        self.kd_kp = torch.as_tensor(parts.arm_kd / parts.arm_kp, **f64).expand(N, -1).clone()
        self.q0 = torch.zeros(N, N_ARM, **f64)
        self.ref_rel = torch.zeros(N, 6, device=dev)
        self.depth_prev = torch.zeros(N, **f64)  # the rewards' previous nail depth and tool pose in the hand
        self.hih_prev = torch.zeros(N, 6, device=dev)
        self.knots = torch.as_tensor([0.0] + [1e3 * h for h in LOOKAHEAD], **f64)  # ms
        # what the policy tick and the observation read (refreshed by _update_command)
        self.s_now = torch.full((N,), parts.s_hold, device=dev)
        self.base = torch.zeros(N, 5, N_JOINTS, device=dev)
        self.lookahead = torch.zeros(N, (N_JOINTS + N_ARM) * len(LOOKAHEAD), device=dev)
        self.forecast = torch.zeros(N, 13, device=dev)
        z64 = torch.zeros(N, **f64)
        self.truth = {"valid": torch.zeros(N, dtype=torch.bool, device=dev), "t_event": z64, "f": z64, "amp": z64,
                      "regime": torch.zeros(N, dtype=torch.long, device=dev),
                      "phase": torch.zeros(N, dtype=torch.long, device=dev), "t_c_pred": z64}
        self.trigger = torch.zeros(N, dtype=torch.bool, device=dev)
        self.metrics["entry_hit_cpu"] = torch.zeros(N, device=dev)
        self._hit_cpu = torch.as_tensor(self.lib.outcome["hit"].astype(float), dtype=torch.float32, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.lookahead, self.forecast], dim=1)

    # ------------------------------------------------------------------ time maps
    def u(self, tau: torch.Tensor) -> torch.Tensor:
        """Library time (s) at env time tau."""
        return self.warp * torch.clamp(tau - self.delay, min=0.0)

    def index(self, tau: torch.Tensor) -> torch.Tensor:
        """The library's 1 ms sample at env time tau, per env."""
        i = torch.floor(self.u(tau) * 1000.0 + 1e-6).long()
        return torch.minimum(i, self.lib.len[self.entry] - 1)

    def _tick(self, tau: torch.Tensor):
        """The arm's latest command tick at or before tau: (started, CPU tick coordinate kf, library time u_j)."""
        e = self.entry
        ph = self.lib.tick_ms[e].double() * 1e-3
        j = torch.floor((tau - self.delay - ph) / TICK + 1e-6)
        u_j = self.warp * (ph + j * TICK)
        kf = (u_j * 1e3 - self.lib.tick_ms[e].double()) / 10.0
        return j >= 0, kf, u_j

    def done(self) -> torch.Tensor:
        """The entry has played out."""
        return self.u(self.rt.t) >= (self.lib.len[self.entry] - 1).double() * 1e-3

    # ------------------------------------------------------------------ per substep: the arm stream
    def arm_target(self, tau: torch.Tensor) -> torch.Tensor:
        e, lib = self.entry, self.lib
        started, kf, _ = self._tick(tau)
        k0 = torch.floor(kf + 1e-6).long().clamp(0, lib.nT - 1)
        k1 = (k0 + 1).clamp(max=lib.nT - 1)
        fr = (kf - k0.double()).clamp(0.0, 1.0)[:, None]

        def interp(x):
            a, b = x[e, k0], x[e, k1]
            return a + fr * (b - a)

        q, qd, droop = interp(lib.q_tick), interp(lib.qd_tick), interp(lib.droop_tick)
        q = torch.where(started[:, None], q, lib.arm_q[e, 0])
        qd = torch.where(started[:, None], qd, lib.arm_qd[e, 0])
        droop = torch.where(started[:, None], droop, lib.arm_droop[e, 0])
        a, w = self.amp[:, None], self.warp[:, None]
        return (self.q0 + a * (q - self.q0) + droop + self.kd_kp * a * w * qd).float()

    # ------------------------------------------------------------------ per policy tick
    def _update_metrics(self) -> None:
        pass

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids  # every env's outputs are pure functions of its time (the forecast's state only moves on its grid)
        rt, lib, e = self.rt, self.lib, self.entry
        tau = rt.t
        i = self.index(tau)
        self.s_now = lib.s[e, i]
        self.base = self.synergy.mit_command(self.s_now, rt.rb_q.value.float())
        # lookahead on the arm's ticks, re-sampled at the warped horizons and scaled
        started, _, u_j = self._tick(tau)
        il = torch.where(started, torch.floor(u_j * 1000.0 + 1e-6).long(), torch.zeros_like(i))
        il = torch.minimum(il.clamp(min=0), lib.len[e] - 1)
        disp = lib.look[e, il].double()  # (N, H, 7)
        vals = torch.cat([torch.zeros_like(disp[:, :1]), disp], dim=1)  # knots 0, 10, 25, 50 ms
        x = self.knots[1:][None] * self.warp[:, None]  # (N, H) the horizons in library time
        seg = torch.searchsorted(self.knots, x.contiguous(), right=True).clamp(1, len(self.knots) - 1) - 1
        x0, x1 = self.knots[seg], self.knots[seg + 1]
        v0 = vals.gather(1, seg[..., None].expand(-1, -1, N_ARM))
        v1 = vals.gather(1, (seg + 1)[..., None].expand(-1, -1, N_ARM))
        arm = v0 + ((x - x0) / (x1 - x0))[..., None] * (v1 - v0)
        arm = arm * self.amp[:, None, None]
        look = torch.cat([torch.zeros(self.num_envs, len(LOOKAHEAD), N_JOINTS, device=self.device), arm.float()], 2)
        self.lookahead = look.reshape(self.num_envs, -1)
        # forecast truth in env time, then the forecast model
        valid = lib.fc_valid[e, i]
        t_ev = self.delay + lib.fc_t_event[e, i] / self.warp
        self.truth = {"valid": valid, "t_event": t_ev, "regime": lib.fc_regime[e, i].clamp(min=0),
                      "f": lib.fc_f[e, i] * self.warp, "amp": lib.fc_amp[e, i] * self.amp * self.warp,
                      "phase": lib.phase[e, i], "t_c_pred": self.delay + lib.t_c_pred[e] / self.warp}
        self.forecast = self.forecaster(tau, valid, torch.where(valid, t_ev, torch.zeros_like(t_ev)),
                                        self.truth["regime"], self.truth["f"], self.truth["amp"]).float()

    # ------------------------------------------------------------------ resets
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        c, lib, dev = self.cfg, self.lib, self.device
        n = len(env_ids)
        if c.assign == "sequential":
            e = env_ids % lib.E
        elif c.assign == "random":
            e = torch.randint(0, lib.E, (n,), device=dev)
        else:
            raise ValueError(f"unknown assign {c.assign!r} (expected 'random' or 'sequential')")
        self.entry[env_ids] = e
        if c.perturb:
            def u(lo_hi):
                return torch.empty(n, dtype=torch.float64, device=dev).uniform_(*lo_hi)

            self.warp[env_ids], self.amp[env_ids], self.delay[env_ids] = (u(c.time_warp), u(c.amplitude),
                                                                          u(c.start_delay))
        else:
            self.warp[env_ids], self.amp[env_ids], self.delay[env_ids] = 1.0, 1.0, 0.0
        self.q0[env_ids] = lib.arm_q[e, 0]
        self.ref_rel[env_ids] = lib.hih0[e]
        self.hih_prev[env_ids] = lib.hih0[e]
        if self.rt.nail is not None:
            self.depth_prev[env_ids] = lib.qpos[e, self.rt.nail.qadr].double()
        if c.apply_params:
            self.apply_params(env_ids, e)
        self.write_state(env_ids, e)
        self.forecaster.reset(env_ids)
        self.metrics["entry_hit_cpu"][env_ids] = self._hit_cpu[e]

    def write_state(self, env_ids: torch.Tensor, e: torch.Tensor) -> None:
        """The CPU's state at the entry's start: qpos, qvel, warm start, the drives' state; time 0."""
        lib, d = self.lib, self._env.sim.data
        d.qpos[env_ids] = lib.qpos[e]
        d.qvel[env_ids] = lib.qvel[e]
        d.qacc_warmstart[env_ids] = lib.qacc_ws[e]
        d.time[env_ids] = 0.0
        drive = self.rt.drive
        st = {k: v[e] for k, v in lib.drive.items()}
        st = {k: v.to(getattr(drive, k).dtype) for k, v in st.items()}
        drive.set_state(st, env_ids)
        robot = self.rt.robot
        robot.data.joint_pos_target[env_ids[:, None], self.rt.arm_ids] = self.arm_target(self.rt.t)[env_ids]

    def apply_params(self, env_ids: torch.Tensor, e: torch.Tensor) -> None:
        """The entry episode's randomization (base_library.episode_params / model_constants) onto these envs."""
        P, model, rt = self.lib.params, self._env.sim.model, self.rt
        ids = env_ids
        b = self.hammer_bid
        model.body_mass[ids, b] = P["hammer_mass"][e].float()
        model.body_ipos[ids, b] = P["hammer_ipos"][e].float()
        model.body_iquat[ids, b] = P["hammer_iquat"][e].float()
        model.body_inertia[ids, b] = P["hammer_inertia"][e].float()
        model.body_invweight0[ids, b] = P["hammer_invweight0"][e].float()
        model.dof_invweight0[ids[:, None], self.hammer_dofs] = P["hammer_dof_invweight0"][e].float()
        mu = P["hand_friction"][e].float()
        sp = self.skin_pairs
        if len(sp):
            model.pair_friction[ids[:, None], sp, 0] = mu[:, None].expand(-1, len(sp))
            model.pair_friction[ids[:, None], sp, 1] = mu[:, None].expand(-1, len(sp))
        so = self.soft_pairs
        if len(so):
            model.pair_solref[ids[:, None], so, 0] = P["hand_solref_tc"][e].float()[:, None].expand(-1, len(so))
            model.pair_solref[ids[:, None], so, 1] = 1.0
        kp, kd = P["arm_kp"][e], P["arm_kd"][e]
        act = self.arm_act
        model.actuator_gainprm[ids[:, None], act, 0] = kp.float()
        model.actuator_biasprm[ids[:, None], act, 1] = -kp.float()
        model.actuator_biasprm[ids[:, None], act, 2] = -kd.float()
        self.kd_kp[ids] = kd / kp
        if rt.nail is not None:
            rt.nail.set_params(ids, resistance_0=P["nail_r0"][e], resistance_per_m=P["nail_r_per_m"][e],
                               vdr_zeta0=P["vdr_zeta0"][e], face_tc=P["face_tc"][e])
            rt.nail.reset(ids)
        lpf = P["sdk_lpf_hz"][e]
        i_rated = P["i_rated"][e][:, self.group]  # (n, J)
        rt.drive.set_params(ids, cmd_latency=P["cmd_latency"][e], sdk_lpf_hz=lpf, tau_i=P["tau_i"][e],
                            i_rated=i_rated, effort_limit_ratio=P["effort_limit_ratio"][e][:, None],
                            t_amb=P["t_amb"][e])
        rt.scan.set_params(ids, gain_mismatch=P["ts_gain_mismatch"][e], offset_lsb=P["ts_offset_lsb"][e],
                           f_half=P["ts_f_half"][e], bandwidth=P["ts_bandwidth"][e],
                           bridge_latency=(P["ts_bridge_lo"][e], P["ts_bridge_hi"][e]),
                           drop_prob=P["ts_drop_prob"][e], noise_scale=P["noise_scale"][e])
        rt.load.skin_depth[ids] = P["ts_skin_depth"][e].float()
        rt.rb_i.set_params(ids, noise_scale=P["noise_scale"][e])

    def param_vector(self) -> torch.Tensor:
        """The randomization each env runs with, scaled to O(1) (teacher / critic observation)."""
        P, e = self.lib.params, self.entry
        cols = [P["head_mass"][e] / 0.5, P["hand_friction"][e], P["hand_solref_tc"][e] / 0.003,
                P["nail_r0"][e] / 300.0, P["nail_r_per_m"][e] / 1e4, P["vdr_zeta0"][e], P["face_tc"][e] / 0.003,
                P["cmd_latency"][e] / 0.003, P["sdk_lpf_hz"][e] / 300.0, P["tau_i"][e] / 0.0005,
                P["i_rated"][e][:, 0], P["effort_limit_ratio"][e], (P["t_amb"][e] - 25.0) / 10.0,
                P["ts_gain_mismatch"][e] / 0.15, P["ts_offset_lsb"][e] / 6.0, P["ts_f_half"][e] / 8.0,
                P["ts_bandwidth"][e] / 500.0, P["ts_skin_depth"][e] / 0.003, P["ts_drop_prob"][e] / 0.01,
                P["ts_bridge_hi"][e] / 0.001, P["servo_p_mult"][e], torch.log(P["noise_scale"][e]),
                self.warp, self.amp, self.delay / 0.1]
        return torch.stack(cols, dim=1).float()


PARAM_DIM = 25
assert math.isfinite(TICK)
