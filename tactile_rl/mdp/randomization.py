"""Domain randomization on the GPU, drawn per env at reset: the shared table (ResidualDRCfg, which the CPU validator
draws per seed with sim.randomize.residual_dr) plus the plan's DR-table entries ResidualDRCfg does not carry
(`DRExtraCfg`).

Groups, each its own reset event (so a task picks what it randomizes), all writing one per-env state (`TactileDR`):

- `pair_solref_solimp`: the hand-tool contact pairs' friction (mu, and the torsional coefficient), contact time
  constant (solref, also on the hand's self pairs, as the testbed's gripper.hand_solref) and impedance d0 (solimp);
- `randomize_wuji_drive`: command latency (+ jitter), SDK low-pass (or off), current-loop time constant, rated current
  (Kt error, residual_dr's i_rated scale), effort limit, ambient temperature, current-readback noise scale; the hand
  joints' friction loss, damping and armature;
- `randomize_taxelscan`: per-taxel gain and offset errors (a new skin), the divider's f_half, skin bandwidth and
  depth, dead taxels, frame drops, the bridge latency's upper end, the residual loop's phase to the boards' SYNC;
- `randomize_plant`: the hammer head's mass (the hammer body's inertial and the constants MuJoCo derives from it, from
  a table compiled on the CPU), the arm servos' P multiplier (kd keeps its damping ratio, as robots.py) and a kd
  scale; the nail (resistance, its slope, the face's time constant and restitution) when the scene has one;
- `randomize_forecast`: the forecast's error model (bias, jitter, regime confusion, frequency and amplitude errors,
  latency, update rate, dropout, episode masking).

`TactileDR.scale` (the curriculum's) shrinks every range toward its nominal value: draw = nominal + scale (U(lo, hi) -
nominal); 0 is the nominal model, 1 the full table. The forecast has its own mode: "clean" (exact, timely, never
masked), "noisy" (the table) with `mask_prob` of whole episodes masked.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields

from tactile_rl.mdp.actions import base_command_term, hand_drive
from tactile_rl.scene.entities import ENTITY, reduced_scene_spec
from tactile_sim import names
from tactile_sim.config import ResidualDRCfg
from tactile_sim.sim.training_parts import training_parts

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

PREFIX = f"{ENTITY}/"
HEAD_MASSES = np.linspace(0.20, 0.75, 12)  # kg, the inertial table's grid (covers ResidualDRCfg.head_mass)


@dataclass(frozen=True)
class DRExtraCfg:
    """The plan's DR-table entries that ResidualDRCfg does not carry."""

    hand_torsion: tuple[float, float] = (0.01, 0.03)
    hand_solimp_d0: tuple[float, float] = (0.85, 0.95)
    joint_friction_scale: tuple[float, float] = (0.5, 1.5)
    joint_damping_scale: tuple[float, float] = (0.5, 1.5)
    armature_scale: tuple[float, float] = (0.7, 1.3)
    cmd_jitter: tuple[float, float] = (0.0, 0.00025)  # s
    dead_taxel_frac: tuple[float, float] = (0.0, 0.02)
    rt_phase: tuple[float, float] = (0.0, 0.001)  # s, the residual loop's phase to SYNC
    arm_kd_scale: tuple[float, float] = (0.7, 1.3)
    # the forecast (behaviors.forecast's error model)
    fc_bias_sigma: tuple[float, float] = (0.005, 0.030)
    fc_jitter_sigma: tuple[float, float] = (0.005, 0.020)
    fc_class_confusion: tuple[float, float] = (0.0, 0.20)
    fc_freq_err: tuple[float, float] = (0.05, 0.20)
    fc_amp_err: tuple[float, float] = (0.10, 0.40)
    fc_latency: tuple[float, float] = (0.05, 0.25)  # s
    fc_update_rate: tuple[float, float] = (2.0, 10.0)  # Hz
    fc_dropout: tuple[float, float] = (0.0, 0.10)
    fc_mask_prob: float = 0.25


# ---------------------------------------------------------------------------------------------- the hammer's inertial


@functools.lru_cache(maxsize=8)
def hammer_inertial_table(scene_key: str, masses: tuple[float, ...] = tuple(HEAD_MASSES)) -> dict[str, np.ndarray]:
    """The hammer body's inertial (mass, ipos, iquat, inertia) and MuJoCo's derived constants (body_invweight0, its
    free joint's dof_invweight0) for each head mass in `masses`: the training spec compiled on the CPU with the head
    geom's mass changed (MuJoCo's compiler and mj_setConst, as the testbed's model for that mass)."""
    parts = training_parts(scene_key)
    out: dict[str, list] = {k: [] for k in ("mass", "ipos", "iquat", "inertia", "invweight0", "dof_invweight0")}
    for mh in masses:
        spec = reduced_scene_spec(scene_key)
        spec.geom(names.HAMMER_HEAD_GEOM).mass = float(mh)
        m = spec.compile()
        b = m.body(parts.hammer_body).id
        j = m.joint(parts.hammer_joint)
        a = int(j.dofadr[0])
        out["mass"].append(float(m.body_mass[b]))
        out["ipos"].append(m.body_ipos[b].copy())
        out["iquat"].append(m.body_iquat[b].copy())
        out["inertia"].append(m.body_inertia[b].copy())
        out["invweight0"].append(m.body_invweight0[b].copy())
        out["dof_invweight0"].append(m.dof_invweight0[a:a + 6].copy())
    res = {k: np.asarray(v, float) for k, v in out.items()}
    res["head_mass"] = np.asarray(masses, float)
    return res


def _interp_table(table: dict[str, np.ndarray], head_mass: torch.Tensor) -> dict[str, torch.Tensor]:
    """Linear interpolation of the inertial table at each env's head mass (iquat renormalized)."""
    dev = head_mass.device
    grid = torch.as_tensor(table["head_mass"], dtype=torch.float64, device=dev)
    x = head_mass.double().clamp(float(grid[0]), float(grid[-1]))
    i = torch.searchsorted(grid, x, right=True).clamp(1, len(grid) - 1) - 1
    w = ((x - grid[i]) / (grid[i + 1] - grid[i]))[:, None]
    out = {}
    for k in ("mass", "ipos", "iquat", "inertia", "invweight0", "dof_invweight0"):
        v = torch.as_tensor(table[k], dtype=torch.float64, device=dev)
        v = v[:, None] if v.ndim == 1 else v
        r = v[i] + w * (v[i + 1] - v[i])
        out[k] = r.squeeze(-1) if table[k].ndim == 1 else r
    out["iquat"] = out["iquat"] / out["iquat"].norm(dim=-1, keepdim=True)
    return out


# ---------------------------------------------------------------------------------------------- the state


class TactileDR:
    """Every env's drawn parameters (float64, nominal until drawn) and the model indices the events write."""

    KEYS = ("head_mass", "hand_friction", "hand_torsion", "hand_solref_tc", "hand_solimp_d0", "cmd_latency",
            "cmd_jitter", "sdk_lpf_hz", "tau_i", "kt_scale", "effort_limit_ratio", "t_amb", "noise_scale",
            "joint_friction_scale", "joint_damping_scale", "armature_scale", "ts_gain_mismatch", "ts_offset_lsb",
            "ts_f_half", "ts_bandwidth", "ts_skin_depth", "ts_drop_prob", "ts_bridge_hi", "dead_taxel_frac",
            "rt_phase", "servo_p_mult", "arm_kd_scale", "nail_r0", "nail_r_per_m", "face_tc", "vdr_zeta0",
            "fc_bias_sigma", "fc_jitter_sigma", "fc_class_confusion", "fc_freq_err", "fc_amp_err", "fc_latency",
            "fc_update_rate", "fc_dropout", "fc_mask_prob")

    def __init__(self, env: ManagerBasedRlEnv, scene_key: str, cfg: ResidualDRCfg | None = None,
                 extra: DRExtraCfg | None = None):
        self.env = env
        self.scene_key = scene_key
        self.parts = parts = training_parts(scene_key)
        sc = parts.cfg
        self.cfg = cfg or sc.residual_dr
        self.extra = extra or DRExtraCfg()
        self.N = N = env.num_envs
        self.dev = dev = env.device
        self.scale = 1.0
        self.forecast_mode = "noisy"  # or "clean"
        self.forecast_masking = True
        m = env.sim.mj_model
        self.m = m
        self.skin_pairs = torch.as_tensor(parts.skin_pairs, dtype=torch.long, device=dev)
        self.soft_pairs = torch.as_tensor(parts.skin_pairs + parts.self_pairs, dtype=torch.long, device=dev)
        self.hand_dofs = torch.as_tensor([int(m.joint(PREFIX + j).dofadr[0]) for j in parts.hand_joints],
                                         dtype=torch.long, device=dev)
        self.arm_act = torch.as_tensor([m.actuator(PREFIX + a).id for a in parts.arm_actuators], dtype=torch.long,
                                       device=dev)
        self.hammer_bid = m.body(PREFIX + parts.hammer_body).id
        a = int(m.joint(PREFIX + parts.hammer_joint).dofadr[0])
        self.hammer_dofs = torch.as_tensor(range(a, a + 6), dtype=torch.long, device=dev)
        f64 = dict(dtype=torch.float64, device=dev)
        # nominal values (the model as compiled; the configuration for the rest)
        w, s, p = sc.wuji, sc.sensors, sc.plant
        sp = parts.skin_pairs[0] if parts.skin_pairs else None
        self.nominal = {
            "head_mass": float(sc.hammer.head_mass),
            "hand_friction": float(m.pair_friction[sp, 0]) if sp is not None else 1.0,
            "hand_torsion": float(m.pair_friction[sp, 2]) if sp is not None else 0.02,
            "hand_solref_tc": float(m.pair_solref[sp, 0]) if sp is not None else 0.002,
            "hand_solimp_d0": float(m.pair_solimp[sp, 0]) if sp is not None else 0.9,
            "cmd_latency": float(w.cmd_latency), "cmd_jitter": float(w.cmd_jitter),
            "sdk_lpf_hz": float(w.sdk_lpf_hz or 0.0), "tau_i": float(w.tau_i), "kt_scale": 1.0,
            "effort_limit_ratio": float(w.effort_limit_ratio), "t_amb": float(w.t_amb), "noise_scale": 1.0,
            "joint_friction_scale": 1.0, "joint_damping_scale": 1.0, "armature_scale": 1.0,
            "ts_gain_mismatch": float(s.ts_gain_mismatch), "ts_offset_lsb": float(s.ts_offset_lsb),
            "ts_f_half": float(s.ts_f_half), "ts_bandwidth": float(s.ts_bandwidth),
            "ts_skin_depth": float(s.ts_skin_depth), "ts_drop_prob": float(s.ts_drop_prob),
            "ts_bridge_hi": float(s.ts_bridge_latency[1]), "dead_taxel_frac": 0.0, "rt_phase": 0.0,
            "servo_p_mult": 1.0, "arm_kd_scale": 1.0, "nail_r0": float(p.resistance_0),
            "nail_r_per_m": float(p.resistance_per_m), "face_tc": float(sc.hammer.face_solref[0]),
            "vdr_zeta0": float(p.vdr_zeta0),
            "fc_bias_sigma": 0.0, "fc_jitter_sigma": 0.0, "fc_class_confusion": 0.0, "fc_freq_err": 0.0,
            "fc_amp_err": 0.0, "fc_latency": 0.05, "fc_update_rate": 10.0, "fc_dropout": 0.0, "fc_mask_prob": 0.0}
        self.v = {k: torch.full((N,), self.nominal[k], **f64) for k in self.KEYS}
        # model nominal arrays the scales multiply
        self.jf0 = torch.as_tensor(m.dof_frictionloss[parts_dofs(m, parts)], **f64)
        self.jd0 = torch.as_tensor(m.dof_damping[parts_dofs(m, parts)], **f64)
        self.ja0 = torch.as_tensor(m.dof_armature[parts_dofs(m, parts)], **f64)
        self.kp0 = torch.as_tensor(m.actuator_gainprm[self.arm_act.cpu().numpy(), 0], **f64)
        self.kd0 = torch.as_tensor(-m.actuator_biasprm[self.arm_act.cpu().numpy(), 2], **f64)
        self.i_rated0 = None  # the drive's nominal rated currents, set on first use
        self._table = None

    # ------------------------------------------------------------------ draws
    def _range(self, key: str, lohi) -> tuple[float, float]:
        nom = self.nominal[key]
        lo, hi = float(lohi[0]), float(lohi[1])
        return nom + self.scale * (lo - nom), nom + self.scale * (hi - nom)

    def draw(self, env_ids: torch.Tensor, key: str, lohi, log: bool = False) -> torch.Tensor:
        lo, hi = self._range(key, lohi)
        n = len(env_ids)
        u = torch.rand(n, dtype=torch.float64, device=self.dev)
        if log and lo > 0 and hi > 0:
            x = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        else:
            x = lo + u * (hi - lo)
        self.v[key][env_ids] = x
        return x

    def table(self) -> dict[str, np.ndarray]:
        if self._table is None:
            self._table = hammer_inertial_table(self.scene_key)
        return self._table

    def param_vector(self) -> torch.Tensor:
        """(N, PARAM_DIM) the drawn parameters, scaled to O(1) (teacher / critic observation)."""
        v = self.v
        cols = [v["head_mass"] / 0.5, v["hand_friction"], v["hand_torsion"] / 0.02, v["hand_solref_tc"] / 0.003,
                (v["hand_solimp_d0"] - 0.9) / 0.05, v["cmd_latency"] / 0.003, v["sdk_lpf_hz"] / 300.0,
                v["tau_i"] / 0.0005, v["kt_scale"], v["effort_limit_ratio"], (v["t_amb"] - 25.0) / 10.0,
                torch.log(v["noise_scale"]), v["joint_friction_scale"], v["joint_damping_scale"], v["armature_scale"],
                v["ts_gain_mismatch"] / 0.15, v["ts_offset_lsb"] / 6.0, v["ts_f_half"] / 8.0,
                v["ts_bandwidth"] / 500.0, v["ts_skin_depth"] / 0.003, v["ts_drop_prob"] / 0.01,
                v["dead_taxel_frac"] / 0.02, v["rt_phase"] / 0.001, v["servo_p_mult"], v["arm_kd_scale"],
                v["nail_r0"] / 100.0, v["nail_r_per_m"] / 5e3, v["face_tc"] / 0.003, v["vdr_zeta0"],
                v["fc_bias_sigma"] / 0.03, v["fc_latency"] / 0.25, v["fc_mask_prob"]]
        return torch.stack(cols, dim=1).float()


PARAM_DIM = 32


def parts_dofs(m, parts) -> np.ndarray:
    return np.array([int(m.joint(PREFIX + j).dofadr[0]) for j in parts.hand_joints], dtype=int)


def tactile_dr(env: ManagerBasedRlEnv, scene_key: str | None = None) -> TactileDR:
    """The env's DR state, created by the first event that asks."""
    st = getattr(env, "_tactile_dr", None)
    if st is None:
        if scene_key is None:
            raise RuntimeError("the DR state is created with the scene key")
        st = TactileDR(env, scene_key)
        env._tactile_dr = st
    return st


def _ids(env: ManagerBasedRlEnv, env_ids) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


# ---------------------------------------------------------------------------------------------- events

CONTACT_FIELDS = ("pair_friction", "pair_solref", "pair_solimp")
DRIVE_FIELDS = ("dof_frictionloss", "dof_damping", "dof_armature")
PLANT_FIELDS = ("body_mass", "body_ipos", "body_iquat", "body_inertia", "body_invweight0", "dof_invweight0",
                "actuator_gainprm", "actuator_biasprm", "pair_solref", "dof_frictionloss")


@requires_model_fields(*CONTACT_FIELDS)
def pair_solref_solimp(env: ManagerBasedRlEnv, env_ids, scene_key: str) -> None:
    """The hand-tool contact: friction (mu, torsion), time constant (also the hand's self pairs), impedance d0."""
    dr = tactile_dr(env, scene_key)
    ids = _ids(env, env_ids)
    c, x = dr.cfg, dr.extra
    mu = dr.draw(ids, "hand_friction", c.hand_friction).float()
    tor = dr.draw(ids, "hand_torsion", x.hand_torsion).float()
    tc = dr.draw(ids, "hand_solref_tc", c.hand_solref_tc).float()
    d0 = dr.draw(ids, "hand_solimp_d0", x.hand_solimp_d0).float()
    model = env.sim.model
    sp, so = dr.skin_pairs, dr.soft_pairs
    if len(sp):
        n = len(sp)
        model.pair_friction[ids[:, None], sp, 0] = mu[:, None].expand(-1, n)
        model.pair_friction[ids[:, None], sp, 1] = mu[:, None].expand(-1, n)
        model.pair_friction[ids[:, None], sp, 2] = tor[:, None].expand(-1, n)
        model.pair_solimp[ids[:, None], sp, 0] = d0[:, None].expand(-1, n)
    if len(so):
        model.pair_solref[ids[:, None], so, 0] = tc[:, None].expand(-1, len(so))
        model.pair_solref[ids[:, None], so, 1] = 1.0


@requires_model_fields(*DRIVE_FIELDS)
def randomize_wuji_drive(env: ManagerBasedRlEnv, env_ids, scene_key: str) -> None:
    """The WUJI drives (latency, low-pass, current loop, Kt, effort limit, ambient, readback noise) and the hand
    joints' friction loss, damping and armature."""
    dr = tactile_dr(env, scene_key)
    ids = _ids(env, env_ids)
    c, x = dr.cfg, dr.extra
    drive = hand_drive(env.scene[ENTITY])
    if dr.i_rated0 is None:
        dr.i_rated0 = drive.i_rated[0].clone()
    lat = dr.draw(ids, "cmd_latency", c.cmd_latency)
    jit = dr.draw(ids, "cmd_jitter", x.cmd_jitter)
    # the SDK low-pass is off nominally: the scale makes it rarer, not a lower cutoff
    on = torch.rand(len(ids), dtype=torch.float64, device=dr.dev) < c.sdk_lpf_prob * dr.scale
    u = torch.rand(len(ids), dtype=torch.float64, device=dr.dev)
    lpf = torch.where(on, c.sdk_lpf_hz[0] + u * (c.sdk_lpf_hz[1] - c.sdk_lpf_hz[0]), torch.zeros_like(u))
    dr.v["sdk_lpf_hz"][ids] = lpf
    tau_i = dr.draw(ids, "tau_i", c.tau_i)
    kt = dr.draw(ids, "kt_scale", c.kt_scale)
    eff = dr.draw(ids, "effort_limit_ratio", c.effort_limit_ratio)
    t_amb = dr.draw(ids, "t_amb", c.t_amb)
    ns = dr.draw(ids, "noise_scale", c.sensor_noise_scale, log=True)
    drive.set_params(ids, cmd_latency=lat, cmd_jitter=jit, sdk_lpf_hz=lpf, tau_i=tau_i,
                     i_rated=dr.i_rated0[None].double() * kt[:, None], effort_limit_ratio=eff[:, None], t_amb=t_amb)
    drive.T[ids] = t_amb[:, None]  # the windings start at ambient
    rt = getattr(env, "_tactile_rt", None)
    if rt is not None:
        rt.rb_i.set_params(ids, noise_scale=ns)
    fs = dr.draw(ids, "joint_friction_scale", x.joint_friction_scale)
    ds = dr.draw(ids, "joint_damping_scale", x.joint_damping_scale)
    as_ = dr.draw(ids, "armature_scale", x.armature_scale)
    model = env.sim.model
    hd = dr.hand_dofs
    model.dof_frictionloss[ids[:, None], hd] = (dr.jf0[None] * fs[:, None]).float()
    model.dof_damping[ids[:, None], hd] = (dr.jd0[None] * ds[:, None]).float()
    model.dof_armature[ids[:, None], hd] = (dr.ja0[None] * as_[:, None]).float()


def randomize_taxelscan(env: ManagerBasedRlEnv, env_ids, scene_key: str) -> None:
    """The TaxelScan readout and skin (a new physical skin per draw), dead taxels and the loop's phase to SYNC."""
    dr = tactile_dr(env, scene_key)
    ids = _ids(env, env_ids)
    c, x = dr.cfg, dr.extra
    rt = getattr(env, "_tactile_rt", None)
    if rt is None:
        return
    g = dr.draw(ids, "ts_gain_mismatch", c.ts_gain_mismatch)
    o = dr.draw(ids, "ts_offset_lsb", c.ts_offset_lsb)
    fh = dr.draw(ids, "ts_f_half", c.ts_f_half)
    bw = dr.draw(ids, "ts_bandwidth", c.ts_bandwidth)
    sd = dr.draw(ids, "ts_skin_depth", c.ts_skin_depth)
    dp = dr.draw(ids, "ts_drop_prob", c.ts_drop_prob)
    bh = dr.draw(ids, "ts_bridge_hi", c.ts_bridge_hi)
    dead = dr.draw(ids, "dead_taxel_frac", x.dead_taxel_frac)
    ph = dr.draw(ids, "rt_phase", x.rt_phase)
    lo = torch.full_like(bh, float(dr.parts.cfg.sensors.ts_bridge_latency[0]))
    rt.scan.set_params(ids, gain_mismatch=g, offset_lsb=o, f_half=fh, bandwidth=bw, bridge_latency=(lo, bh),
                       drop_prob=dp, noise_scale=dr.v["noise_scale"][ids], dead_frac=dead, phase=ph)
    rt.load.skin_depth[ids] = sd.float()


@requires_model_fields(*PLANT_FIELDS)
def randomize_plant(env: ManagerBasedRlEnv, env_ids, scene_key: str) -> None:
    """The hammer head's mass (inertial and derived constants), the arm servos, and the nail when there is one."""
    dr = tactile_dr(env, scene_key)
    ids = _ids(env, env_ids)
    c, x = dr.cfg, dr.extra
    model = env.sim.model
    hm = dr.draw(ids, "head_mass", c.head_mass)
    t = _interp_table(dr.table(), hm)
    b = dr.hammer_bid
    model.body_mass[ids, b] = t["mass"].float()
    model.body_ipos[ids, b] = t["ipos"].float()
    model.body_iquat[ids, b] = t["iquat"].float()
    model.body_inertia[ids, b] = t["inertia"].float()
    model.body_invweight0[ids, b] = t["invweight0"].float()
    model.dof_invweight0[ids[:, None], dr.hammer_dofs] = t["dof_invweight0"].float()
    pm = dr.draw(ids, "servo_p_mult", c.servo_p_mult)
    ks = dr.draw(ids, "arm_kd_scale", x.arm_kd_scale)
    kp = dr.kp0[None] * pm[:, None]
    kd = dr.kd0[None] * torch.sqrt(pm)[:, None] * ks[:, None]
    act = dr.arm_act
    model.actuator_gainprm[ids[:, None], act, 0] = kp.float()
    model.actuator_biasprm[ids[:, None], act, 1] = -kp.float()
    model.actuator_biasprm[ids[:, None], act, 2] = -kd.float()
    cmd = base_command_term(env)
    if cmd is not None and hasattr(cmd, "set_arm_gains"):
        cmd.set_arm_gains(ids, kp, kd)
    rt = getattr(env, "_tactile_rt", None)
    if rt is not None and rt.nail is not None:
        r0 = dr.draw(ids, "nail_r0", c.nail_resistance_0)
        rpm = dr.draw(ids, "nail_r_per_m", c.nail_resistance_per_m)
        tc = dr.draw(ids, "face_tc", c.face_timeconst)
        z0 = dr.draw(ids, "vdr_zeta0", c.vdr_zeta0)
        rt.nail.set_params(ids, resistance_0=r0, resistance_per_m=rpm, vdr_zeta0=z0, face_tc=tc)
        rt.nail.reset(ids)


def randomize_forecast(env: ManagerBasedRlEnv, env_ids, scene_key: str) -> None:
    """The forecast's error model per env ("clean" mode: exact and never masked)."""
    dr = tactile_dr(env, scene_key)
    ids = _ids(env, env_ids)
    x = dr.extra
    cmd = base_command_term(env)
    if cmd is None:
        return
    fc = cmd.forecaster
    if dr.forecast_mode == "clean":
        for k in ("fc_bias_sigma", "fc_jitter_sigma", "fc_class_confusion", "fc_freq_err", "fc_amp_err",
                  "fc_dropout", "fc_mask_prob"):
            dr.v[k][ids] = 0.0
        dr.v["fc_latency"][ids] = dr.nominal["fc_latency"]
        dr.v["fc_update_rate"][ids] = dr.nominal["fc_update_rate"]
    else:
        # the forecast's ranges are not shrunk by the DR scale: the stage that turns them on wants them whole
        for key, lohi in (("fc_bias_sigma", x.fc_bias_sigma), ("fc_jitter_sigma", x.fc_jitter_sigma),
                          ("fc_class_confusion", x.fc_class_confusion), ("fc_freq_err", x.fc_freq_err),
                          ("fc_amp_err", x.fc_amp_err), ("fc_latency", x.fc_latency),
                          ("fc_update_rate", x.fc_update_rate), ("fc_dropout", x.fc_dropout)):
            u = torch.rand(len(ids), dtype=torch.float64, device=dr.dev)
            dr.v[key][ids] = lohi[0] + u * (lohi[1] - lohi[0])
        dr.v["fc_mask_prob"][ids] = x.fc_mask_prob if dr.forecast_masking else 0.0
    v = dr.v
    fc.set_params(ids, mask_prob=v["fc_mask_prob"][ids], dropout=v["fc_dropout"][ids],
                  bias_sigma=v["fc_bias_sigma"][ids], jitter_sigma=v["fc_jitter_sigma"][ids],
                  class_confusion=v["fc_class_confusion"][ids], freq_err=v["fc_freq_err"][ids],
                  amp_err=v["fc_amp_err"][ids], update_rate=v["fc_update_rate"][ids], latency=v["fc_latency"][ids])


__all__ = ["DRExtraCfg", "PARAM_DIM", "TactileDR", "hammer_inertial_table", "pair_solref_solimp", "randomize_forecast",
           "randomize_plant", "randomize_taxelscan", "randomize_wuji_drive", "tactile_dr"]
