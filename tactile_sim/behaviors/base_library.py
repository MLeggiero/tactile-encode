"""The base library: scripted hammer swings recorded on the CPU testbed for the GPU training env (tactile_rl).

`export_library` runs SwingBase episodes (behaviors.base_sources) with a zero residual on the reduced residual scene
(sim.training_parts.residual_training_config: 0.25 ms physics, no noslip, as MuJoCo Warp runs it), one residual_dr
draw per episode, and records every millisecond (the residual's 1 kHz clock, after every controller has run):

- the arm's 100 Hz servo stream as PositionL1 writes it: the joint target q, its velocity feedforward qd and the
  gravity droop (the servo holds q + droop + kd / kp qd until the next command tick);
- the hand base: the grip synergy's scalar s (the base MIT command is WrapSynergy.mit_command(s, q_read)) and the
  grip setpoint the grip loop tracks;
- the base's lookahead block, arm part (joint targets at the LOOKAHEAD times relative to the current target);
- the forecast truth SwingBase hands its ForecastModel (valid, event time, regime, frequency, amplitude);
- the swing phase and the nail depth.

Each swing is one entry, from its windup to the next windup (or the episode's end), with the state at its start
(qpos, qvel, the solver's warm start, the WUJI drives' full state), the episode's randomization (the residual_dr
draw, and the model constants it changes: the hammer's inertial, the arm servo gains, the face/nail stiffness), and
the CPU's outcome of the swing (hit, nail advance, peak force and width of the blow, slip; logging.strike_metrics).

`replay_entry` plays an entry open loop on a CPU World the way the GPU env does (tactile_rl.mdp.commands): the arm
stream on the 100 Hz command ticks, the synergy at s every millisecond, the nail plant every step. Comparing a replay
with the closed-loop run that recorded it isolates the replay's own error; comparing the GPU with the CPU replay
isolates the simulators'.

    python -m tactile_sim.behaviors.base_library export --episodes 20 --strikes 10 --workers 14 \
        --out runs/base_library/hammer_v1.npz
    python -m tactile_sim.behaviors.base_library replay --lib runs/base_library/hammer_v1.npz --entries 0:20
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from tactile_sim import names
from tactile_sim.config import SimConfig
from tactile_sim.control.interface import HandCommand
from tactile_sim.control.obs_spec import LOOKAHEAD, N_ARM, N_JOINTS, REGIMES

LIBRARY_VERSION = 1
PHASES = ("approach", "windup", "swing", "post", "settle", "done")
# per-episode randomization, flat: what tactile_rl applies per env (see episode_params)
PARAM_KEYS = ("head_mass", "hand_friction", "hand_solref_tc", "nail_r0", "nail_r_per_m", "vdr_zeta0", "face_tc",
              "cmd_latency", "sdk_lpf_hz", "tau_i", "effort_limit_ratio", "t_amb", "ts_gain_mismatch", "ts_offset_lsb",
              "ts_f_half", "ts_bandwidth", "ts_skin_depth", "ts_drop_prob", "ts_bridge_lo", "ts_bridge_hi",
              "servo_p_mult", "noise_scale")
DRIVE_STATE = ("t", "host", "host_valid", "filt", "filt_valid", "line_t", "line_cmd", "line_head", "line_n",
               "last_arrival", "active", "next_sdk", "next_servo", "tau_star", "i", "T", "fault", "n_sdk")
MAX_INFLIGHT = 8  # commands in transit kept in a drive state (tactile_rl's TorchWujiDrive line length)
OUTCOMES = ("hit", "drop", "advance", "peak", "peak2", "width", "impulse", "v_face", "t_contact", "slip_trans",
            "slip_rot")


# ---------------------------------------------------------------------------------------------- configuration


def episode_config(seed: int, rate: float = 1000.0, timestep: float = 0.00025, randomize: bool = True,
                   **sections) -> tuple[SimConfig, dict | None]:
    """The configuration of library episode `seed`: residual_training_config with that seed's residual_dr draw
    (none if not `randomize`)."""
    from tactile_sim.sim.randomize import apply_residual_dr, residual_dr
    from tactile_sim.sim.training_parts import residual_training_config

    cfg = residual_training_config(rate, timestep, **sections)
    if not randomize:
        return cfg, None
    draw = residual_dr(cfg, np.random.default_rng(seed))
    return apply_residual_dr(cfg, draw), draw


def episode_params(cfg: SimConfig, draw: dict | None) -> dict[str, float]:
    """The randomized parameters of an episode, flat (PARAM_KEYS), read back from its configuration."""
    s, w = cfg.sensors, cfg.wuji
    return {"head_mass": cfg.hammer.head_mass, "hand_friction": cfg.gripper.hand_friction,
            "hand_solref_tc": cfg.gripper.hand_solref[0], "nail_r0": cfg.plant.resistance_0,
            "nail_r_per_m": cfg.plant.resistance_per_m, "vdr_zeta0": cfg.plant.vdr_zeta0,
            "face_tc": cfg.hammer.face_solref[0], "cmd_latency": w.cmd_latency, "sdk_lpf_hz": w.sdk_lpf_hz or 0.0,
            "tau_i": w.tau_i, "effort_limit_ratio": w.effort_limit_ratio, "t_amb": w.t_amb,
            "ts_gain_mismatch": s.ts_gain_mismatch, "ts_offset_lsb": s.ts_offset_lsb, "ts_f_half": s.ts_f_half,
            "ts_bandwidth": s.ts_bandwidth, "ts_skin_depth": s.ts_skin_depth, "ts_drop_prob": s.ts_drop_prob,
            "ts_bridge_lo": s.ts_bridge_latency[0], "ts_bridge_hi": s.ts_bridge_latency[1],
            "servo_p_mult": cfg.arm.servo_p_mult,
            "noise_scale": float(draw["_runtime"]["sensor_noise_scale"]) if draw else 1.0}


def base_signature(cfg: SimConfig) -> dict:
    """What a recorded base depends on beyond the scene: the swing's grip schedule and the grip loop (the residual
    experiments' base holds a constant grip, open loop, once residual_config says so). A library recorded under other
    settings is stale (tactile_rl warns)."""
    return {"grip_loop": cfg.controller.grip_loop, "grip_profile": cfg.swing.grip_profile,
            "grip_margin_step": cfg.swing.grip_margin_step, "grip_hold": cfg.controller.grip_hold,
            "post_switch": cfg.swing.post_switch, "impact_sources": list(cfg.controller.impact_sources)}


def model_constants(w) -> dict[str, np.ndarray]:
    """The model constants an episode's randomization changes, read from its CPU model: the hammer body's inertial
    and the derived constants MuJoCo computes from it (mj_setConst), the arm servo gains, the face/nail stiffness."""
    m = w.model
    b = w.hammer_body
    dofs = slice(w.hammer_dofadr, w.hammer_dofadr + 6)
    return {"hammer_mass": np.array(m.body_mass[b]), "hammer_ipos": m.body_ipos[b].copy(),
            "hammer_iquat": m.body_iquat[b].copy(), "hammer_inertia": m.body_inertia[b].copy(),
            "hammer_invweight0": m.body_invweight0[b].copy(), "hammer_dof_invweight0": m.dof_invweight0[dofs].copy(),
            "arm_kp": w.servo_kp.copy(), "arm_kd": w.servo_kd.copy(), "k_face": np.array(w.plant.k_face),
            "i_rated": np.asarray(w.cfg.wuji.i_rated, float)}


# ---------------------------------------------------------------------------------------------- drive state


def drive_state(drive, t0: float) -> dict[str, np.ndarray]:
    """A WujiMitDrive's full state in tactile_rl's TorchWujiDrive layout, times relative to t0 (the drive at t0 is the
    drive at time 0 of the entry)."""
    n = drive.n
    host = drive.host.array() if drive.host is not None else np.zeros((5, n))
    line = list(drive.line)
    if len(line) > MAX_INFLIGHT:
        raise ValueError(f"{len(line)} commands in transit, more than {MAX_INFLIGHT}")
    line_t = np.zeros(MAX_INFLIGHT)
    line_cmd = np.zeros((MAX_INFLIGHT, 5, n))
    for k, (ta, x) in enumerate(line):
        line_t[k], line_cmd[k] = ta - t0, x
    return {"t": np.array(0.0), "host": host, "host_valid": np.array(drive.host is not None),
            "filt": drive.filt.copy() if drive.filt is not None else np.zeros((5, n)),
            "filt_valid": np.array(drive.filt is not None), "line_t": line_t, "line_cmd": line_cmd,
            "line_head": np.array(0), "line_n": np.array(len(line)),
            "last_arrival": np.array(drive.last_arrival - t0), "active": drive.active.copy(),
            "next_sdk": np.array(drive.next_sdk - t0), "next_servo": np.array(drive.next_servo - t0),
            "tau_star": drive.tau_star.copy(), "i": drive.i.copy(), "T": drive.T.copy(), "fault": drive.fault.copy(),
            "n_sdk": np.array(drive.n_sdk)}


def set_drive_state(drive, st: dict[str, np.ndarray], t0: float = 0.0) -> None:
    """The inverse of drive_state onto a WujiMitDrive whose clock reads t0 at the entry's start."""
    from collections import deque

    drive.host = HandCommand.from_array(t0, st["host"], src="library") if bool(st["host_valid"]) else None
    drive.filt = np.array(st["filt"], float) if bool(st["filt_valid"]) else None
    n_line = int(st["line_n"])
    head = int(st["line_head"])
    drive.line = deque((float(st["line_t"][(head + k) % MAX_INFLIGHT]) + t0,
                        np.array(st["line_cmd"][(head + k) % MAX_INFLIGHT], float)) for k in range(n_line))
    drive.last_arrival = float(st["last_arrival"]) + t0
    drive.active = np.array(st["active"], float)
    drive.next_sdk = float(st["next_sdk"]) + t0
    drive.next_servo = float(st["next_servo"]) + t0
    drive.tau_star = np.array(st["tau_star"], float)
    drive.i = np.array(st["i"], float)
    drive.T = np.array(st["T"], float)
    drive.fault = np.array(st["fault"], bool)
    drive.n_sdk = int(st["n_sdk"])


# ---------------------------------------------------------------------------------------------- recording


class _Recorder:
    """Hooks a SwingBase episode: the arm's servo commands, the base's outputs and a 1 kHz sample of everything."""

    def __init__(self, ep, base):
        self.ep, self.tb, self.base = ep, ep.tb, base
        w = self.w = ep.tb.world
        self.arm = (w.q_hover.copy(), np.zeros(N_ARM), w._hover_droop.copy())
        set_arm = w.set_arm_position

        def recording_set_arm(q, qd=None, droop=None):
            self.arm = (np.array(q, float), np.zeros(N_ARM) if qd is None else np.array(qd, float),
                        np.zeros(N_ARM) if droop is None else np.array(droop, float))
            set_arm(q, qd, droop)

        w.set_arm_position = recording_set_arm
        self.truth = None
        truth = base.truth

        def recording_truth(t):
            self.truth = truth(t)
            return self.truth

        base.truth = recording_truth
        self.look = np.zeros((N_JOINTS + N_ARM) * len(LOOKAHEAD))

        def base_fn(t):
            out = base(t)
            self.look = np.asarray(out[1], float)
            return out

        self.tb.residual.base_fn = base_fn
        self.rows: dict[str, list] = {k: [] for k in ("t", "arm_q", "arm_qd", "arm_droop", "s", "grip_sp", "look",
                                                      "fc_valid", "fc_t_event", "fc_regime", "fc_f", "fc_amp",
                                                      "phase", "depth")}
        self.starts: list[dict] = []
        self._last_phase = None

    def __call__(self, t: float) -> None:
        sw, w = self.ep.swing, self.w
        phase = "done" if sw.done else sw.phase
        if phase == "windup" and self._last_phase != "windup":
            d = w.data
            self.starts.append({"t0": float(t), "k": len(self.rows["t"]), "strike": int(sw.k),
                                "qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "qacc_ws": d.qacc_warmstart.copy(),
                                "s0": float(w.hand.s), "drive": drive_state(w.hand.drive, t),
                                "depth0": float(w.plant.depth), "hih0": np.concatenate(w.hammer_in_hand())})
        self._last_phase = phase
        r = self.rows
        q, qd, droop = self.arm
        r["t"].append(t)
        r["arm_q"].append(q)
        r["arm_qd"].append(qd)
        r["arm_droop"].append(droop)
        r["s"].append(float(w.hand.s))
        r["grip_sp"].append(float(getattr(self.tb.l1, "grip_setpoint", np.nan)))
        look = self.look.reshape(len(LOOKAHEAD), N_JOINTS + N_ARM)[:, N_JOINTS:]
        r["look"].append(look.reshape(-1).copy())
        tr = self.truth
        valid = tr is not None and tr.valid
        r["fc_valid"].append(valid)
        r["fc_t_event"].append(tr.t_event if valid else np.nan)
        r["fc_regime"].append(REGIMES.index(tr.regime) if valid else -1)
        r["fc_f"].append(tr.f_dom if valid else 0.0)
        r["fc_amp"].append(tr.amp if valid else 0.0)
        r["phase"].append(PHASES.index(phase))
        r["depth"].append(float(w.plant.depth))


def record_episode(seed: int, n_strikes: int = 10, rate: float = 1000.0, timestep: float = 0.00025,
                   randomize: bool = True) -> dict:
    """One library episode (see module doc): its streams, swing starts, outcomes and randomization."""
    from tactile_sim.behaviors.base_sources import SwingBase
    from tactile_sim.episode import Episode

    cfg, draw = episode_config(seed, rate, timestep, randomize)
    t_wall = time.time()
    ep = Episode(cfg, seed=seed)
    tb = ep.tb
    params = episode_params(cfg, draw)
    tb.sensors.set_noise_scale(params["noise_scale"])
    base = SwingBase(tb, ep.swing, rng=np.random.default_rng(10_000 + seed))
    rec = _Recorder(ep, base)
    tb.sched.register("library", 1000.0, rec)
    peak2: dict[int, float] = {}  # each blow's second largest force sample (the peak without a one-step spike)
    on_end = ep.swing.on_strike_end

    def strike_end(plan):
        fs = np.sort(np.asarray(ep._series_f, float))
        peak2[plan.idx] = float(fs[-2]) if len(fs) > 1 else 0.0
        on_end(plan)

    ep.swing.on_strike_end = strike_end
    res = ep.run(n_strikes)
    rows = {k: np.asarray(v) for k, v in rec.rows.items()}
    strikes = {s.idx: s for s in res.strikes}
    t_end = float(rows["t"][-1]) + 1e-3
    entries = []
    for j, st in enumerate(rec.starts):
        k1 = rec.starts[j + 1]["k"] if j + 1 < len(rec.starts) else len(rows["t"])
        s = strikes.get(st["strike"])
        t1 = rec.starts[j + 1]["t0"] if j + 1 < len(rec.starts) else t_end
        out = {"hit": False, "drop": bool(tb.grip.dropped), "advance": float(rows["depth"][k1 - 1] - st["depth0"]),
               "peak": 0.0, "peak2": 0.0, "width": 0.0, "impulse": 0.0, "v_face": float("nan"), "t_contact": float("nan"),
               "slip_trans": float("nan"), "slip_rot": float("nan"), "t_swing": float("nan"),
               "t_c_pred": float("nan"), "v_cmd": float("nan")}
        if s is not None:
            out.update(hit=bool(s.hit), drop=bool(s.drop), peak=float(s.peak_force_truth),
                       peak2=peak2.get(st["strike"], 0.0), width=float(s.pulse_width),
                       impulse=float(s.impulse), v_face=float(s.v_strike_actual),
                       t_contact=float(s.t_contact_truth - st["t0"]), slip_trans=float(s.slip_trans),
                       slip_rot=float(s.slip_rot), t_swing=float(s.t_swing_start - st["t0"]),
                       t_c_pred=float(s.t_c_pred - st["t0"]), v_cmd=float(s.v_cmd))
        entries.append({**st, "len": k1 - st["k"], "duration": t1 - st["t0"], "outcome": out})
    return {"seed": seed, "cfg": cfg.to_json(), "params": params, "const": model_constants(tb.world),
            "rows": rows, "entries": entries, "summary": res.summary, "wall_s": time.time() - t_wall}


def _stream(rows: dict, k0: int, n: int, t0: float) -> dict[str, np.ndarray]:
    sl = slice(k0, k0 + n)
    out = {k: rows[k][sl] for k in ("arm_q", "arm_qd", "arm_droop", "s", "grip_sp", "look", "fc_valid", "fc_regime",
                                    "fc_f", "fc_amp", "phase", "depth")}
    out["fc_t_event"] = rows["fc_t_event"][sl] - t0  # relative to the entry's start
    return out


def assemble(episodes: list[dict], meta: dict | None = None) -> dict[str, np.ndarray]:
    """The library arrays (see module doc and tactile_rl.mdp.commands.BaseLibrary for the layout)."""
    streams: dict[str, list] = {}
    ent: dict[str, list] = {}
    ep_arrays: dict[str, list] = {"ep_seed": [], "ep_cfg": [], "ep_wall_s": []}
    for k in PARAM_KEYS:
        ep_arrays[f"ep_{k}"] = []
    offset = 0
    for i, ep in enumerate(episodes):
        ep_arrays["ep_seed"].append(ep["seed"])
        ep_arrays["ep_cfg"].append(ep["cfg"])
        ep_arrays["ep_wall_s"].append(ep["wall_s"])
        for k in PARAM_KEYS:
            ep_arrays[f"ep_{k}"].append(ep["params"][k])
        for k, v in ep["const"].items():
            ep_arrays.setdefault(f"ep_{k}", []).append(v)
        for e in ep["entries"]:
            if e["len"] < 2:
                continue
            for k, v in _stream(ep["rows"], e["k"], e["len"], e["t0"]).items():
                streams.setdefault(f"s_{k}", []).append(v)
            rec = {"episode": i, "strike": e["strike"], "t0": e["t0"], "start": offset, "len": e["len"],
                   "duration": e["duration"], "qpos": e["qpos"], "qvel": e["qvel"], "qacc_ws": e["qacc_ws"],
                   "s0": e["s0"], "depth0": e["depth0"], "hih0": e["hih0"], **e["outcome"]}
            rec.update({f"drive_{k}": v for k, v in e["drive"].items()})
            for k, v in rec.items():
                ent.setdefault(f"e_{k}", []).append(v)
            offset += e["len"]
    out = {k: np.concatenate(v) for k, v in streams.items()}
    out.update({k: np.asarray(v) for k, v in ent.items()})
    out.update({k: np.asarray(v) for k, v in ep_arrays.items()})
    base = {}
    if episodes:
        from tactile_sim.sim.training_parts import config_from_key

        base = {"base": base_signature(config_from_key(episodes[0]["cfg"]))}
    out["meta"] = np.array(json.dumps({"version": LIBRARY_VERSION, "lookahead": LOOKAHEAD, "regimes": REGIMES,
                                       "phases": PHASES, "params": PARAM_KEYS, **base, **(meta or {})}))
    return out


def _record_job(args) -> dict:
    return record_episode(*args)


def export_library(out: str | Path, episodes: int = 20, strikes: int = 10, workers: int = 14, seed0: int = 0,
                   rate: float = 1000.0, timestep: float = 0.00025, randomize: bool = True) -> dict[str, np.ndarray]:
    """Record `episodes` episodes of `strikes` swings (seeds seed0 ...) in parallel and write the library."""
    from tactile_sim.assets.fetch_hands import hand_xml
    from tactile_sim.model.hands.wuji_grasp import wrap_grasp

    cfg, _ = episode_config(seed0, rate, timestep, randomize)
    wrap_grasp(cfg, hand_xml("wuji2"))  # every draw shares the nominal grasp: synthesize it once, serially
    jobs = [(seed0 + i, strikes, rate, timestep, randomize) for i in range(episodes)]
    t0 = time.time()
    if workers <= 1:
        eps = [_record_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            eps = list(ex.map(_record_job, jobs))
    lib = assemble(eps, {"episodes": episodes, "strikes": strikes, "rate": rate, "timestep": timestep,
                         "randomize": randomize, "seed0": seed0, "wall_s": time.time() - t0})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **lib)
    return lib


def load_library(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        lib = {k: z[k] for k in z.files}
    meta = json.loads(str(lib["meta"]))
    if meta["version"] != LIBRARY_VERSION:
        raise ValueError(f"{path}: library version {meta['version']}, this code reads {LIBRARY_VERSION}")
    return lib


def entry_stream(lib: dict, e: int) -> dict[str, np.ndarray]:
    a, n = int(lib["e_start"][e]), int(lib["e_len"][e])
    return {k[2:]: lib[k][a:a + n] for k in lib if k.startswith("s_")}


def entry_drive_state(lib: dict, e: int) -> dict[str, np.ndarray]:
    return {k: lib[f"e_drive_{k}"][e] for k in DRIVE_STATE}


# ---------------------------------------------------------------------------------------------- replay


def arm_tick_phase_ms(t0: float) -> int:
    """Milliseconds from an entry's start to the arm's first 100 Hz command tick (PositionL1 commands on the absolute
    10 ms grid)."""
    return (10 - int(round(t0 * 1000.0)) % 10) % 10


def strike_outcome(t: np.ndarray, f: np.ndarray, depth: np.ndarray, hih: tuple[np.ndarray, np.ndarray],
                   t_swing: float, v_face: float = float("nan"), t_contact: float = float("nan")) -> dict:
    """An entry's outcome from per-step face/nail force, nail depth and the tool pose in the hand at the swing's
    start and the entry's end (the testbed's StrikeRecord metrics: pulse_stats, grasp_slip). `peak2` is the second
    largest force sample: the peak without a single-step solver spike."""
    from tactile_sim.logging.strike_metrics import grasp_slip
    from tactile_sim.sim.truth import pulse_stats

    sel = t >= t_swing
    hit = bool(np.any(f[sel] > 0))
    st = pulse_stats(t[sel], np.where(f[sel] > 0, f[sel], 0.0)) if hit else None
    s_tr, s_rot = grasp_slip(hih[0][:3], hih[0][3:], hih[1][:3], hih[1][3:])
    i0 = int(np.searchsorted(t, t_swing))
    pk = np.sort(f[sel])[-2:] if hit else np.zeros(2)
    return {"hit": hit, "advance": float(depth[-1] - depth[min(i0, len(depth) - 1)]),
            "peak": st["peak"] if st else 0.0, "peak2": float(pk[0]) if len(pk) > 1 else 0.0,
            "width": st["width"] if st else 0.0,
            "impulse": st["impulse"] if st else 0.0, "t_contact": t_contact, "v_face": v_face,
            "slip_trans": s_tr, "slip_rot": s_rot}


def replay_entry(lib: dict, e: int, world=None, physics: dict | None = None, seed: int | None = None) -> dict:
    """Play entry `e` open loop on a CPU World of its episode's configuration (`physics` overrides PhysicsCfg, e.g.
    the 8 kHz validator's {"timestep": 0.000125, "noslip_iterations": 5}). Returns the outcome (strike_outcome) and
    the per-step traces. Pass `world` to reuse one World for the entries of one episode. The drives' command jitter is
    drawn from `seed` (default: the entry's index), so a replay does not depend on what the World played before: a
    stiff blow is sensitive to it (a rare single-step spike of the face/nail force)."""
    from tactile_sim.sim.training_parts import config_from_key
    from tactile_sim.sim.truth import pair_force
    from tactile_sim.sim.world import World

    ep = int(lib["e_episode"][e])
    cfg = config_from_key(str(lib["ep_cfg"][ep]))
    if physics:
        cfg = cfg.replace(physics=physics)
    w = world or World(cfg)
    m, d = w.model, w.data
    st = entry_stream(lib, e)
    t0 = float(lib["e_t0"][e])
    mujoco.mj_resetData(m, d)
    d.qpos[:] = lib["e_qpos"][e]
    d.qvel[:] = lib["e_qvel"][e]
    d.qacc_warmstart[:] = lib["e_qacc_ws"][e]
    d.time = 0.0
    w.step_count = 0
    hand = w.hand
    hand.ext_cmd = None
    set_drive_state(hand.drive, entry_drive_state(lib, e))
    hand.drive.rng = np.random.default_rng(e if seed is None else seed)
    plant = w.plant
    plant._in_contact, plant._last_contact_t = False, -np.inf
    mujoco.mj_forward(m, d)
    g_face, g_nail = m.geom(names.HAMMER_FACE_GEOM).id, m.geom(names.NAIL_HEAD_GEOM).id
    face_site = w.site[names.HAMMER_FACE_SITE]
    n_ms = int(lib["e_len"][e])
    per_ms = int(round(1e-3 / w.dt))
    phase = arm_tick_phase_ms(t0)
    kd_kp = w.servo_kd / w.servo_kp
    ts, fs, depth = [], [], []
    t_contact, v_face = float("nan"), float("nan")
    hih0 = np.concatenate(w.hammer_in_hand())
    for i in range(n_ms):
        if i == 0 or (i - phase) % 10 == 0:
            d.ctrl[w.arm_act] = st["arm_q"][i] + st["arm_droop"][i] + kd_kp * st["arm_qd"][i]
        hand.drive.set_host_command(HandCommand(w.t, *hand.synergy.mit_command(float(st["s"][i]), d.qpos[hand.qadr]),
                                                src="library"))
        for _ in range(per_ms):
            w.step()
            f, _ = pair_force(m, d, g_face, g_nail)
            if f > 0 and not math.isfinite(t_contact):
                t_contact = w.t
                v_face = float(w.site_velocity(face_site)[:3] @ plant.axis)
            ts.append(w.t)
            fs.append(f)
            depth.append(plant.depth)
    hih1 = np.concatenate(w.hammer_in_hand())
    t_swing = float(lib["e_t_swing"][e]) if np.isfinite(lib["e_t_swing"][e]) else 0.0
    out = strike_outcome(np.array(ts), np.array(fs), np.array(depth), (hih0, hih1), t_swing, v_face, t_contact)
    return {"outcome": out, "t": np.array(ts), "f": np.array(fs), "depth": np.array(depth)}


def compare_outcomes(ref: dict[str, np.ndarray], other: dict[str, np.ndarray]) -> dict[str, float]:
    """Hit rates and the per-entry differences of advance, peak force and pulse width on entries both hit."""
    hit_r, hit_o = np.asarray(ref["hit"], bool), np.asarray(other["hit"], bool)
    both = hit_r & hit_o
    out = {"n": int(len(hit_r)), "hit_rate_ref": float(hit_r.mean()), "hit_rate": float(hit_o.mean()),
           "hit_agree": float(np.mean(hit_r == hit_o))}
    for k in ("advance", "peak", "peak2", "width"):
        if k not in ref or k not in other:
            continue
        a, b = np.asarray(ref[k], float)[both], np.asarray(other[k], float)[both]
        if not len(a):
            continue
        out[f"{k}_ref_mean"], out[f"{k}_mean"] = float(a.mean()), float(b.mean())
        rel = np.abs(b - a) / np.maximum(np.abs(a), 1e-9)
        out[f"{k}_rel_median"], out[f"{k}_rel_p90"] = float(np.median(rel)), float(np.percentile(rel, 90))
    return out


def _replay_job(args) -> list[dict]:
    path, entries, physics = args
    from tactile_sim.sim.training_parts import config_from_key
    from tactile_sim.sim.world import World

    lib = load_library(path)
    out, world, ep_built = [], None, None
    for e in entries:
        ep = int(lib["e_episode"][e])
        if ep != ep_built:
            cfg = config_from_key(str(lib["ep_cfg"][ep]))
            world, ep_built = World(cfg.replace(physics=physics) if physics else cfg), ep
        out.append({"entry": e, **replay_entry(lib, e, world, physics)["outcome"]})
    return out


def replay_library(path: str | Path, entries: list[int] | None = None, workers: int = 14,
                   physics: dict | None = None) -> list[dict]:
    """replay_entry on many entries in parallel (one World per episode per worker)."""
    lib = load_library(path)
    entries = list(range(len(lib["e_t0"]))) if entries is None else list(entries)
    by_ep: dict[int, list[int]] = {}
    for e in entries:
        by_ep.setdefault(int(lib["e_episode"][e]), []).append(e)
    jobs = [(str(path), es, physics) for es in by_ep.values()]
    if workers <= 1:
        res = [_replay_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(_replay_job, jobs))
    rows = sorted((r for rs in res for r in rs), key=lambda r: r["entry"])
    return rows


def table(rows: list[dict], keys=OUTCOMES) -> dict[str, np.ndarray]:
    return {k: np.array([r[k] for r in rows]) for k in keys if rows and k in rows[0]}


def library_outcomes(lib: dict, entries: list[int] | None = None) -> dict[str, np.ndarray]:
    """The closed-loop CPU outcome the library recorded for each entry."""
    idx = np.arange(len(lib["e_t0"])) if entries is None else np.asarray(entries)
    return {k: np.asarray(lib[f"e_{k}"])[idx] for k in OUTCOMES if f"e_{k}" in lib}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("export", "replay", "info"))
    ap.add_argument("--out", default="runs/base_library/hammer_v1.npz")
    ap.add_argument("--lib", default="runs/base_library/hammer_v1.npz")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--strikes", type=int, default=10)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--entries", default=None, help="a:b range of entries to replay")
    ap.add_argument("--validator", action="store_true", help="replay at the 8 kHz validator physics (noslip 5)")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        lib = export_library(a.out, a.episodes, a.strikes, a.workers, a.seed0, randomize=not a.no_randomize)
        print(f"{len(lib['e_t0'])} entries from {a.episodes} episodes -> {a.out}")
        print(json.dumps({k: float(np.mean(v)) for k, v in library_outcomes(lib).items()}, indent=1))
        return 0
    lib = load_library(a.lib)
    if a.cmd == "info":
        print(json.loads(str(lib["meta"])))
        print(f"{len(lib['e_t0'])} entries, {len(lib['s_s'])} ms of streams")
        print(json.dumps({k: float(np.nanmean(v)) for k, v in library_outcomes(lib).items()}, indent=1))
        return 0
    entries = None
    if a.entries:
        lo, hi = (int(x) for x in a.entries.split(":"))
        entries = list(range(lo, min(hi, len(lib["e_t0"]))))
    physics = {"timestep": 0.000125, "noslip_iterations": 5} if a.validator else None
    rows = replay_library(a.lib, entries, a.workers, physics)
    idx = [r["entry"] for r in rows]
    print(json.dumps(compare_outcomes(library_outcomes(lib, idx), table(rows)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DRIVE_STATE", "OUTCOMES", "PARAM_KEYS", "PHASES", "assemble", "compare_outcomes",
           "drive_state", "entry_drive_state", "entry_stream", "episode_config", "episode_params", "export_library",
           "load_library", "replay_entry", "replay_library", "set_drive_state"]
