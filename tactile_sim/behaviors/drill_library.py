"""The drill's base library: scripted drill feeds recorded on the CPU testbed for the GPU env (tactile_rl), in the
hammer library's layout (behaviors.base_library), so tactile_rl.mdp.commands.BaseTrajectoryCommand replays either.

`export_drill_library` runs DrillBase episodes (behaviors.base_sources: the position-arm feed brings the bit to the
screw head or the board, presses, pulls the trigger with the index finger, follows the bit in, releases once the
clutch ratchets or the bit is through, backs out) with a zero residual on the reduced scene at the training physics
(sim.training_parts.residual_training_config(task="drill"): Vega U + WUJI, the 12 V pistol driver, the trigger
finger's friction capped), one residual_dr draw per episode, and records every millisecond:

- the arm's 100 Hz servo stream (target, velocity feedforward, droop), as the hammer's;
- the hand base: the grip synergy's scalar s and the trigger finger's own level s_trig (WrapSynergy.levels), and
  whether the base allows the trigger (the residual's interlock);
- the lookahead's arm part and the forecast truth DrillBase hands its ForecastModel (DrillPlant.event_truth: the
  seating and the clutch's ratchet, the breakthrough);
- the feed's phase; the plant's truth (state, depth, spindle speed, chuck torque, clutch clicks) and the tool's pose
  in the hand, to compare a replay against.

One entry per episode, from its first millisecond, with the state then (qpos, qvel, warm start, the drives' state),
the episode's randomization (base_library.episode_params, the drill's draw as drill_<name>, the tool's inertial and
the arm gains) and the CPU's outcome (seated or through, when, the clutch's clicks, the peak chuck torque, cam-outs,
the tool's slip in the hand).

    python -m tactile_sim.behaviors.drill_library export --episodes 20 --mode screw --workers 10 \
        --out runs/base_library/drill_screw_v1.npz
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tactile_sim.behaviors.base_library import LIBRARY_VERSION, PARAM_KEYS, base_signature, drive_state, episode_params
from tactile_sim.config import SimConfig
from tactile_sim.control.obs_spec import LOOKAHEAD, N_ARM, N_JOINTS, REGIMES

DRILL_PHASES = ("start", "approach", "press", "drive", "release", "retract", "done")
# the drill's randomized parameters (sim.randomize._drill_dr): the drivetrain's and the screw's / the hole's
DRILL_PARAMS = ("free_speed", "stall_torque", "spindle_inertia", "gear_ratio", "clutch_detents", "clutch_ring_ratio",
                "clutch_duty", "clutch_overshoot", "strip_torque", "screw_twist_off", "cut_energy", "exit_len")
DRILL_T_MAX = 10.0  # s, as the residual experiment's drill episodes
OUTCOMES = ("done", "t_done", "state", "clicks", "peak_torque", "cam_outs", "max_slip_trans", "max_slip_rot",
            "net_slip_trans", "net_slip_rot", "lost", "depth")


def drill_episode_config(seed: int, mode: str = "screw", rate: float = 1000.0, timestep: float = 0.00025,
                         randomize: bool = True) -> tuple[SimConfig, dict | None]:
    from tactile_sim.sim.randomize import apply_residual_dr, residual_dr
    from tactile_sim.sim.training_parts import residual_training_config

    cfg = residual_training_config(rate, timestep, task="drill").replace(drill={"mode": mode})
    if not randomize:
        return cfg, None
    draw = residual_dr(cfg, np.random.default_rng(seed))
    return apply_residual_dr(cfg, draw), draw


def drill_constants(w) -> dict[str, np.ndarray]:
    """The model constants a drill episode's randomization changes: the tool body's inertial (the pistol's masses)
    and what MuJoCo derives from it, the arm servo gains, the drives' rated currents (base_library.model_constants
    without the nail)."""
    m = w.model
    b = w.hammer_body
    dofs = slice(w.hammer_dofadr, w.hammer_dofadr + 6)
    return {"hammer_mass": np.array(m.body_mass[b]), "hammer_ipos": m.body_ipos[b].copy(),
            "hammer_iquat": m.body_iquat[b].copy(), "hammer_inertia": m.body_inertia[b].copy(),
            "hammer_invweight0": m.body_invweight0[b].copy(), "hammer_dof_invweight0": m.dof_invweight0[dofs].copy(),
            "arm_kp": w.servo_kp.copy(), "arm_kd": w.servo_kd.copy(), "i_rated": np.asarray(w.cfg.wuji.i_rated, float)}


class _DrillRecorder:
    """Hooks a DrillBase episode: the arm's servo commands, the base's outputs, the plant's forecast truth, and a
    1 kHz sample of everything (the state at the first one)."""

    def __init__(self, tt, base):
        self.tt, self.tb, self.base = tt, tt.tb, base
        w = self.w = tt.tb.world
        self.arm = (w.q_hover.copy(), np.zeros(N_ARM), w._hover_droop.copy())
        set_arm = w.set_arm_position

        def recording_set_arm(q, qd=None, droop=None):
            self.arm = (np.array(q, float), np.zeros(N_ARM) if qd is None else np.array(qd, float),
                        np.zeros(N_ARM) if droop is None else np.array(droop, float))
            set_arm(q, qd, droop)

        w.set_arm_position = recording_set_arm
        self.truth = None
        plant = w.plant
        event_truth = plant.event_truth

        def recording_truth(t):
            self.truth = event_truth(t)
            return self.truth

        plant.event_truth = recording_truth
        self.look = np.zeros((N_JOINTS + N_ARM) * len(LOOKAHEAD))

        def base_fn(t):
            out = base(t)
            self.look = np.asarray(out[1], float)
            return out

        self.tb.residual.base_fn = base_fn
        self.rows: dict[str, list] = {k: [] for k in (
            "t", "arm_q", "arm_qd", "arm_droop", "s", "s_trig", "trigger", "grip_sp", "look", "fc_valid", "fc_t_event",
            "fc_regime", "fc_f", "fc_amp", "phase", "depth", "drill_state", "spindle_speed", "drill_torque", "clicks",
            "hih")}
        self.start: dict | None = None

    def __call__(self, t: float) -> None:
        w, plant = self.w, self.w.plant
        if self.start is None:
            d = w.data
            self.start = {"t0": float(t), "k": 0, "strike": 0, "qpos": d.qpos.copy(), "qvel": d.qvel.copy(),
                          "qacc_ws": d.qacc_warmstart.copy(), "s0": float(w.hand.s),
                          "drive": drive_state(w.hand.drive, t), "depth0": float(plant.depth),
                          "hih0": np.concatenate(w.hammer_in_hand())}
        r = self.rows
        q, qd, droop = self.arm
        r["t"].append(t)
        r["arm_q"].append(q)
        r["arm_qd"].append(qd)
        r["arm_droop"].append(droop)
        r["s"].append(float(w.hand.s))
        r["s_trig"].append(float(w.hand.synergy.s_trigger))
        r["trigger"].append(bool(self.tt.behavior.trigger > 0))  # the base allows the trigger (the interlock)
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
        r["phase"].append(DRILL_PHASES.index(self.tt.behavior.phase))
        r["depth"].append(float(plant.depth))
        from tactile_sim.model.plants.drill import STATES

        r["drill_state"].append(STATES.index(plant.state))
        r["spindle_speed"].append(float(plant.drive.w))
        r["drill_torque"].append(float(plant.tau_out))
        r["clicks"].append(int(plant.drive.clicks))
        r["hih"].append(np.concatenate(w.hammer_in_hand()))


def record_drill_episode(seed: int, mode: str = "screw", rate: float = 1000.0, timestep: float = 0.00025,
                         randomize: bool = True, t_max: float = DRILL_T_MAX) -> dict:
    """One drill library episode (see module doc)."""
    from tactile_sim.behaviors.base_sources import DrillBase
    from tactile_sim.tool_task import ToolTask

    cfg, draw = drill_episode_config(seed, mode, rate, timestep, randomize)
    t_wall = time.time()
    tt = ToolTask(cfg, seed=seed, t_max=t_max)
    tb = tt.tb
    params = episode_params(cfg, draw)
    params.update({f"drill_{k}": float(getattr(cfg.drill, k)) for k in DRILL_PARAMS})
    tb.sensors.set_noise_scale(params["noise_scale"])
    base = DrillBase(tb, tt.behavior, rng=np.random.default_rng(10_000 + seed))
    rec = _DrillRecorder(tt, base)
    tb.sched.register("library", 1000.0, rec)
    res = tt.run()
    rows = {k: np.asarray(v) for k, v in rec.rows.items()}
    s = res.summary
    st = rec.start
    t_done = float(s["t_done"]) - st["t0"] if np.isfinite(s["t_done"]) else float("nan")
    out = {"done": bool(s["done"]), "t_done": t_done, "state": int(rows["drill_state"][-1]),
           "clicks": int(rows["clicks"][-1]), "peak_torque": float(s["peak_torque"]), "cam_outs": int(s["cam_outs"]),
           "max_slip_trans": float(s["max_slip_trans"]), "max_slip_rot": float(s["max_slip_rot"]),
           "net_slip_trans": float(s["net_slip_trans"]), "net_slip_rot": float(s["net_slip_rot"]),
           "lost": bool(s["lost"]), "depth": float(rows["depth"][-1])}
    entry = {**st, "len": len(rows["t"]), "duration": float(rows["t"][-1] - st["t0"]) + 1e-3, "outcome": out}
    return {"seed": seed, "mode": mode, "cfg": cfg.to_json(), "params": params, "const": drill_constants(tb.world),
            "rows": rows, "entries": [entry], "summary": s, "wall_s": time.time() - t_wall}


STREAMS = ("arm_q", "arm_qd", "arm_droop", "s", "s_trig", "trigger", "grip_sp", "look", "fc_valid", "fc_regime",
           "fc_f", "fc_amp", "phase", "depth", "drill_state", "spindle_speed", "drill_torque", "clicks", "hih")


def assemble_drill(episodes: list[dict], meta: dict | None = None) -> dict[str, np.ndarray]:
    """The library arrays in base_library.assemble's layout (s_* streams, e_* entries, ep_* episodes), with the
    drill's streams, outcomes and parameters."""
    streams: dict[str, list] = {}
    ent: dict[str, list] = {}
    ep_arrays: dict[str, list] = {"ep_seed": [], "ep_cfg": [], "ep_wall_s": []}
    keys = list(PARAM_KEYS) + [f"drill_{k}" for k in DRILL_PARAMS]
    for k in keys:
        ep_arrays[f"ep_{k}"] = []
    offset = 0
    nan = float("nan")
    for i, ep in enumerate(episodes):
        ep_arrays["ep_seed"].append(ep["seed"])
        ep_arrays["ep_cfg"].append(ep["cfg"])
        ep_arrays["ep_wall_s"].append(ep["wall_s"])
        for k in keys:
            ep_arrays[f"ep_{k}"].append(ep["params"][k])
        for k, v in ep["const"].items():
            ep_arrays.setdefault(f"ep_{k}", []).append(v)
        for e in ep["entries"]:
            sl = slice(e["k"], e["k"] + e["len"])
            for k in STREAMS:
                streams.setdefault(f"s_{k}", []).append(ep["rows"][k][sl])
            streams.setdefault("s_fc_t_event", []).append(ep["rows"]["fc_t_event"][sl] - e["t0"])
            rec = {"episode": i, "strike": e["strike"], "t0": e["t0"], "start": offset, "len": e["len"],
                   "duration": e["duration"], "qpos": e["qpos"], "qvel": e["qvel"], "qacc_ws": e["qacc_ws"],
                   "s0": e["s0"], "depth0": e["depth0"], "hih0": e["hih0"], "t_swing": nan, "t_c_pred": nan,
                   **e["outcome"]}
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
                                       "phases": DRILL_PHASES, "params": keys, "task": "drill", **base,
                                       **(meta or {})}))
    return out


def _record_job(args) -> dict:
    return record_drill_episode(*args)


def export_drill_library(out: str | Path, episodes: int = 20, mode: str = "screw", workers: int = 10, seed0: int = 0,
                         rate: float = 1000.0, timestep: float = 0.00025, randomize: bool = True,
                         t_max: float = DRILL_T_MAX) -> dict[str, np.ndarray]:
    from tactile_sim.assets.fetch_hands import hand_xml
    from tactile_sim.model.hands.wuji_grasp import wrap_grasp

    cfg, _ = drill_episode_config(seed0, mode, rate, timestep, randomize)
    wrap_grasp(cfg, hand_xml("wuji2"))  # the nominal grasp, once, serially
    jobs = [(seed0 + i, mode, rate, timestep, randomize, t_max) for i in range(episodes)]
    t0 = time.time()
    if workers <= 1:
        eps = [_record_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            eps = list(ex.map(_record_job, jobs))
    lib = assemble_drill(eps, {"episodes": episodes, "mode": mode, "rate": rate, "timestep": timestep,
                               "randomize": randomize, "seed0": seed0, "wall_s": time.time() - t0})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **lib)
    return lib


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - export script
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("export",))
    ap.add_argument("--out", default="runs/base_library/drill_screw_v1.npz")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--mode", default="screw", choices=("screw", "hole"))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--t-max", type=float, default=DRILL_T_MAX)
    a = ap.parse_args(argv)
    lib = export_drill_library(a.out, a.episodes, a.mode, a.workers, a.seed0, t_max=a.t_max)
    print(f"{len(lib['e_t0'])} episodes -> {a.out}")
    print(json.dumps({k: float(np.nanmean(lib[f"e_{k}"].astype(float))) for k in OUTCOMES}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
