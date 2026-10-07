"""Slip ablation: which lever keeps a held tool from turning or sliding in the hand, at equal grip effort?

    python -m tactile_sim.experiments.slip_ablation list --stage screen
    python -m tactile_sim.experiments.slip_ablation run --stage screen --workers 14
    python -m tactile_sim.experiments.slip_ablation run --stage full --workers 14 --platform fr3_wuji --task hammer

Each condition changes one factor from a common baseline: constant grip at the hand's hold setpoint, no anticipatory
ramp, no cross-strike grip adaptation (see docs/slip_ablation_plan.md). One JSON file per episode at
runs/slip_ablation/<stage>/<platform>/<task>/<condition>/seed<k>.json; a run skips files that exist, so it resumes
where it stopped. Stage "screen": randomization off, seed 0. Stage "full": domain randomization on, seeds 0-4 (0-9 on
FR3 + WUJI, whose wrap is close to failing and the noisiest). A seed fixes the randomized hammer, nail, pads and
sensors, so conditions are compared pairwise. The WUJI platforms run twice: grip regulated on the taxel-patch force
("closed") and grip held as a commanded effort ("_open").
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tactile_sim.config import SimConfig, hand_config, task_config

ROOT = Path("runs/slip_ablation")
STAGES = ("screen", "full")
FULL_SEEDS = {"fr3_wuji": 10, "fr3_wuji_open": 10}  # other platforms: 5
LAGS = (0.0, 0.010, 0.030)  # s, grip actuation lag for the reactive and anticipatory conditions
N_STRIKES = 10
TOOL_T_MAX = {"saw": 30.0, "drill_screw": 10.0, "drill_hole": 15.0}

PLATFORMS = {
    "fr3_wuji": {"hand": "wuji2", "robot": "fr3", "grip_loop": "closed"},
    "fr3_wuji_open": {"hand": "wuji2", "robot": "fr3", "grip_loop": "open"},
    "fr3_franka": {"hand": "franka", "robot": "fr3", "grip_loop": "closed"},
    "vega_wuji": {"hand": "wuji2", "robot": "vega_1u", "grip_loop": "closed"},
    "vega_wuji_open": {"hand": "wuji2", "robot": "vega_1u", "grip_loop": "open"},
}
# the saw and driver grasps exist for the Franka Hand only (config.TASK_OVERRIDES)
TASKS = {"fr3_wuji": ("hammer",), "fr3_wuji_open": ("hammer",),
         "fr3_franka": ("hammer", "saw", "drill_screw", "drill_hole"), "vega_wuji": ("hammer",),
         "vega_wuji_open": ("hammer",)}


def seeds(stage: str, platform: str) -> tuple[int, ...]:
    return (0,) if stage == "screen" else tuple(range(FULL_SEEDS.get(platform, 5)))

# grip setpoints in each hand's grip-command units: Franka Hand N per pad (70 N continuous), WUJI N summed over its
# taxel patches (~89 N at full synergy, hold capped at 70 % of that)
GRIP = {
    "franka": {"base": 55.0, "levels": (25.0, 40.0, 55.0, 70.0), "low": 25.0},
    "wuji2": {"base": 40.0, "levels": (20.0, 30.0, 40.0, 50.0, 60.0), "low": 20.0},
    "tool": {"base": 60.0, "levels": (30.0, 45.0, 60.0, 70.0), "low": 30.0},
}
# Reactive-loop thresholds (ReactiveCfg fields) per hand and task, each just above the largest excursion seen without
# slip in log-mode baseline runs (seed 0, 6 strikes / one tool episode; docs/slip_ablation_plan.md, "Calibration"):
# hammer swings and saw strokes must not trigger, impacts and genuine creep must.
_DEG = math.pi / 180.0
THRESHOLDS: dict[str, dict[str, float]] = {
    # Franka Hand, hammer: swing peaks omega 0.50 rad/s, v 0.053 m/s, CoP speed 1.5 m/s, accel 114 m/s^2, pattern
    # rotation 1.2 deg; impacts reach omega 9.8 rad/s, CoP speed 3.2 m/s, accel 310 m/s^2
    "franka_hammer": {"omega_thresh": 2.5, "v_thresh": 0.08, "cop_speed_thresh": 1.8, "load_drop_thresh": 0.2,
                      "accel_thresh": 140.0, "drift_rot_thresh": 1.0 * _DEG, "drift_trans_thresh": 0.0005,
                      "cop_drift_thresh": 0.0025, "pattern_rot_thresh": 1.5 * _DEG},
    # Franka Hand, saw and driver: strokes flex the pads at up to omega 1.9 rad/s and turn the pressure pattern by
    # 0.3 deg peak to peak while the handle creeps 0.18 deg/s (pattern 0.04 deg/s); the CoP ripples 1.9 mm and does
    # not follow the creep, so its drift cue is effectively off
    "franka_tool": {"omega_thresh": 2.5, "v_thresh": 0.08, "cop_speed_thresh": 1.8, "load_drop_thresh": 0.2,
                    "accel_thresh": 140.0, "drift_rot_thresh": 1.0 * _DEG, "drift_trans_thresh": 0.0005,
                    "cop_drift_thresh": 0.0025, "pattern_rot_thresh": 0.5 * _DEG},
    # WUJI Hand 2 (FR3 and Vega U): swing peaks omega 1.2 rad/s, v 0.03 m/s, CoP speed 0.27 m/s, accel 20 m/s^2,
    # pattern rotation 1.6 deg; the patches lose up to 99 % of their load in every swing (the windup's deceleration
    # moves the handle onto the bare fingers), so the load-drop cue is off
    "wuji2_hammer": {"omega_thresh": 1.5, "v_thresh": 0.04, "cop_speed_thresh": 0.35, "load_drop_thresh": 2.0,
                     "accel_thresh": 30.0, "drift_rot_thresh": 1.0 * _DEG, "drift_trans_thresh": 0.0005,
                     "cop_drift_thresh": 0.0005, "pattern_rot_thresh": 2.0 * _DEG},
}


@dataclass
class Condition:
    name: str
    factor: str  # B baseline, F fixed grip, A anticipatory, R reactive, W wrist, K stiffness, L grip location, ...
    level: str
    over: dict = field(default_factory=dict)  # config overrides: {section: {field: value}}


def _merge(*parts: dict) -> dict:
    out: dict = {}
    for d in parts:
        for sec, vals in d.items():
            out.setdefault(sec, {}).update(vals)
    return out


def _grip(f: float) -> dict:
    return {"controller": {"grip_hold": float(f)}}


def _anticipatory(gain: float) -> dict:
    return {"swing": {"grip_profile": "template", "grip_pre_gain": float(gain)}}


def _reactive(mode: str, key: str, latch: bool = False) -> dict:
    return {"reactive": {"mode": mode, "latch": latch, **THRESHOLDS[key]}}


def _lag(lag: float) -> dict:
    return {"gripper": {"grip_lag": lag}} if lag > 0.0 else {}


def _lag_tag(lag: float) -> tuple[str, str]:
    """(condition-name suffix, label suffix) for a grip lag."""
    return ("", "") if lag <= 0.0 else (f"_lag{lag * 1e3:.0f}", f", {lag * 1e3:.0f} ms grip lag")


def reactive_conditions(key: str, g: dict, unit: str) -> list[Condition]:
    """Reflex (200 ms boost) and latched (decays over ~2 s) loops, each with the oracle and the taxel detector:
    at the baseline grip for every grip lag, at the low grip without lag."""
    out = []
    for latch, tag, kind in ((False, "R", "reflex"), (True, "RL", "latched")):
        for mode in ("oracle", "taxel"):
            for f, lags in ((g["base"], LAGS), (g["low"], (0.0,))):
                for lag in lags:
                    sfx, lbl = _lag_tag(lag)
                    out.append(Condition(f"{tag}{mode}@{f:g}{sfx}", tag, f"{mode} {kind} loop, grip {f:g} {unit}{lbl}",
                                         _merge(_grip(f), _reactive(mode, key, latch), _lag(lag))))
    return out


def _wrist(k: str) -> dict:
    """Passive wrist between the F/T body and the hand: hinges about the flange x and y axes. The damping keeps the
    default's damping ratio; a soft wrist gets room to swing before its stops."""
    if k == "rigid":
        return {"arm": {"flex_mode": "rigid"}}
    kr = float(k)
    return {"arm": {"flex_mode": "wrist", "wrist_k_rot": kr, "wrist_d_rot": 1.5 * math.sqrt(kr / 1500.0),
                    "wrist_range": 0.1 if kr >= 1500.0 else 0.6}}


def hammer_conditions(platform: str) -> list[Condition]:
    p = PLATFORMS[platform]
    hand, vega = p["hand"], p["robot"].startswith("vega")
    g = GRIP[hand]
    c = [Condition("B", "B", f"constant grip {g['base']:g} N")]
    c += [Condition(f"F{f:g}", "F", f"constant grip {f:g} N", _grip(f)) for f in g["levels"] if f != g["base"]]
    for k in (6.0, 12.0, 24.0):
        for lag in (LAGS if k == 12.0 else (0.0,)):
            sfx, lbl = _lag_tag(lag)
            c.append(Condition(f"A{k:g}{sfx}", "A", f"anticipatory ramp, pre-load gain {k:g} N/(kg m/s){lbl}",
                               _merge(_anticipatory(k), _lag(lag))))
    c += reactive_conditions(f"{hand}_hammer", g, "N")
    # the FR3 model has a 1500 N m/rad wrist by default, the Vega a rigid one
    for k in (("1500", "400", "100", "30") if vega else ("rigid", "400", "100", "30")):
        c.append(Condition(f"W{k}", "W", f"wrist {k}" + ("" if k == "rigid" else " N m/rad"), _wrist(k)))
    if vega:  # position servos: the stiffness knob is dexcontrol's P-gain multiplier
        c += [Condition(f"P{m:g}", "K", f"servo P gain x{m:g}", {"arm": {"servo_p_mult": m}}) for m in (0.25, 2.0)]
    else:
        c += [Condition(f"K{k:g}", "K", f"controller rotational stiffness {k:g} N m/rad",
                        {"controller": {"k_rot": (k, k, k)}, "swing": {"strike_k_rot": k}}) for k in (30.0, 1000.0)]
    c += [Condition(f"L{gl * 1000:.0f}", "L", f"grip {gl * 1000:.0f} mm from the head axis",
                    {"hammer": {"grip_from_head": gl}}) for gl in (0.13, 0.16, 0.22)]
    c.append(Condition("adapt", "adapt", "cross-strike grip margin +10 N after a slip",
                       {"swing": {"grip_margin_step": 10.0}}))
    for lag in (0.0, LAGS[-1]):
        sfx, lbl = _lag_tag(lag)
        c.append(Condition(f"A12+Roracle{sfx}", "combo", f"anticipatory ramp + oracle reflex loop{lbl}",
                           _merge(_anticipatory(12.0), _reactive("oracle", f"{hand}_hammer"), _lag(lag))))
    c.append(Condition("A12+W100", "combo", "anticipatory ramp + 100 N m/rad wrist",
                       _merge(_anticipatory(12.0), _wrist("100"))))
    return c


def tool_conditions(task: str) -> list[Condition]:
    g = GRIP["tool"]
    c = [Condition("B", "B", f"constant grip {g['base']:g} N per pad")]
    c += [Condition(f"F{f:g}", "F", f"constant grip {f:g} N per pad", _grip(f)) for f in g["levels"] if f != g["base"]]
    c += reactive_conditions("franka_tool", g, "N per pad")
    c += [Condition(f"W{k}", "W", f"wrist {k}" + ("" if k == "rigid" else " N m/rad"), _wrist(k))
          for k in ("rigid", "400", "100")]
    c += [Condition(f"K{k:g}", "K", f"controller rotational stiffness {k:g} N m/rad",
                    {"controller": {"k_rot": (k, k, k)}}) for k in (30.0, 1000.0)]
    return c


def conditions(platform: str, task: str) -> list[Condition]:
    return hammer_conditions(platform) if task == "hammer" else tool_conditions(task)


def base_config(platform: str, task: str) -> SimConfig:
    """Platform and task defaults, with the arm servoing the flange-side frame (wrist_control = "flange") in every
    condition, so the wrist compliance stays a passive element and all factors share one control frame."""
    p = PLATFORMS[platform]
    cfg = hand_config(p["hand"], SimConfig(), robot=p["robot"]).replace(arm={"wrist_control": "flange"},
                                                                        controller={"grip_loop": p["grip_loop"]})
    if task == "hammer":
        return cfg.replace(swing={"grip_profile": "constant", "grip_margin_step": 0.0})
    kind, _, mode = task.partition("_")
    cfg = task_config(kind, cfg)
    return cfg.replace(drill={"mode": mode}) if kind == "drill" else cfg


@dataclass
class Job:
    stage: str
    platform: str
    task: str
    cond: Condition
    seed: int

    @property
    def dr(self) -> bool:
        return self.stage == "full"

    def path(self, root: Path | str = ROOT) -> Path:
        return Path(root) / self.stage / self.platform / self.task / self.cond.name / f"seed{self.seed}.json"

    def config(self) -> SimConfig:
        cfg = base_config(self.platform, self.task)
        if self.cond.over:
            cfg = cfg.replace(**self.cond.over)
        return cfg.replace(dr={"enabled": True}) if self.dr else cfg

    def dr_sample(self, cfg: SimConfig):
        from tactile_sim.sim.randomize import sample_dr

        return sample_dr(cfg, np.random.default_rng(self.seed))

    def episode_config(self) -> SimConfig:
        """The config an episode runs with, randomization applied (as Episode does)."""
        cfg = self.config()
        if not cfg.dr.enabled:
            return cfg
        from tactile_sim.sim.randomize import apply_dr

        return apply_dr(cfg, self.dr_sample(cfg))

    def meta(self) -> dict:
        return {"stage": self.stage, "platform": self.platform, "task": self.task, "condition": self.cond.name,
                "factor": self.cond.factor, "level": self.cond.level, "seed": self.seed, "dr": self.dr,
                "overrides": self.cond.over}

    @property
    def cost(self) -> float:
        """Relative wall time, for scheduling the slow jobs first."""
        if self.task != "hammer":
            return {"saw": 1.5, "drill_screw": 0.3, "drill_hole": 0.6}[self.task]
        return {"fr3_wuji": 4.0, "fr3_wuji_open": 4.0, "vega_wuji": 5.0, "vega_wuji_open": 5.0,
                "fr3_franka": 0.8}[self.platform]


def make_jobs(stage: str, platforms=None, tasks=None, only=None) -> list[Job]:
    jobs = []
    for pf in platforms or PLATFORMS:
        for task in TASKS[pf]:
            if tasks and task not in tasks:
                continue
            for c in conditions(pf, task):
                if only and c.name not in only:
                    continue
                jobs += [Job(stage, pf, task, c, s) for s in seeds(stage, pf)]
    return jobs


# ---------------------------------------------------------------------- running
def _jsonable(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _strike_row(r) -> dict:
    row = {"idx": r.idx, "hit": bool(r.hit), "drop": bool(r.drop), "slip_rot_deg": math.degrees(r.slip_rot),
           "slip_trans_mm": 1e3 * r.slip_trans, "advance_mm": 1e3 * r.depth_inc, "depth_mm": 1e3 * r.depth_after,
           "v_face": r.v_strike_actual, "v_cmd": r.v_cmd, "flag_latency_ms": 1e3 * r.flag_latency,
           "flag_source": r.flag_source, "peak_force": r.peak_force_truth, "pulse_width_ms": 1e3 * r.pulse_width,
           "grip_at_contact": r.grip_at_contact, "grip_peak": r.grip_peak, "t_swing": r.t_swing_start,
           "t_contact": r.t_contact_truth}
    row.update({f"limit_{k}": v for k, v in r.limits.items()})
    for k, v in r.extra.items():
        if k.startswith("slip_") and k.endswith("_rot"):
            row[f"{k}_deg"] = math.degrees(v) if v == v else v
        elif k.startswith("slip_") and k.endswith("_trans"):
            row[f"{k}_mm"] = 1e3 * v
        else:
            row[k] = v
    return row


def run_job(job: Job, root: str = str(ROOT)) -> dict:
    out = job.path(root)
    if out.exists():
        return {"status": "skip", "path": str(out), "wall_s": 0.0}
    t0 = time.time()
    payload: dict = {}
    try:
        cfg = job.config()
        if job.task == "hammer":
            from tactile_sim.episode import Episode

            ep = Episode(cfg, seed=job.seed, record_truth=False)
            res = ep.run(N_STRIKES)
            payload.update(summary=res.summary, strikes=[_strike_row(r) for r in res.strikes])
            rx = ep.tb.reactive
        else:
            from tactile_sim.sim.randomize import apply_dr, apply_runtime_dr
            from tactile_sim.tool_task import ToolTask

            s = job.dr_sample(cfg) if cfg.dr.enabled else None  # ToolTask itself does not randomize
            tt = ToolTask(apply_dr(cfg, s) if s is not None else cfg, seed=job.seed, t_max=TOOL_T_MAX[job.task])
            if s is not None:
                apply_runtime_dr(tt.tb, s)
            res = tt.run()
            payload.update(summary=res.summary, strikes=[])
            if s is not None:
                payload["summary"].update({f"dr_{k}": v for k, v in s.as_dict().items()})
            rx = tt.tb.reactive
        if rx is not None:
            payload["reactive"] = {"n": len(rx.detections), "ticks": rx.ticks_fired,
                                   "detections": [[d.t, d.source] for d in rx.detections]}
        status = "ok"
    except Exception:  # noqa: BLE001 - record the failure, keep the grid going
        payload["error"] = traceback.format_exc()
        status = "error"
    payload.update(job=job.meta(), status=status, wall_s=time.time() - t0)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, default=_jsonable))
    tmp.replace(out if status == "ok" else out.with_suffix(".error.json"))
    return {"status": status, "path": str(out), "wall_s": payload["wall_s"]}


def _synthesize(cfg_dict: dict) -> str:
    from tactile_sim.assets.fetch_hands import hand_xml
    from tactile_sim.model.hands.wuji_grasp import wrap_grasp

    wrap_grasp(SimConfig.from_dict(cfg_dict), hand_xml("wuji2"))
    return "ok"


def prewarm(jobs: list[Job], workers: int) -> None:
    """Build every cached asset the jobs need before they run in parallel: YCB hammer variants per grip location
    (serially) and WUJI grasp keyframes (one synthesis per cache key, in parallel)."""
    from tactile_sim.assets.fetch_hands import hand_xml
    from tactile_sim.assets.ycb import ensure_hammer
    from tactile_sim.model.hands.wuji_grasp import cache_path

    for gl in sorted({j.config().hammer.grip_from_head for j in jobs if j.task == "hammer"}):
        ensure_hammer(gl, download=False)
    wuji = hand_xml("wuji2")
    todo: dict[str, dict] = {}
    for j in jobs:
        cfg = j.episode_config()
        if cfg.gripper.hand == "wuji2" and j.task == "hammer":
            cp = cache_path(cfg, wuji)
            if not cp.exists():
                todo.setdefault(str(cp), cfg.to_dict())
    if not todo:
        return
    print(f"prewarm: {len(todo)} WUJI grasp syntheses", flush=True)
    with ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=mp.get_context("spawn")) as ex:
        for f in as_completed([ex.submit(_synthesize, d) for d in todo.values()]):
            f.result()


def run_all(jobs: list[Job], root: Path, workers: int) -> None:
    # one process per core: keep each worker's linear-algebra libraries single-threaded (spawned workers inherit this)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    pending = sorted((j for j in jobs if not j.path(root).exists()), key=lambda j: -j.cost)
    print(f"{len(jobs)} jobs, {len(jobs) - len(pending)} done, {len(pending)} to run on {workers} workers", flush=True)
    if not pending:
        return
    prewarm(pending, workers)
    t0 = time.time()
    done = n_err = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
        futs = {ex.submit(run_job, j, str(root)): j for j in pending}
        for f in as_completed(futs):
            j = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 - a worker died (e.g. out of memory)
                r = {"status": f"crash {e!r}", "wall_s": float("nan")}
            done += 1
            n_err += r["status"] != "ok"
            el = time.time() - t0
            eta = el / done * (len(pending) - done)
            print(f"[{done}/{len(pending)}] {r['status']:5s} {j.platform}/{j.task}/{j.cond.name}/seed{j.seed} "
                  f"{r['wall_s']:.0f}s  elapsed {el / 60:.1f} min, eta {eta / 60:.1f} min", flush=True)
    print(f"finished: {done} jobs, {n_err} errors, {(time.time() - t0) / 60:.1f} min", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "run"])
    ap.add_argument("--stage", choices=list(STAGES), default="screen")
    ap.add_argument("--platform", action="append", choices=list(PLATFORMS))
    ap.add_argument("--task", action="append", choices=sorted({t for ts in TASKS.values() for t in ts}))
    ap.add_argument("--only", action="append", help="condition name (repeatable)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    jobs = make_jobs(args.stage, args.platform, args.task, args.only)
    if args.cmd == "list":
        by: dict[tuple[str, str], list[str]] = {}
        for j in jobs:
            names = by.setdefault((j.platform, j.task), [])
            if j.cond.name not in names:
                names.append(j.cond.name)
        for (pf, task), names in by.items():
            print(f"{pf}/{task}: {len(names)} conditions x {len(seeds(args.stage, pf))} seeds: {' '.join(names)}")
        done = sum(j.path(args.root).exists() for j in jobs)
        print(f"total {len(jobs)} jobs, {done} done")
        return 0
    run_all(jobs, args.root, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
