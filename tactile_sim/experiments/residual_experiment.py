"""Residual experiments in the 8 kHz CPU testbed (the plan's "The experiment (all in sim)").

    python -m tactile_sim.experiments.residual_experiment list --stage E1
    python -m tactile_sim.experiments.residual_experiment run --stage E1 --workers 14 [--policy NAME=PATH ...]
    python -m tactile_sim.experiments.residual_experiment report --stage E1
    python -m tactile_sim.experiments.residual_experiment run --stage E1hold --policy s1k=student_1k.onnx  # holds

Every episode is the Vega U + WUJI under residual_config: MIT drives, the TaxelScan fast profile, a scripted base,
and the residual loop. --task hammer (default): the light hammer, the scripted swing as the base (SwingBase:
synergy hand command, arm lookahead, forecast), `strikes` strikes. --task drill: the pistol driver driving a
screw, the scripted feed as the base (DrillBase: the trigger pulled by the index finger, the plant's seating and
clutch forecast), one screw per episode. A seed fixes one residual_dr draw, shared by every condition, so
conditions are compared pairwise. One JSON per episode at runs/residual/[drill/]<stage>/<condition>/seed<k>.json;
a run skips files that exist.

Conditions (E1): "base" (the residual's base alone: constant grip, no anticipation, residual zero), "template_oracle"
(the human grip template timed by the sim-only impact flag), "reactive_taxel" (the slip-ablation taxel detector
raising the grip), "squeeze" (a constant grip at the hand's pre-impact level), and one "residual:<name>" per policy
given with --policy. The drill has no oracle template (its loads have no impact flag).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tactile_sim.config import SimConfig, fast_config, residual_config

ROOT = Path("runs/residual")
# seeds x strikes per episode; "fast" runs the 4 kHz physics (smoke tests only)
STAGES = {"smoke": {"seeds": 2, "strikes": 3, "fast": True}, "E1": {"seeds": 20, "strikes": 10, "fast": False},
          "E1hold": {"seeds": 20, "strikes": 10, "fast": False}, "hold_smoke": {"seeds": 1, "strikes": 2, "fast": True},
          "E3": {"seeds": 20, "strikes": 10, "fast": False}, "E4": {"seeds": 20, "strikes": 10, "fast": False},
          "E5": {"seeds": 10, "strikes": 10, "fast": False}, "E6": {"seeds": 20, "strikes": 10, "fast": False}}
# E4: the observation blocks each sensing variant cannot see
SENSING = {"touch": ("hand_current",), "current": ("taxel_codes", "patch_features", "board_status"),
           "proprio": ("hand_current", "taxel_codes", "patch_features", "board_status"), "both": ()}
# E5: hardware requirement sweeps, one parameter at a time on top of each seed's randomization
SWEEPS = {
    "cmd_latency": ("wuji", "cmd_latency", (0.0005, 0.001, 0.002, 0.003, 0.005)),
    "sdk_lpf_hz": ("wuji", "sdk_lpf_hz", (None, 300.0, 100.0, 50.0)),
    "effort_limit": ("wuji", "effort_limit_ratio", (0.8, 1.0, 1.2)),
    "tau_i": ("wuji", "tau_i", (0.0001, 0.0005, 0.002)),
    "taxel_rate": ("sensors", "ts_rate", (1000.0, 770.0, 500.0, 250.0)),
    "taxel_latency": ("sensors", "ts_bridge_latency", ((0.000125, 0.001), (0.001, 0.002), (0.003, 0.005))),
    "taxel_bits": ("sensors", "ts_code_bits", (8, 6)),
}
# E6: held-out physics, outside the training ranges
HELD_OUT = {"heavy_head": {"hammer": {"head_mass": 0.8}}, "slick_skin": {"gripper": {"hand_friction": 0.5}},
            "stiff_nail": {"plant": {"resistance_0": 250.0}}, "soft_skin": {"gripper": {"hand_solref": (0.006, 1.0)}}}
# (the drill's: a heavier battery; a clutch set above the training range on a stronger 4.5 mm screw; a brushless
# M12-class drill, stronger, faster and heavier-geared than the 12 V class trained on)
HELD_OUT_DRILL = {"heavy_battery": {"drill": {"pistol_battery_mass": 0.75}},
                  "slick_skin": {"gripper": {"hand_friction": 0.5}},
                  "strong_clutch": {"drill": {"clutch_torque": 2.4, "screw_twist_off": 3.8, "strip_torque": 5.0}},
                  "brushless_drill": {"drill": {"stall_torque": 24.0, "free_speed": 49.5, "spindle_inertia": 1.4e-2,
                                                "gear_ratio": 50.0}},
                  "soft_skin": {"gripper": {"hand_solref": (0.006, 1.0)}}}
TASKS = ("hammer", "drill")
# E1hold: the base at these fractions of its grip (the GPU curriculum's hold levels; ResidualCfg.hold_level)
HOLDS = (0.3, 0.5, 0.8, 1.0)
LOST_MM, LOST_DEG = 20.0, 30.0  # the GPU env's lost grasp (terminations.tool_left_hand), from the swing's start
SLIP_LPF_HZ = 30.0  # the slip path's low-pass, as the GPU runtime's
DRILL_T_MAX = 10.0  # s per screw


@dataclass(frozen=True)
class Condition:
    name: str
    overrides: dict = field(default_factory=dict)
    policy: str = "zero"  # "zero" or a path to an exported policy


def e1_conditions(policies: dict[str, str] | None = None, task: str = "hammer") -> list[Condition]:
    const = {"swing": {"grip_profile": "constant"}}
    out = [
        Condition("base", const),
        Condition("template_oracle", {"swing": {"grip_profile": "template", "post_switch": "flag"},
                                      "controller": {"impact_sources": ("ft", "accel")}}),
        # the taxel detector without its accelerometer channel: the real hand has no accelerometer
        Condition("reactive_taxel", {**const, "reactive": {"mode": "taxel", "accel_thresh": 1e9}}),
        Condition("squeeze", {**const, "controller": {"grip_hold": 60.0}}),
    ]
    for name, path in (policies or {}).items():
        from tactile_sim.control.policy_io import PolicyMeta

        rate = PolicyMeta.read(path).rate  # the loop runs at the policy's own rate
        out.append(Condition(f"residual:{name}", {**const, "residual": {"rate": rate}}, policy=path))
    if task == "drill":
        out = [c for c in out if c.name != "template_oracle"]
    return out


@dataclass(frozen=True)
class Job:
    stage: str
    condition: Condition
    seed: int
    task: str = "hammer"

    def path(self, root: Path) -> Path:
        safe = self.condition.name.replace(":", "_").replace("@", "__").replace("=", "-")
        return task_root(root, self.task) / self.stage / safe / f"seed{self.seed}.json"

    def config(self) -> tuple[SimConfig, dict]:
        from tactile_sim.sim.randomize import apply_residual_dr, residual_dr

        st = STAGES[self.stage]
        cfg = residual_config(base=fast_config() if st["fast"] else None, task=self.task)
        dr = residual_dr(cfg, np.random.default_rng(self.seed))
        cfg = apply_residual_dr(cfg, dr)
        if self.condition.overrides:
            cfg = cfg.replace(**self.condition.overrides)
        return cfg, dr


def task_root(root: Path, task: str) -> Path:
    """Where a task's results live: the hammer's at the root (as before the drill existed), the drill's under drill/."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r} (expected one of {TASKS})")
    return root if task == "hammer" else root / task


def stage_conditions(stage: str, policies: dict[str, str] | None = None, task: str = "hammer") -> list[Condition]:
    """The conditions a stage runs. E1/E3/smoke: the baselines plus every policy (E3 passes one per rate); E4: every
    policy (each run with its sidecar's mask); E5: the base and each policy under every sweep point; E6: the base
    and each policy under each held-out physics."""
    if stage in ("smoke", "E1", "E3"):
        return e1_conditions(policies, task)
    if stage in ("E1hold", "hold_smoke"):  # the base and each policy with the base at each hold level
        conds = [c for c in e1_conditions(policies, task) if c.name == "base" or c.name.startswith("residual:")]
        return [Condition(f"{c.name}@hold={h:g}", _merge(c.overrides, {"residual": {"hold_level": h}}), c.policy)
                for h in HOLDS for c in conds]
    pol = e1_conditions(policies, task)
    base = pol[0]
    pol = [c for c in pol if c.name.startswith("residual:")]
    if stage == "E4":
        out = []
        for c in pol:
            from tactile_sim.control.policy_io import PolicyMeta

            mask = tuple(PolicyMeta.read(c.policy).extra.get("mask_blocks", ()))
            out.append(Condition(c.name, _merge(c.overrides, {"residual": {"mask_blocks": mask}}), c.policy))
        return out
    out = []
    if stage == "E5":
        for key, (sec, fld, values) in SWEEPS.items():
            for v in values:
                tag = "off" if v is None else (f"{v[0]:g}-{v[1]:g}" if isinstance(v, tuple) else f"{v:g}")
                ov = {sec: {fld: v}}
                if key == "taxel_rate":  # slower frames are not stale: scale the guard with the frame period
                    ov["residual"] = {"stale_taxel": max(0.005, 2.0 / v + 0.002)}
                for c in [base] + pol:
                    out.append(Condition(f"{c.name}@{key}={tag}", _merge(c.overrides, ov), c.policy))
    elif stage == "E6":
        for key, ov in (HELD_OUT_DRILL if task == "drill" else HELD_OUT).items():
            for c in [base] + pol:
                out.append(Condition(f"{c.name}@{key}", _merge(c.overrides, ov), c.policy))
    return out


def _merge(a: dict, b: dict) -> dict:
    out = {k: dict(v) for k, v in a.items()}
    for k, v in b.items():
        out.setdefault(k, {}).update(v)
    return out


def make_jobs(stage: str, policies: dict[str, str] | None = None, only: str | None = None,
              task: str = "hammer") -> list[Job]:
    st = STAGES[stage]
    conds = stage_conditions(stage, policies, task)
    if only:
        conds = [c for c in conds if c.name == only]
    return [Job(stage, c, s, task) for c in conds for s in range(st["seeds"])]


def load_policy(spec: str):
    from tactile_sim.control.residual_l2 import make_policy

    return make_policy(spec)


class Recorder:
    """1 kHz: drive effort (I^2 t, peak current) and the residual's offsets, for the reaction-latency metric."""

    def __init__(self, tb):
        self.tb = tb
        self.hand = tb.world.hand
        self.t, self.off = [], []
        self.i2t = 0.0
        self.i_peak = 0.0
        # per millisecond: time, the tool's pose in the hand, the 30 Hz low-passed pose's step (mm, deg), the drives'
        # mean (i / i_rated)^2 (the GPU runtime's I^2 t integrand)
        self.ms_t, self.ms_pose, self.ms_path, self.ms_i2 = [], [], [], []
        self._lp = None
        self._a = 1.0 - math.exp(-2.0 * math.pi * SLIP_LPF_HZ * 1e-3)

    def __call__(self, t: float) -> None:
        d = self.hand.drive
        r = d.i / d.i_rated
        self.i2t += float(np.sum(r * r)) * 1e-3
        self.i_peak = max(self.i_peak, float(np.max(np.abs(r))))
        p, rv = self.tb.world.hammer_in_hand()
        x = np.concatenate([p, rv])
        if self._lp is None:
            self._lp, step = x.copy(), (0.0, 0.0)
        else:
            lp = self._lp + self._a * (x - self._lp)
            dx = lp - self._lp
            step = (1e3 * float(np.linalg.norm(dx[:3])), math.degrees(float(np.linalg.norm(dx[3:]))))
            self._lp = lp
        self.ms_t.append(t)
        self.ms_pose.append(x)
        self.ms_path.append(step)
        self.ms_i2.append(float(np.mean(r * r)) * 1e-3)
        res = self.tb.residual
        if res is not None:
            sp, jp = res.spec, res.jp
            n = np.concatenate([res.off[0] / np.asarray(sp.dq_max)[jp.group], res.off[1] / jp.kp_nom,
                                res.off[2] / (sp.tau_frac * jp.tau_rated)])
            self.t.append(t)
            self.off.append(n)

    def reaction_ms(self, t_contact: float, thresh: float = 0.05, window: float = 0.1) -> float:
        """Time from contact until the residual's normalized offsets first move by `thresh` (NaN if they do not
        within `window`)."""
        if not self.t or not np.isfinite(t_contact):
            return float("nan")
        t = np.asarray(self.t)
        off = np.asarray(self.off)
        k0 = int(np.searchsorted(t, t_contact))
        if k0 >= len(t):
            return float("nan")
        moved = np.max(np.abs(off[k0:] - off[k0]), axis=1) > thresh
        ks = np.flatnonzero(moved & (t[k0:] <= t_contact + window))
        return float(1e3 * (t[k0 + ks[0]] - t_contact)) if len(ks) else float("nan")

    def window(self, t0: float, t1: float) -> dict:
        """From the swing's start t0 to the strike's end t1: the slip path (mm, deg), I^2 t (mean over the joints),
        and whether the tool moved 2 cm or 30 degrees from its pose at t0 (the GPU env's lost grasp)."""
        t = np.asarray(self.ms_t)
        m = (t > t0) & (t <= t1)
        if not m.any() or not np.isfinite(t1):
            return {"path_mm": float("nan"), "path_deg": float("nan"), "i2t_w": float("nan"), "lost_w": False}
        path = np.asarray(self.ms_path)[m]
        pose = np.asarray(self.ms_pose)
        k0 = max(int(np.searchsorted(t, t0)) - 1, 0)
        dp = np.linalg.norm(pose[m, :3] - pose[k0, :3], axis=1) * 1e3
        dr = np.degrees(np.linalg.norm(pose[m, 3:] - pose[k0, 3:], axis=1))
        return {"path_mm": float(path[:, 0].sum()), "path_deg": float(path[:, 1].sum()),
                "i2t_w": float(np.asarray(self.ms_i2)[m].sum()),
                "lost_w": bool((dp > LOST_MM).any() or (dr > LOST_DEG).any())}


def _strike_row(r, rec: Recorder, t_end: float = float("nan")) -> dict:
    return {"idx": r.idx, "hit": bool(r.hit), "drop": bool(r.drop), "slip_rot_deg": math.degrees(r.slip_rot),
            "slip_trans_mm": 1e3 * r.slip_trans, "advance_mm": 1e3 * r.depth_inc, "peak_force": r.peak_force_truth,
            "pulse_width_ms": 1e3 * r.pulse_width, "v_face": r.v_strike_actual, "t_contact": r.t_contact_truth,
            "reaction_ms": rec.reaction_ms(r.t_contact_truth), "t_swing_start": r.t_swing_start, "t_end": t_end,
            **rec.window(r.t_swing_start, t_end)}


def _hammer_episode(cfg: SimConfig, job: Job):
    """The scripted swing: testbed, and run(recorder) -> summary and per-strike rows."""
    from tactile_sim.behaviors.base_sources import SwingBase
    from tactile_sim.episode import Episode

    ep = Episode(cfg, seed=job.seed)
    ep.tb.residual.base_fn = SwingBase(ep.tb, ep.swing, rng=np.random.default_rng(10_000 + job.seed))

    def run(rec: Recorder) -> dict:
        res = ep.run(STAGES[job.stage]["strikes"])
        ends = {name: t for name, t in ep.events if name.startswith("strike_end_")}
        return {"summary": res.summary,
                "strikes": [_strike_row(r, rec, ends.get(f"strike_end_{r.idx}", float("nan"))) for r in res.strikes]}

    return ep.tb, run


def _drill_episode(cfg: SimConfig, job: Job):
    """The scripted drill feed driving one screw: testbed, and run(recorder) -> summary (seated or not, cam-outs,
    the tool's slip in the hand, push, effort) and the reaction to the seating (the clutch's first ratchet)."""
    from tactile_sim.behaviors.base_sources import DrillBase
    from tactile_sim.tool_task import ToolTask

    tt = ToolTask(cfg, seed=job.seed, t_max=DRILL_T_MAX)
    tt.tb.residual.base_fn = DrillBase(tt.tb, tt.behavior, rng=np.random.default_rng(10_000 + job.seed))

    def run(rec: Recorder) -> dict:
        s = dict(tt.run().summary)
        s["aim_turn_deg"] = math.degrees(tt.behavior.aim_correction[0])
        return {"summary": s, "events": [{"kind": "seated", "t": s["t_done"], "reaction_ms": rec.reaction_ms(
            s["t_done"])}]}

    return tt.tb, run


def run_job(job: Job, root: str = str(ROOT)) -> dict:
    out = job.path(Path(root))
    if out.exists():
        return {"status": "skip", "path": str(out)}
    t0 = time.time()
    cfg, dr = job.config()
    tb, run = (_drill_episode if job.task == "drill" else _hammer_episode)(cfg, job)
    tb.sensors.set_noise_scale(dr["_runtime"]["sensor_noise_scale"])
    policy = load_policy(job.condition.policy)
    tb.residual.check_policy(policy)
    tb.residual.policy = policy
    rec = Recorder(tb)
    tb.sched.register("recorder", 1000.0, rec)
    result = run(rec)
    r = tb.residual
    payload = {
        "job": {"task": job.task, "stage": job.stage, "condition": job.condition.name,
                "policy": job.condition.policy, "seed": job.seed},
        **result,
        "residual": {"ticks": r.n_ticks, "stale": r.n_stale, "nonfinite": r.n_nonfinite, "i2t": rec.i2t,
                     "i_peak_ratio": rec.i_peak, "t_max_winding": float(tb.world.hand.drive.T.max())},
        "dropped": bool(tb.grip.dropped), "dr": dr, "wall_s": time.time() - t0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, default=_jsonable))
    tmp.rename(out)
    return {"status": "ok", "path": str(out), "wall_s": payload["wall_s"]}


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def prewarm(jobs: list[Job]) -> None:
    """Synthesize every grasp keyframe the jobs need once, serially, before they run in parallel."""
    from tactile_sim.assets.fetch_hands import hand_xml
    from tactile_sim.model.hands.wuji_grasp import cache_path, wrap_grasp

    hand = hand_xml("wuji2")
    seen = set()
    for j in jobs:
        cfg, _ = j.config()
        key = cache_path(cfg, hand)
        if key not in seen:
            seen.add(key)
            wrap_grasp(cfg, hand)


def run_all(jobs: list[Job], root: Path, workers: int) -> None:
    todo = [j for j in jobs if not j.path(root).exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run on {workers} workers", flush=True)
    if not todo:
        return
    prewarm(todo)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
        futs = {ex.submit(run_job, j, str(root)): j for j in todo}
        for k, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                r = f.result()
                msg = f"ok {r.get('wall_s', 0):.0f}s"
            except Exception:  # noqa: BLE001 - report and keep going
                msg = "ERROR\n" + traceback.format_exc()
            print(f"[{k}/{len(todo)}] {j.condition.name}/seed{j.seed} {msg} elapsed {(time.time() - t0) / 60:.1f} min",
                  flush=True)


def load_results(stage: str, root: Path = ROOT, task: str = "hammer") -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((task_root(root, task) / stage).glob("*/seed*.json"))]


METRICS = {"hammer": ("slip_rot_deg", "slip_trans_mm", "advance_mm", "drops", "i2t", "i_peak"),
           "drill": ("success", "cam_outs", "slip_rot_deg", "slip_trans_mm", "t_seat", "drops", "i2t", "i_peak")}


def per_seed(results: list[dict]) -> dict[str, dict[int, dict]]:
    """condition -> seed -> metrics: per strike averaged over the episode (hammer), or the episode's (drill)."""
    out: dict[str, dict[int, dict]] = {}
    for r in results:
        effort = {"i2t": r["residual"]["i2t"], "i_peak": r["residual"]["i_peak_ratio"]}
        if r["job"].get("task", "hammer") == "drill":
            s = r["summary"]
            m = {"success": float(s["done"] and not s["lost"]), "cam_outs": float(s["cam_outs"]),
                 "slip_rot_deg": math.degrees(s["max_slip_rot"]), "slip_trans_mm": 1e3 * s["max_slip_trans"],
                 "t_seat": float(s["t_done"]), "drops": float(s["lost"]), **effort}
        else:
            st = r["strikes"]
            if not st:
                continue
            m = {"slip_rot_deg": float(np.mean([x["slip_rot_deg"] for x in st])),
                 "slip_trans_mm": float(np.mean([x["slip_trans_mm"] for x in st])),
                 "advance_mm": float(np.mean([x["advance_mm"] for x in st])), "drops": float(r["dropped"]),
                 **effort}
        out.setdefault(r["job"]["condition"], {})[r["job"]["seed"]] = m
    return out


def paired_bootstrap(a: list[float], b: list[float], n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of b - a over paired seeds, with a 95 % bootstrap interval."""
    d = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def report(stage: str, root: Path = ROOT, ref: str = "base", task: str = "hammer") -> str:
    data = per_seed(load_results(stage, root, task))
    if ref not in data:
        return f"no '{ref}' results under {task_root(root, task) / stage}"
    keys = METRICS[task]
    lines = ["| condition | n | " + " | ".join(keys) + " |", "|---" * (len(keys) + 2) + "|"]
    for cond, seeds in sorted(data.items()):
        cells = []
        for k in keys:
            mean = np.nanmean([seeds[s][k] for s in seeds]) if any(np.isfinite(seeds[s][k]) for s in seeds) \
                else float("nan")
            # pairs where both are finite (a screw that never seated has no seating time)
            common = [s for s in sorted(set(seeds) & set(data[ref])) if np.isfinite(seeds[s][k]) and
                      np.isfinite(data[ref][s][k])]
            if cond == ref or len(common) < 2:
                cells.append(f"{mean:.3g}")
            else:
                d, lo, hi = paired_bootstrap([data[ref][s][k] for s in common], [seeds[s][k] for s in common])
                cells.append(f"{mean:.3g} ({d:+.2g} [{lo:+.2g}, {hi:+.2g}])")
        lines.append(f"| {cond} | {len(seeds)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def hold_table(stage: str = "E1hold", root: Path = ROOT) -> tuple[str, list[dict]]:
    """E1hold's per condition (policy and hold level) over every strike of every seed: strikes; lost grasps per
    strike (the testbed's drop, which interrupts the strike and ends the episode, or the tool 2 cm / 30 degrees from
    its pose at the swing's start); the episodes a drop ended (the plan's episode success) and the first strikes lost
    (the GPU evaluation's first swings); net slip and slip path per kept strike; I^2 t from the swing's start to the
    strike's end (mean over the joints, as the GPU's); the nail's advance; and each against the zero residual at the
    full hold."""
    rows = []
    by: dict[str, list[dict]] = {}
    for r in load_results(stage, root, "hammer"):
        by.setdefault(r["job"]["condition"], []).append(r)

    def share(xs) -> float:
        return float(np.mean(xs)) if len(xs) else float("nan")

    for cond, eps in sorted(by.items()):
        name, _, h = cond.rpartition("@hold=")
        st = [x for r in eps for x in r["strikes"]]
        for x in st:
            x["lost"] = bool(x.get("lost_w", False) or x["drop"])
        kept = [x for x in st if not x["lost"]]

        def mean(key, xs=kept):
            v = [x[key] for x in xs if x.get(key) is not None and np.isfinite(x[key])]
            return float(np.mean(v)) if v else float("nan")

        rows.append({"policy": name, "hold": float(h), "episodes": len(eps), "strikes": len(st),
                     "lost": share([x["lost"] for x in st]),
                     "episodes_dropped": share([bool(r["dropped"]) for r in eps]),
                     "lost_first": share([x["lost"] for x in st if x["idx"] == 0]),
                     "net_mm": mean("slip_trans_mm"), "net_deg": mean("slip_rot_deg"), "path_mm": mean("path_mm"),
                     "path_deg": mean("path_deg"), "i2t_w": mean("i2t_w"), "advance_mm": mean("advance_mm"),
                     "hit": share([x["hit"] for x in st])})
    base = {r["hold"]: r for r in rows if r["policy"] == "base"}
    lines = ["| policy | hold | episodes / strikes | lost per strike | episodes ended by a drop | first strike lost | "
             "net slip mm / deg | slip path mm / deg | I2t (swing to end) | advance mm | hits | against zero at the "
             "same hold: net mm, deg, I2t | against zero at the full hold: net mm, deg, I2t |",
             "|---" * 13 + "|"]

    def against(r, z):
        if z is None:
            return ""
        return f"{r['net_mm'] / z['net_mm']:.2f}, {r['net_deg'] / z['net_deg']:.2f}, {r['i2t_w'] / z['i2t_w']:.2f}"

    for r in sorted(rows, key=lambda r: (r["hold"], r["policy"])):
        lines.append(f"| {r['policy']} | {r['hold']:g} | {r['episodes']} / {r['strikes']} | {100 * r['lost']:.1f} % | "
                     f"{100 * r['episodes_dropped']:.0f} % | {100 * r['lost_first']:.0f} % | {r['net_mm']:.3f} / "
                     f"{r['net_deg']:.3f} | {r['path_mm']:.2f} / {r['path_deg']:.2f} | {r['i2t_w']:.3f} | "
                     f"{r['advance_mm']:.2f} | {100 * r['hit']:.1f} % | {against(r, base.get(r['hold']))} | "
                     f"{against(r, base.get(1.0))} |")
    return "\n".join(lines), rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("list", "run", "report"))
    ap.add_argument("--stage", default="smoke", choices=sorted(STAGES))
    ap.add_argument("--task", default="hammer", choices=TASKS)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--policy", action="append", default=[], help="NAME=PATH of an exported residual policy")
    ap.add_argument("--only", default=None, help="run one condition")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    policies = dict(p.split("=", 1) for p in a.policy)
    jobs = make_jobs(a.stage, policies, a.only, a.task)
    root = Path(a.root)
    if a.cmd == "list":
        for j in jobs:
            print(j.condition.name, j.seed, "done" if j.path(root).exists() else "")
    elif a.cmd == "run":
        run_all(jobs, root, a.workers)
    elif a.stage in ("E1hold", "hold_smoke"):
        print(hold_table(a.stage, root)[0])
    else:
        print(report(a.stage, root, task=a.task))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
