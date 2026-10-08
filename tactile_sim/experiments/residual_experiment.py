"""Residual experiments in the 8 kHz CPU testbed (the plan's "The experiment (all in sim)").

    python -m tactile_sim.experiments.residual_experiment list --stage E1
    python -m tactile_sim.experiments.residual_experiment run --stage E1 --workers 14 [--policy NAME=PATH ...]
    python -m tactile_sim.experiments.residual_experiment report --stage E1

Every episode is the Vega U + WUJI hammering with the light hammer under residual_config: MIT drives, the TaxelScan
fast profile, the scripted swing as the base (SwingBase: synergy hand command, arm lookahead, forecast), and the
residual loop. A seed fixes one residual_dr draw, shared by every condition, so conditions are compared pairwise.
One JSON per episode at runs/residual/<stage>/<condition>/seed<k>.json; a run skips files that exist.

Conditions (E1): "base" (the residual's base alone: constant grip, no anticipation, residual zero), "template_oracle"
(the human grip template timed by the sim-only impact flag), "reactive_taxel" (the slip-ablation taxel detector
raising the grip), "squeeze" (a constant grip at the hand's pre-impact level), and one "residual:<name>" per policy
given with --policy.
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
            "stiff_nail": {"plant": {"resistance_0": 400.0}}, "soft_skin": {"gripper": {"hand_solref": (0.006, 1.0)}}}


@dataclass(frozen=True)
class Condition:
    name: str
    overrides: dict = field(default_factory=dict)
    policy: str = "zero"  # "zero" or a path to an exported policy


def e1_conditions(policies: dict[str, str] | None = None) -> list[Condition]:
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
    return out


@dataclass(frozen=True)
class Job:
    stage: str
    condition: Condition
    seed: int

    def path(self, root: Path) -> Path:
        safe = self.condition.name.replace(":", "_").replace("@", "__").replace("=", "-")
        return root / self.stage / safe / f"seed{self.seed}.json"

    def config(self) -> tuple[SimConfig, dict]:
        from tactile_sim.sim.randomize import apply_residual_dr, residual_dr

        st = STAGES[self.stage]
        cfg = residual_config(base=fast_config() if st["fast"] else None)
        dr = residual_dr(cfg, np.random.default_rng(self.seed))
        cfg = apply_residual_dr(cfg, dr)
        if self.condition.overrides:
            cfg = cfg.replace(**self.condition.overrides)
        return cfg, dr


def stage_conditions(stage: str, policies: dict[str, str] | None = None) -> list[Condition]:
    """The conditions a stage runs. E1/E3/smoke: the baselines plus every policy (E3 passes one per rate); E4: every
    policy (each run with its sidecar's mask); E5: the base and each policy under every sweep point; E6: the base
    and each policy under each held-out physics."""
    if stage in ("smoke", "E1", "E3"):
        return e1_conditions(policies)
    pol = e1_conditions(policies)
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
        for key, ov in HELD_OUT.items():
            for c in [base] + pol:
                out.append(Condition(f"{c.name}@{key}", _merge(c.overrides, ov), c.policy))
    return out


def _merge(a: dict, b: dict) -> dict:
    out = {k: dict(v) for k, v in a.items()}
    for k, v in b.items():
        out.setdefault(k, {}).update(v)
    return out


def make_jobs(stage: str, policies: dict[str, str] | None = None, only: str | None = None) -> list[Job]:
    st = STAGES[stage]
    conds = stage_conditions(stage, policies)
    if only:
        conds = [c for c in conds if c.name == only]
    return [Job(stage, c, s) for c in conds for s in range(st["seeds"])]


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

    def __call__(self, t: float) -> None:
        d = self.hand.drive
        r = d.i / d.i_rated
        self.i2t += float(np.sum(r * r)) * 1e-3
        self.i_peak = max(self.i_peak, float(np.max(np.abs(r))))
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


def _strike_row(r, rec: Recorder) -> dict:
    return {"idx": r.idx, "hit": bool(r.hit), "drop": bool(r.drop), "slip_rot_deg": math.degrees(r.slip_rot),
            "slip_trans_mm": 1e3 * r.slip_trans, "advance_mm": 1e3 * r.depth_inc, "peak_force": r.peak_force_truth,
            "pulse_width_ms": 1e3 * r.pulse_width, "v_face": r.v_strike_actual, "t_contact": r.t_contact_truth,
            "reaction_ms": rec.reaction_ms(r.t_contact_truth)}


def run_job(job: Job, root: str = str(ROOT)) -> dict:
    from tactile_sim.behaviors.base_sources import SwingBase
    from tactile_sim.episode import Episode

    out = job.path(Path(root))
    if out.exists():
        return {"status": "skip", "path": str(out)}
    t0 = time.time()
    cfg, dr = job.config()
    ep = Episode(cfg, seed=job.seed)
    tb = ep.tb
    tb.sensors.set_noise_scale(dr["_runtime"]["sensor_noise_scale"])
    tb.residual.base_fn = SwingBase(tb, ep.swing, rng=np.random.default_rng(10_000 + job.seed))
    policy = load_policy(job.condition.policy)
    tb.residual.check_policy(policy)
    tb.residual.policy = policy
    rec = Recorder(tb)
    tb.sched.register("recorder", 1000.0, rec)
    res = ep.run(STAGES[job.stage]["strikes"])
    r = tb.residual
    payload = {
        "job": {"stage": job.stage, "condition": job.condition.name, "policy": job.condition.policy, "seed": job.seed},
        "summary": res.summary, "strikes": [_strike_row(s, rec) for s in res.strikes],
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


def load_results(stage: str, root: Path = ROOT) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((root / stage).glob("*/seed*.json"))]


def per_seed(results: list[dict]) -> dict[str, dict[int, dict]]:
    """condition -> seed -> metrics averaged over that episode's strikes."""
    out: dict[str, dict[int, dict]] = {}
    for r in results:
        s = r["strikes"]
        if not s:
            continue
        m = {"slip_rot_deg": float(np.mean([x["slip_rot_deg"] for x in s])),
             "slip_trans_mm": float(np.mean([x["slip_trans_mm"] for x in s])),
             "advance_mm": float(np.mean([x["advance_mm"] for x in s])),
             "drops": float(r["dropped"]), "i2t": r["residual"]["i2t"], "i_peak": r["residual"]["i_peak_ratio"]}
        out.setdefault(r["job"]["condition"], {})[r["job"]["seed"]] = m
    return out


def paired_bootstrap(a: list[float], b: list[float], n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of b - a over paired seeds, with a 95 % bootstrap interval."""
    d = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def report(stage: str, root: Path = ROOT, ref: str = "base") -> str:
    data = per_seed(load_results(stage, root))
    if ref not in data:
        return f"no '{ref}' results under {root / stage}"
    keys = ("slip_rot_deg", "slip_trans_mm", "advance_mm", "drops", "i2t", "i_peak")
    lines = ["| condition | n | " + " | ".join(keys) + " |", "|---" * (len(keys) + 2) + "|"]
    for cond, seeds in sorted(data.items()):
        common = sorted(set(seeds) & set(data[ref]))
        cells = []
        for k in keys:
            mean = np.mean([seeds[s][k] for s in seeds])
            if cond == ref or not common:
                cells.append(f"{mean:.3g}")
            else:
                d, lo, hi = paired_bootstrap([data[ref][s][k] for s in common], [seeds[s][k] for s in common])
                cells.append(f"{mean:.3g} ({d:+.2g} [{lo:+.2g}, {hi:+.2g}])")
        lines.append(f"| {cond} | {len(seeds)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("list", "run", "report"))
    ap.add_argument("--stage", default="smoke", choices=sorted(STAGES))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--policy", action="append", default=[], help="NAME=PATH of an exported residual policy")
    ap.add_argument("--only", default=None, help="run one condition")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    policies = dict(p.split("=", 1) for p in a.policy)
    jobs = make_jobs(a.stage, policies, a.only)
    root = Path(a.root)
    if a.cmd == "list":
        for j in jobs:
            print(j.condition.name, j.seed, "done" if j.path(root).exists() else "")
    elif a.cmd == "run":
        run_all(jobs, root, a.workers)
    else:
        print(report(a.stage, root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
