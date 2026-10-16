"""P3, the plan's closed-loop statistics: the same base-library strikes under four conditions, compared as
distributions against the 8 kHz validator.

Conditions (every entry of the library, or `--entries`, played once from the CPU's state at its windup, unperturbed,
with its episode's randomization; base_library.replay_entry on the CPU, parity.strikes.gpu_replay on the GPU):

- cpu_8k_noslip5: the validator physics (0.125 ms, noslip 5; multiccd as the library's configuration says);
- cpu_4k: the training physics on the CPU (0.25 ms, no noslip);
- gpu_4k: MuJoCo Warp at 0.25 ms;
- gpu_2k: MuJoCo Warp at 0.5 ms.

Per condition and metric (nail advance per strike, peak force, pulse width, slip in the hand per strike, translation
and rotation): the mean with a 90 % bootstrap interval, and against the validator the plan's pass test: the means
within 10 % (slip 15 %), the 90 % intervals overlapping, and a two-sample Kolmogorov-Smirnov p > 0.01.

    python -m tactile_rl.parity.p3 --lib runs/base_library/hammer_v2.npz [--entries 0:400] [--cpu-workers 14] \
        [--out p3.json]

The CPU conditions belong in the validator's environment (MuJoCo 3.14, .venv) and the GPU ones in .venv-rl: run the
script once in each with `--skip` and merge the two outputs with `--combine gpu.json cpu.json`.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from tactile_sim.behaviors.base_library import OUTCOMES, load_library, replay_library, table

PHYSICS = {"cpu_8k_noslip5": {"timestep": 0.000125, "noslip_iterations": 5},
           "cpu_4k": {"timestep": 0.00025, "noslip_iterations": 0}}
GPU_DT = {"gpu_4k": 0.00025, "gpu_2k": 0.0005}
REF = "cpu_8k_noslip5"
# metric: (scale for the report, relative tolerance on the mean)
METRICS = {"advance": (1e3, 0.10), "peak2": (1.0, 0.10), "width": (1e3, 0.10), "slip_trans": (1e3, 0.15),
           "slip_rot": (180.0 / np.pi, 0.15)}
UNITS = {"advance": "mm", "peak2": "N", "width": "ms", "slip_trans": "mm", "slip_rot": "deg"}


def bootstrap_ci(x: np.ndarray, level: float = 0.90, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    a = (1.0 - level) / 2.0
    return float(np.quantile(means, a)), float(np.quantile(means, 1.0 - a))


def compare(tabs: dict[str, dict[str, np.ndarray]], ref: str = REF) -> dict:
    """The P3 report: per condition and metric, mean [90 % CI] and the pass tests against `ref` (on the strikes that
    hit in both)."""
    from scipy.stats import ks_2samp

    out: dict = {"conditions": {}, "pass": {}}
    hit_ref = np.asarray(tabs[ref]["hit"], bool)
    for name, tab in tabs.items():
        hit = np.asarray(tab["hit"], bool)
        both = hit & hit_ref
        row = {"n": int(len(hit)), "hit_rate": float(hit.mean())}
        for k, (scale, tol) in METRICS.items():
            x = np.asarray(tab[k], float)[hit] * scale
            mean = float(np.mean(x))
            lo, hi = bootstrap_ci(x)
            m = {"mean": mean, "ci90": [lo, hi], "unit": UNITS[k]}
            if name != ref:
                r = np.asarray(tabs[ref][k], float)[hit_ref] * scale
                r_lo, r_hi = bootstrap_ci(r)
                rel = abs(mean - float(np.mean(r))) / max(abs(float(np.mean(r))), 1e-12)
                ks = ks_2samp(x, r)
                paired = np.asarray(tab[k], float)[both] * scale - np.asarray(tabs[ref][k], float)[both] * scale
                m.update(rel_diff=rel, within_tol=bool(rel <= tol), ci_overlap=bool(lo <= r_hi and r_lo <= hi),
                         ks_p=float(ks.pvalue), ks_ok=bool(ks.pvalue > 0.01),
                         paired_abs_median=float(np.median(np.abs(paired))) if len(paired) else float("nan"))
                m["pass"] = bool(m["within_tol"] and m["ci_overlap"] and m["ks_ok"])
            row[k] = m
        out["conditions"][name] = row
        if name != ref:
            out["pass"][name] = bool(all(row[k]["pass"] for k in METRICS))
    return out


def format_table(rep: dict) -> str:
    names = list(rep["conditions"])
    lines = ["| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    lines.append("| strikes (hit) | " + " | ".join(f"{c['n']} ({100 * c['hit_rate']:.0f} %)"
                                                    for c in rep["conditions"].values()) + " |")
    for k in METRICS:
        cells = []
        for name in names:
            m = rep["conditions"][name][k]
            s = f"{m['mean']:.3g} [{m['ci90'][0]:.3g}, {m['ci90'][1]:.3g}]"
            if "pass" in m:
                s += f" {'ok' if m['pass'] else 'FAIL'} ({100 * m['rel_diff']:.1f} %, KS p {m['ks_p']:.2g})"
            cells.append(s)
        lines.append(f"| {k} ({UNITS[k]}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - report script
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--entries", default=None, help="a:b range (default: every entry)")
    ap.add_argument("--cpu-workers", type=int, default=14)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip", default="", help="comma list of conditions to skip")
    ap.add_argument("--combine", nargs="+", default=None, help="merge these outputs' tables and report")
    a = ap.parse_args(argv)
    if a.combine:
        tabs, walls = {}, {}
        for path in a.combine:
            with open(path) as fh:
                d = json.load(fh)
            tabs.update({k: {c: np.asarray(v) for c, v in t.items()} for k, t in d["tables"].items()})
            walls.update(d["report"].get("wall_s", {}))
        tabs = {k: tabs[k] for k in [REF, "cpu_4k", "gpu_4k", "gpu_2k"] if k in tabs}
        rep = compare(tabs)
        rep["wall_s"] = walls
        print(format_table(rep))
        print(json.dumps(rep["pass"]))
        if a.out:
            with open(a.out, "w") as fh:
                json.dump({"report": rep, "tables": {k: {c: np.asarray(v).tolist() for c, v in t.items()}
                                                     for k, t in tabs.items()}}, fh)
        return 0
    from tactile_rl.parity.strikes import gpu_replay

    n_all = len(load_library(a.lib)["e_t0"])
    if a.entries:
        lo, hi = (int(x) for x in a.entries.split(":"))
        entries = list(range(lo, min(hi, n_all)))
    else:
        entries = list(range(n_all))
    skip = set(filter(None, a.skip.split(",")))
    tabs, walls = {}, {}
    for name, dt in GPU_DT.items():
        if name in skip:
            continue
        t0 = time.time()
        g = gpu_replay(a.lib, entries, timestep=dt)
        tabs[name] = table(g["rows"], OUTCOMES)
        walls[name] = time.time() - t0
        print(f"{name}: {len(entries)} strikes in {walls[name]:.0f} s", flush=True)
    for name, phys in PHYSICS.items():
        if name in skip:
            continue
        t0 = time.time()
        rows = replay_library(a.lib, entries, a.cpu_workers, phys)
        tabs[name] = table(rows, OUTCOMES)
        walls[name] = time.time() - t0
        print(f"{name}: {len(entries)} strikes in {walls[name]:.0f} s", flush=True)
    tabs = {k: tabs[k] for k in [REF, "cpu_4k", "gpu_4k", "gpu_2k"] if k in tabs}
    rep = compare(tabs) if REF in tabs else {"conditions": {}, "pass": {}}
    rep["wall_s"] = walls
    if REF in tabs:
        print(format_table(rep))
        print(json.dumps(rep["pass"]))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"entries": entries, "report": rep,
                       "tables": {k: {c: np.asarray(v).tolist() for c, v in t.items()} for k, t in tabs.items()}},
                      fh)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
