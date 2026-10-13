"""Rerun some of a sweep's validator episodes from this tree and compare them with the sweep's own files: the same
code and inputs give the same JSON (wall time aside), so a tree that changed something else (another observation
layout, other training code) can add conditions to a sweep another tree began.

    python -m tactile_sim.experiments.rerun_check --src ROOT --dst ROOT2 --stage Efric --policy s250=PATH \\
        --condition "base@hold=1@mu=0.6" --condition "residual:s250@hold=1@mu=0.6" [--seed 0] [--workers 2]

Prints per episode the values compared and those that differ (the first few), then CONSISTENT or DIFFERENT. Exits 0
when it could compare every episode (whatever the verdict), 2 when one is missing in either root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tactile_sim.experiments.residual_experiment import HAMMER, make_jobs, run_all


def leaves(x, path: str = ""):
    """(path, value) of every scalar in a JSON payload, wall time aside."""
    if isinstance(x, dict):
        for k, v in x.items():
            if k != "wall_s":
                yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def compare(a: dict, b: dict) -> tuple[int, list[str], float]:
    """Values compared, the paths that differ (NaN equals NaN), the largest numeric difference."""
    la, lb = dict(leaves(a)), dict(leaves(b))

    def same(x, y):
        return x == y or (isinstance(x, float) and isinstance(y, float) and x != x and y != y)

    diff = sorted(k for k in set(la) | set(lb) if not same(la.get(k), lb.get(k)))
    nums = [abs(la[k] - lb[k]) for k in diff
            if isinstance(la.get(k), (int, float)) and isinstance(lb.get(k), (int, float))]
    return len(la), diff, max(nums, default=0.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the sweep's --root")
    ap.add_argument("--dst", required=True, help="where this tree's reruns go")
    ap.add_argument("--stage", default="Efric")
    ap.add_argument("--policy", action="append", default=[], help="NAME=PATH, as the sweep ran it")
    ap.add_argument("--condition", action="append", required=True)
    ap.add_argument("--seed", type=int, action="append", default=None)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    seeds = set(a.seed or [0])
    policies = dict(p.split("=", 1) for p in a.policy)
    jobs = [j for j in make_jobs(a.stage, policies, None, "hammer", HAMMER)
            if j.condition.name in set(a.condition) and j.seed in seeds]
    if len(jobs) != len(a.condition) * len(seeds):
        print(f"expected {len(a.condition) * len(seeds)} episodes, found {[j.condition.name for j in jobs]}")
        return 2
    src, dst = Path(a.src), Path(a.dst)
    run_all(jobs, dst, a.workers)
    bad = 0
    for j in jobs:
        if not (j.path(src).exists() and j.path(dst).exists()):
            print(f"{j.condition.name} seed {j.seed}: missing ({j.path(src)} or {j.path(dst)})")
            return 2
        n, diff, worst = compare(json.loads(j.path(src).read_text()), json.loads(j.path(dst).read_text()))
        print(f"{j.condition.name} seed {j.seed}: {n} values, {len(diff)} differ (largest numeric difference "
              f"{worst:g})")
        for k in diff[:10]:
            print(f"  {k}")
        bad += bool(diff)
    print("CONSISTENT" if not bad else "DIFFERENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
