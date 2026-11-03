"""What a run of the planned demo generator produced (vla.demos --generator planned): per task, how many episodes
succeeded, failed in simulation, failed the quality gates (vla.quality) or were rejected by the pre-check; why; the
success rate by how far the episode started from the task and how far its windup apex left the nominal one; and the
coverage of each sampled parameter, intended against achieved. Each parameter's unit range is cut into COVERAGE_BINS
equal bins; a bin holding under half its share of the episodes is flagged. Achieved coverage is shown twice: after
the pre-check (the plans that were simulated) and among the exported demos (successful and passing the gates), so a
gap can be traced to infeasibility or to the simulation and the gates.

    python -m tactile_sim.vla.coverage runs/vla_demos/hammer_planned runs/vla_demos/drill_planned
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

START_BINS = (0.0, 0.04, 0.08, 0.12, np.inf)  # m, start offset from the hover / pre-contact pose
APEX_BINS = (0.0, 0.03, 0.06, np.inf)  # m, windup apex offset from the nominal apex
COVERAGE_BINS = 5


def read_metas(dirs: list[Path]) -> list[dict]:
    out = []
    for d in dirs:
        for f in sorted(Path(d).glob("episode_*.npz")):
            if f.name.endswith(".part.npz"):
                continue
            with np.load(f) as z:
                out.append(json.loads(str(z["meta"])))
    return out


def _bin(x: float, edges) -> str:
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if lo <= x < hi:
            return f"{100 * lo:.0f}-{100 * hi:.0f} cm" if np.isfinite(hi) else f">= {100 * lo:.0f} cm"
    return "?"


def summarize(metas: list[dict]) -> dict:
    tasks: dict[str, dict] = {}
    for m in metas:
        t = tasks.setdefault(m["task"], {"n": 0, "status": Counter(), "reasons": Counter(), "quality": Counter(),
                                         "attempts": [], "u_accepted": [], "u_exported": [],
                                         "by_start": defaultdict(lambda: [0, 0]),
                                         "by_apex": defaultdict(lambda: [0, 0])})
        t["n"] += 1
        q = m.get("quality", {})
        status = "rejected" if m.get("status") == "rejected" else ("success" if m.get("success") else "failed")
        if status == "success" and not q.get("passed", True):
            status = "fails quality"
            for r in q.get("reasons", []):
                t["quality"][r.split(":")[0]] += 1
        t["status"][status] += 1
        att = m.get("attempts", [])
        t["attempts"].append(len(att))
        for a in att:
            if not a["ok"]:
                t["reasons"][f"{a['reason']} ({a['where'].split('[')[0].split(' ')[0]})"] += 1
        if status == "rejected":
            continue
        u = next((a["u"] for a in reversed(att) if a.get("ok") and "u" in a), None)
        if u is not None:
            t["u_accepted"].append(u)
            if status == "success":
                t["u_exported"].append(u)
        ok = status == "success"
        if "start_offset" in m:
            b = t["by_start"][_bin(float(np.linalg.norm(m["start_offset"])), START_BINS)]
            b[0] += ok
            b[1] += 1
        plan = m.get("plan") or {}
        if "apex_offset" in plan:
            b = t["by_apex"][_bin(float(np.linalg.norm(plan["apex_offset"])), APEX_BINS)]
            b[0] += ok
            b[1] += 1
    return tasks


def report(metas: list[dict]) -> str:
    lines = []
    for task, t in summarize(metas).items():
        st = t["status"]
        lines.append(f"{task}: {t['n']} episodes: {st['success']} exported, {st['fails quality']} failed the quality "
                     f"gates, {st['failed']} failed in simulation, {st['rejected']} rejected by the pre-check")
        for reason, n in t["quality"].most_common():
            lines.append(f"  failing the gates: {n:4d}  {reason}")
        if t["attempts"]:
            lines.append(f"  candidates per episode: mean {np.mean(t['attempts']):.1f}, max {max(t['attempts'])}")
        for reason, n in t["reasons"].most_common():
            lines.append(f"  rejected candidates: {n:4d}  {reason}")
        for name, key in (("start offset", "by_start"), ("windup apex offset", "by_apex")):
            for b in sorted(t[key]):
                k, n = t[key][b]
                lines.append(f"  success by {name} {b:>10}: {k}/{n} ({100 * k / n:.0f} %)")
        lines += coverage_lines(t["u_accepted"], "after the pre-check")
        lines += coverage_lines(t["u_exported"], "exported")
    return "\n".join(lines)


def coverage_lines(us: list, label: str) -> list[str]:
    """Per sampled parameter: the episodes in each of COVERAGE_BINS equal bins of its unit range, flagged when a bin
    holds under half its share."""
    from tactile_sim.vla.variation import PARAM_NAMES

    if not us:
        return []
    u = np.asarray(us, float)
    expect = len(u) / COVERAGE_BINS
    out = [f"  coverage {label} ({len(u)} episodes; {COVERAGE_BINS} bins per parameter, expected {expect:.1f} each):"]
    for d in range(u.shape[1]):
        counts = np.bincount(np.minimum((u[:, d] * COVERAGE_BINS).astype(int), COVERAGE_BINS - 1),
                             minlength=COVERAGE_BINS)
        flag = "  UNDER-COVERED" if counts.min() < 0.5 * expect else ""
        out.append(f"    {PARAM_NAMES[d]:>13}: {' '.join(f'{c:3d}' for c in counts)}{flag}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", help="directories of recorded demos (vla.demos --out)")
    a = ap.parse_args(argv)
    print(report(read_metas([Path(d) for d in a.dirs])))


if __name__ == "__main__":
    main()
