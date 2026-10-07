"""Tables and plots for the slip ablation (tactile_sim.experiments.slip_ablation).

    python -m tactile_sim.experiments.slip_report --stage screen

Writes runs/slip_ablation/<stage>/report/: report.md (a table per platform and task), strikes.csv (every strike),
episodes.csv (every episode) and frontier_<platform>_<task>.png (slip against grip effort, the mean commanded
grip: the fixed-grip conditions trace the frontier; a lever helps only if it lands below it).

Hammer: per-strike metrics over strikes 2-10 (strike 1 is the setting tap). Tools: per-episode metrics. In the full
stage every condition is compared with the baseline paired by seed (same randomized world): mean difference and a
bootstrap 95% interval over seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from tactile_sim.experiments.slip_ablation import ROOT, conditions

HAMMER_COLS = [  # (column, label, format)
    ("slip_rot_deg", "tilt/strike (deg)", ".2f"),
    ("slip_trans_mm", "slide/strike (mm)", ".2f"),
    ("slip_impact_rot_deg", "tilt in 100 ms after contact (deg)", ".2f"),
    ("effort_mean", "grip effort (N)", ".1f"),
    ("grip_mean", "contact load (N)", ".1f"),
    ("advance_mm", "nail advance (mm)", ".2f"),
    ("grasp_peak_pitch", "peak grasp pitch moment (N m)", ".1f"),
    ("grasp_peak_twist", "peak grasp twist (N m)", ".2f"),
    ("aim_offset_mm", "aim offset (mm)", ".1f"),
    ("flag_latency_ms", "flag latency (ms)", ".2f"),
    ("reactive_n", "reactive detections", ".1f"),
    ("drops", "drops", ".1f"),
]
TOOL_COLS = [
    ("net_slip_rot_deg", "net tilt (deg)", ".2f"),
    ("max_slip_rot_deg", "max tilt (deg)", ".2f"),
    ("net_slip_trans_mm", "net slide (mm)", ".2f"),
    ("mean_effort", "grip effort (N)", ".1f"),
    ("mean_grip", "measured grip (N)", ".1f"),
    ("t_done", "time to goal (s)", ".2f"),
    ("reactive_n", "reactive detections", ".1f"),
    ("lost", "tool lost (fraction)", ".2f"),
]
PRIMARY = {"hammer": "slip_rot_deg", "tool": "net_slip_rot_deg"}
COLORS = {"B": "#222222", "F": "#222222", "A": "#1f77b4", "R": "#d62728", "RL": "#ff9896", "W": "#2ca02c",
          "K": "#9467bd",
          "L": "#ff7f0e", "adapt": "#8c564b", "combo": "#e377c2"}


def load(stage_dir: Path) -> list[dict]:
    out = []
    for p in sorted(stage_dir.glob("*/*/*/seed*.json")):
        if p.name.endswith(".error.json"):
            continue
        d = json.loads(p.read_text())
        if d.get("status") == "ok":
            out.append(d)
    return out


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def episode_rows(results: list[dict]) -> list[dict]:
    """One row per episode: job meta + per-episode means (hammer: strikes 2-10) or the tool summary."""
    rows = []
    for d in results:
        j = d["job"]
        row = {k: j[k] for k in ("platform", "task", "condition", "factor", "level", "seed")}
        if j["task"] == "hammer":
            st = [s for s in d["strikes"] if s["idx"] >= 1]
            for col, _, _ in HAMMER_COLS:
                vals = []
                for s in st:
                    if col == "aim_offset_mm":
                        vals.append(1e3 * _f(s.get("aim_offset")))
                    elif col == "reactive_n":
                        vals.append(_f(s.get("reactive_n", 0)))
                    else:
                        vals.append(_f(s.get(col)))
                vals = np.array(vals, dtype=float)
                row[col] = float(np.nanmean(vals)) if np.isfinite(vals).any() else math.nan
            row["n_strikes"] = len(st)
            row["hit_rate"] = float(np.mean([s["hit"] for s in st])) if st else math.nan
            row["drops"] = int(sum(s["drop"] for s in d["strikes"]))
            row["depth_mm"] = _f(d["strikes"][-1]["depth_mm"]) if d["strikes"] else math.nan
            row["hand_stop_load_max"] = max((_f(s.get("limit_hand_stop_load")) for s in d["strikes"]),
                                            default=math.nan)
        else:
            s = d["summary"]
            # a tool that left the hand: its slip and effort (the grip loop winds up) mean nothing; flag it instead
            lost = bool(s.get("lost", s.get("dropped", False) or _f(s["net_slip_trans"]) > 0.05))
            na = math.nan
            row.update(net_slip_rot_deg=na if lost else math.degrees(_f(s["net_slip_rot"])),
                       max_slip_rot_deg=na if lost else math.degrees(_f(s["max_slip_rot"])),
                       net_slip_trans_mm=na if lost else 1e3 * _f(s["net_slip_trans"]),
                       mean_grip=na if lost else _f(s["mean_grip"]),
                       mean_effort=na if lost else _f(s.get("mean_effort")),
                       t_done=_f(s["t_done"]), done=bool(s["done"]), reactive_n=_f(s.get("reactive_n", 0)),
                       lost=float(lost))
        rows.append(row)
    return rows


def strike_rows(results: list[dict]) -> list[dict]:
    rows = []
    for d in results:
        j = d["job"]
        for s in d["strikes"]:
            rows.append({**{k: j[k] for k in ("platform", "task", "condition", "factor", "seed")}, **s})
    return rows


def _boot_ci(x: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return math.nan, math.nan
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def table(rows: list[dict], platform: str, task: str, stage: str) -> list[str]:
    kind = "hammer" if task == "hammer" else "tool"
    cols = HAMMER_COLS if kind == "hammer" else TOOL_COLS
    prim = PRIMARY[kind]
    sub = [r for r in rows if r["platform"] == platform and r["task"] == task]
    order = [c.name for c in conditions(platform, task)]
    base = {r["seed"]: r for r in sub if r["condition"] == "B"}
    head = "| condition | level | " + " | ".join(lbl for _, lbl, _ in cols)
    if stage == "full":
        head += f" | {prim} vs B, paired (95% CI)"
    out = [head + " |", "|" + "---|" * (head.count("|") - 1)]
    for name in order:
        rs = [r for r in sub if r["condition"] == name]
        if not rs:
            continue
        cells = []
        for col, _, fmt in cols:
            v = np.array([r.get(col, math.nan) for r in rs], dtype=float)
            m = float(np.nanmean(v)) if np.isfinite(v).any() else math.nan
            cells.append("-" if not np.isfinite(m) else format(m, fmt))
        line = f"| {name} | {rs[0]['level']} | " + " | ".join(cells)
        if stage == "full":
            diffs = np.array([r[prim] - base[r["seed"]][prim] for r in rs if r["seed"] in base], dtype=float)
            if name == "B" or not len(diffs):
                line += " | -"
            else:
                lo, hi = _boot_ci(diffs)
                line += f" | {np.nanmean(diffs):+.2f} ({lo:+.2f}, {hi:+.2f})"
        out.append(line + " |")
    return out


def frontier_plot(rows: list[dict], platform: str, task: str, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    kind = "hammer" if task == "hammer" else "tool"
    prim = PRIMARY[kind]
    gcol = "effort_mean" if kind == "hammer" else "mean_effort"
    sub = [r for r in rows if r["platform"] == platform and r["task"] == task]
    if not sub:
        return False
    agg = {}
    for r in sub:
        agg.setdefault(r["condition"], []).append(r)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    pts = {}
    for name, rs in agg.items():
        x = float(np.nanmean([r[gcol] for r in rs]))
        y = float(np.nanmean([r[prim] for r in rs]))
        pts[name] = (x, y, rs[0]["factor"])
    fr = sorted((v[0], v[1]) for k, v in pts.items() if v[2] in ("B", "F"))
    if fr:
        ax.plot([p[0] for p in fr], [p[1] for p in fr], "-o", color="#888888", lw=1.5, ms=5,
                label="fixed grip (frontier)", zorder=1)
    for name, (x, y, fac) in pts.items():
        if fac == "F":
            continue
        ax.scatter([x], [y], color=COLORS.get(fac, "#555555"), s=40, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("grip effort: mean commanded grip over the strike (N)" if kind == "hammer"
                  else "grip effort: mean commanded grip (N)")
    ax.set_ylabel("tilt in the hand per strike (deg)" if kind == "hammer" else "net tilt in the hand (deg)")
    ax.set_title(f"{platform} / {task}: slip vs grip effort")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="screen")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    sd = args.root / args.stage
    results = load(sd)
    out = sd / "report"
    out.mkdir(parents=True, exist_ok=True)
    eps = episode_rows(results)
    _write_csv(eps, out / "episodes.csv")
    _write_csv(strike_rows(results), out / "strikes.csv")
    errors = sorted(str(p.relative_to(sd)) for p in sd.glob("*/*/*/*.error.json"))
    lines = [f"# Slip ablation: {args.stage}", "", f"{len(results)} episodes ({len(errors)} failed)."]
    if errors:
        lines += ["", "Failed: " + ", ".join(errors)]
    for pf, task in sorted({(r["platform"], r["task"]) for r in eps}):
        lines += ["", f"## {pf} / {task}", ""]
        lines += table(eps, pf, task, args.stage)
        png = out / f"frontier_{pf}_{task}.png"
        if frontier_plot(eps, pf, task, png):
            lines += ["", f"![frontier]({png.name})"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'report.md'} ({len(results)} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
