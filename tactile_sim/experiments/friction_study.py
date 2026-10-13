"""The friction study (residual_experiment's stage Efric): the hand's friction on the tool fixed per condition, the
base at two hold levels, the zero residual and each student; outcomes and the residual's own activity.

    python -m tactile_sim.experiments.friction_study --root runs/friction/e --out runs/friction/reports [--stage Efric]

Writes report.md (the tables below) and four figures:
- outcomes.png: per friction level and hold, lost strikes, net slip, slip path, nail advance and I2t per policy
  (means over seeds with 95 % bootstrap intervals);
- activity_hold<h>.png: the residual's mean and 95th-percentile |dq| and |dtau| per joint group;
- squeeze_hold<h>.png: the closing joints' feedforward offset (close_dir x dtau) around a blow's contact;
- reseat.png: the tool's in-hand rotation per blow: the excursion during the blow and what stays after it.

Tables: absolute values per condition; each policy against the zero residual at the same friction and hold, paired by
seed (residual_experiment.paired_bootstrap); the activity per joint group; the re-seating. A seed's strike metrics are
the means over its kept strikes (a dropped strike has no slip); lost per strike counts the testbed's drops and the tool
2 cm / 30 deg from its pose at the swing's start; an episode dropped before its first strike counts one lost strike.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from tactile_sim.experiments.residual_experiment import (
    ACTIVITY_GROUPS,
    SQUEEZE_MS,
    load_results,
    paired_bootstrap,
)

STRIKE_KEYS = (("slip_trans_mm", "net slip mm"), ("slip_rot_deg", "net slip deg"), ("path_deg", "slip path deg"),
               ("advance_mm", "advance mm"), ("i2t_w", "I2t"))


def parse(cond: str) -> tuple[str, float, float]:
    """'residual:s250@hold=1@mu=0.2' -> ('s250', 1.0, 0.2); the zero residual is 'zero'."""
    name, _, mu = cond.rpartition("@mu=")
    name, _, h = name.rpartition("@hold=")
    return ("zero" if name == "base" else name.removeprefix("residual:")), float(h), float(mu)


def seed_metrics(r: dict) -> dict:
    st = r["strikes"]
    lost = [bool(x.get("lost_w", False) or x["drop"]) for x in st]
    kept = [x for x, lo in zip(st, lost, strict=True) if not lo]
    m = {"lost": float(np.mean(lost)) if lost else float(bool(r.get("dropped", False))),
         "dropped": float(bool(r.get("dropped", False))), "strikes": len(st),
         "first_lost": float(lost[0]) if lost else float(bool(r.get("dropped", False)))}
    for k, _ in STRIKE_KEYS:
        v = [x[k] for x in kept if x.get(k) is not None and np.isfinite(x[k])]
        m[k] = float(np.mean(v)) if v else float("nan")
    a = r.get("activity", {}) or {}
    for g, v in (a.get("groups") or {}).items():
        for k, x in v.items():
            m[f"{g}|{k}"] = x
    rs = a.get("reseat") or []
    if rs:
        before = np.array([b["before"][0] for b in rs])
        peak = np.array([b["peak"][0] for b in rs])
        after = np.array([b["after"][0] for b in rs])
        m["blow_excursion_deg"] = float(np.nanmean(peak - before))
        m["blow_kept_deg"] = float(np.nanmean(after - before))
        m["blow_recovered_deg"] = float(np.nanmean(peak - after))
        m["drift_end_deg"] = float(rs[-1]["end"][0]) if np.isfinite(rs[-1]["end"][0]) else float(after[-1])
    m["_squeeze"] = a.get("squeeze") or {}
    m["_blows"] = int(a.get("squeeze_blows", 0) or 0)
    return m


def collect(root: Path, stage: str) -> dict[tuple[str, float, float], dict[int, dict]]:
    out: dict[tuple[str, float, float], dict[int, dict]] = {}
    for r in load_results(stage, root, "hammer"):
        out.setdefault(parse(r["job"]["condition"]), {})[r["job"]["seed"]] = seed_metrics(r)
    return out


def mean_ci(x, n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float("nan"), float("nan")
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def policies_of(data) -> list[str]:
    pols = sorted({p for p, _, _ in data}, key=lambda p: (p != "zero", p))
    return pols


def squeeze_curve(seeds: dict[int, dict], group: str = "closing") -> np.ndarray | None:
    cs, ws = [], []
    for m in seeds.values():
        c = m["_squeeze"].get(group)
        if c is not None and m["_blows"] > 0:
            cs.append(np.asarray(c, float))
            ws.append(m["_blows"])
    if not cs:
        return None
    cs, ws = np.stack(cs), np.asarray(ws, float)
    ok = np.isfinite(cs)
    return np.where(ok, cs, 0.0).T @ ws / np.maximum(ok.T @ ws, 1e-12)


# ------------------------------------------------------------------------------------------------ tables
def tables(data) -> str:
    pols, mus, holds = policies_of(data), sorted({k[2] for k in data}), sorted({k[1] for k in data}, reverse=True)
    lines = ["## Outcomes per condition (means over seeds; strike metrics over kept strikes)", "",
             "| mu | hold | policy | seeds | strikes | lost per strike | episodes dropped | first strike lost | "
             "net slip mm / deg | slip path deg | advance mm | I2t |", "|---" * 12 + "|"]
    for mu in mus:
        for h in holds:
            for p in pols:
                s = data.get((p, h, mu))
                if not s:
                    continue
                v = list(s.values())

                def mean(k, v=v):
                    x = [m[k] for m in v if np.isfinite(m[k])]
                    return float(np.mean(x)) if x else float("nan")

                lines.append(f"| {mu:g} | {h:g} | {p} | {len(v)} | {sum(m['strikes'] for m in v)} | "
                             f"{100 * mean('lost'):.1f} % | {100 * mean('dropped'):.0f} % | "
                             f"{100 * mean('first_lost'):.0f} % | {mean('slip_trans_mm'):.3f} / "
                             f"{mean('slip_rot_deg'):.2f} | {mean('path_deg'):.2f} | {mean('advance_mm'):.2f} | "
                             f"{mean('i2t_w'):.3f} |")
    lines += ["", "## Each policy against the zero residual at the same friction and hold (paired by seed): "
              "difference [95 % CI]", "",
              "| mu | hold | policy | lost per strike | " + " | ".join(lab for _, lab in STRIKE_KEYS) + " |",
              "|---" * (4 + len(STRIKE_KEYS)) + "|"]
    for mu in mus:
        for h in holds:
            ref = data.get(("zero", h, mu), {})
            for p in pols:
                if p == "zero" or (p, h, mu) not in data:
                    continue
                s = data[(p, h, mu)]
                cells = []
                for k in ["lost"] + [k for k, _ in STRIKE_KEYS]:
                    common = [x for x in sorted(set(s) & set(ref)) if np.isfinite(s[x][k]) and np.isfinite(ref[x][k])]
                    if len(common) < 2:
                        cells.append("")
                        continue
                    d, lo, hi = paired_bootstrap([ref[x][k] for x in common], [s[x][k] for x in common])
                    fmt = "{:+.3f}" if k != "path_deg" else "{:+.2f}"
                    cells.append(f"{fmt.format(d)} [{fmt.format(lo)}, {fmt.format(hi)}]")
                lines.append(f"| {mu:g} | {h:g} | {p} | " + " | ".join(cells) + " |")
    lines += ["", "## The residual's activity per joint group (mean over seeds of the episode's mean and 95th "
              "percentile over the strike windows)", "",
              "| mu | hold | policy | " + " | ".join(f"{g}: abs(dq) deg mean / p95, abs(dtau) Nm mean / p95"
                                                       for g in ACTIVITY_GROUPS) + " |",
              "|---" * (3 + len(ACTIVITY_GROUPS)) + "|"]
    for mu in mus:
        for h in holds:
            for p in pols:
                if p == "zero" or (p, h, mu) not in data:
                    continue
                v = list(data[(p, h, mu)].values())
                cells = []
                for g in ACTIVITY_GROUPS:
                    def gm(k, g=g, v=v):
                        x = [m.get(f"{g}|{k}", float("nan")) for m in v]
                        x = [y for y in x if np.isfinite(y)]
                        return float(np.mean(x)) if x else float("nan")

                    cells.append(f"{math.degrees(gm('dq_mean')):.2f} / {math.degrees(gm('dq_p95')):.2f}, "
                                 f"{gm('dtau_mean'):.3f} / {gm('dtau_p95'):.3f}")
                lines.append(f"| {mu:g} | {h:g} | {p} | " + " | ".join(cells) + " |")
    lines += ["", "## Re-seating: the tool's in-hand rotation per blow (deg; from its pose at the episode's first "
              "swing start)", "",
              "| mu | hold | policy | excursion in the blow (peak - before) | kept after 300 ms (after - before) | "
              "recovered (peak - after) | drift at the last strike's end |", "|---" * 7 + "|"]
    for mu in mus:
        for h in holds:
            for p in pols:
                s = data.get((p, h, mu))
                if not s:
                    continue
                cells = []
                for k in ("blow_excursion_deg", "blow_kept_deg", "blow_recovered_deg", "drift_end_deg"):
                    mval, lo, hi = mean_ci([m.get(k, float("nan")) for m in s.values()])
                    cells.append(f"{mval:.2f} [{lo:.2f}, {hi:.2f}]")
                lines.append(f"| {mu:g} | {h:g} | {p} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ figures
def figures(data, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pols, mus, holds = policies_of(data), sorted({k[2] for k in data}), sorted({k[1] for k in data}, reverse=True)
    colors = {p: c for p, c in zip(pols, ["#444444", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"],
                                    strict=False)}
    written = []
    # outcomes
    keys = (("lost", "lost per strike (%)", 100.0), ("slip_rot_deg", "net slip per strike (deg)", 1.0),
            ("path_deg", "slip path per strike (deg)", 1.0), ("advance_mm", "nail advance per strike (mm)", 1.0),
            ("i2t_w", "I2t per strike", 1.0))
    fig, axs = plt.subplots(len(holds), len(keys), figsize=(4.0 * len(keys), 3.2 * len(holds)), squeeze=False)
    for i, h in enumerate(holds):
        for j, (k, lab, sc) in enumerate(keys):
            ax = axs[i][j]
            for p in pols:
                xs, ys, lo, hi = [], [], [], []
                for mu in mus:
                    s = data.get((p, h, mu))
                    if not s:
                        continue
                    mval, a, b = mean_ci([m[k] for m in s.values()])
                    xs.append(mu)
                    ys.append(sc * mval)
                    lo.append(sc * (mval - a) if np.isfinite(a) else 0.0)
                    hi.append(sc * (b - mval) if np.isfinite(b) else 0.0)
                if xs:
                    ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", ms=3, capsize=2, label=p, color=colors[p])
            ax.set_xscale("log")
            ax.set_xticks(mus)
            ax.set_xticklabels([f"{m:g}" for m in mus])
            ax.set_xlabel("hand friction mu")
            ax.set_title(f"{lab}, hold {h:g}", fontsize=9)
            ax.grid(alpha=0.3)
    axs[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "outcomes.png", dpi=120)
    plt.close(fig)
    written.append("outcomes.png")
    # activity per joint group
    act_pols = [p for p in pols if p != "zero"]
    for h in holds:
        if not act_pols:
            break
        cols = (("dq_mean", "mean |dq| (deg)", math.degrees(1)), ("dq_p95", "p95 |dq| (deg)", math.degrees(1)),
                ("dtau_mean", "mean |dtau| (Nm)", 1.0), ("dtau_p95", "p95 |dtau| (Nm)", 1.0))
        fig, axs = plt.subplots(len(act_pols), len(cols), figsize=(4.0 * len(cols), 3.0 * len(act_pols)),
                                squeeze=False)
        for i, p in enumerate(act_pols):
            for j, (k, lab, sc) in enumerate(cols):
                ax = axs[i][j]
                for g in ACTIVITY_GROUPS:
                    xs, ys = [], []
                    for mu in mus:
                        s = data.get((p, h, mu))
                        if not s:
                            continue
                        v = [m.get(f"{g}|{k}", float("nan")) for m in s.values()]
                        v = [y for y in v if np.isfinite(y)]
                        if v:
                            xs.append(mu)
                            ys.append(sc * float(np.mean(v)))
                    if xs:
                        ax.plot(xs, ys, marker="o", ms=3, label=g)
                ax.set_xscale("log")
                ax.set_xticks(mus)
                ax.set_xticklabels([f"{m:g}" for m in mus])
                ax.set_title(f"{p}: {lab}, hold {h:g}", fontsize=9)
                ax.grid(alpha=0.3)
        axs[0][0].legend(fontsize=7)
        fig.tight_layout()
        name = f"activity_hold{h:g}.png"
        fig.savefig(out / name, dpi=120)
        plt.close(fig)
        written.append(name)
    # squeeze around contact
    tt = np.arange(SQUEEZE_MS[0], SQUEEZE_MS[1] + 1)
    for h in holds:
        fig, axs = plt.subplots(1, len(mus), figsize=(3.2 * len(mus), 3.0), squeeze=False, sharey=True)
        for j, mu in enumerate(mus):
            ax = axs[0][j]
            for p in act_pols:
                s = data.get((p, h, mu))
                c = squeeze_curve(s) if s else None
                if c is not None:
                    ax.plot(tt, c, label=p, color=colors[p])
            ax.axvline(0.0, color="k", lw=0.5)
            ax.axhline(0.0, color="k", lw=0.5)
            ax.set_title(f"mu {mu:g}, hold {h:g}", fontsize=9)
            ax.set_xlabel("ms from contact")
            ax.grid(alpha=0.3)
        axs[0][0].set_ylabel("closing feedforward offset (Nm)")
        axs[0][0].legend(fontsize=8)
        fig.tight_layout()
        name = f"squeeze_hold{h:g}.png"
        fig.savefig(out / name, dpi=120)
        plt.close(fig)
        written.append(name)
    # re-seating
    keys = (("blow_excursion_deg", "rotation during the blow (deg)"),
            ("blow_kept_deg", "rotation kept 300 ms after (deg)"), ("blow_recovered_deg", "recovered (deg)"))
    fig, axs = plt.subplots(len(holds), len(keys), figsize=(4.0 * len(keys), 3.0 * len(holds)), squeeze=False)
    for i, h in enumerate(holds):
        for j, (k, lab) in enumerate(keys):
            ax = axs[i][j]
            for p in pols:
                xs, ys = [], []
                for mu in mus:
                    s = data.get((p, h, mu))
                    if s:
                        mval, _, _ = mean_ci([m.get(k, float("nan")) for m in s.values()])
                        if np.isfinite(mval):
                            xs.append(mu)
                            ys.append(mval)
                if xs:
                    ax.plot(xs, ys, marker="o", ms=3, label=p, color=colors[p])
            ax.set_xscale("log")
            ax.set_xticks(mus)
            ax.set_xticklabels([f"{m:g}" for m in mus])
            ax.set_title(f"{lab}, hold {h:g}", fontsize=9)
            ax.grid(alpha=0.3)
    axs[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "reseat.png", dpi=120)
    plt.close(fig)
    written.append("reseat.png")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="residual_experiment's --root of the sweep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="Efric")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    data = collect(Path(a.root), a.stage)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    text = "# Friction study\n\n" + tables(data)
    if not a.no_figures and data:
        text += "\n\nFigures: " + ", ".join(figures(data, out))
    (out / "report.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
