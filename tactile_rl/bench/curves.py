"""Learning curves of Tactile-Residual teachers against simulated seconds, from their tensorboard logs.

Simulated seconds are summed over the envs: an iteration collects `envs` x `rollout_s` of them (4096 x 0.128 s =
524 s at every rate). Per iteration: the episode return, the curriculum's stage, and per kind (the load generator,
the hammer; a kind's means are separated from the logged means with the share of hammer episodes among the episodes
that ended): net slip (per event; per strike that kept the grasp), slip path per episode, lost grasps, I^2 t per
episode, the load generator's events within the target. A CSV per run and one PNG overlaying the runs.

    python -m tactile_rl.bench.curves OUT_PREFIX RUN_DIR[:LABEL] [RUN_DIR[:LABEL] ...] [--envs 4096] [--rollout-s 0.128]
"""

from __future__ import annotations

import argparse
import csv
import glob

import numpy as np

TAGS = {"return": "Train/mean_reward", "stage": "Curriculum/stage/stage", "capped": "Curriculum/stage/capped",
        "cur_ok": "Curriculum/stage/events_ok_frac", "cur_drop": "Curriculum/stage/drop_rate",
        "hammer": "Metrics/base/hammer", "lg_net_mm": "Metrics/base/slip_per_event_mm",
        "lg_net_deg": "Metrics/base/slip_per_event_deg", "lg_path_ev_mm": "Metrics/base/path_per_event_mm",
        "lg_ok": "Metrics/base/events_ok_frac", "level": "Metrics/base/hold_level",
        "advance": "Metrics/base/hammer_advance_mm", "dropped": "Episode_Metrics/dropped",
        "path_mm": "Episode_Metrics/slip_mm", "path_deg": "Episode_Metrics/slip_deg", "i2t": "Episode_Metrics/i2t",
        "h_dropped": "Episode_Metrics/hammer_dropped", "h_path_mm": "Episode_Metrics/hammer_slip_mm",
        "h_path_deg": "Episode_Metrics/hammer_slip_deg", "h_i2t": "Episode_Metrics/hammer_i2t",
        "h_strike": "Episode_Metrics/hammer_strike", "h_net_mm": "Episode_Metrics/hammer_net_mm",
        "h_net_deg": "Episode_Metrics/hammer_net_deg", "std": "Policy/mean_std", "lr": "Loss/learning_rate"}


def load_run(run: str, envs: int = 4096, rollout_s: float = 0.128) -> dict[str, np.ndarray]:
    """Per iteration: the logged scalars and the per-kind series derived from them."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(glob.glob(f"{run}/events.out.tfevents.*"))
    data: dict[str, dict[int, float]] = {}
    for f in files:
        ev = EventAccumulator(f, size_guidance={"scalars": 0})
        ev.Reload()
        have = set(ev.Tags()["scalars"])
        for k, tag in TAGS.items():
            if tag in have:
                data.setdefault(k, {}).update({e.step: e.value for e in ev.Scalars(tag)})
    its = np.array(sorted(data.get("return", {})), int)
    out = {"iteration": its, "sim_s": (its + 1) * envs * rollout_s}
    for k in TAGS:
        out[k] = np.array([data.get(k, {}).get(int(i), np.nan) for i in its], float)
    fh = np.nan_to_num(out["hammer"], nan=0.0)
    fl = 1.0 - fh
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(fl > 0.02, 1.0 / fl, np.nan)
        hm = np.where(fh > 0.02, 1.0 / fh, np.nan)
        hs = np.where(out["h_strike"] > 1e-3, 1.0 / out["h_strike"], np.nan)
        h0 = np.nan_to_num(out["h_dropped"], nan=0.0)
        out.update({
            "lg_net_mm_k": out["lg_net_mm"] * lg, "lg_net_deg_k": out["lg_net_deg"] * lg, "lg_ok_k": out["lg_ok"] * lg,
            "lg_path_ev_mm_k": out["lg_path_ev_mm"] * lg,
            "lg_lost": (out["dropped"] - h0) * lg,
            "lg_path_mm": (out["path_mm"] - np.nan_to_num(out["h_path_mm"], nan=0.0)) * lg,
            "lg_i2t": (out["i2t"] - np.nan_to_num(out["h_i2t"], nan=0.0)) * lg,
            "h_lost": out["h_dropped"] * hm, "h_path_mm_k": out["h_path_mm"] * hm, "h_i2t_k": out["h_i2t"] * hm,
            "h_net_mm_k": out["h_net_mm"] * hs, "h_net_deg_k": out["h_net_deg"] * hs,
            "h_advance_k": out["advance"] * hm})
    return out


def windowed(runs: dict[str, dict], keys: list[str], window_s: float = 50_000.0) -> str:
    """A markdown table of window means against simulated seconds."""
    lines = []
    for label, r in runs.items():
        lines.append(f"\n{label}\n")
        lines.append("| simulated s | stage | " + " | ".join(keys) + " |")
        lines.append("|---" * (len(keys) + 2) + "|")
        edges = np.arange(0.0, float(r["sim_s"].max()) + window_s, window_s)
        for a, b in zip(edges[:-1], edges[1:], strict=False):
            m = (r["sim_s"] > a) & (r["sim_s"] <= b)
            if not m.any():
                continue
            cells = []
            for k in keys:
                v = r[k][m]
                v = v[np.isfinite(v)]
                cells.append(f"{v.mean():.4g}" if len(v) else "-")
            st = r["stage"][m]
            st = st[np.isfinite(st)]
            stage = f"C{int(np.floor(st.max() + 1e-6))}" if len(st) else "-"
            lines.append(f"| {a / 1e3:.0f}k-{b / 1e3:.0f}k | {stage} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


PANELS = [("return", "Mean episode return", None), ("stage", "Curriculum stage", None),
          ("lg_net_mm_k", "Load generator: net slip per event (mm)", None),
          ("h_net_mm_k", "Hammer: net slip per strike (mm)", None),
          ("lg_path_mm", "Load generator: slip path per episode (mm)", None),
          ("h_path_mm_k", "Hammer: slip path per episode (mm)", None),
          ("lg_lost", "Load generator: lost grasps", "pct"), ("h_lost", "Hammer: lost grasps", "pct"),
          ("lg_i2t", "Load generator: I²t per episode", None), ("h_i2t_k", "Hammer: I²t per episode", None)]


def plot(runs: dict[str, dict], out_png: str, title: str = "Tactile-Residual teachers") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = ["#2a78d6", "#d9622b", "#2f9e5b", "#8a4fd1"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.titlecolor": ink, "figure.facecolor": surface,
                         "axes.facecolor": surface, "savefig.facecolor": surface})
    fig, axes = plt.subplots(5, 2, figsize=(11, 13), sharex=True)
    for ax, (k, name, fmt) in zip(axes.flat, PANELS, strict=True):
        for c, (label, r) in zip(colors, runs.items(), strict=False):
            x = r["sim_s"] / 1e3
            y = r[k]
            if k == "stage":
                ax.plot(x, np.floor(y + 1e-6), color=c, linewidth=2, drawstyle="steps-post", label=label)
                continue
            good = np.isfinite(y)
            if good.sum() > 9:
                idx = np.flatnonzero(good)
                ys = np.full_like(y, np.nan)
                ys[idx] = np.convolve(np.pad(y[idx], 4, mode="edge"), np.ones(9) / 9, mode="valid")
                ax.plot(x, y, color=c, alpha=0.2, linewidth=1)
                ax.plot(x, ys, color=c, linewidth=2, label=label)
            elif good.any():
                ax.plot(x, y, color=c, linewidth=2, label=label)
        ax.set_title(name, fontsize=10, loc="left")
        ax.grid(True, color=grid, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        if k == "stage":
            ax.set_yticks([0, 1, 2, 3], ["C0", "C1", "C2", "C3"])
        if fmt == "pct":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes.flat[1].legend(frameon=False, loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("simulated seconds (thousands, all envs)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=ink)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - report script
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--rollout-s", type=float, default=0.128)
    ap.add_argument("--window-s", type=float, default=50_000.0)
    ap.add_argument("--title", default="Tactile-Residual teachers (4096 envs, GRU 256)")
    a = ap.parse_args(argv)
    runs = {}
    for spec in a.runs:
        path, _, label = spec.partition(":")
        runs[label or path.rstrip("/").split("/")[-1]] = load_run(path, a.envs, a.rollout_s)
    for label, r in runs.items():
        with open(f"{a.out}_{label}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            keys = list(r)
            w.writerow(keys)
            for i in range(len(r["iteration"])):
                w.writerow([r[k][i] for k in keys])
    plot(runs, f"{a.out}.png", a.title)
    print(windowed(runs, ["return", "lg_net_mm_k", "lg_net_deg_k", "lg_ok_k", "lg_path_ev_mm_k", "lg_lost", "lg_i2t",
                          "h_net_mm_k", "h_net_deg_k", "h_path_mm_k", "h_lost", "h_i2t_k", "h_advance_k", "std"],
                   a.window_s))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
