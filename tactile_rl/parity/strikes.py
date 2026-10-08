"""Strikes on the GPU against the CPU, on the same base-library entries (and the nail plant on matched strikes).

Every entry plays once, unperturbed and with its episode's randomization, in its own GPU world of the hammer task
(BaseTrajectoryCommand, "sequential" assignment), from the CPU's state at its windup: the arm's 100 Hz stream, the
grip synergy, the nail plant's hooks and the WUJI drives, with a zero residual. Per entry, as the testbed's
StrikeRecord measures them (tactile_sim.behaviors.base_library.strike_outcome): hit, nail advance, peak face/nail
force and pulse width (pulse_stats), contact onset time and face speed at onset, slip in the hand (swing start to the
entry's end).

Compared with the CPU's closed-loop run that recorded the entry (the library's outcome) and with the CPU replaying it
open loop the same way (base_library.replay_entry), which separates the replay's own error from the simulators'.

    python -m tactile_rl.parity.strikes --lib runs/base_library/hammer_v1.npz [--entries 0:200] [--cpu-workers 14]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from tactile_sim.behaviors.base_library import (
    OUTCOMES,
    compare_outcomes,
    library_outcomes,
    load_library,
    replay_library,
    strike_outcome,
    table,
)


def gpu_replay(library: str, entries: list[int], device: str = "cuda:0", rate: float = 1000.0) -> dict:
    """Replay `entries` on the GPU (one world each); returns the outcome table and the capacity peaks."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.mdp.events import CapacityPeaks
    from tactile_rl.tasks.hammer import hammer_env_cfg

    lib = load_library(library)
    lens = lib["e_len"][entries].astype(int)
    n = len(entries)
    cfg = hammer_env_cfg(rate, library=library, num_envs=n, perturb=False, apply_params=True, entries=tuple(entries),
                         assign="sequential", episode_s=float(lens.max()) * 1e-3 + 0.05, nconmax=64, njmax=192)
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("time_out",)}
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        env.reset()
        cmd = env.command_manager.get_term("base")
        rt = cmd.rt
        dev = env.device
        steps = int(lens.max())
        spm = rt.steps_per_ms
        T = steps * spm
        f = torch.zeros(n, T, device=dev)
        depth = torch.zeros(n, T, device=dev)
        v_ax = torch.zeros(n, T, device=dev)
        hih = torch.zeros(n, steps + 1, 6, device=dev)
        peaks = CapacityPeaks(dev)
        k = {"i": 0}

        def rec(r):
            i = k["i"]
            if i < T:
                f[:, i] = r.nail.force
                depth[:, i] = r.nail.depth()
                v_ax[:, i] = r.nail.v_axis.float()
            k["i"] = i + 1
            peaks.update(env)

        rt.recorders.append(rec)
        hih[:, 0] = rt.hammer_in_hand()
        zero = torch.zeros(n, 60, device=dev)
        t0 = time.time()
        for s in range(steps):
            env.step(zero)
            hih[:, s + 1] = rt.hammer_in_hand()
        torch.cuda.synchronize()
        wall = time.time() - t0
        f, depth, v_ax, hih = (x.cpu().numpy().astype(float) for x in (f, depth, v_ax, hih))
    finally:
        env.close()
    dt = 1e-3 / spm
    t = (np.arange(T) + 1) * dt
    rows = []
    for j, e in enumerate(entries):
        n_t = lens[j] * spm
        ts = float(lib["e_t_swing"][e]) if np.isfinite(lib["e_t_swing"][e]) else 0.0
        i_sw = min(int(round(ts * 1e3)), lens[j])
        on = np.flatnonzero(f[j, :n_t] > 0)
        t_c = float(t[on[0]]) if len(on) else float("nan")
        v_c = float(v_ax[j, on[0]]) if len(on) else float("nan")
        out = strike_outcome(t[:n_t], f[j, :n_t], depth[j, :n_t], (hih[j, i_sw], hih[j, lens[j]]), ts, v_c, t_c)
        rows.append({"entry": e, **out})
    return {"rows": rows, "peaks": peaks.report(n), "wall_s": wall, "sim_s": steps * 1e-3}


def report(cpu_lib: dict, cpu_replay: dict | None, gpu: dict) -> dict:
    out = {"gpu_vs_cpu_recorded": compare_outcomes(cpu_lib, gpu)}
    if cpu_replay is not None:
        out["cpu_replay_vs_cpu_recorded"] = compare_outcomes(cpu_lib, cpu_replay)
        out["gpu_vs_cpu_replay"] = compare_outcomes(cpu_replay, gpu)
    for name, tab in (("cpu_recorded", cpu_lib), ("cpu_replay", cpu_replay), ("gpu", gpu)):
        if tab is None:
            continue
        hit = np.asarray(tab["hit"], bool)
        out[f"{name}_means"] = {k: float(np.nanmean(np.asarray(tab[k], float)[hit])) for k in
                                ("advance", "peak", "peak2", "width", "v_face", "slip_trans", "slip_rot") if k in tab}
    both = np.asarray(cpu_lib["hit"], bool) & np.asarray(gpu["hit"], bool)
    dtc = (np.asarray(gpu["t_contact"]) - np.asarray(cpu_lib["t_contact"]))[both]
    dvf = (np.asarray(gpu["v_face"]) - np.asarray(cpu_lib["v_face"]))[both]
    out["contact_time_diff_ms"] = {"median": float(np.median(dtc) * 1e3), "p90_abs": float(np.percentile(np.abs(dtc),
                                                                                                         90) * 1e3)}
    out["v_face_diff"] = {"median": float(np.median(dvf)), "p90_abs": float(np.percentile(np.abs(dvf), 90))}
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - report script
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", default="runs/base_library/hammer_v1.npz")
    ap.add_argument("--entries", default=None, help="a:b range")
    ap.add_argument("--cpu-workers", type=int, default=14, help="0 skips the CPU replay")
    ap.add_argument("--out", default=None, help="write the per-entry tables and the report here (JSON)")
    a = ap.parse_args(argv)
    lib = load_library(a.lib)
    n_all = len(lib["e_t0"])
    if a.entries:
        lo, hi = (int(x) for x in a.entries.split(":"))
        entries = list(range(lo, min(hi, n_all)))
    else:
        entries = list(range(n_all))
    gpu = gpu_replay(a.lib, entries)
    cpu_rows = replay_library(a.lib, entries, a.cpu_workers) if a.cpu_workers > 0 else None
    tabs = {"cpu_recorded": library_outcomes(lib, entries), "gpu": table(gpu["rows"], OUTCOMES),
            "cpu_replay": table(cpu_rows, OUTCOMES) if cpu_rows else None}
    rep = report(tabs["cpu_recorded"], tabs["cpu_replay"], tabs["gpu"])
    rep["gpu_capacity_peaks"] = gpu["peaks"]
    rep["gpu_wall_s"] = gpu["wall_s"]
    print(json.dumps(rep, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"entries": entries, "report": rep,
                       "tables": {k: {c: np.asarray(v).tolist() for c, v in t.items()} for k, t in tabs.items() if t}},
                      fh)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
