"""Strikes on the GPU against the CPU, on the same base-library entries (and the nail plant on matched strikes).

Every entry plays once, unperturbed and with its episode's randomization, in its own GPU world of the hammer task
(BaseTrajectoryCommand, "sequential" assignment), from the CPU's state at its windup: the arm's 100 Hz stream, the
grip synergy, the nail plant's hooks and the WUJI drives, with a zero residual. Per entry, as the testbed's
StrikeRecord measures them (tactile_sim.behaviors.base_library.strike_outcome): hit, nail advance, peak face/nail
force and pulse width (pulse_stats), contact onset time and face speed at onset, slip in the hand (swing start to the
entry's end).

Compared with the CPU's closed-loop run that recorded the entry (the library's outcome) and with the CPU replaying it
open loop the same way (base_library.replay_entry), which separates the replay's own error from the simulators'.

    python -m tactile_rl.parity.strikes --lib runs/base_library/hammer_v1.npz [--entries 0:200] [--cpu-workers 14] \
        [--validator] [--perturbed 4]
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


def gpu_replay(library: str, entries: list[int], device: str = "cuda:0", rate: float = 1000.0,
               perturb: bool = False, copies: int = 1, timestep: float = 0.00025, hold_level: float = 1.0) -> dict:
    """Replay `entries` on the GPU at physics `timestep`, `copies` worlds each (perturbed bases draw their own time
    warp, amplitude and start delay), the hand base at `hold_level` of the library's grip, zero residual; returns the
    outcome table (with each world's perturbation, and `lost`: the tool 2 cm or 30 degrees from its start in the hand
    at some point) and the capacity peaks."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.mdp.events import CapacityPeaks
    from tactile_rl.tasks.hammer import hammer_env_cfg

    lib = load_library(library)
    n_e = len(entries)
    n = n_e * copies
    lens_e = lib["e_len"][entries].astype(int)
    horizon = float(lens_e.max()) * 1e-3 / (0.8 if perturb else 1.0) + (0.1 if perturb else 0.0)
    cfg = hammer_env_cfg(rate, library=library, timestep=timestep, num_envs=n, perturb=perturb, apply_params=True,
                         entries=tuple(entries), assign="sequential", episode_s=horizon + 0.05, nconmax=64,
                         njmax=192, hold_level=(hold_level, hold_level))
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("time_out",)}
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        env.reset()
        cmd = env.command_manager.get_term("base")
        rt = cmd.rt
        dev = env.device
        warp, amp, delay = (x.cpu().numpy().astype(float) for x in (cmd.warp, cmd.amp, cmd.delay))
        idx = cmd.entry.cpu().numpy()  # each world's entry, positions in `entries`
        # each world's entry in env time: the swing starts at d + t_swing / w, the entry ends at d + len / w
        lens = np.ceil((lens_e[idx] * 1e-3 / warp + delay) * 1e3).astype(int)
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
    for j in range(n):
        e = entries[idx[j]]
        n_t = lens[j] * spm
        ts = float(lib["e_t_swing"][e]) if np.isfinite(lib["e_t_swing"][e]) else 0.0
        ts = delay[j] + ts / warp[j]
        i_sw = min(int(round(ts * 1e3)), lens[j])
        on = np.flatnonzero(f[j, :n_t] > 0)
        t_c = float(t[on[0]]) if len(on) else float("nan")
        v_c = float(v_ax[j, on[0]]) if len(on) else float("nan")
        out = strike_outcome(t[:n_t], f[j, :n_t], depth[j, :n_t], (hih[j, i_sw], hih[j, lens[j]]), ts, v_c, t_c)
        d = hih[j, :lens[j] + 1] - hih[j, :1]
        lost = bool(((np.linalg.norm(d[:, :3], axis=1) > 0.02) | (np.linalg.norm(d[:, 3:], axis=1) > 0.5236)).any())
        rows.append({"entry": e, "warp": warp[j], "amp": amp[j], "delay": delay[j], "lost": lost, **out})
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
    ap.add_argument("--validator", action="store_true", help="also replay on the CPU at 8 kHz with noslip 5")
    ap.add_argument("--perturbed", type=int, default=0, help="also play every entry this many times, perturbed")
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
    if a.validator and a.cpu_workers > 0:  # the same entries replayed at the 8 kHz validator physics (noslip 5)
        val = table(replay_library(a.lib, entries, a.cpu_workers, {"timestep": 0.000125, "noslip_iterations": 5}),
                    OUTCOMES)
        tabs["cpu_validator"] = val
        rep["gpu_vs_cpu_validator"] = compare_outcomes(val, tabs["gpu"])
        rep["cpu_validator_vs_cpu_replay"] = compare_outcomes(tabs["cpu_replay"], val)
        hit = np.asarray(val["hit"], bool)
        rep["cpu_validator_means"] = {k: float(np.nanmean(np.asarray(val[k], float)[hit])) for k in
                                      ("advance", "peak", "peak2", "width", "v_face", "slip_trans", "slip_rot")}
    if a.perturbed > 0:  # perturbed bases: does the swing still hit the nail?
        pg = gpu_replay(a.lib, entries, perturb=True, copies=a.perturbed)
        pt = table(pg["rows"], OUTCOMES + ("warp", "amp", "delay"))
        hit = np.asarray(pt["hit"], bool)
        adv = np.asarray(pt["advance"], float)
        w, amp = np.asarray(pt["warp"]), np.asarray(pt["amp"])
        rep["gpu_perturbed"] = {
            "worlds": int(len(hit)), "hit_rate": float(hit.mean()), "advance_mean": float(adv[hit].mean()),
            "advance_by_speed": {f"{lo:.2f}-{hi:.2f}": float(adv[hit & (w * amp >= lo) & (w * amp < hi)].mean())
                                 for lo, hi in ((0.7, 0.9), (0.9, 1.1), (1.1, 1.35))},
            "peak2_mean": float(np.asarray(pt["peak2"], float)[hit].mean()),
            "unperturbed_advance_mean": float(np.asarray(tabs["gpu"]["advance"], float)[np.asarray(tabs["gpu"]["hit"],
                                                                                                   bool)].mean())}
        tabs["gpu_perturbed"] = pt
    print(json.dumps(rep, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"entries": entries, "report": rep,
                       "tables": {k: {c: np.asarray(v).tolist() for c, v in t.items()} for k, t in tabs.items() if t}},
                      fh)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
