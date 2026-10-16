"""Zero-residual drill episodes on the GPU against the CPU that recorded them.

Each env plays one episode of a drill library (behaviors.drill_library) open loop, as BaseTrajectoryCommand plays a
hammer swing: the CPU's state at the episode's start, its randomization (the pistol's masses, the hand's friction
with the trigger finger's cap, contact softness, the drives, the TaxelScan, the arm gains, the drill's draw), the arm's
servo stream on its 100 Hz ticks, the grip synergy at the recorded scalar with the trigger finger at its own level,
the 12 V drill plant every physics step (tactile_rl.plants.drill). Per episode, against the CPU's recording: whether
and when the screw seated (or the bit went through), the clutch's clicks, the peak chuck torque, cam-outs, the final
depth, the tool's slip in the hand (largest and net), and the depth and spindle-speed traces.

The load noise and the Torx lobes' free turn are drawn on each side from its own generator, so episodes agree in
distribution, not draw for draw (tactile_rl.parity.drill_plant replays the CPU's draws and agrees exactly).

    python -m tactile_rl.parity.drill_episodes --lib runs/base_library/drill_screw_v1.npz [--entries 0:20]
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np


def gpu_replay(library: str, entries: list[int] | None = None, rate: float = 1000.0, device: str = "cuda:0") -> dict:
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.tasks.drill import drill_env_cfg
    from tactile_sim.behaviors.base_library import load_library

    lib = load_library(library)
    meta = json.loads(str(lib["meta"]))
    mode = meta.get("mode", "screw")
    n_all = len(lib["e_t0"])
    entries = list(range(n_all)) if entries is None else list(entries)
    n = len(entries)
    lens = lib["e_len"][entries].astype(int)
    horizon = float(lens.max()) * 1e-3 + 0.05
    cfg = drill_env_cfg(rate, library, mode, num_envs=n, episode_s=horizon + 0.1, entries=tuple(entries))
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        rt = env._tactile_rt
        laws = rt.drill.laws
        cmd = env.command_manager.get_term("base")
        zero = torch.zeros(n, env.action_manager.total_action_dim, device=env.device)
        steps = int(math.ceil(horizon / env.step_dt))
        rec = {k: [] for k in ("depth", "w", "tau", "state", "clicks", "hih", "done")}
        with torch.inference_mode():
            env.reset()
            for _ in range(steps):
                env.step(zero)
                rec["depth"].append(laws.depth.cpu().numpy())
                rec["w"].append(laws.drive.w.cpu().numpy())
                rec["tau"].append(laws.tau_out.cpu().numpy())
                rec["state"].append(laws.state.cpu().numpy())
                rec["clicks"].append(laws.drive.clicks.cpu().numpy())
                rec["hih"].append(rt.hammer_in_hand().cpu().numpy())
                rec["done"].append(laws.completed.cpu().numpy())
            hih0 = cmd.lib.hih0.cpu().numpy()
    finally:
        env.close()
    tr = {k: np.stack(v) for k, v in rec.items()}  # (T, n, ...)
    dt = 1.0 / rate
    rows = []
    for j, e in enumerate(entries):
        L = int(lens[j])
        sl = slice(0, min(L, tr["depth"].shape[0]))
        done = tr["done"][sl, j]
        t_done = float((np.argmax(done) + 1) * dt) if done.any() else float("nan")
        d = tr["hih"][sl, j] - hih0[j]
        trans = np.linalg.norm(d[:, :3], axis=1)
        rot = np.linalg.norm(d[:, 3:], axis=1)
        # the CPU's traces over the same milliseconds
        a = int(lib["e_start"][e])
        cpu_depth = lib["s_depth"][a:a + L]
        cpu_w = lib["s_spindle_speed"][a:a + L]
        k = min(len(cpu_depth), sl.stop)
        rows.append({
            "entry": e, "done_gpu": bool(done.any()), "done_cpu": bool(lib["e_done"][e]),
            "t_done_gpu": t_done, "t_done_cpu": float(lib["e_t_done"][e]),
            "clicks_gpu": int(tr["clicks"][sl.stop - 1, j]), "clicks_cpu": int(lib["e_clicks"][e]),
            "peak_torque_gpu": float(tr["tau"][sl, j].max()), "peak_torque_cpu": float(lib["e_peak_torque"][e]),
            "depth_gpu": float(tr["depth"][sl.stop - 1, j]), "depth_cpu": float(lib["e_depth"][e]),
            "max_slip_trans_gpu": float(trans.max()), "max_slip_trans_cpu": float(lib["e_max_slip_trans"][e]),
            "max_slip_rot_gpu": float(rot.max()), "max_slip_rot_cpu": float(lib["e_max_slip_rot"][e]),
            "net_slip_trans_gpu": float(trans[-1]), "net_slip_trans_cpu": float(lib["e_net_slip_trans"][e]),
            "net_slip_rot_gpu": float(rot[-1]), "net_slip_rot_cpu": float(lib["e_net_slip_rot"][e]),
            "depth_rms_mm": float(np.sqrt(np.mean((tr["depth"][:k, j] - cpu_depth[:k]) ** 2)) * 1e3),
            "w_rms": float(np.sqrt(np.mean((tr["w"][:k, j] - cpu_w[:k]) ** 2)))})
    return {"rows": rows, "mode": mode}


def summary(rows: list[dict]) -> dict[str, float]:
    def col(k):
        return np.array([r[k] for r in rows], float)

    out = {"episodes": len(rows), "done_gpu": float(col("done_gpu").mean()), "done_cpu": float(col("done_cpu").mean()),
           "done_agree": float(np.mean(col("done_gpu") == col("done_cpu")))}
    both = (col("done_gpu") > 0) & (col("done_cpu") > 0)
    if both.any():
        dt = col("t_done_gpu")[both] - col("t_done_cpu")[both]
        out.update(t_done_cpu=float(col("t_done_cpu")[both].mean()), t_done_gpu=float(col("t_done_gpu")[both].mean()),
                   t_done_diff=float(dt.mean()), t_done_absdiff=float(np.abs(dt).mean()))
    for k in ("clicks", "peak_torque", "depth", "max_slip_trans", "max_slip_rot", "net_slip_trans", "net_slip_rot"):
        g, c = col(f"{k}_gpu"), col(f"{k}_cpu")
        out[f"{k}_gpu"], out[f"{k}_cpu"] = float(np.nanmean(g)), float(np.nanmean(c))
    out["depth_rms_mm"] = float(col("depth_rms_mm").mean())
    out["w_rms"] = float(col("w_rms").mean())
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - report script
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--entries", default=None, help="a:b")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    entries = None
    if a.entries:
        lo, hi = (int(x) for x in a.entries.split(":"))
        entries = list(range(lo, hi))
    res = gpu_replay(a.lib, entries)
    s = summary(res["rows"])
    print(json.dumps(s, indent=1))
    print("| entry | done GPU / CPU | t done (s) GPU / CPU | clicks GPU / CPU | peak torque (Nm) GPU / CPU | max slip "
          "(mm / deg) GPU / CPU | depth rms (mm) |")
    print("|---" * 7 + "|")
    for r in res["rows"]:
        print(f"| {r['entry']} | {r['done_gpu']:d} / {r['done_cpu']:d} | {r['t_done_gpu']:.2f} / "
              f"{r['t_done_cpu']:.2f} | {r['clicks_gpu']} / {r['clicks_cpu']} | {r['peak_torque_gpu']:.2f} / "
              f"{r['peak_torque_cpu']:.2f} | "
              f"{1e3 * r['max_slip_trans_gpu']:.2f} / {math.degrees(r['max_slip_rot_gpu']):.2f}, "
              f"{1e3 * r['max_slip_trans_cpu']:.2f} / {math.degrees(r['max_slip_rot_cpu']):.2f} | "
              f"{r['depth_rms_mm']:.3f} |")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"summary": s, **res}, fh, indent=1)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
