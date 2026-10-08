"""Throughput of the GPU training env: zero-residual hold of the hammer (the full hand/handle contact set) on
Tactile-Hold-VegaWuji-1000Hz, for several world counts and physics steps; or (`--task hammer`) the hammer task,
perturbed library swings with every sensor, plant and observation hook running.

Per configuration, each in its own process: physics steps per second (batched, and summed over worlds), simulated
seconds per wall second (summed over worlds), the env's GPU memory (beyond the CUDA context), MuJoCo Warp's buffer
peaks (broadphase candidates and contacts per world, constraint rows in the fullest world, for the hold and for the
startup settle) and any overflow, and the hammer's drift in the hand after 1 s simulated next to the CPU testbed's
(same reduced scene, without noslip as on the GPU, and with the validator's noslip 5).

    python -m tactile_rl.bench.throughput                              # 1024/2048/4096 worlds x 0.25/0.5 ms
    python -m tactile_rl.bench.throughput --envs 4096 --dt 0.00025 --nconmax 40 --njmax 96
    python -m tactile_rl.bench.throughput --task hammer --seconds 0.2 --rate 1000 250 --library <lib.npz>
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time

import numpy as np
import torch


def cpu_drift(timestep: float, noslip: int, seconds: float = 1.0) -> dict[str, float]:
    """The testbed's World on the reduced scene: reset, then hold (arm servos at the hover targets, the grip synergy at
    the hold scalar through the MIT drives) for `seconds`; the hammer's motion in the hand."""
    from tactile_sim.sim.training_parts import training_config
    from tactile_sim.sim.world import World

    w = World(training_config(timestep, noslip_iterations=noslip))
    w.reset()
    p0, r0 = w.hammer_in_hand()
    t0 = time.perf_counter()
    for _ in range(int(round(seconds / w.dt))):
        w.hand.tick(w.t)
        w._hold_arm()
        w.step()
    wall = time.perf_counter() - t0
    p1, r1 = w.hammer_in_hand()
    return {"drift_mm": 1e3 * float(np.linalg.norm(p1 - p0)), "drift_deg": float(np.degrees(np.linalg.norm(r1 - r0))),
            "sim_s_per_wall_s": seconds / wall}


def bench(num_envs: int, timestep: float, seconds: float = 1.0, nconmax: int | None = None, njmax: int | None = None,
          device: str = "cuda:0") -> dict:
    """One configuration; run it in a fresh process (`--one`) for a clean memory figure."""
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.mdp.events import CapacityPeaks, tactile_state
    from tactile_rl.mdp.kinematics import hammer_in_hand
    from tactile_rl.tasks.hold import NCONMAX, NJMAX, hold_env_cfg

    nconmax = NCONMAX if nconmax is None else nconmax
    njmax = NJMAX if njmax is None else njmax
    torch.zeros(1, device=device)  # the CUDA context, outside the env's share
    torch.cuda.synchronize()
    free0, total = torch.cuda.mem_get_info()
    t_build = time.perf_counter()
    cfg = hold_env_cfg(1000.0, timestep=timestep, num_envs=num_envs, episode_s=100.0, nconmax=nconmax, njmax=njmax)
    key = cfg.actions["residual"].scene_key
    env = ManagerBasedRlEnv(cfg, device=device)
    build_s = time.perf_counter() - t_build
    try:
        env.reset()
        a = torch.zeros(num_envs, env.action_manager.total_action_dim, device=env.device)
        n_steps = int(round(seconds / env.step_dt))
        for _ in range(10):  # warm up (first launches, caches)
            env.step(a)
        env.reset()
        torch.cuda.synchronize()
        ref = hammer_in_hand(env, key)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            env.step(a)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        free1, _ = torch.cuda.mem_get_info()
        d = hammer_in_hand(env, key) - ref
        drift_mm = 1e3 * torch.linalg.norm(d[:, :3], dim=1)
        drift_deg = torch.rad2deg(torch.linalg.norm(d[:, 3:], dim=1))
        # buffer peaks and overflow over a further 0.25 s (monitored apart, so the timing above carries no extra work)
        peaks = CapacityPeaks(env.device)
        for _ in range(max(1, n_steps // 4)):
            env.step(a)
            peaks.update(env)
        hold_peaks = peaks.report(num_envs)
        settle = tactile_state(env).settle_peaks
        settle_peaks = settle.report(num_envs) if settle is not None else {}
        substeps = n_steps * env.cfg.decimation
        out = {
            "num_envs": num_envs, "dt_ms": timestep * 1e3, "decimation": env.cfg.decimation, "nconmax": nconmax,
            "njmax": njmax, "build_s": build_s, "wall_s": wall,
            "physics_steps_per_s": substeps / wall, "world_steps_per_s": substeps * num_envs / wall,
            "policy_steps_per_s": n_steps * num_envs / wall, "sim_s_per_wall_s": num_envs * seconds / wall,
            "env_gpu_gb": (free0 - free1) / 1e9, "device_used_gb": (total - free1) / 1e9,
            "drift_mm_median": float(drift_mm.median()), "drift_mm_max": float(drift_mm.max()),
            "drift_deg_median": float(drift_deg.median()), "drift_deg_max": float(drift_deg.max()),
            "hold_peaks": hold_peaks, "settle_peaks": settle_peaks,
            "terminated": int(env.termination_manager.terminated.sum()),
        }
    finally:
        env.close()
        del env
        gc.collect()
    return out


def bench_hammer(num_envs: int, rate: float, library: str | None, seconds: float = 0.2,
                 device: str = "cuda:0") -> dict:
    """The hammer task (Tactile-Hammer-VegaWuji-<rate>Hz, perturbed library swings, zero residual): env steps per
    second, simulated seconds per wall second, GPU memory and MuJoCo Warp's buffer peaks."""
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.mdp.events import CapacityPeaks
    from tactile_rl.tasks.hammer import hammer_env_cfg

    torch.zeros(1, device=device)
    torch.cuda.synchronize()
    free0, total = torch.cuda.mem_get_info()
    t_build = time.perf_counter()
    cfg = hammer_env_cfg(rate, library=library, num_envs=num_envs)
    env = ManagerBasedRlEnv(cfg, device=device)
    build_s = time.perf_counter() - t_build
    try:
        env.reset()
        a = torch.zeros(num_envs, env.action_manager.total_action_dim, device=env.device)
        n_steps = max(1, int(round(seconds / env.step_dt)))
        for _ in range(20):  # warm up (first launches, the compiled load model)
            env.step(a)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            env.step(a)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        free1, _ = torch.cuda.mem_get_info()
        peaks = CapacityPeaks(env.device)
        for _ in range(max(1, n_steps // 4)):
            env.step(a)
            peaks.update(env)
        out = {"task": "hammer", "num_envs": num_envs, "rate": rate, "dt_ms": env.physics_dt * 1e3,
               "decimation": env.cfg.decimation, "build_s": build_s, "wall_s": wall,
               "ms_per_env_step": 1e3 * wall / n_steps, "sim_s_per_wall_s": num_envs * n_steps * env.step_dt / wall,
               "env_gpu_gb": (free0 - free1) / 1e9, "peaks": peaks.report(num_envs),
               "taxel_contact_overflow": int(env._tactile_rt.load.overflow)}
    finally:
        env.close()
        del env
        gc.collect()
    return out


def bench_in_subprocess(num_envs: int, timestep: float, seconds: float, nconmax: int | None,
                        njmax: int | None, task: str = "hold", rate: float = 1000.0,
                        library: str | None = None) -> dict:
    cmd = [sys.executable, "-m", "tactile_rl.bench.throughput", "--one", str(num_envs), repr(timestep),
           "--seconds", repr(seconds), "--task", task, "--rate", repr(rate)]
    if library:
        cmd += ["--library", library]
    if nconmax is not None:
        cmd += ["--nconmax", str(nconmax)]
    if njmax is not None:
        cmd += ["--njmax", str(njmax)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("{")]
    if res.returncode != 0 or not lines:
        tail = "\n".join((res.stdout + res.stderr).splitlines()[-15:])
        return {"num_envs": num_envs, "dt_ms": timestep * 1e3, "error": tail}
    return json.loads(lines[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--dt", type=float, nargs="+", default=[0.00025, 0.0005])
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--nconmax", type=int, default=None)
    ap.add_argument("--njmax", type=int, default=None)
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU reference drift")
    ap.add_argument("--json", type=str, default=None, help="also write the results here")
    ap.add_argument("--task", choices=("hold", "hammer"), default="hold")
    ap.add_argument("--rate", type=float, nargs="+", default=[1000.0], help="hammer: policy rates")
    ap.add_argument("--library", type=str, default=None, help="hammer: the base library ($TACTILE_BASE_LIBRARY)")
    ap.add_argument("--one", type=float, nargs=2, metavar=("ENVS", "DT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one is not None:  # a single configuration in this process (the parent collects the JSON line)
        if args.task == "hammer":
            r = bench_hammer(int(args.one[0]), args.rate[0], args.library, args.seconds)
        else:
            r = bench(int(args.one[0]), args.one[1], args.seconds, args.nconmax, args.njmax)
        print(json.dumps(r), flush=True)
        return
    if args.task == "hammer":  # python -m tactile_rl.bench.throughput --task hammer --seconds 0.2 --rate 1000 250
        print("| worlds | policy (Hz) | ms per env step | sim s / wall s | env GPU mem (GB) | peaks: broadphase, "
              "contacts / world; nefc |")
        print("|---|---|---|---|---|---|")
        for rate in args.rate:
            for n in args.envs:
                r = bench_in_subprocess(n, 0.00025, args.seconds, None, None, "hammer", rate, args.library)
                if "error" in r:
                    print(f"{n} worlds, {rate:g} Hz failed:\n{r['error']}")
                    continue
                p = r["peaks"]
                print(f"| {n} | {rate:g} | {r['ms_per_env_step']:.1f} | {r['sim_s_per_wall_s']:.1f} | "
                      f"{r['env_gpu_gb']:.2f} | {p['broadphase_per_world']:.1f}, {p['contacts_per_world']:.1f}; "
                      f"{p['nefc_max']} |", flush=True)
        return
    rows = []
    for dt in args.dt:
        cpu = {} if args.no_cpu else {f"noslip{n}": cpu_drift(dt, n, args.seconds) for n in (0, 5)}
        for n in args.envs:
            r = bench_in_subprocess(n, dt, args.seconds, args.nconmax, args.njmax)
            r["cpu"] = cpu
            rows.append(r)
            print(json.dumps(r), flush=True)
    rows_ok = [r for r in rows if "error" not in r]
    print()
    print("| worlds | dt (ms) | physics steps/s | world-steps/s | sim s / wall s | env GPU mem (GB) | "
          "drift 1 s median / max (mm) | CPU drift noslip 0 / 5 (mm) | peaks hold: broadphase, contacts /world; nefc "
          "| peaks settle | overflow |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows_ok:
        c = r["cpu"]
        cpu_s = f"{c['noslip0']['drift_mm']:.4f} / {c['noslip5']['drift_mm']:.4f}" if c else "-"
        hp, sp = r["hold_peaks"], r["settle_peaks"]
        print(f"| {r['num_envs']} | {r['dt_ms']:g} | {r['physics_steps_per_s']:.0f} | {r['world_steps_per_s']:.3g} | "
              f"{r['sim_s_per_wall_s']:.1f} | {r['env_gpu_gb']:.2f} | {r['drift_mm_median']:.4f} / "
              f"{r['drift_mm_max']:.4f} | {cpu_s} | {hp['broadphase_per_world']:.1f}, {hp['contacts_per_world']:.1f}; "
              f"{hp['nefc_max']} | {sp.get('broadphase_per_world', float('nan')):.1f}, "
              f"{sp.get('contacts_per_world', float('nan')):.1f}; {sp.get('nefc_max', '-')} | "
              f"{hp['overflow'] | sp.get('overflow', 0)} |")
    for r in rows:
        if "error" in r:
            print(f"{r['num_envs']} worlds, {r['dt_ms']:g} ms failed:\n{r['error']}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
