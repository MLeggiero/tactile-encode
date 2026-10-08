"""Calibrating the load generator's band to the hand: one event per episode, of one kind, at an amplitude from a grid,
with the base at a hold level (fraction of the settled hold), zero residual, the full randomization. Per kind, every
(amplitude, hold level) cell runs in its own envs at once; per cell: the grasps lost (before the event: the hold alone
cannot keep the tool; or by the event's window's end), the net slip per event (the tool's pose in the hand at the
window's end against at its start; kept grasps) and the slip path over the window.

Kinds: impulse (half-sine, 0.5-5 ms, within 20 degrees of the strike axis), periodic force along the strike axis or
torque about the handle (1-100 Hz, drifting amplitude, LoadGenCfg's durations), step (a breakthrough or an appearing
push), bind (a torque spike about the handle), push (a steady force along the strike axis); durations, frequencies
and directions as LoadGenCfg draws them, only the amplitude fixed per cell.

    python -m tactile_rl.bench.loadgen_calibration [--envs 4096] [--kinds impulse,bind] [--out calib.json]
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

# kind: (TorchLoadGen kind index, torque?, the amplitude grid: N or Nm)
GRID = {
    "impulse": (0, False, (25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 450.0)),
    "periodic_force": (1, False, (2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0)),
    "periodic_torque": (1, True, (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)),
    "step": (2, False, (5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0)),
    "bind": (3, True, (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)),
    "push": (4, False, (5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0, 100.0)),
}
HOLDS = (0.3, 0.5, 0.8, 1.0)
T0 = 0.4  # s: the event's start (the reset has settled, the drift reference is set)
SETTLE = 0.05


def _durations(cfg, kind: int, n: int, dev) -> torch.Tensor:
    lo, hi = {0: (cfg.impulse_ms[0] * 1e-3, cfg.impulse_ms[1] * 1e-3), 1: cfg.periodic_s, 2: cfg.step_hold_s,
              3: (cfg.bind_ms[0] * 1e-3, cfg.bind_ms[1] * 1e-3), 4: cfg.push_s}[kind]
    return lo + (hi - lo) * torch.rand(n, dtype=torch.float64, device=dev)


def set_single_events(cmd, kind: int, torque: bool, amp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One event per env (slot 0) of `kind` at amplitudes `amp` (N,); returns each env's window (start, end)."""
    g, cfg = cmd.gen, cmd.gen.cfg
    n, dev = g.n_env, g.dev
    f64 = dict(dtype=torch.float64, device=dev)
    g.valid[:] = False
    g.valid[:, 0] = True
    g.kind[:, 0] = kind
    dur = _durations(cfg, kind, n, dev)
    g.dur[:, 0] = dur
    g.amp[:, 0] = amp.double()
    g.amp_scale[:] = 1.0
    d = torch.zeros(n, 3, **f64)
    d[:, 2] = 1.0
    if kind == 0:
        d = g._cone(n, math.radians(cfg.cone_deg))
    g.dir[:, 0] = d
    g.torque[:, 0] = torque
    g.freq[:, 0] = torch.exp(math.log(cfg.periodic_hz[0]) + torch.rand(n, **f64) *
                             (math.log(cfg.periodic_hz[1]) - math.log(cfg.periodic_hz[0]))) if kind == 1 else 0.0
    g.phase[:, 0] = 2 * math.pi * torch.rand(n, **f64) if kind == 1 else 0.0
    walk = torch.cumsum(torch.randn(n, g.D, **f64) * cfg.drift * math.sqrt(0.01), dim=1)
    g.drift[:, 0] = torch.clamp(1.0 + walk, 0.2, 2.0) if kind == 1 else 1.0
    g.n_drift[:, 0] = (torch.ceil(dur / 0.01).long() + 1) if kind == 1 else 1
    sign = torch.where(torch.rand(n, **f64) < 0.7, 1.0, -1.0).double() if kind == 2 else torch.ones(n, **f64)
    g.sign[:, 0] = sign
    breakthrough = (kind == 2) & (sign > 0)
    g.t0[:, 0] = torch.where(breakthrough, T0 + dur, torch.full_like(dur, T0))
    start = torch.full_like(dur, T0)
    end = T0 + dur + SETTLE  # a breakthrough's push holds from T0 and vanishes at T0 + dur; the others run from T0
    return start, end


def run_kind(env, name: str, holds=HOLDS) -> list[dict]:
    from tactile_rl.mdp.actions import hand_drive

    kind, torque, amps = GRID[name]
    N, dev = env.num_envs, env.device
    cells = [(a, h) for a in amps for h in holds]
    per = N // len(cells)
    if per < 8:
        raise ValueError(f"{N} envs for {len(cells)} cells")
    cell = torch.arange(N, device=dev) // per
    use = cell < len(cells)
    cell = cell.clamp(max=len(cells) - 1)
    amp_of = torch.as_tensor([a for a, _ in cells], dtype=torch.float64, device=dev)[cell]
    hold_of = torch.as_tensor([h for _, h in cells], dtype=torch.float32, device=dev)[cell]
    torch.manual_seed(0)
    env.reset()
    cmd = env.command_manager.get_term("base")
    rt = env._tactile_rt
    drive = hand_drive(env.scene["robot"])
    start, end = set_single_events(cmd, kind, torque, amp_of)
    cmd.level[:] = hold_of
    cmd.s_now[:] = hold_of * cmd.s_hold
    zero = torch.zeros(N, env.action_manager.total_action_dim, device=dev)
    t_end = float(end.max()) + 0.01
    steps = int(math.ceil(t_end / env.step_dt))
    alive = torch.ones(N, dtype=torch.bool, device=dev)  # no termination yet in this episode
    lost_pre = torch.zeros(N, dtype=torch.bool, device=dev)
    lost = torch.zeros(N, dtype=torch.bool, device=dev)
    got0 = torch.zeros(N, dtype=torch.bool, device=dev)
    got1 = torch.zeros(N, dtype=torch.bool, device=dev)
    p0 = torch.zeros(N, 6, device=dev)
    p1 = torch.zeros(N, 6, device=dev)
    path = torch.zeros(N, 2, device=dev)
    i_peak = torch.zeros(N, device=dev)
    for _ in range(steps):
        t_prev = rt.t.clone()
        env.step(zero)
        t = rt.t  # (an env that terminated was reset inside the step: its clock restarted)
        term = env.termination_manager.get_term("tool_dropped")
        pre = t_prev + env.step_dt < start + 1e-9
        lost_pre |= alive & term & pre
        lost |= alive & term
        alive &= ~(env.termination_manager.terminated | env.termination_manager.time_outs)
        hih = rt.hammer_in_hand()
        new0 = alive & ~got0 & (t >= start)
        p0 = torch.where(new0[:, None], hih, p0)
        got0 |= new0
        win = alive & got0 & ~got1
        path[:, 0] += torch.where(win, rt.acc["slip_mm"], torch.zeros_like(path[:, 0]))
        path[:, 1] += torch.where(win, rt.acc["slip_deg"], torch.zeros_like(path[:, 1]))
        i_peak = torch.where(win, torch.maximum(i_peak, (drive.i.abs() / drive.i_rated).max(1).values), i_peak)
        new1 = alive & got0 & ~got1 & (t >= end)
        p1 = torch.where(new1[:, None], hih, p1)
        got1 |= new1
    d = p1 - p0
    net_mm = torch.linalg.norm(d[:, :3], dim=1) * 1e3
    net_deg = torch.rad2deg(torch.linalg.norm(d[:, 3:], dim=1))
    out = []
    for c, (a, h) in enumerate(cells):
        m = use & (cell == c)
        kept = m & got1 & ~lost
        n = int(m.sum())
        out.append({"kind": name, "amp": a, "hold": h, "n": n, "lost_before": float(lost_pre[m].float().mean()),
                    "lost": float(lost[m].float().mean()),
                    "net_mm": float(net_mm[kept].mean()) if bool(kept.any()) else float("nan"),
                    "net_deg": float(net_deg[kept].mean()) if bool(kept.any()) else float("nan"),
                    "net_mm_p90": float(net_mm[kept].quantile(0.9)) if bool(kept.any()) else float("nan"),
                    "path_mm": float(path[kept, 0].mean()) if bool(kept.any()) else float("nan"),
                    "path_deg": float(path[kept, 1].mean()) if bool(kept.any()) else float("nan"),
                    "i_peak": float(i_peak[kept].mean()) if bool(kept.any()) else float("nan")})
    return out


def band_check(env, holds=HOLDS, episode_s: float = 3.0, seed: int = 0) -> list[dict]:
    """The whole band as the load generator draws it (every kind, Poisson events from 0.2 s), one episode per env,
    the envs split over the hold levels, zero residual: per hold level the episodes that lose the grasp, and over the
    events of the kept ones the net slip, the fraction within 0.7 mm / 1.4 degrees and the slip path."""
    from tactile_rl.mdp.actions import hand_drive

    N, dev = env.num_envs, env.device
    per = N // len(holds)
    grp = (torch.arange(N, device=dev) // per).clamp(max=len(holds) - 1)
    hold_of = torch.as_tensor(holds, dtype=torch.float32, device=dev)[grp]
    cmd = env.command_manager.get_term("base")
    cmd.enabled, cmd.amp_scale = True, 1.0
    torch.manual_seed(seed)
    env.reset()
    rt = env._tactile_rt
    drive = hand_drive(env.scene["robot"])
    cmd.level[:] = hold_of
    cmd.s_now[:] = hold_of * cmd.s_hold
    zero = torch.zeros(N, env.action_manager.total_action_dim, device=dev)
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    lost = torch.zeros(N, dtype=torch.bool, device=dev)
    i2t = torch.zeros(N, device=dev)
    i_peak = torch.zeros(N, device=dev)
    for _ in range(int(round((episode_s - 0.01) / env.step_dt))):
        env.step(zero)
        lost |= alive & env.termination_manager.get_term("tool_dropped")
        alive &= ~(env.termination_manager.terminated | env.termination_manager.time_outs)
        i2t += torch.where(alive, rt.acc["i2"], torch.zeros_like(i2t))
        i_peak = torch.where(alive, torch.maximum(i_peak, (drive.i.abs() / drive.i_rated).max(1).values), i_peak)
    e = cmd.ep  # events, within target, net mm, net deg, path mm, path deg (the kept episodes' are intact)
    out = []
    for k, h in enumerate(holds):
        m = grp == k
        kept = m & alive
        ev = float(e[kept, 0].sum())
        n = max(ev, 1.0)
        out.append({"hold": h, "episodes": int(m.sum()), "lost": float(lost[m].float().mean()),
                    "events_per_episode": ev / max(int(kept.sum()), 1), "within_target": float(e[kept, 1].sum()) / n,
                    "net_mm": float(e[kept, 2].sum()) / n, "net_deg": float(e[kept, 3].sum()) / n,
                    "path_mm": float(e[kept, 4].sum()) / n, "path_deg": float(e[kept, 5].sum()) / n,
                    "i2t": float(i2t[kept].mean()), "i_peak": float(i_peak[kept].mean())})
    return out


def table(rows: list[dict], name: str) -> str:
    holds = sorted({r["hold"] for r in rows if r["kind"] == name})
    amps = sorted({r["amp"] for r in rows if r["kind"] == name})
    unit = "Nm" if GRID[name][1] else "N"
    lines = [f"| {name} ({unit}) | " + " | ".join(f"hold {h:g}: lost / net mm / net deg" for h in holds) + " |",
             "|---" * (len(holds) + 1) + "|"]
    for a in amps:
        cells = []
        for h in holds:
            r = next(x for x in rows if x["kind"] == name and x["amp"] == a and x["hold"] == h)
            cells.append(f"{100 * r['lost']:.0f} % / {r['net_mm']:.2f} / {r['net_deg']:.2f}")
        lines.append(f"| {a:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - calibration script
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--kinds", default=",".join(GRID))
    ap.add_argument("--out", default=None)
    ap.add_argument("--band", action="store_true", help="check LoadGenCfg's band instead of sweeping amplitudes")
    a = ap.parse_args(argv)
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.mdp.randomization import tactile_dr
    from tactile_rl.tasks.common import TEACHER_GROUPS
    from tactile_rl.tasks.loadgen import loadgen_env_cfg, loadgen_scene_key

    env = ManagerBasedRlEnv(loadgen_env_cfg(1000.0, num_envs=a.envs, curriculum=False, obs_groups=TEACHER_GROUPS,
                                            episode_s=T0 + 2.6), device="cuda:0")
    rows = []
    try:
        dr = tactile_dr(env, loadgen_scene_key())
        dr.scale = 1.0
        if a.band:
            rows = band_check(env)
            print("| hold level | episodes | lost grasps | events per kept episode | within 0.7 mm / 1.4 deg | "
                  "net slip per event (mm / deg) | slip path per event (mm / deg) | I2t per episode | "
                  "peak current / rated |")
            print("|---|---|---|---|---|---|---|---|---|")
            for r in rows:
                print(f"| {r['hold']:g} | {r['episodes']} | {100 * r['lost']:.1f} % | {r['events_per_episode']:.2f} | "
                      f"{100 * r['within_target']:.1f} % | {r['net_mm']:.2f} / {r['net_deg']:.2f} | "
                      f"{r['path_mm']:.2f} / {r['path_deg']:.2f} | {r['i2t']:.2f} | {r['i_peak']:.2f} |", flush=True)
        for name in ([] if a.band else a.kinds.split(",")):
            t0 = time.time()
            rows += run_kind(env, name)
            print(table(rows, name), flush=True)
            print(f"({time.time() - t0:.0f} s)\n", flush=True)
    finally:
        env.close()
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["GRID", "HOLDS", "run_kind", "set_single_events", "table"]
assert np.isfinite(T0)
