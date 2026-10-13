"""Calibrating the load generator's band to the hand: one event per episode, of one kind, at an amplitude from a grid,
with the base at a hold level (fraction of the settled hold), zero residual, the full randomization. Per kind, every
(amplitude, hold level) cell runs in its own envs at once; per cell: the grasps lost (before the event: the hold alone
cannot keep the tool; or by the event's window's end), the net slip per event (the tool's pose in the hand at the
window's end against at its start; kept grasps) and the slip path over the window.

Kinds: impulse (half-sine, 0.5-5 ms, within cone_deg of the strike axis), periodic force along the strike axis or
torque about the handle (1-100 Hz, drifting amplitude, LoadGenCfg's durations), step (a breakthrough or an appearing
push), bind (a torque spike about the handle, or about any tool axis), push (a steady force along the strike axis),
torque impulse (a half-sine torque about any tool axis); durations, frequencies and directions as the configuration
draws them (`--wide`: the training's wide loads, loadgen_command.wide_loads), only the amplitude fixed per cell.

`--friction MU` fixes the hand/tool friction (else the full randomization's), `--holds` the hold levels. `--choose
FILE` picks the training band from the table (choose_band: per kind, from an amplitude no event of which loses the
grasp to the one that loses it `--target-lost` of the time, at the first hold level) and writes it with the table;
loadgen_command.wide_loads reads it ($TACTILE_LOADGEN_BAND).

    python -m tactile_rl.bench.loadgen_calibration [--envs 4096] [--kinds impulse,bind] [--out calib.json]
    python -m tactile_rl.bench.loadgen_calibration --wide --friction 0.6 --holds 0.5 --choose band.json
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
    "impulse": (0, False, (10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 450.0, 700.0)),
    "periodic_force": (1, False, (1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 90.0)),
    "periodic_torque": (1, True, (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)),
    "step": (2, False, (2.5, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0)),
    "bind": (3, True, (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)),
    "push": (4, False, (2.5, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0, 100.0, 150.0)),
    # (a blow's own moment about the grasp: 300 N at the face, 0.17 m out, is 51 Nm)
    "torque_impulse": (5, True, (0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0, 40.0)),
}
# each kind's band key (LoadGenCfg) and the low end's share of the high end (LoadGenCfg's own ranges' shape)
BAND_OF = {"impulse": ("impulse_n", 14.0 / 200.0), "periodic_force": ("periodic_n", 1.25 / 10.0),
           "periodic_torque": ("periodic_nm", 0.05 / 0.4), "step": ("step_n", 2.7 / 12.0),
           "bind": ("bind_nm", 0.1 / 0.6), "push": ("push_n", 1.5 / 12.0),
           "torque_impulse": ("torque_impulse_nm", 0.2 / 2.0)}
HOLDS = (0.3, 0.5, 0.8, 1.0)
TARGET_LOST = 0.4  # the band's strongest events lose the grasp this often (at the calibration's hold and friction)
T0 = 0.4  # s: the event's start (the reset has settled, the drift reference is set)
SETTLE = 0.05


def _durations(cfg, kind: int, n: int, dev) -> torch.Tensor:
    lo, hi = {0: (cfg.impulse_ms[0] * 1e-3, cfg.impulse_ms[1] * 1e-3), 1: cfg.periodic_s, 2: cfg.step_hold_s,
              3: (cfg.bind_ms[0] * 1e-3, cfg.bind_ms[1] * 1e-3), 4: cfg.push_s,
              5: (cfg.torque_impulse_ms[0] * 1e-3, cfg.torque_impulse_ms[1] * 1e-3)}[kind]
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
    ex = torch.zeros(n, 3, **f64)
    ex[:, 0] = 1.0
    any_axis = kind == 5 or (kind == 3 and cfg.bind_axes == "xyz")
    g.tax[:, 0] = g._axis(n) if any_axis else ex
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


def run_kind(env, name: str, holds=HOLDS, friction: float | None = None) -> list[dict]:
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
    if friction is not None:
        _check_friction(env, friction)
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
        lb, lt = float(lost_pre[m].float().mean()), float(lost[m].float().mean())
        out.append({"kind": name, "amp": a, "hold": h, "n": n, "lost_before": lb, "lost": lt,
                    "lost_event": (lt - lb) / (1.0 - lb) if lb < 1.0 else float("nan"),
                    "net_mm": float(net_mm[kept].mean()) if bool(kept.any()) else float("nan"),
                    "net_deg": float(net_deg[kept].mean()) if bool(kept.any()) else float("nan"),
                    "net_mm_p90": float(net_mm[kept].quantile(0.9)) if bool(kept.any()) else float("nan"),
                    "path_mm": float(path[kept, 0].mean()) if bool(kept.any()) else float("nan"),
                    "path_deg": float(path[kept, 1].mean()) if bool(kept.any()) else float("nan"),
                    "i_peak": float(i_peak[kept].mean()) if bool(kept.any()) else float("nan")})
    return out


def _check_friction(env, mu: float) -> None:
    """The reset's draw put `mu` on every env's hand/tool pairs (a degenerate range at the full DR scale)."""
    from tactile_rl.mdp.randomization import tactile_dr

    dr = getattr(env, "_tactile_dr", None) or tactile_dr(env)
    got = dr.v["hand_friction"]
    if not bool(torch.allclose(got, torch.full_like(got, mu), atol=1e-9)):
        raise RuntimeError(f"hand friction {float(got.min()):.3f}-{float(got.max()):.3f}, expected {mu}")


def _lost_at(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([r["amp"] for r in rows], float)
    lost = np.array([r["lost_event"] for r in rows], float)
    o = np.argsort(a)
    return a[o], np.nan_to_num(lost[o], nan=1.0)


def choose_band(rows: list[dict], hold: float, target: float = TARGET_LOST) -> dict:
    """The training band from a calibration table at `hold`: per kind, the high end the amplitude whose events lose
    the grasp `target` of the time (of the grasps that held until the event; log-linear between the grid's cells),
    the low end the kind's reference share of it (BAND_OF), lowered to the largest amplitude below which no event
    lost the grasp. Returns {"band": {key: [lo, hi]}, "notes": {kind: what limited it}}."""
    band, notes = {}, {}
    for name, (key, share) in BAND_OF.items():
        rs = [r for r in rows if r["kind"] == name and abs(r["hold"] - hold) < 1e-9]
        if not rs:
            continue
        a, lost = _lost_at(rs)
        above = np.nonzero(lost >= target)[0]
        if len(above) == 0:
            hi, note = float(a[-1]), f"never {target:.0%} lost (at most {lost.max():.0%}): the grid's top"
        elif above[0] == 0:
            hi, note = float(a[0]), f"{lost[0]:.0%} lost at the grid's bottom"
        else:
            i = int(above[0])
            x0, x1, y0, y1 = math.log(a[i - 1]), math.log(a[i]), lost[i - 1], lost[i]
            hi = float(math.exp(x0 + (target - y0) / max(y1 - y0, 1e-9) * (x1 - x0)))
            note = f"{target:.0%} lost between {a[i - 1]:g} ({y0:.0%}) and {a[i]:g} ({y1:.0%})"
        lo = share * hi
        zero = 0
        while zero < len(a) and lost[zero] == 0.0:
            zero += 1
        if zero == 0:
            lo = min(lo, 0.5 * float(a[0]))
            note += f"; every amplitude lost some ({lost[0]:.0%} at {a[0]:g}): low end half the grid's bottom"
        elif lo > a[zero - 1]:
            lo = float(a[zero - 1])
            note += f"; low end {lo:g}, the largest without a loss"
        band[key] = [float(lo), float(hi)]
        notes[name] = note
    return {"band": band, "notes": notes}


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
    ap.add_argument("--band", action="store_true", help="check the configuration's band instead of sweeping amplitudes")
    ap.add_argument("--wide", action="store_true", help="the training's wide loads (cone, axes; band of "
                    "$TACTILE_LOADGEN_BAND for --band)")
    ap.add_argument("--friction", type=float, default=None, help="fix the hand/tool friction")
    ap.add_argument("--holds", default=",".join(f"{h:g}" for h in HOLDS))
    ap.add_argument("--rate", type=float, default=1000.0)
    ap.add_argument("--choose", default=None, help="write the chosen band (and the table) to this JSON file")
    ap.add_argument("--target-lost", type=float, default=TARGET_LOST)
    a = ap.parse_args(argv)
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.mdp.loadgen_command import wide_loads
    from tactile_rl.mdp.randomization import tactile_dr
    from tactile_rl.tasks.common import TEACHER_GROUPS
    from tactile_rl.tasks.loadgen import loadgen_env_cfg, loadgen_scene_key
    from tactile_sim.config import LoadGenCfg

    holds = tuple(float(h) for h in a.holds.split(","))
    loads = (wide_loads() if a.band else wide_loads({})) if a.wide else LoadGenCfg()
    mu = None if a.friction is None else (a.friction, a.friction)
    env = ManagerBasedRlEnv(loadgen_env_cfg(a.rate, num_envs=a.envs, curriculum=False, obs_groups=TEACHER_GROUPS,
                                            episode_s=T0 + 2.6, loadgen=loads, hand_friction=mu), device="cuda:0")
    rows = []
    try:
        dr = tactile_dr(env, loadgen_scene_key(a.rate, env.physics_dt))
        dr.scale = 1.0
        if a.band:
            rows = band_check(env, holds)
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
            rows += run_kind(env, name, holds, a.friction)
            print(table(rows, name), flush=True)
            print(f"({time.time() - t0:.0f} s)\n", flush=True)
    finally:
        env.close()
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)
    if a.choose and not a.band:
        out = choose_band(rows, holds[0], a.target_lost)
        out.update({"hold": holds[0], "friction": a.friction, "target_lost": a.target_lost, "rate": a.rate,
                    "envs": a.envs, "wide": a.wide, "rows": rows})
        with open(a.choose, "w") as fh:
            json.dump(out, fh, indent=1)
        print("band:", json.dumps(out["band"]), flush=True)
        for k, v in out["notes"].items():
            print(f"  {k}: {v}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["BAND_OF", "GRID", "HOLDS", "choose_band", "run_kind", "set_single_events", "table"]
assert np.isfinite(T0)
