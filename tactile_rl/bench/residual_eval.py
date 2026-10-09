"""Teachers against the zero residual at matched hold levels, on the hammer and on the load generator (milestone 4).

Each policy plays in Tactile-Residual at its own rate under C3's settings (the full randomization, the noisy and
sometimes masked forecast, the load generator's full band), the base fixed at one hold level per run, one episode per
env and no resets: every hammer env plays one library swing (entries in order, perturbed) to its end, every
load-generator env 3 s of loads. Every run starts from the same seed, so the policies meet the same entries, draws and
schedules. A trained policy plays its mean action; "grip" is the strongest constant residual toward closing (the
authority bound: every joint's offset, stiffness and feedforward at their limits).

Per run: lost grasps (the tool 2 cm or 30 degrees from its reference, the training's drop), net slip (hammer: the
tool's pose in the hand at the entry's end against its start, per strike; load generator: per event, and the share
within 0.7 mm / 1.4 degrees), slip path (the 30 Hz low-passed pose's path at 1 kHz, per strike or per event), I^2 t per
episode (the hand's mean (i / i_rated)^2 integrated), the RMS current, the peak current after the first 0.1 s (the
largest |i| / i_rated over the joints, every substep; the reset state's opening transient excluded) and the share of
joint-time at the current limit (|i| >= 0.99 rated). Hammer runs add hits (the face met the nail, any strike) and the
nail's advance per strike.
Slip and current are means over the kept episodes.

    python -m tactile_rl.bench.residual_eval --library LIB --policies zero@1000,zero@250,t1k=CKPT@1000,t250=CKPT@250 \
        [--holds 0.3,0.5,0.8,1.0] [--tasks hammer,loadgen] [--envs 1024] [--out eval.json]
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

HOLDS = (0.3, 0.5, 0.8, 1.0)
T_PEAK = 0.1  # s: the peak current counts from here
LOADGEN_S = 3.0  # s: a load-generator episode (training's)


def parse_policies(spec: str) -> list[dict]:
    """"zero@1000,grip@1000,t1k=path.pt@1000" -> [{"name", "checkpoint" (None: zero or grip), "rate"}]."""
    out = []
    for item in spec.split(","):
        body, rate = item.rsplit("@", 1)
        name, _, ckpt = body.partition("=")
        out.append({"name": f"{name}@{float(rate):g}", "checkpoint": ckpt or None, "rate": float(rate)})
    return out


class ConstantGrip:
    """The strongest constant residual toward closing (the authority bound): every joint's position offset, stiffness
    and feedforward at their limits in the synergy's closing direction (pre-tanh actions of arctanh(0.999))."""

    def __init__(self, env):
        act = env.action_manager.get_term("residual")
        cd = torch.as_tensor(np.asarray(act.parts.synergy.close_dir, float), dtype=torch.float32, device=env.device)
        big = float(np.arctanh(0.999))
        self.a = torch.cat([big * cd, torch.full_like(cd, big), big * cd])[None].repeat(env.num_envs, 1)

    def reset(self) -> None:
        pass

    def __call__(self, obs) -> torch.Tensor:
        del obs
        return self.a


def _horizon(library: str) -> float:
    """The longest entry as perturbed (the slowest time warp, the longest start delay), s."""
    from tactile_rl.mdp.commands import BaseTrajectoryCommandCfg
    from tactile_sim.behaviors.base_library import load_library

    c = BaseTrajectoryCommandCfg(scene_key="", library=library)
    return float(np.max(load_library(library)["e_len"])) * 1e-3 / c.time_warp[0] + c.start_delay[1]


def make_env(rate: float, library: str, num_envs: int, episode_s: float, device: str = "cuda:0"):
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.tasks.common import TEACHER_GROUPS
    from tactile_rl.tasks.residual import residual_env_cfg

    cfg = residual_env_cfg(rate, library=library, num_envs=num_envs, start_stage=3, max_stage=3,
                           obs_groups=TEACHER_GROUPS, episode_s=episode_s)
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}  # one episode, no resets
    cfg.curriculum = {}  # C3's settings are set here
    cfg.metrics = {}  # (the training's read the terminations)
    env = ManagerBasedRlEnv(cfg, device=device)
    env.command_manager.get_term("base").ham.cfg.assign = "sequential"
    return env


def run(env, policy, task: str, hold: float, seed: int = 0) -> dict:
    """One episode per env of `task` ("hammer" or "loadgen") with the base at `hold`; `policy`: None (zero) or an
    actor (rsl_rl RNNModel)."""
    from tensordict import TensorDict

    from tactile_rl.mdp.curriculum import STAGES
    from tactile_rl.mdp.randomization import tactile_dr
    from tactile_rl.mdp.terminations import tool_left_hand
    from tactile_rl.tasks.loadgen import loadgen_scene_key

    s = STAGES[3]
    N, dev = env.num_envs, env.device
    cmd = env.command_manager.get_term("base")
    rate = 1.0 / env.step_dt
    dr = tactile_dr(env, loadgen_scene_key(rate, env.physics_dt))
    dr.scale, dr.forecast_mode, dr.forecast_masking = s["dr_scale"], s["forecast"], s["mask"]
    cmd.enabled, cmd.amp_scale = s["loads"], s["amp_scale"]
    cmd.hold_level = (hold, hold)
    cmd.hammer_frac = 1.0 if task == "hammer" else 0.0
    rt = env._tactile_rt
    drive = rt.drive
    ham = task == "hammer"
    # per substep: the peak current and the time at the limit while an episode is under way
    active = torch.zeros(N, dtype=torch.bool, device=dev)
    peak = torch.zeros(N, device=dev)
    at_lim = torch.zeros(N, device=dev)
    n_sub = torch.zeros(N, device=dev)

    def rec(r):
        rel = (drive.i.abs() / drive.i_rated)
        on = active & (r.t >= T_PEAK)
        peak.copy_(torch.where(on, torch.maximum(peak, rel.max(1).values), peak))
        at_lim.add_(torch.where(on, (rel >= 0.99).float().mean(1), torch.zeros_like(peak)))
        n_sub.add_(on.float())

    out = {}
    with torch.inference_mode():
        torch.manual_seed(seed)
        obs, _ = env.reset()
        cmd.take_totals()
        if policy is not None:
            policy.reset()
        assert bool((cmd.is_hammer == ham).all())
        rt.recorders.append(rec)
        try:
            alive = torch.ones(N, dtype=torch.bool, device=dev)  # not lost
            going = torch.ones(N, dtype=torch.bool, device=dev)  # the episode under way
            lost = torch.zeros(N, dtype=torch.bool, device=dev)
            i2t = torch.zeros(N, device=dev)
            t_ep = torch.zeros(N, device=dev)
            path = torch.zeros(N, 2, device=dev)
            net = torch.zeros(N, 2, device=dev)
            adv = torch.zeros(N, device=dev)
            blows = torch.zeros(N, dtype=torch.long, device=dev)
            depth0 = rt.nail.depth().double().clone() if rt.nail is not None else None
            t_end = LOADGEN_S - 0.01
            zero = torch.zeros(N, env.action_manager.total_action_dim, device=dev)
            k = 0
            max_steps = int(math.ceil(env.max_episode_length_s / env.step_dt)) - 2
            while k < max_steps and bool(going.any()):
                active.copy_(going)
                a = zero if policy is None else policy(TensorDict({"teacher": obs["teacher"]}, batch_size=[N]))
                obs, *_ = env.step(a)
                k += 1
                i2t += torch.where(going, rt.acc["i2"], torch.zeros_like(i2t))
                blows = torch.where(going, rt.n_blows, blows)
                t_ep += going.float() * env.step_dt
                path += torch.where(going[:, None], torch.stack([rt.acc["slip_mm"], rt.acc["slip_deg"]], 1),
                                    torch.zeros_like(path))
                gone = going & tool_left_hand(env)
                lost |= gone
                alive &= ~gone
                going &= ~gone
                if ham:
                    fin = going & cmd.ham.done()
                    d = rt.hammer_in_hand() - cmd.ham.ref_rel
                    nn_ = torch.stack([torch.linalg.norm(d[:, :3], dim=1) * 1e3,
                                       torch.rad2deg(torch.linalg.norm(d[:, 3:], dim=1))], 1)
                    net = torch.where(fin[:, None], nn_, net)
                    if depth0 is not None:
                        adv = torch.where(fin, ((rt.nail.depth().double() - depth0) * 1e3).float(), adv)
                    going &= ~fin
                elif k * env.step_dt >= t_end:
                    going[:] = False
            active.zero_()
        finally:
            rt.recorders.remove(rec)
        kept = alive
        n_kept = max(int(kept.sum()), 1)

        def km(x):
            return float(x[kept].mean()) if bool(kept.any()) else float("nan")

        r = {"task": task, "hold": hold, "episodes": N, "kept": int(kept.sum()), "lost": float(lost.float().mean()),
             "i2t": km(i2t), "i_peak": km(peak), "at_limit": km(at_lim / n_sub.clamp(min=1.0)),
             "i_rms": km(torch.sqrt(i2t / t_ep.clamp(min=1e-6))), "duration_s": km(t_ep), "steps": k}
        if ham:
            unfinished = int((going & alive).sum())
            r.update({"strikes": N, "unfinished": unfinished, "net_mm": km(net[:, 0]), "net_deg": km(net[:, 1]),
                      "path_mm": km(path[:, 0]), "path_deg": km(path[:, 1]), "advance_mm": km(adv),
                      "hit": float((blows > 0).float().mean())})
        else:
            e = cmd.lg.ep  # events, within target, net mm, net deg, path mm, path deg (the kept episodes' intact)
            ev = float(e[kept, 0].sum())
            n = max(ev, 1.0)
            r.update({"events": ev, "events_per_episode": ev / n_kept, "within_target": float(e[kept, 1].sum()) / n,
                      "net_mm": float(e[kept, 2].sum()) / n, "net_deg": float(e[kept, 3].sum()) / n,
                      "path_mm": float(e[kept, 4].sum()) / n, "path_deg": float(e[kept, 5].sum()) / n})
        out = r
    return out


def evaluate(library: str, policies: list[dict], holds=HOLDS, tasks=("hammer", "loadgen"), num_envs: int = 1024,
             seed: int = 0, device: str = "cuda:0", log=print) -> list[dict]:
    from tactile_rl.export.onnx_export import load_actor

    rows = []
    episode_s = max(LOADGEN_S, _horizon(library)) + 0.05
    for rate in sorted({p["rate"] for p in policies}, reverse=True):
        env = make_env(rate, library, num_envs, episode_s, device)
        try:
            for p in [p for p in policies if p["rate"] == rate]:
                if p["checkpoint"] is not None:
                    actor = load_actor(p["checkpoint"], device=device).to(device)
                else:
                    actor = ConstantGrip(env) if p["name"].startswith("grip@") else None
                for task in tasks:
                    for h in holds:
                        t0 = time.time()
                        r = run(env, actor, task, float(h), seed)
                        r.update({"policy": p["name"], "rate": rate, "wall_s": time.time() - t0})
                        rows.append(r)
                        log(json.dumps(r), flush=True)
        finally:
            env.close()
    return rows


def tables(rows: list[dict]) -> str:
    out = []
    for task in ("hammer", "loadgen"):
        rs = [r for r in rows if r["task"] == task]
        if not rs:
            continue
        unit = "strike" if task == "hammer" else "event"
        extra = " | hits | advance (mm)" if task == "hammer" else " | within 0.7 mm / 1.4 deg"
        out.append(f"\n{task}\n\n| policy | hold | lost grasps | net slip per {unit} (mm / deg) | slip path per {unit} "
                   f"(mm / deg) | I2t per episode | RMS current / rated | peak current / rated | time at the limit"
                   f"{extra} |")
        out.append("|---" * (11 if task == "hammer" else 10) + "|")
        for r in sorted(rs, key=lambda r: (r["hold"], r["policy"])):
            tail = (f" | {100 * r['hit']:.1f} % | {r['advance_mm']:.2f}" if task == "hammer"
                    else f" | {100 * r['within_target']:.1f} %")
            out.append(f"| {r['policy']} | {r['hold']:g} | {100 * r['lost']:.1f} % | {r['net_mm']:.3f} / "
                       f"{r['net_deg']:.3f} | {r['path_mm']:.2f} / {r['path_deg']:.2f} | {r['i2t']:.3f} | "
                       f"{r['i_rms']:.3f} | {r['i_peak']:.2f} | {100 * r['at_limit']:.2f} %{tail} |")
    # the efficiency comparison: each policy at hold h against the zero residual at the full hold (same rate)
    out.append("\nagainst the zero residual at the full hold (its rate): net slip and current ratios\n")
    out.append("| task | policy | hold | net slip mm (ratio) | net slip deg (ratio) | lost grasps | I2t (ratio) | "
               "RMS current (ratio) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["task"], r["policy"], r["hold"])):
        ref = [z for z in rows if z["task"] == r["task"] and z["policy"].startswith("zero@")
               and z["rate"] == r["rate"] and z["hold"] == 1.0]
        if not ref or r["policy"].startswith("zero@") and r["hold"] == 1.0:
            continue
        z = ref[0]
        out.append(f"| {r['task']} | {r['policy']} | {r['hold']:g} | "
                   f"{r['net_mm']:.3f} ({r['net_mm'] / z['net_mm']:.2f}) "
                   f"| {r['net_deg']:.3f} ({r['net_deg'] / z['net_deg']:.2f}) | {100 * r['lost']:.1f} % | "
                   f"{r['i2t']:.3f} ({r['i2t'] / z['i2t']:.2f}) | {r['i_rms']:.3f} ({r['i_rms'] / z['i_rms']:.2f}) |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - report script
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True)
    ap.add_argument("--policies", default="zero@1000")
    ap.add_argument("--holds", default=",".join(f"{h:g}" for h in HOLDS))
    ap.add_argument("--tasks", default="hammer,loadgen")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = evaluate(a.library, parse_policies(a.policies), [float(h) for h in a.holds.split(",")],
                    a.tasks.split(","), a.envs, a.seed)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)
    print(tables(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
