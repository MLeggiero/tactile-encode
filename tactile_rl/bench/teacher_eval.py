"""A load-generator teacher against the zero residual on the same episodes.

The trained policy plays its mean action (no exploration noise), the zero residual plays the base alone; every mode
starts from the same seed, so the bank states, the randomization, the schedules and the noise draws match. The
curriculum stage's settings are fixed (loads, amplitude, the base's hold level, the randomization's scale, the
forecast); a mode suffixed "@hold" runs the base at the full settled hold instead. Per mode: events, the fraction
within the slip target (net slip), net slip per event, lost grasps, I^2 t and slip path per episode.

    python -m tactile_rl.bench.teacher_eval CHECKPOINT [--envs 1024] [--stage 1] \
        [--modes zero,teacher,zero@hold,teacher@hold] [--desired-kl 0.02]
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import asdict

import torch


def evaluate(checkpoint: str, num_envs: int = 1024, stage: int = 1, modes=("zero", "teacher"),
             episode_s: float = 3.0, seed: int = 7, desired_kl: float = 0.02, device: str = "cuda:0") -> dict:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from tactile_rl.mdp.curriculum import STAGES
    from tactile_rl.mdp.randomization import tactile_dr
    from tactile_rl.tasks.common import TEACHER_GROUPS
    from tactile_rl.tasks.loadgen import loadgen_env_cfg, loadgen_scene_key
    from tactile_rl.tasks.rl_cfg import teacher_runner_cfg

    env = ManagerBasedRlEnv(loadgen_env_cfg(1000.0, num_envs=num_envs, curriculum=False, obs_groups=TEACHER_GROUPS,
                                            episode_s=episode_s), device=device)
    out = {}
    try:
        s = STAGES[stage]
        dr = tactile_dr(env, loadgen_scene_key())
        dr.scale, dr.forecast_mode, dr.forecast_masking = s["dr_scale"], s["forecast"], s["mask"]
        cmd = env.command_manager.get_term("base")
        cmd.enabled, cmd.amp_scale = s["loads"], s["amp_scale"]
        venv = RslRlVecEnvWrapper(env)
        runner = MjlabOnPolicyRunner(venv, asdict(teacher_runner_cfg(1000.0, "loadgen", desired_kl=desired_kl)),
                                     tempfile.mkdtemp(), device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device)
        rt = env._tactile_rt
        steps = int(round((episode_s - 0.01) / env.step_dt))
        zero = torch.zeros(num_envs, env.action_manager.total_action_dim, device=env.device)
        for mode in modes:
            with torch.inference_mode():  # the policy's hidden state lives in inference mode
                cmd.hold_level = (1.0, 1.0) if mode.endswith("@hold") else tuple(s["hold"])
                torch.manual_seed(seed)
                venv.reset()
                obs = venv.get_observations()
                policy.reset()
                cmd.take_totals()
                lost = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
                i2 = torch.zeros(num_envs, device=env.device)
                path = torch.zeros(num_envs, 2, device=env.device)
                for _ in range(steps):
                    obs, _, dones, _ = venv.step(policy(obs) if mode.startswith("teacher") else zero)
                    policy.reset(dones)
                    lost |= env.termination_manager.get_term("tool_dropped")
                    i2 += rt.acc["i2"]
                    path[:, 0] += rt.acc["slip_mm"]
                    path[:, 1] += rt.acc["slip_deg"]
                ev, ok, mm, deg = (float(x) for x in cmd.take_totals().cpu())
                n = max(ev, 1.0)
                out[mode] = {"events": ev, "within_target": ok / n, "net_mm": mm / n, "net_deg": deg / n,
                             "lost": float(lost.float().mean()), "i2t": float(i2.mean()),
                             "path_mm": float(path[:, 0].mean()), "path_deg": float(path[:, 1].mean())}
    finally:
        env.close()
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - report script
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--modes", default="zero,teacher")
    ap.add_argument("--desired-kl", type=float, default=0.02)
    a = ap.parse_args(argv)
    res = evaluate(a.checkpoint, a.envs, a.stage, a.modes.split(","), desired_kl=a.desired_kl)
    print("| mode | events | within target | net slip per event (mm / deg) | lost grasps | I2t per episode | "
          "slip path per episode (mm / deg) |")
    print("|---|---|---|---|---|---|---|")
    for mode, r in res.items():
        print(f"| C{a.stage} {mode} | {r['events']:.0f} | {100 * r['within_target']:.1f} % | {r['net_mm']:.3f} / "
              f"{r['net_deg']:.3f} | {100 * r['lost']:.2f} % | {r['i2t']:.3f} | {r['path_mm']:.2f} / "
              f"{r['path_deg']:.2f} |")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
