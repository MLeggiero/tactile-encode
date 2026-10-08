"""Registered tasks (mjlab registry). Importing this module registers Tactile-Hold-VegaWuji-{1000,500,250,100}Hz; the
scene export runs when an env is built, not here."""

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from tactile_rl.tasks.hold import RATES, hold_env_cfg, task_id
from tactile_rl.tasks.rl_cfg import ppo_runner_cfg

for _rate in RATES:
    if task_id(_rate) not in list_tasks():
        register_mjlab_task(task_id=task_id(_rate), env_cfg=hold_env_cfg(_rate),
                            play_env_cfg=hold_env_cfg(_rate, num_envs=16, episode_s=10.0),
                            rl_cfg=ppo_runner_cfg(_rate))

__all__ = ["RATES", "hold_env_cfg", "ppo_runner_cfg", "task_id"]
