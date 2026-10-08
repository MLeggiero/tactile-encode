"""Registered tasks (mjlab registry). Importing this module registers Tactile-Hold-VegaWuji-{1000,500,250,100}Hz and
Tactile-Hammer-VegaWuji-{1000,500,250,100}Hz; the scene export (and, for the hammer tasks, loading the base library)
runs when an env is built, not here."""

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from tactile_rl.tasks.hammer import hammer_env_cfg, hammer_task_id
from tactile_rl.tasks.hold import RATES, hold_env_cfg, task_id
from tactile_rl.tasks.rl_cfg import ppo_runner_cfg

for _rate in RATES:
    if task_id(_rate) not in list_tasks():
        register_mjlab_task(task_id=task_id(_rate), env_cfg=hold_env_cfg(_rate),
                            play_env_cfg=hold_env_cfg(_rate, num_envs=16, episode_s=10.0),
                            rl_cfg=ppo_runner_cfg(_rate))
    if hammer_task_id(_rate) not in list_tasks():
        register_mjlab_task(task_id=hammer_task_id(_rate), env_cfg=hammer_env_cfg(_rate),
                            play_env_cfg=hammer_env_cfg(_rate, num_envs=16),
                            rl_cfg=ppo_runner_cfg(_rate, experiment="tactile_hammer",
                                                  obs_groups={"actor": ("teacher",), "critic": ("critic",)}))

__all__ = ["RATES", "hammer_env_cfg", "hammer_task_id", "hold_env_cfg", "ppo_runner_cfg", "task_id"]
