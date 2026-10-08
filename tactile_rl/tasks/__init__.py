"""Registered tasks (mjlab registry). Importing this module registers, at 1000, 500, 250 and 100 Hz:

- Tactile-Hold-VegaWuji-{rate}Hz (milestone 1's hold; MLP PPO);
- Tactile-Hammer-VegaWuji-{rate}Hz (the base library's swings; the GRU teacher);
- Tactile-LoadGen-VegaWuji-{rate}Hz (the hold under synthetic loads with the curriculum; the GRU teacher).

The scene export (and, for the hammer tasks, loading the base library) runs when an env is built, not here."""

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from tactile_rl.tasks.common import TEACHER_GROUPS
from tactile_rl.tasks.hammer import hammer_env_cfg, hammer_task_id
from tactile_rl.tasks.hold import RATES, hold_env_cfg, task_id
from tactile_rl.tasks.loadgen import loadgen_env_cfg, loadgen_task_id
from tactile_rl.tasks.rl_cfg import ppo_runner_cfg, teacher_runner_cfg

# the registered hammer and load-generator tasks train the teacher: their envs return the teacher's group only
for _rate in RATES:
    if task_id(_rate) not in list_tasks():
        register_mjlab_task(task_id=task_id(_rate), env_cfg=hold_env_cfg(_rate),
                            play_env_cfg=hold_env_cfg(_rate, num_envs=16, episode_s=10.0),
                            rl_cfg=ppo_runner_cfg(_rate))
    if hammer_task_id(_rate) not in list_tasks():
        register_mjlab_task(task_id=hammer_task_id(_rate), env_cfg=hammer_env_cfg(_rate, obs_groups=TEACHER_GROUPS),
                            play_env_cfg=hammer_env_cfg(_rate, num_envs=16, obs_groups=TEACHER_GROUPS),
                            rl_cfg=teacher_runner_cfg(_rate, "hammer"))
    if loadgen_task_id(_rate) not in list_tasks():
        register_mjlab_task(task_id=loadgen_task_id(_rate),
                            env_cfg=loadgen_env_cfg(_rate, obs_groups=TEACHER_GROUPS),
                            play_env_cfg=loadgen_env_cfg(_rate, num_envs=16, start_stage=2, curriculum=False,
                                                         obs_groups=TEACHER_GROUPS),
                            rl_cfg=teacher_runner_cfg(_rate, "loadgen"))

__all__ = ["RATES", "hammer_env_cfg", "hammer_task_id", "hold_env_cfg", "loadgen_env_cfg", "loadgen_task_id",
           "ppo_runner_cfg", "task_id", "teacher_runner_cfg"]
