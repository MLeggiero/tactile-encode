"""The drill on the GPU (12 V model): the Vega U holds the pistol driver in the WUJI (the trigger part pulled by the
index finger, that finger's friction capped), and the base is a scripted drill feed from the CPU's drill library
(behaviors.drill_library) replayed by BaseTrajectoryCommand: the arm's servo stream, the grip synergy's scalar and
the trigger finger's own level. The plant is tactile_rl.plants.drill (the 12 V drivetrain, the screw's or the hole's
laws) on the reduced scene at 0.25 ms (residual_training_config(task="drill")).

No drill teacher yet: the env serves the zero-residual parity of whole drill episodes against the CPU
(tactile_rl.parity.drill_episodes) and returns the student's group; the drill's rewards come with its teacher.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.viewer import ViewerConfig

from tactile_rl.mdp.actions import ResidualHandActionCfg, SubstepHooksCfg
from tactile_rl.mdp.commands import BaseTrajectoryCommandCfg
from tactile_rl.mdp.events import tactile_model_fields
from tactile_rl.mdp.rewards import residual_l2
from tactile_rl.mdp.terminations import base_done
from tactile_rl.scene.entities import ENTITY, mujoco_cfg, robot_entity_cfg
from tactile_rl.sensors.hub import TactileHubCfg
from tactile_rl.tasks.common import observation_groups
from tactile_rl.tasks.hold import decimation_for
from tactile_sim.sim.training_parts import config_key, residual_training_config

# MuJoCo Warp's buffers: the pistol's housing, grip, battery and trigger against the wrap, with room
NCONMAX = 96
NJMAX = 320


def drill_training_config(rate: float = 1000.0, timestep: float = 0.00025, mode: str = "screw"):
    return residual_training_config(rate, timestep, task="drill").replace(drill={"mode": mode})


def drill_scene_key(rate: float = 1000.0, timestep: float = 0.00025, mode: str = "screw") -> str:
    return config_key(drill_training_config(rate, timestep, mode))


def drill_env_cfg(rate: float, library: str, mode: str = "screw", timestep: float = 0.00025, num_envs: int = 64,
                  episode_s: float = 11.0, perturb: bool = False, apply_params: bool = True,
                  entries: tuple[int, ...] | None = None, assign: str = "sequential",
                  obs_groups: tuple[str, ...] = ("student",), nconmax: int | None = NCONMAX,
                  njmax: int | None = NJMAX) -> ManagerBasedRlEnvCfg:
    key = drill_scene_key(rate, timestep, mode)
    groups = {k: v for k, v in observation_groups("drill", ("student",)).items() if k in obs_groups}
    return ManagerBasedRlEnvCfg(
        decimation=decimation_for(rate, timestep),
        scene=SceneCfg(num_envs=num_envs, env_spacing=2.0, entities={ENTITY: robot_entity_cfg(key)},
                       sensors=(TactileHubCfg(),)),
        commands={"base": BaseTrajectoryCommandCfg(scene_key=key, library=library, perturb=perturb,
                                                   apply_params=apply_params, task="drill", entries=entries,
                                                   assign=assign)},
        observations=groups,
        actions={
            "residual": ResidualHandActionCfg(entity_name=ENTITY, scene_key=key, task="drill"),
            "substep": SubstepHooksCfg(entity_name=ENTITY, scene_key=key),
        },
        events={"tactile_fields": EventTermCfg(func=tactile_model_fields, mode="startup")},
        rewards={"residual_l2": RewardTermCfg(func=residual_l2, weight=-0.01)},
        terminations={"time_out": TerminationTermCfg(func=time_out, time_out=True),
                      "base_done": TerminationTermCfg(func=base_done, time_out=True)},
        sim=SimulationCfg(nconmax=nconmax, njmax=njmax, mujoco=mujoco_cfg(drill_training_config(rate, timestep, mode))),
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name=ENTITY, body_name="hand",
                            distance=0.8, elevation=-20.0, azimuth=135.0),
        episode_length_s=episode_s,
    )


__all__ = ["drill_env_cfg", "drill_scene_key", "drill_training_config"]
