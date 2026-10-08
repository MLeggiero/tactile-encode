"""Tactile-Hammer-VegaWuji-{rate}Hz: the Vega U hammers the small nail with the light hammer; the base is a scripted
swing from the CPU's base library (BaseTrajectoryCommand), and the residual adjusts the WUJI's MIT commands at `rate`.

The scene is residual_training_config(rate) on the reduced scene: Vega U + WUJI Hand 2 with MIT drives, the TaxelScan
fast profile, the light hammer and the small nail, 0.25 ms physics; decimation = 1 / (rate x timestep). Every episode
starts in the CPU's state at a library entry's windup, with that entry's randomization, and plays it (perturbed in
time, amplitude and start, unless `perturb` is off) until it ends.

Observation groups: `student` is obs_spec's layout exactly (ResidualHandL2.observe, stack_for(rate) deep, `mask_blocks`
zeroed); `teacher` and `critic` add the privileged state (tactile_rl.mdp.observations). Rewards are placeholders (nail
advance, slip path, effort, residual size); the plan's rewards, curriculum and teacher architecture come later.

The library: `library`, else $TACTILE_BASE_LIBRARY, else runs/base_library/hammer_v1.npz (python -m
tactile_sim.behaviors.base_library export).
"""

from __future__ import annotations

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.viewer import ViewerConfig

from tactile_rl.mdp.actions import ResidualHandActionCfg, SubstepHooksCfg
from tactile_rl.mdp.commands import BaseTrajectoryCommandCfg
from tactile_rl.mdp.events import tactile_model_fields
from tactile_rl.mdp.observations import (
    drive_internals,
    event_truth,
    nail_state,
    patch_contact,
    randomization,
    student_obs,
    tool_in_hand,
)
from tactile_rl.mdp.rewards import hand_i2, nail_advance, residual_l2, slip_path
from tactile_rl.mdp.terminations import base_done, nonfinite_state, thermal_fault, tool_left_hand
from tactile_rl.scene.entities import ENTITY, mujoco_cfg, robot_entity_cfg
from tactile_rl.sensors.hub import TactileHubCfg
from tactile_rl.tasks.hold import RATES, decimation_for
from tactile_sim.sim.training_parts import config_key, residual_training_config

DEFAULT_LIBRARY = "runs/base_library/hammer_v1.npz"
# MuJoCo Warp's buffers for the hammer scene (the light hammer's capsule handle makes one pair per hand link): above
# the peaks of the library's swings (tactile_rl.parity.strikes reports them)
NCONMAX = 40
NJMAX = 128


def library_path(library: str | None = None) -> str:
    return library or os.environ.get("TACTILE_BASE_LIBRARY", DEFAULT_LIBRARY)


def hammer_scene_key(rate: float = 1000.0, timestep: float = 0.00025) -> str:
    return config_key(residual_training_config(rate, timestep))


def hammer_env_cfg(rate: float = 1000.0, library: str | None = None, timestep: float = 0.00025,
                   num_envs: int = 4096, episode_s: float = 2.5, perturb: bool = True, apply_params: bool = True,
                   mask_blocks: tuple[str, ...] = (), entries: tuple[int, ...] | None = None,
                   assign: str = "random", nconmax: int | None = NCONMAX,
                   njmax: int | None = NJMAX) -> ManagerBasedRlEnvCfg:
    key = hammer_scene_key(rate, timestep)
    priv = {"tool_in_hand": ObservationTermCfg(func=tool_in_hand),
            "patch_contact": ObservationTermCfg(func=patch_contact),
            "event_truth": ObservationTermCfg(func=event_truth),
            "nail_state": ObservationTermCfg(func=nail_state),
            "drive_internals": ObservationTermCfg(func=drive_internals),
            "randomization": ObservationTermCfg(func=randomization)}
    student = {"obs": ObservationTermCfg(func=student_obs)}
    return ManagerBasedRlEnvCfg(
        decimation=decimation_for(rate, timestep),
        scene=SceneCfg(num_envs=num_envs, env_spacing=2.0, entities={ENTITY: robot_entity_cfg(key)},
                       sensors=(TactileHubCfg(),)),
        commands={"base": BaseTrajectoryCommandCfg(scene_key=key, library=library_path(library), perturb=perturb,
                                                   apply_params=apply_params, task="hammer", mask_blocks=mask_blocks,
                                                   entries=entries, assign=assign)},
        observations={"student": ObservationGroupCfg(dict(student)),
                      "teacher": ObservationGroupCfg({**student, **priv}),
                      "critic": ObservationGroupCfg({**student, **priv})},
        actions={
            "residual": ResidualHandActionCfg(entity_name=ENTITY, scene_key=key, task="hammer"),
            "substep": SubstepHooksCfg(entity_name=ENTITY, scene_key=key),  # after the residual: sees its command
        },
        events={"tactile_fields": EventTermCfg(func=tactile_model_fields, mode="startup")},
        rewards={
            "nail_advance": RewardTermCfg(func=nail_advance, weight=1.0),
            "slip_path": RewardTermCfg(func=slip_path, weight=-1.0),
            "hand_i2": RewardTermCfg(func=hand_i2, weight=-0.05),
            "residual_l2": RewardTermCfg(func=residual_l2, weight=-0.01),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "base_done": TerminationTermCfg(func=base_done, time_out=True),
            "tool_dropped": TerminationTermCfg(func=tool_left_hand),
            "thermal_fault": TerminationTermCfg(func=thermal_fault),
            "nonfinite": TerminationTermCfg(func=nonfinite_state),
        },
        sim=SimulationCfg(nconmax=nconmax, njmax=njmax, mujoco=mujoco_cfg(residual_training_config(rate, timestep))),
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name=ENTITY, body_name="hand",
                            distance=0.8, elevation=-20.0, azimuth=135.0),
        episode_length_s=episode_s,
    )


def hammer_scene_env_cfg(num_envs: int, rate: float = 1000.0, timestep: float = 0.00025, nconmax: int | None = 128,
                         njmax: int | None = 512) -> ManagerBasedRlEnvCfg:
    """The hammer task's scene and model (per-world plant and contact fields included) without its base, sensors or
    resets: for the parity scripts, which load states into the worlds themselves."""
    from tactile_rl.mdp.observations import hand_current

    key = hammer_scene_key(rate, timestep)
    return ManagerBasedRlEnvCfg(
        decimation=decimation_for(rate, timestep),
        scene=SceneCfg(num_envs=num_envs, env_spacing=2.0, entities={ENTITY: robot_entity_cfg(key)}),
        observations={"actor": ObservationGroupCfg({"i": ObservationTermCfg(func=hand_current)}),
                      "critic": ObservationGroupCfg({"i": ObservationTermCfg(func=hand_current)})},
        actions={"residual": ResidualHandActionCfg(entity_name=ENTITY, scene_key=key, task="hammer"),
                 "substep": SubstepHooksCfg(entity_name=ENTITY, scene_key=key)},
        events={"tactile_fields": EventTermCfg(func=tactile_model_fields, mode="startup")},
        rewards={"residual_l2": RewardTermCfg(func=residual_l2, weight=-0.01)},
        terminations={"time_out": TerminationTermCfg(func=time_out, time_out=True)},
        sim=SimulationCfg(nconmax=nconmax, njmax=njmax, mujoco=mujoco_cfg(residual_training_config(rate, timestep))),
        episode_length_s=1.0,
    )


def hammer_task_id(rate: float) -> str:
    return f"Tactile-Hammer-VegaWuji-{rate:g}Hz"


__all__ = ["DEFAULT_LIBRARY", "RATES", "hammer_env_cfg", "hammer_scene_key", "hammer_task_id", "library_path"]
