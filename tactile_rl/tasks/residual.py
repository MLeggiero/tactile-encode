"""Tactile-Residual-VegaWuji-{rate}Hz: the residual's training curriculum through C3 in one task.

C0-C2 are the load generator's stages (Tactile-LoadGen); C3 mixes hammer episodes (the base library's swings, their
perturbations and randomization, the hand base at a hold level of 0.3-0.8) 70/30 with the load generator's full band
(MixedBaseCommand). The scene is the hammer's (residual_training_config, the nail live); the reset randomization runs
on every env and a hammer env's library draw is written over it. Rewards are the hammer's (nail advance and rebound
are zero on a load-generator env), the terminations the hammer's (an entry that played out ends its episode).

Stages pass on the curriculum's gates (80 % of events within the slip target, < 1 % drops) or, past `stage_cap_s`
simulated seconds (summed over the envs) in a stage, advance anyway (TactileCurriculum): by default C1 and C2 take at
most 120 000 s each at 4096 envs (229 iterations of 4096 envs x 0.128 s; the cap scales with the env count:
`cap_ref_envs`), so a budget of 1000 iterations spends at least half of it in C3.

The training distribution (2026-10-13; each a parameter, the evaluation sets its own): the hand/tool friction drawn
per episode from TRAIN_HAND_FRICTION on every env (the hammer's entries keep their start states, settled at the
keyframes' friction), the load generator's wide loads (loadgen_command.wide_loads: impulses up to 60 degrees off the
strike axis, binds and torque impulses about every tool axis, the band of $TACTILE_LOADGEN_BAND), and VLA-style hand
commands (action_map.TorchHandTable) in VLA_FRAC of the episodes.

Metrics: the load generator's per event (Metrics/base/*, zero on hammer envs), the share of hammer episodes
(Metrics/base/hammer) and the nail's advance (Metrics/base/hammer_advance_mm); per episode the lost grasp, the slip
path and I^2 t over all envs (Episode_Metrics/dropped, slip_mm, slip_deg, i2t) and over the hammer's
(Episode_Metrics/hammer_dropped, ...: the load generator's share is the difference), and the hammer's net slip over the
strikes that kept the grasp (Episode_Metrics/hammer_net_mm, hammer_net_deg, over hammer_strike).

Observations carry mjlab's NaN guard ("warn": a non-finite value is reported in the log and zeroed) so one env cannot
end a long run; tests build with "error".

The library: `library`, else $TACTILE_BASE_LIBRARY, else runs/base_library/hammer_v1.npz.
"""

from __future__ import annotations

import os

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.viewer import ViewerConfig

from tactile_rl.mdp.actions import ResidualHandActionCfg, SubstepHooksCfg, base_command_term
from tactile_rl.mdp.curriculum import TactileCurriculum
from tactile_rl.mdp.events import build_state_bank, reset_from_state_bank, tactile_model_fields
from tactile_rl.mdp.loadgen_command import wide_loads
from tactile_rl.mdp.mixed_command import MixedBaseCommandCfg
from tactile_rl.scene.entities import ENTITY, mujoco_cfg, robot_entity_cfg
from tactile_rl.sensors.hub import TactileHubCfg
from tactile_rl.tasks.common import ALL_GROUPS, dr_events, observation_groups, reward_terms, termination_terms
from tactile_rl.tasks.hammer import library_path
from tactile_rl.tasks.hold import decimation_for
from tactile_rl.tasks.loadgen import _dropped, _i2t, _slip_deg, _slip_mm, loadgen_scene_key
from tactile_sim.config import LoadGenCfg
from tactile_sim.control.obs_spec import OBS_SPEC_VERSION
from tactile_sim.sim.training_parts import residual_training_config

# MuJoCo Warp's buffers: the hammer's strikes and the load generator's holds (each task's peaks, with room)
NCONMAX = 128  # the YCB hammer (12 handle slices, the fingers colliding): see tactile_rl.bench.throughput
NJMAX = 512
STAGE_CAP_S = 120_000.0  # simulated s (all envs) a gated stage may take, at CAP_REF_ENVS envs
CAP_REF_ENVS = 4096
TRAIN_HAND_FRICTION = (0.2, 2.0)  # the hand/tool friction drawn per episode (the CPU's ResidualDRCfg: 1.1-2.0)
VLA_FRAC = 0.4  # the share of episodes whose hand base is VLA-style


def _hammer(env):
    return base_command_term(env).is_hammer.float()


def _hammer_dropped(env):
    return _dropped(env) * _hammer(env)


def _hammer_slip_mm(env):
    return _slip_mm(env) * _hammer(env)


def _hammer_slip_deg(env):
    return _slip_deg(env) * _hammer(env)


def _hammer_i2t(env):
    return _i2t(env) * _hammer(env)


def _hammer_strike(env):
    """1 on the step a hammer entry plays out with the tool in the hand (an episode's last step)."""
    tm = env.termination_manager
    return (base_command_term(env).is_hammer & tm.get_term("base_done") & ~tm.get_term("tool_dropped")).float()


def _hammer_net(env) -> torch.Tensor:
    cmd = base_command_term(env)
    return cmd.rt.hammer_in_hand() - cmd.ham.ref_rel


def _hammer_net_mm(env):
    """The tool's net slip in the hand over a strike that kept the grasp (its entry's end against its start), mm."""
    return torch.linalg.norm(_hammer_net(env)[:, :3], dim=1) * 1e3 * _hammer_strike(env)


def _hammer_net_deg(env):
    return torch.rad2deg(torch.linalg.norm(_hammer_net(env)[:, 3:], dim=1)) * _hammer_strike(env)


def residual_env_cfg(rate: float = 1000.0, library: str | None = None, timestep: float = 0.00025,
                     num_envs: int = 4096, episode_s: float = 3.0, start_stage: int | None = None, max_stage: int = 3,
                     min_episodes: int = 2000, min_events: int = 2000, obs_groups: tuple[str, ...] = ALL_GROUPS,
                     nconmax: int | None = NCONMAX, njmax: int | None = NJMAX,
                     stage_cap_s: float | None = STAGE_CAP_S, nan_policy: str = "warn",
                     student_version: int = OBS_SPEC_VERSION, cap_ref_envs: int | None = CAP_REF_ENVS,
                     hand_friction: tuple[float, float] | None = TRAIN_HAND_FRICTION, vla_frac: float = VLA_FRAC,
                     loadgen: LoadGenCfg | None = None) -> ManagerBasedRlEnvCfg:
    """The task's env (see the module doc); `loadgen` None: wide_loads() (the band of $TACTILE_LOADGEN_BAND)."""
    if start_stage is None:  # $TACTILE_START_STAGE: a resumed teacher continues in its stage (the registered task's)
        start_stage = int(os.environ.get("TACTILE_START_STAGE", "0"))
    key = loadgen_scene_key(rate, timestep)
    events = {"tactile_fields": EventTermCfg(func=tactile_model_fields, mode="startup"),
              "state_bank": EventTermCfg(func=build_state_bank, mode="startup", params={"scene_key": key}),
              "reset_from_bank": EventTermCfg(func=reset_from_state_bank, mode="reset", params={"scene_key": key}),
              **dr_events(key, hand_friction)}
    cur = {"stage": CurriculumTermCfg(func=TactileCurriculum, params={
        "scene_key": key, "start_stage": start_stage, "max_stage": max_stage, "min_episodes": min_episodes,
        "min_events": min_events, "stage_cap_s": stage_cap_s, "cap_ref_envs": cap_ref_envs})}
    metrics = {"dropped": MetricsTermCfg(func=_dropped, reduce="last"),
               "slip_mm": MetricsTermCfg(func=_slip_mm, reduce="sum"),
               "slip_deg": MetricsTermCfg(func=_slip_deg, reduce="sum"),
               "i2t": MetricsTermCfg(func=_i2t, reduce="sum"),
               "hammer_dropped": MetricsTermCfg(func=_hammer_dropped, reduce="last"),
               "hammer_slip_mm": MetricsTermCfg(func=_hammer_slip_mm, reduce="sum"),
               "hammer_slip_deg": MetricsTermCfg(func=_hammer_slip_deg, reduce="sum"),
               "hammer_i2t": MetricsTermCfg(func=_hammer_i2t, reduce="sum"),
               "hammer_strike": MetricsTermCfg(func=_hammer_strike, reduce="sum"),
               "hammer_net_mm": MetricsTermCfg(func=_hammer_net_mm, reduce="sum"),
               "hammer_net_deg": MetricsTermCfg(func=_hammer_net_deg, reduce="sum")}
    return ManagerBasedRlEnvCfg(
        decimation=decimation_for(rate, timestep),
        scene=SceneCfg(num_envs=num_envs, env_spacing=2.0, entities={ENTITY: robot_entity_cfg(key)},
                       sensors=(TactileHubCfg(),)),
        commands={"base": MixedBaseCommandCfg(scene_key=key, library=library_path(library),
                                              loadgen=wide_loads() if loadgen is None else loadgen,
                                              hand_friction=hand_friction, vla_frac=vla_frac)},
        observations=observation_groups("mixed", obs_groups, nan_policy, student_version),
        actions={
            "residual": ResidualHandActionCfg(entity_name=ENTITY, scene_key=key, task="hammer"),
            "substep": SubstepHooksCfg(entity_name=ENTITY, scene_key=key),
        },
        events=events,
        rewards=reward_terms("mixed"),
        terminations=termination_terms("mixed"),
        curriculum=cur,
        metrics=metrics,
        sim=SimulationCfg(nconmax=nconmax, njmax=njmax, mujoco=mujoco_cfg(residual_training_config(rate, timestep))),
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name=ENTITY, body_name="hand",
                            distance=0.8, elevation=-20.0, azimuth=135.0),
        episode_length_s=episode_s,
    )


def residual_task_id(rate: float) -> str:
    return f"Tactile-Residual-VegaWuji-{rate:g}Hz"


__all__ = ["CAP_REF_ENVS", "TRAIN_HAND_FRICTION", "VLA_FRAC", "residual_env_cfg", "residual_task_id"]
