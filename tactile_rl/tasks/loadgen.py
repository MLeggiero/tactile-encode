"""Tactile-LoadGen-VegaWuji-{rate}Hz: the Vega U holds the light hammer at the hover pose while synthetic loads hit
the tool's working point (impulses, periodic forces and torques, steps, binds, pushes: tactile_rl.plants.loadgen,
the mirror of tactile_sim.model.loadgen), and the residual adjusts the WUJI's MIT commands at `rate`.

The scene is residual_training_config(rate) (as the hammer task's: Vega U + WUJI Hand 2 with MIT drives, the
TaxelScan fast profile, the light hammer, 0.25 ms physics; the nail is there but idle). Episodes start from the bank of
settled grasps (reset_from_state_bank), randomized at reset (pair_solref_solimp, randomize_wuji_drive,
randomize_taxelscan, randomize_plant, randomize_forecast); the base is the hold (LoadGenCommand) with a forecast of
the schedule's next event. The curriculum (C0 hold, C1 low amplitude with a clean forecast, C2 the full band, the
full randomization, a noisy forecast masked in 25 % of episodes) sets the loads, the randomization's scale and the
forecast. Rewards: the tool's slip path and drift in the hand, the effort penalties, the drop (-50) and the thermal
fault (-20), integrated over the substeps.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.viewer import ViewerConfig

from tactile_rl.mdp.actions import ResidualHandActionCfg, SubstepHooksCfg
from tactile_rl.mdp.curriculum import TactileCurriculum
from tactile_rl.mdp.events import build_state_bank, reset_from_state_bank, tactile_model_fields
from tactile_rl.mdp.loadgen_command import LoadGenCommandCfg
from tactile_rl.scene.entities import ENTITY, mujoco_cfg, robot_entity_cfg
from tactile_rl.sensors.hub import TactileHubCfg
from tactile_rl.tasks.common import ALL_GROUPS, dr_events, observation_groups, reward_terms, termination_terms
from tactile_rl.tasks.hold import decimation_for
from tactile_sim.config import LoadGenCfg
from tactile_sim.sim.training_parts import config_key, residual_training_config

# MuJoCo Warp's buffers for the held light hammer under loads (tactile_rl.bench.throughput reports the peaks)
NCONMAX = 128
NJMAX = 512


def _dropped(env) -> torch.Tensor:
    return env.termination_manager.get_term("tool_dropped").float()


def _slip_mm(env):
    return env._tactile_rt.acc["slip_mm"]


def _slip_deg(env):
    return env._tactile_rt.acc["slip_deg"]


def _i2t(env):
    return env._tactile_rt.acc["i2"]


def loadgen_scene_key(rate: float = 1000.0, timestep: float = 0.00025) -> str:
    return config_key(residual_training_config(rate, timestep))


def loadgen_env_cfg(rate: float = 1000.0, timestep: float = 0.00025, num_envs: int = 4096, episode_s: float = 3.0,
                    start_stage: int = 0, max_stage: int = 2, curriculum: bool = True, randomize: bool = True,
                    min_episodes: int = 2000, min_events: int = 2000, nconmax: int | None = NCONMAX,
                    njmax: int | None = NJMAX, obs_groups: tuple[str, ...] = ALL_GROUPS,
                    loadgen: LoadGenCfg | None = None,
                    hand_friction: tuple[float, float] | None = None) -> ManagerBasedRlEnvCfg:
    """The load generator's task; `loadgen` its loads (None: LoadGenCfg's), `hand_friction` the hand/tool friction's
    range (None: the scene's ResidualDRCfg; a degenerate range fixes it)."""
    key = loadgen_scene_key(rate, timestep)
    events = {"tactile_fields": EventTermCfg(func=tactile_model_fields, mode="startup"),
              "state_bank": EventTermCfg(func=build_state_bank, mode="startup", params={"scene_key": key}),
              "reset_from_bank": EventTermCfg(func=reset_from_state_bank, mode="reset", params={"scene_key": key})}
    if randomize:
        events.update(dr_events(key, hand_friction))
    cur = {}
    if curriculum:
        cur["stage"] = CurriculumTermCfg(func=TactileCurriculum, params={
            "scene_key": key, "start_stage": start_stage, "max_stage": max_stage, "min_episodes": min_episodes,
            "min_events": min_events})
    metrics = {"dropped": MetricsTermCfg(func=_dropped, reduce="last"),
               "slip_mm": MetricsTermCfg(func=_slip_mm, reduce="sum"),
               "slip_deg": MetricsTermCfg(func=_slip_deg, reduce="sum"),
               "i2t": MetricsTermCfg(func=_i2t, reduce="sum")}
    return ManagerBasedRlEnvCfg(
        decimation=decimation_for(rate, timestep),
        scene=SceneCfg(num_envs=num_envs, env_spacing=2.0, entities={ENTITY: robot_entity_cfg(key)},
                       sensors=(TactileHubCfg(),)),
        commands={"base": LoadGenCommandCfg(scene_key=key, loadgen=LoadGenCfg() if loadgen is None else loadgen)},
        observations=observation_groups("loadgen", obs_groups),
        actions={
            "residual": ResidualHandActionCfg(entity_name=ENTITY, scene_key=key, task="loadgen"),
            "substep": SubstepHooksCfg(entity_name=ENTITY, scene_key=key),  # after the residual: sees its command
        },
        events=events,
        rewards=reward_terms("loadgen"),
        terminations=termination_terms("loadgen"),
        curriculum=cur,
        metrics=metrics,
        sim=SimulationCfg(nconmax=nconmax, njmax=njmax, mujoco=mujoco_cfg(residual_training_config(rate, timestep))),
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name=ENTITY, body_name="hand",
                            distance=0.8, elevation=-20.0, azimuth=135.0),
        episode_length_s=episode_s,
    )


def loadgen_task_id(rate: float) -> str:
    return f"Tactile-LoadGen-VegaWuji-{rate:g}Hz"


__all__ = ["loadgen_env_cfg", "loadgen_scene_key", "loadgen_task_id"]
