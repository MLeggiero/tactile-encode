"""Tactile-Hold-VegaWuji-{rate}Hz: the Vega U holds the hammer at the hover pose while the residual adjusts the WUJI's
MIT commands at `rate` on top of the grip synergy. Milestone 1's minimal task (placeholder observation and reward)
that exercises the scene, the drives, the action map, the reset bank and rollouts at scale.

Physics at `timestep` (0.25 ms by default); decimation = 1 / (rate x timestep): the drives, the arm servos and every
hook run on the physics clock, only the policy's tick changes with the rate.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import last_action, time_out
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.viewer import ViewerConfig

from tactile_rl.mdp.actions import ResidualHandActionCfg, SubstepHooksCfg
from tactile_rl.mdp.events import build_state_bank, reset_from_state_bank
from tactile_rl.mdp.observations import hand_current, hand_joint_pos
from tactile_rl.mdp.rewards import hammer_drift, hand_i2, residual_l2
from tactile_rl.mdp.terminations import nonfinite_state, thermal_fault, tool_dropped
from tactile_rl.scene.entities import ENTITY, mujoco_cfg, robot_entity_cfg
from tactile_sim.sim.training_parts import config_key, training_config

RATES = (1000.0, 500.0, 250.0, 100.0)
# MuJoCo Warp's buffers, just above the peaks of the zero-residual hold and its settle (tactile_rl.bench.throughput
# reports them; the welded settle is the peak: 45 broadphase candidates and 10 contacts per world, 77 constraint rows
# in the fullest world at 0.25 ms; 50 and 83 at 0.5 ms). Strikes, the load generator and other plants will need more.
NCONMAX = 52  # broadphase candidates (and contacts) per world; MuJoCo Warp pools them over all worlds
NJMAX = 88  # constraint rows per world


def decimation_for(rate: float, timestep: float) -> int:
    dec = round(1.0 / (rate * timestep))
    if dec < 1 or abs(dec * rate * timestep - 1.0) > 1e-9:
        raise ValueError(f"a {rate:g} Hz policy needs a whole number of {timestep * 1e3:g} ms physics steps")
    return dec


def hold_env_cfg(rate: float = 1000.0, timestep: float = 0.00025, num_envs: int = 4096, episode_s: float = 1.0,
                 nconmax: int | None = NCONMAX, njmax: int | None = NJMAX) -> ManagerBasedRlEnvCfg:
    key = config_key(training_config(timestep))
    obs = {
        "hand_q": ObservationTermCfg(func=hand_joint_pos, params={"scene_key": key}),
        "hand_current": ObservationTermCfg(func=hand_current),
        "last_action": ObservationTermCfg(func=last_action, params={"action_name": "residual"}),
    }
    return ManagerBasedRlEnvCfg(
        decimation=decimation_for(rate, timestep),
        scene=SceneCfg(num_envs=num_envs, env_spacing=2.0, entities={ENTITY: robot_entity_cfg(key)}),
        observations={"actor": ObservationGroupCfg(dict(obs)), "critic": ObservationGroupCfg(dict(obs))},
        actions={
            "residual": ResidualHandActionCfg(entity_name=ENTITY, scene_key=key),
            "substep": SubstepHooksCfg(entity_name=ENTITY, scene_key=key),  # after the residual: sees its command
        },
        events={
            "state_bank": EventTermCfg(func=build_state_bank, mode="startup", params={"scene_key": key}),
            "reset_from_bank": EventTermCfg(func=reset_from_state_bank, mode="reset", params={"scene_key": key}),
        },
        rewards={
            "hammer_drift": RewardTermCfg(func=hammer_drift, weight=-0.1, params={"scene_key": key}),
            "hand_i2": RewardTermCfg(func=hand_i2, weight=-0.05),
            "residual_l2": RewardTermCfg(func=residual_l2, weight=-0.01),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            "tool_dropped": TerminationTermCfg(func=tool_dropped, params={"scene_key": key}),
            "thermal_fault": TerminationTermCfg(func=thermal_fault),
            "nonfinite": TerminationTermCfg(func=nonfinite_state),
        },
        sim=SimulationCfg(nconmax=nconmax, njmax=njmax, mujoco=mujoco_cfg(training_config(timestep))),
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name=ENTITY, body_name="hand",
                            distance=0.8, elevation=-20.0, azimuth=135.0),
        episode_length_s=episode_s,
    )


def task_id(rate: float) -> str:
    return f"Tactile-Hold-VegaWuji-{rate:g}Hz"
