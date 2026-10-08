"""Reward terms.

The plan's rewards are integrated over the physics substeps (the tactile runtime's integrands, tactile_rl.mdp.runtime),
so a policy at 100 Hz and one at 1 kHz collect the same reward for the same motion. mjlab's reward manager multiplies
every term by the policy step (scale_by_dt), so each term here returns its integral over the step divided by the step:
weight x term x step_dt is then weight x the integral. Events (nail advance, a drop) return their amount divided by
the step, so they count once whatever the rate.

- penalties: I^2 t, torque roughness about its 50 Hz low-pass, torque above rated, load near the hard stops, thermal
  margin (and the thermal fault's termination penalty), the residual's size;
- hammer: nail advance (mm), slip path in the hand (mm, degrees), drift from the reference pose (mm + degrees, per
  second), rebound energy over [t_c + 5, t_c + 60] ms (J), the drop's penalty;
- load generator: slip, drift, drop and the effort terms.

Milestone 1's hold terms (`hammer_drift`, `hand_i2`, `residual_l2`) stay for the hold tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tactile_rl.mdp.actions import base_command_term, hand_drive
from tactile_rl.mdp.events import tactile_state
from tactile_rl.mdp.kinematics import hammer_in_hand
from tactile_rl.scene.entities import ENTITY

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _rate(env: ManagerBasedRlEnv, key: str) -> torch.Tensor:
    rt = env._tactile_rt
    return rt.acc[key] / env.step_dt


# ---------------------------------------------------------------------------------------------- penalties (integrated)


def effort_i2t(env: ManagerBasedRlEnv) -> torch.Tensor:
    """d/dt of the integral of the hand's mean (i / i_rated)^2."""
    return _rate(env, "i2")


def torque_roughness(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The hand torque about its 50 Hz low-pass, per rated torque, squared and averaged over the joints."""
    return _rate(env, "smooth")


def torque_peak(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Torque above the rated torque, per rated, squared and averaged over the joints."""
    return _rate(env, "peak")


def hard_stop(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Joints within the stop margin of their range limits (fraction of the margin, squared, averaged)."""
    return _rate(env, "hardstop")


def thermal_margin(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Windings within 20 K of their fault temperature (fraction, squared, averaged)."""
    return _rate(env, "thermal")


def residual_size(env: ManagerBasedRlEnv, action_name: str = "residual") -> torch.Tensor:
    """The squashed residual action's mean square (held over the policy step)."""
    a = env.action_manager.get_term(action_name).a_sq
    return torch.mean(a * a, dim=1)


# ---------------------------------------------------------------------------------------------- the tool in the hand


def slip_mm(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The tool's path in the hand over the step (mm), per second of the step."""
    return _rate(env, "slip_mm")


def slip_deg(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The tool's rotation path in the hand over the step (degrees), per second of the step."""
    return _rate(env, "slip_deg")


def pose_drift(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The tool's distance from the episode's reference pose in the hand: mm + degrees."""
    return _rate(env, "drift")


def rebound_energy(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The tool's kinetic energy relative to the hand over [5, 60] ms after a blow's onset (window mean, J)."""
    return _rate(env, "rebound")


def nail_advance(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Nail driven during the step (mm), per second of the step."""
    cmd = base_command_term(env)
    depth = cmd.rt.nail.depth().double()
    r = (depth - cmd.depth_prev) * 1e3
    cmd.depth_prev = depth.clone()
    return (r / env.step_dt).float()


def termination_penalty(env: ManagerBasedRlEnv, term: str) -> torch.Tensor:
    """1 / step on the step a termination term fires (the weight is the penalty); zero if the task has no such term."""
    tm = env.termination_manager
    if term not in tm.active_terms:
        return torch.zeros(env.num_envs, device=env.device)
    return tm.get_term(term).float() / env.step_dt


# ---------------------------------------------------------------------------------------------- milestone 1 (hold)


def hammer_drift(env: ManagerBasedRlEnv, scene_key: str, pos_scale: float = 1e-3, rot_scale: float = 0.0175
                 ) -> torch.Tensor:
    """Distance of the hammer's pose in the hand from its settled pose at reset: mm + degrees by default."""
    d = hammer_in_hand(env, scene_key) - tactile_state(env).ref_rel
    return torch.linalg.norm(d[:, :3], dim=1) / pos_scale + torch.linalg.norm(d[:, 3:], dim=1) / rot_scale


def hand_i2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Mean squared motor current over the hand, in units of the rated current."""
    drive = hand_drive(env.scene[ENTITY])
    return torch.mean((drive.i / drive.i_rated) ** 2, dim=1)


def residual_l2(env: ManagerBasedRlEnv, action_name: str = "residual") -> torch.Tensor:
    """Squared size of the residual's squashed action (milestone 1's residual-size penalty)."""
    a = env.action_manager.get_term(action_name).raw_action
    return torch.sum(torch.tanh(a) ** 2, dim=1)


# the plan's weights (per mm, per degree, per joule; the effort terms set so a hold at the rated current costs about
# what 1 mm of slip does per second; the hard stops lightly: the base's wrap already rests some joints on their stops)
WEIGHTS_EFFORT = {"effort_i2t": -1.0, "torque_roughness": -10.0, "torque_peak": -10.0, "hard_stop": -0.1,
                  "thermal_margin": -10.0, "residual_size": -0.1}
WEIGHTS_HAMMER = {"nail_advance": 1.0, "slip_mm": -1.0, "slip_deg": -1.0, "pose_drift": -1.0, "rebound_energy": -10.0}
WEIGHTS_LOADGEN = {"slip_mm": -1.0, "slip_deg": -1.0, "pose_drift": -1.0}
DROP_PENALTY = -50.0
THERMAL_FAULT_PENALTY = -20.0
