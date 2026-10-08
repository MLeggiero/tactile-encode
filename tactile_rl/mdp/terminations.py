"""Termination terms: the tool left the hand, a drive faulted thermally, or the state went non-finite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tactile_rl.mdp.actions import hand_drive
from tactile_rl.mdp.events import tactile_state
from tactile_rl.mdp.kinematics import hammer_in_hand
from tactile_rl.scene.entities import ENTITY

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def tool_dropped(env: ManagerBasedRlEnv, scene_key: str, max_shift: float = 0.02) -> torch.Tensor:
    """The hammer's grasp frame moved more than `max_shift` (m) from its settled place in the hand."""
    d = hammer_in_hand(env, scene_key) - tactile_state(env).ref_rel
    return torch.linalg.norm(d[:, :3], dim=1) > max_shift


def thermal_fault(env: ManagerBasedRlEnv) -> torch.Tensor:
    """A hand drive has faulted (winding at t_max; torque off until reset)."""
    return hand_drive(env.scene[ENTITY]).fault.any(dim=1)


def nonfinite_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    d = env.sim.data
    return ~(torch.isfinite(d.qpos).all(dim=1) & torch.isfinite(d.qvel).all(dim=1))


def base_done(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The base's library entry has played out (a truncation, like a time-out)."""
    from tactile_rl.mdp.actions import base_command_term

    return base_command_term(env).done()


def tool_left_hand(env: ManagerBasedRlEnv, max_shift: float = 0.02) -> torch.Tensor:
    """The hammer's grasp frame moved more than `max_shift` (m) in the hand since the episode's start (the hammer
    tasks: the start is the library entry's, not a bank state's)."""
    from tactile_rl.mdp.actions import base_command_term

    cmd = base_command_term(env)
    d = cmd.rt.hammer_in_hand() - cmd.ref_rel
    return torch.linalg.norm(d[:, :3], dim=1) > max_shift
