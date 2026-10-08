"""Reward terms. Milestone 1 has a placeholder hold reward (keep the hammer where the settled grasp put it, at little
effort); the plan's substep-integrated rewards (slip path, nail advance, rebound, I^2t, smoothness) come later."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tactile_rl.mdp.actions import hand_drive
from tactile_rl.mdp.events import tactile_state
from tactile_rl.mdp.kinematics import hammer_in_hand
from tactile_rl.scene.entities import ENTITY

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


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
    """Squared size of the residual's squashed action (the plan's residual-size penalty)."""
    a = env.action_manager.get_term(action_name).raw_action
    return torch.sum(torch.tanh(a) ** 2, dim=1)


# ---------------------------------------------------------------------------------------------- hammer (placeholders)
# The plan's rewards (substep-integrated slip path, nail advance, rebound energy, I^2 t, thermal margin, smoothness)
# are not milestone 2; these stand in so the hammer tasks run end to end.


def nail_advance(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Nail driven since the previous policy step (mm)."""
    from tactile_rl.mdp.actions import base_command_term

    cmd = base_command_term(env)
    depth = cmd.rt.nail.depth().double()
    r = (depth - cmd.depth_prev) * 1e3
    cmd.depth_prev = depth.clone()
    return r.float()


def slip_path(env: ManagerBasedRlEnv) -> torch.Tensor:
    """The hammer's motion in the hand since the previous policy step: mm + degrees."""
    from tactile_rl.mdp.actions import base_command_term

    cmd = base_command_term(env)
    rel = cmd.rt.hammer_in_hand()
    d = rel - cmd.hih_prev
    cmd.hih_prev = rel.clone()
    return torch.linalg.norm(d[:, :3], dim=1) * 1e3 + torch.rad2deg(torch.linalg.norm(d[:, 3:], dim=1))
