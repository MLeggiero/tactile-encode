"""Small batched kinematics helpers on the MuJoCo Warp state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_from_matrix

from tactile_rl.scene.entities import ENTITY
from tactile_sim.sim.training_parts import training_parts

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def rotvec_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """Rotation vector (axis x angle, |angle| <= pi) of rotation matrices (..., 3, 3), as mju_mat2Quat + mju_quat2Vel
    compute it on the testbed."""
    return axis_angle_from_quat(quat_from_matrix(R))


def site_id(env: ManagerBasedRlEnv, name: str) -> int:
    return env.sim.mj_model.site(f"{ENTITY}/{name}").id


def hammer_in_hand(env: ManagerBasedRlEnv, scene_key: str) -> torch.Tensor:
    """(N, 6) the hammer's grasp frame in the TCP frame: position (m) and rotation vector (rad), as
    World.hammer_in_hand. Reads the derived kinematics (valid after a forward pass)."""
    parts = training_parts(scene_key)
    d = env.sim.data
    t, h = site_id(env, parts.tcp_site), site_id(env, parts.hammer_site)
    xpos, xmat = d.site_xpos, d.site_xmat
    Rt = xmat[:, t].reshape(-1, 3, 3)
    Rh = xmat[:, h].reshape(-1, 3, 3)
    p = torch.einsum("nji,nj->ni", Rt, xpos[:, h] - xpos[:, t])
    return torch.cat([p, rotvec_from_matrix(Rt.transpose(1, 2) @ Rh)], dim=-1)
