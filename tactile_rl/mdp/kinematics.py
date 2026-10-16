"""Small batched kinematics helpers on the MuJoCo Warp state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import axis_angle_from_quat

from tactile_rl.scene.entities import ENTITY
from tactile_sim.sim.training_parts import training_parts

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def quat_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """Quaternions (w, x, y, z) of rotation matrices (..., 3, 3): mjlab's quat_from_matrix (pytorch3d's best-conditioned
    candidate), selected with a gather instead of a boolean mask, so no host synchronization."""
    batch = R.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = R.reshape(batch + (9,)).unbind(-1)
    q2 = torch.stack([1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], -1)
    q_abs = torch.sqrt(torch.clamp(q2, min=0.0))
    cand = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], -1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], -1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], -1),
    ], -2) / (2.0 * torch.clamp(q_abs, min=0.1))[..., None]
    idx = q_abs.argmax(-1)[..., None, None].expand(batch + (1, 4))
    return cand.gather(-2, idx).squeeze(-2)


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
