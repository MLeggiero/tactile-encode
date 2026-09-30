"""Task-space kinematics on MuJoCo models: site pose, Jacobian, pose error, task inertia, IK."""

from __future__ import annotations

import mujoco
import numpy as np


def site_pose(data: mujoco.MjData, sid: int) -> tuple[np.ndarray, np.ndarray]:
    return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()


def site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, sid: int, dofs: np.ndarray | None = None) -> np.ndarray:
    """6 x n Jacobian [linear; angular] of a site, optionally restricted to the given dof columns."""
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, sid)
    J = np.vstack([jp, jr])
    return J if dofs is None else J[:, dofs]


def rot_error(R_des: np.ndarray, R: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking R to R_des."""
    q_des = np.zeros(4)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q_des, R_des.reshape(-1))
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    # R_des = R_err * R  ->  R_err = R_des R^T
    q_inv = np.zeros(4)
    mujoco.mju_negQuat(q_inv, q)
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, q_des, q_inv)
    if q_err[0] < 0:
        q_err = -q_err
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q_err, 1.0)
    return v


def pose_error(p_des: np.ndarray, R_des: np.ndarray, p: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.concatenate([p_des - p, rot_error(R_des, R)])


def task_inertia(M: np.ndarray, J: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Operational-space inertia (J M^-1 J^T)^-1 with light damping."""
    Minv_JT = np.linalg.solve(M, J.T)
    return np.linalg.inv(J @ Minv_JT + eps * np.eye(J.shape[0]))


def solve_ik(model: mujoco.MjModel, site_id: int, p_des: np.ndarray, R_des: np.ndarray, qadr: np.ndarray,
             dofs: np.ndarray, q_seed: np.ndarray, qpos_base: np.ndarray | None = None, iters: int = 300,
             tol: float = 1e-6, damping: float = 1e-3, q_rest: np.ndarray | None = None,
             rest_gain: float = 0.05) -> tuple[np.ndarray, float]:
    """Damped least-squares IK with joint limits and a weak pull toward q_rest. Returns (q, residual)."""
    d = mujoco.MjData(model)
    if qpos_base is not None:
        d.qpos[:] = qpos_base
    q = np.array(q_seed, dtype=float)
    lo = model.jnt_range[[model.dof_jntid[i] for i in dofs], 0]
    hi = model.jnt_range[[model.dof_jntid[i] for i in dofs], 1]
    q_rest = q.copy() if q_rest is None else np.asarray(q_rest, dtype=float)
    err_n = np.inf
    for _ in range(iters):
        d.qpos[qadr] = q
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        p, R = site_pose(d, site_id)
        e = pose_error(p_des, R_des, p, R)
        err_n = float(np.linalg.norm(e))
        if err_n < tol:
            break
        J = site_jacobian(model, d, site_id, dofs)
        JJt = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, e)
        # null-space pull toward the rest posture
        N = np.eye(len(dofs)) - J.T @ np.linalg.solve(JJt, J)
        dq += N @ (rest_gain * (q_rest - q))
        q = np.clip(q + dq, lo + 1e-3, hi - 1e-3)
    return q, err_n
