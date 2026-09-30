"""L1 Cartesian impedance law (1 kHz).

    e     = clip(pose_error(x_eq, x))              reference limiter: |x_eq - x| <= delta_max
    edot  = xd_eq - xd_used                        xd_used = reference velocity while gated after impact
    F     = K_t e + D edot + F_ff + Lambda xdd_ff  K_t slew-limited toward the command (passivity)
    tau   = J^T F + N^T (k_null (q_null - q) - d_null qd) + qfrc_bias + J_payload^T (-m_payload g)

K and D are diagonal in the command's stiffness frame R_K (the strike frame). D defaults to a
critically damped choice 2 zeta sqrt(K diag(Lambda)) with Lambda expressed in that frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tactile_sim.config import ControllerCfg
from tactile_sim.control.interface import L2Command
from tactile_sim.control.kinematics import pose_error, task_inertia


def _frame6(R: np.ndarray) -> np.ndarray:
    T = np.zeros((6, 6))
    T[:3, :3] = R
    T[3:, 3:] = R
    return T


def clip_norm(v: np.ndarray, lim: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n <= lim else v * (lim / n)


@dataclass
class ImpedanceOutput:
    tau: np.ndarray
    F: np.ndarray  # task wrench commanded (world)
    F_spring: np.ndarray  # K e part (world)
    e: np.ndarray  # limited pose error
    K: np.ndarray  # stiffness actually applied (R_K frame)
    D: np.ndarray


class CartesianImpedance:
    def __init__(self, cfg: ControllerCfg, dt: float, q_null: np.ndarray):
        self.cfg = cfg
        self.dt = dt
        self.q_null = np.asarray(q_null, dtype=float).copy()
        self.K_t: np.ndarray | None = None

    def reset(self, K0: np.ndarray | None = None) -> None:
        self.K_t = None if K0 is None else np.asarray(K0, dtype=float).copy()

    def slew(self, K_cmd: np.ndarray) -> np.ndarray:
        if self.K_t is None:
            self.K_t = K_cmd.astype(float).copy()
            return self.K_t
        step_t = self.cfg.k_dot_max * self.dt
        # rotational stiffness slews at the same relative rate as translational
        k_ref = max(float(np.max(self.K_t[:3])), 1.0)
        step_r = step_t * max(float(np.max(self.K_t[3:])), 1.0) / k_ref
        lim = np.array([step_t] * 3 + [step_r] * 3)
        self.K_t = self.K_t + np.clip(K_cmd - self.K_t, -lim, lim)
        return self.K_t

    def compute(self, s, cmd: L2Command, xd_used: np.ndarray, payload_tau: np.ndarray) -> ImpedanceOutput:
        c = self.cfg
        e = pose_error(cmd.x_eq, cmd.R_eq, s.tcp_pos, s.tcp_R)
        e = np.concatenate([clip_norm(e[:3], c.delta_max_pos), clip_norm(e[3:], c.delta_max_rot)])
        edot = cmd.xd_eq - xd_used
        K = self.slew(np.asarray(cmd.K, dtype=float))
        Lam = task_inertia(s.M, s.J)
        T = _frame6(cmd.R_K)
        if cmd.D is None:
            lam_k = np.clip(np.diag(T.T @ Lam @ T), 1e-3, None)
            D = 2.0 * c.zeta * np.sqrt(K * lam_k)
        else:
            D = np.asarray(cmd.D, dtype=float)
        Kw = T @ np.diag(K) @ T.T
        Dw = T @ np.diag(D) @ T.T
        F_spring = Kw @ e
        f_lim = np.array([c.f_ff_max[0]] * 3 + [c.f_ff_max[1]] * 3)
        F = F_spring + Dw @ edot + np.clip(cmd.F_ff, -f_lim, f_lim)
        if c.osc_inertia:
            F = F + Lam @ cmd.xdd_ff
        # dynamically consistent null space: N = I - J^T Lambda J M^-1
        Minv = np.linalg.inv(s.M)
        N = np.eye(len(s.q)) - s.J.T @ Lam @ s.J @ Minv
        tau_null = N @ (c.k_null * (self.q_null - s.q) - c.d_null * s.qd)
        tau = s.J.T @ F + tau_null + s.bias + payload_tau
        return ImpedanceOutput(tau, F, F_spring, e, K.copy(), D)
