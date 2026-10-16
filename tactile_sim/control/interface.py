"""The L2 -> L1 boundary: pose + stiffness + feedforward wrench + grasp force (report, section "Layers"), and the
per-joint MIT-mode command a dexterous hand's drives take (HandCommand)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Mode(IntEnum):
    FREE = 0  # ordinary tracking
    ANTE = 1  # ante-impact reference (swing, continued through the expected contact)
    INTERIM = 2  # around the predicted contact, before the flag
    POST = 3  # post-impact reference, after the flag
    HOLD = 4  # supervisor fallback: hold the last pose


@dataclass
class L2Command:
    t: float  # time the command was issued
    x_eq: np.ndarray  # equilibrium TCP position (world)
    R_eq: np.ndarray  # equilibrium TCP orientation (world)
    xd_eq: np.ndarray = field(default_factory=lambda: np.zeros(6))  # reference twist [lin; ang]
    xdd_ff: np.ndarray = field(default_factory=lambda: np.zeros(6))  # reference acceleration
    K: np.ndarray = field(default_factory=lambda: np.array([1500.0, 1500, 1500, 60, 60, 60]))  # in R_K frame
    R_K: np.ndarray = field(default_factory=lambda: np.eye(3))  # stiffness frame (strike frame)
    D: np.ndarray | None = None  # damping in R_K frame; None = critically damped from K and task inertia
    F_ff: np.ndarray = field(default_factory=lambda: np.zeros(6))  # feedforward wrench (world)
    F_grip: float = 40.0  # grasp-force setpoint per pad (N)
    mode: Mode = Mode.FREE
    t_c_pred: float = float("nan")  # predicted contact time
    # Optional reference the L1 evaluates at its own rate (a chunk, in VLA terms). It must provide
    # evaluate(t, t_flag, x_now) -> (x, xd, xdd, mode); x_eq/xd_eq/xdd_ff/mode are overwritten from it.
    ref: object | None = None
    payload_ff: bool = True  # add payload inertia x reference acceleration to the feedforward

    def copy(self) -> L2Command:
        return L2Command(self.t, self.x_eq.copy(), self.R_eq.copy(), self.xd_eq.copy(), self.xdd_ff.copy(),
                         self.K.copy(), self.R_K.copy(), None if self.D is None else self.D.copy(),
                         self.F_ff.copy(), self.F_grip, self.mode, self.t_c_pred, self.ref, self.payload_ff)


@dataclass
class HandCommand:
    """The host's MIT-mode command for every hand joint (WUJI Hand 2 realtime controller). Each driver applies
    tau = kp (q - q_meas) + kd (qd - qd_meas) + tau before clamping its current (tactile_sim.sim.wuji_drive)."""

    t: float  # time the host issued it
    q: np.ndarray  # target angle per joint (rad)
    qd: np.ndarray  # target velocity (rad/s)
    kp: np.ndarray  # Nm/rad
    kd: np.ndarray  # Nms/rad
    tau: np.ndarray  # feedforward torque (Nm)
    src: str = ""  # who issued it: "synergy", "base", "residual", "safety", ...

    FIELDS = ("q", "qd", "kp", "kd", "tau")

    @classmethod
    def zeros(cls, t: float, n: int, src: str = "") -> HandCommand:
        """No torque: zero gains and no feedforward."""
        z = np.zeros(n)
        return cls(t, z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), src)

    @classmethod
    def from_array(cls, t: float, x: np.ndarray, src: str = "") -> HandCommand:
        """From a (5, n) array of rows q, qd, kp, kd, tau."""
        return cls(t, *(np.array(r, dtype=float) for r in x), src)

    def array(self) -> np.ndarray:
        """(5, n): rows q, qd, kp, kd, tau."""
        return np.stack([self.q, self.qd, self.kp, self.kd, self.tau]).astype(float)

    def copy(self) -> HandCommand:
        return HandCommand(self.t, self.q.copy(), self.qd.copy(), self.kp.copy(), self.kd.copy(), self.tau.copy(),
                           self.src)
