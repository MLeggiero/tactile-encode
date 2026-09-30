"""The L2 -> L1 boundary: pose + stiffness + feedforward wrench + grasp force (report, section "Layers")."""

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

    def copy(self) -> L2Command:
        return L2Command(self.t, self.x_eq.copy(), self.R_eq.copy(), self.xd_eq.copy(), self.xdd_ff.copy(),
                         self.K.copy(), self.R_K.copy(), None if self.D is None else self.D.copy(),
                         self.F_ff.copy(), self.F_grip, self.mode, self.t_c_pred)
