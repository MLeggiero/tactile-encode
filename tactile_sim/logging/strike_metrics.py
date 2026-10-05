"""Per-strike force-truth metrics: impact pulse, nail advance, tool slip in the grasp, flag timing."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


def rotvec_diff(rv0: np.ndarray, rv1: np.ndarray) -> float:
    """Angle (rad) between two orientations given as rotation vectors."""
    q0, q1, q0i, dq = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(q0, rv0 / (np.linalg.norm(rv0) or 1.0), float(np.linalg.norm(rv0)))
    mujoco.mju_axisAngle2Quat(q1, rv1 / (np.linalg.norm(rv1) or 1.0), float(np.linalg.norm(rv1)))
    mujoco.mju_negQuat(q0i, q0)
    mujoco.mju_mulQuat(dq, q1, q0i)
    return float(2.0 * np.arccos(np.clip(abs(dq[0]), 0.0, 1.0)))


def grasp_slip(p0: np.ndarray, rv0: np.ndarray, p1: np.ndarray, rv1: np.ndarray) -> tuple[float, float]:
    """Translational (m) and rotational (rad) change of the tool pose in the TCP frame."""
    return float(np.linalg.norm(p1 - p0)), rotvec_diff(rv0, rv1)


STRIKE_FIELDS = [
    ("idx", np.int32), ("t_swing_start", np.float64), ("t_c_pred", np.float64), ("t_contact_truth", np.float64),
    ("t_flag", np.float64), ("flag_latency", np.float64), ("flag_source", "S8"), ("hit", np.bool_),
    ("peak_force_truth", np.float64), ("impulse", np.float64), ("pulse_width", np.float64),
    ("peak_ft_meas", np.float64), ("depth_before", np.float64), ("depth_after", np.float64),
    ("depth_inc", np.float64), ("slip_trans", np.float64), ("slip_rot", np.float64), ("drop", np.bool_),
    ("peak_joint_torque", np.float64, (7,)), ("v_cmd", np.float64), ("v_strike_actual", np.float64),
    ("v_tcp", np.float64),
    ("grip_at_contact", np.float64),
    ("grip_peak", np.float64), ("t_grip_peak_rel", np.float64), ("t_grip_ramp_start_rel", np.float64),
    ("ringing_energy", np.float64), ("pre_energy", np.float64),
    # worst hardware-limit ratios during the strike (1.0 = at the limit; see tactile_sim.limits)
    ("limit_arm_torque_rate", np.float64), ("limit_arm_velocity", np.float64), ("limit_hand_stop_load", np.float64),
]
STRIKE_DTYPE = np.dtype(STRIKE_FIELDS)


@dataclass
class StrikeRecord:
    idx: int
    t_swing_start: float
    t_c_pred: float = float("nan")
    t_contact_truth: float = float("nan")
    t_flag: float = float("nan")
    flag_source: str = ""
    hit: bool = False
    peak_force_truth: float = 0.0
    impulse: float = 0.0
    pulse_width: float = 0.0
    peak_ft_meas: float = 0.0
    depth_before: float = 0.0
    depth_after: float = 0.0
    slip_trans: float = 0.0
    slip_rot: float = 0.0
    drop: bool = False
    peak_joint_torque: np.ndarray = field(default_factory=lambda: np.zeros(7))
    v_cmd: float = 0.0  # commanded strike speed (the first strike may be a setting tap)
    v_strike_actual: float = 0.0  # hammer face speed along the strike axis at contact
    v_tcp: float = 0.0  # hand (TCP) speed along the strike axis at contact
    grip_at_contact: float = float("nan")
    grip_peak: float = float("nan")
    t_grip_peak_rel: float = float("nan")
    t_grip_ramp_start_rel: float = float("nan")
    ringing_energy: float = float("nan")
    pre_energy: float = float("nan")
    limits: dict = field(default_factory=dict)

    @property
    def limit_arm_torque_rate(self) -> float:
        return self.limits.get("arm_torque_rate", float("nan"))

    @property
    def limit_arm_velocity(self) -> float:
        return self.limits.get("arm_velocity", float("nan"))

    @property
    def limit_hand_stop_load(self) -> float:
        return self.limits.get("hand_stop_load", float("nan"))

    @property
    def flag_latency(self) -> float:
        return self.t_flag - self.t_contact_truth

    @property
    def depth_inc(self) -> float:
        return self.depth_after - self.depth_before

    def as_row(self) -> tuple:
        vals = []
        for f in STRIKE_FIELDS:
            v = getattr(self, f[0])
            vals.append(v.encode()[:8] if isinstance(v, str) else v)
        return tuple(vals)


def to_array(records: list[StrikeRecord]) -> np.ndarray:
    return np.array([r.as_row() for r in records], dtype=STRIKE_DTYPE)
