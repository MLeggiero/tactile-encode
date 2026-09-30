"""Joint-level control of a dexterous hand (WUJI Hand 2: MIT mode, 1 kHz).

Per joint: tau = kp (q_ref - q) - kd qd + tau_ff, clipped to the joint's rated torque. The wrap grasp is
one synergy: every closing joint gets tau_ff = s * dir * tau_max, where the grip scalar s (0..1) comes
from the grasp-force loop; joints that only shape the hand (finger abduction) hold their keyframe angle
with a PD term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THUMB_JOINTS = ("r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_thumb_mcp", "r_thumb_ip")


@dataclass
class WrapSynergy:
    joints: list[str]
    tau_max: np.ndarray  # rated torque per joint (Nm)
    close_dir: np.ndarray  # +1 / -1 closes the joint onto the tool; 0 = shaping joint (PD hold)
    q_hold: np.ndarray  # keyframe angles for the shaping joints
    kp: float = 2.0  # Nm/rad on shaping joints
    kd: float = 0.01  # Nms/rad on every joint

    @classmethod
    def for_wuji(cls, joints: list[tuple[str, float]], thumb_close: tuple[int, int, int, int],
                 thumb_preshape: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> WrapSynergy:
        names = [j for j, _ in joints]
        tau_max = np.array([lim for _, lim in joints])
        thumb = dict(zip(THUMB_JOINTS, thumb_close, strict=True))
        d = np.array([thumb[j] if j in thumb else (0 if j.endswith("_abd") else 1) for j in names], dtype=float)
        pre = dict(zip(THUMB_JOINTS, thumb_preshape, strict=True))
        q_hold = np.array([pre.get(j, 0.0) for j in names])
        return cls(names, tau_max, d, q_hold)

    def torque(self, s: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        s = float(np.clip(s, 0.0, 1.0))
        shaping = self.close_dir == 0
        tau = s * self.close_dir * self.tau_max - self.kd * qd
        tau[shaping] += self.kp * (self.q_hold[shaping] - q[shaping])
        return np.clip(tau, -self.tau_max, self.tau_max)
