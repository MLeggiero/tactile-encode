"""Joint-level control of a dexterous hand (WUJI Hand 2: MIT mode, 1 kHz).

Per joint: tau = kp (q_ref - q) - kd qd + tau_ff, clipped to the joint's rated torque. The wrap grasp is
one synergy: every closing joint gets tau_ff = s * dir * w * tau_max, where the grip scalar s (0..1) comes
from the grasp-force loop; joints that only shape the hand (finger abduction) hold their keyframe angle
with a PD term.

The weights w close a finger the way a human hand does ("tendon" synergy). A finger is flexed by two tendons,
the deep flexor (over the knuckle, middle and fingertip joints) and the superficial flexor (over the knuckle
and middle joints); a joint's torque is the tendon tension times its moment arm, ~10 / 7.5 / 5 mm (deep) and
~10 / 7 mm (superficial) at MCP / PIP / DIP, so equal tension in both gives MCP : PIP : DIP = 1 : 0.73 : 0.25.
That is scaled to the joints' ratings (the PIP's 0.3 Nm binds). Driving each joint at its full rating instead
("rated") folds the finger flat at the knuckle (2.0 Nm against 0.3 Nm) and hooks the tip, which no human grip
does. The fingertip joint is also coupled to the middle joint (q_DIP ~ 2/3 q_PIP, the linkage of a human
finger's two distal joints), so a tip neither hooks before the middle segment reaches the tool nor bends back.
The thumb keeps its rated closing torques: its intrinsic muscles drive each joint directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THUMB_JOINTS = ("r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_thumb_mcp", "r_thumb_ip")
# finger flexion torque ratios from the two flexor tendons' moment arms (MCP, PIP, DIP)
TENDON_RATIO = {"mcp_flex": 1.0, "pip": (7.5 + 7.0) / 20.0, "dip": 5.0 / 20.0}


def tendon_weights(names: list[str], tau_max: np.ndarray) -> np.ndarray:
    """Closing torque per joint as a fraction of its rating: each finger's MCP, PIP and DIP in the tendon ratio,
    scaled so the most constrained joint gets its full rating. Other joints get 1."""
    w = np.ones(len(names))
    fingers = {n[: -len("_pip")] for n in names if n.endswith("_pip")}
    for f in fingers:
        idx = {k: names.index(f"{f}_{k}") for k in TENDON_RATIO if f"{f}_{k}" in names}
        scale = min(tau_max[i] / TENDON_RATIO[k] for k, i in idx.items())
        for k, i in idx.items():
            w[i] = scale * TENDON_RATIO[k] / tau_max[i]
    return w


@dataclass
class WrapSynergy:
    joints: list[str]
    tau_max: np.ndarray  # rated torque per joint (Nm)
    close_dir: np.ndarray  # +1 / -1 closes the joint onto the tool; 0 = shaping joint (PD hold)
    q_hold: np.ndarray  # keyframe angles for the shaping joints
    kp: float = 2.0  # Nm/rad on shaping joints
    kd: float = 0.01  # Nms/rad on every joint
    weight: np.ndarray | None = None  # closing torque as a fraction of the rating, per joint (None = 1)
    # DIP follows its PIP (q_dip = ratio * q_pip, as a human finger's linked joints do): (dip index, pip index)
    coupled: list[tuple[int, int]] | None = None
    coupling_ratio: float = 2.0 / 3.0
    coupling_kp: float = 3.0  # Nm/rad

    @classmethod
    def for_wuji(cls, joints: list[tuple[str, float]], thumb_close: tuple[int, int, int, int],
                 thumb_preshape: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
                 synergy: str = "tendon") -> WrapSynergy:
        names = [j for j, _ in joints]
        tau_max = np.array([lim for _, lim in joints])
        thumb = dict(zip(THUMB_JOINTS, thumb_close, strict=True))
        d = np.array([thumb[j] if j in thumb else (0 if j.endswith("_abd") else 1) for j in names], dtype=float)
        pre = dict(zip(THUMB_JOINTS, thumb_preshape, strict=True))
        q_hold = np.array([pre.get(j, 0.0) for j in names])
        if synergy not in ("tendon", "rated"):
            raise ValueError(f"unknown finger synergy {synergy!r}")
        w = tendon_weights(names, tau_max) if synergy == "tendon" else None
        coupled = None
        if synergy == "tendon":
            coupled = [(names.index(n), names.index(n[: -len("_dip")] + "_pip")) for n in names
                       if n.endswith("_dip") and n[: -len("_dip")] + "_pip" in names]
        return cls(names, tau_max, d, q_hold, weight=w, coupled=coupled)

    def torque(self, s: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        s = float(np.clip(s, 0.0, 1.0))
        shaping = self.close_dir == 0
        w = 1.0 if self.weight is None else self.weight
        tau = s * self.close_dir * w * self.tau_max - self.kd * qd
        tau[shaping] += self.kp * (self.q_hold[shaping] - q[shaping])
        for i, j in self.coupled or ():
            # the fingertip joint is driven to its share of the middle joint's flexion, plus its tendon share
            tau[i] += self.coupling_kp * (self.coupling_ratio * q[j] - q[i])
        return np.clip(tau, -self.tau_max, self.tau_max)
