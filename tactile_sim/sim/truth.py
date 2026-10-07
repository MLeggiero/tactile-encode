"""Ground-truth extractors: contact forces between geom sets and poses the sensors never see."""

from __future__ import annotations

import mujoco
import numpy as np


def pair_force(model: mujoco.MjModel, data: mujoco.MjData, g1: int, g2: int) -> tuple[float, int]:
    """Sum of contact normal forces between two geoms (N) and the number of active contacts."""
    f6 = np.zeros(6)
    total = 0.0
    n = 0
    pair = {g1, g2}
    for i in range(data.ncon):
        c = data.contact[i]
        if {c.geom1, c.geom2} == pair and c.efc_address >= 0:
            mujoco.mj_contactForce(model, data, i, f6)
            if f6[0] > 0:
                total += f6[0]
                n += 1
    return total, n


def contact_wrench(model: mujoco.MjModel, data: mujoco.MjData, body_geoms: frozenset[int],
                   other_geoms: frozenset[int], point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Net contact force (N) and moment about `point` (N m), world frame, that `other_geoms` exert on `body_geoms`.

    MuJoCo's contact normal points from geom1 to geom2, and mj_contactForce gives the force on geom2 in the contact
    frame (rows of `frame`), so geom1 receives the opposite."""
    F = np.zeros(3)
    M = np.zeros(3)
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom2 in body_geoms and c.geom1 in other_geoms:
            sign = 1.0
        elif c.geom1 in body_geoms and c.geom2 in other_geoms:
            sign = -1.0
        else:
            continue
        if c.efc_address < 0:
            continue
        mujoco.mj_contactForce(model, data, i, f6)
        R = c.frame.reshape(3, 3)
        f = sign * (R.T @ f6[:3])
        F += f
        M += np.cross(c.pos - point, f) + sign * (R.T @ f6[3:])
    return F, M


def pulse_stats(t: np.ndarray, f: np.ndarray, frac: float = 0.05, max_gap: float = 0.001) -> dict[str, float]:
    """Peak, impulse and width of a force pulse.

    The width is the span of samples above `frac` of the peak around the peak, where drops shorter
    than `max_gap` are bridged: a yielding nail makes the contact chatter (the light nail runs ahead
    of the hammer for a sample or two), which is one physical blow, not several.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    if len(f) == 0 or np.max(f) <= 0:
        return {"peak": 0.0, "impulse": 0.0, "width": 0.0, "t_peak": float("nan"), "t_start": float("nan"),
                "t_end": float("nan")}
    k = int(np.argmax(f))
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    above = f > frac * f[k]
    max_skip = max(1, int(round(max_gap / dt))) if dt > 0 else 1

    def extend(i: int, step: int) -> int:
        last = i
        j = i + step
        miss = 0
        while 0 <= j < len(f):
            if above[j]:
                last, miss = j, 0
            else:
                miss += 1
                if miss > max_skip:
                    break
            j += step
        return last

    lo, hi = extend(k, -1), extend(k, +1)
    a, b = max(lo - 1, 0), min(hi + 2, len(f))  # include the rising and falling edges
    return {"peak": float(f[k]), "impulse": float(np.trapezoid(f[a:b], t[a:b])),
            "width": float((hi - lo + 1) * dt), "t_peak": float(t[k]), "t_start": float(t[lo]),
            "t_end": float(t[hi])}
