"""1-D quintic segments and straight-line path references along a fixed direction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quintic_coeffs(p0: float, v0: float, a0: float, p1: float, v1: float, a1: float, T: float) -> np.ndarray:
    """Coefficients c0..c5 of p(t) = sum c_k t^k meeting position/velocity/acceleration at t=0 and t=T."""
    A = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5],
        [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
        [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
    ], dtype=float)
    return np.linalg.solve(A, np.array([p0, v0, a0, p1, v1, a1], dtype=float))


def eval_poly(c: np.ndarray, t: float) -> tuple[float, float, float]:
    p = c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))))
    v = c[1] + t * (2 * c[2] + t * (3 * c[3] + t * (4 * c[4] + t * 5 * c[5])))
    a = 2 * c[2] + t * (6 * c[3] + t * (12 * c[4] + t * 20 * c[5]))
    return float(p), float(v), float(a)


def min_jerk(p0: float, p1: float, T: float) -> np.ndarray:
    return quintic_coeffs(p0, 0.0, 0.0, p1, 0.0, 0.0, T)


@dataclass
class Segment:
    t0: float
    T: float
    c: np.ndarray

    def eval(self, t: float) -> tuple[float, float, float]:
        return eval_poly(self.c, min(max(t - self.t0, 0.0), self.T))

    @property
    def t_end(self) -> float:
        return self.t0 + self.T


class LinePath:
    """p(t) = origin + direction * s(t), with s(t) given by consecutive segments; fixed orientation.

    Before the first segment the path holds its start; after the last it holds its end
    (velocity/acceleration zero unless `extrapolate` continues at the final velocity).
    """

    def __init__(self, origin: np.ndarray, direction: np.ndarray, R: np.ndarray, segments: list[Segment],
                 extrapolate: bool = False):
        self.origin = np.asarray(origin, dtype=float)
        self.dir = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        self.R = np.asarray(R, dtype=float)
        self.segments = segments
        self.extrapolate = extrapolate

    @property
    def t_end(self) -> float:
        return self.segments[-1].t_end

    def s(self, t: float) -> tuple[float, float, float]:
        segs = self.segments
        if t <= segs[0].t0:
            p, _, _ = segs[0].eval(segs[0].t0)
            return p, 0.0, 0.0
        for seg in segs:
            if t <= seg.t_end:
                return seg.eval(t)
        last = segs[-1]
        p, v, _ = last.eval(last.t_end)
        if self.extrapolate:
            return p + v * (t - last.t_end), v, 0.0
        return p, 0.0, 0.0

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p, v, a = self.s(t)
        x = self.origin + self.dir * p
        xd = np.concatenate([self.dir * v, np.zeros(3)])
        xdd = np.concatenate([self.dir * a, np.zeros(3)])
        return x, xd, xdd


def swing_duration(distance: float, v_end: float, a_max: float) -> float:
    """Duration of a rest-to-v_end quintic over `distance` whose peak acceleration stays under a_max."""
    T = 1.6 * distance / max(v_end, 1e-6)
    for _ in range(60):
        c = quintic_coeffs(0.0, 0.0, 0.0, distance, v_end, 0.0, T)
        ts = np.linspace(0.0, T, 200)
        a_pk = max(abs(eval_poly(c, t)[2]) for t in ts)
        v_pk = max(eval_poly(c, t)[1] for t in ts)
        ok = a_pk <= a_max and v_pk <= 1.25 * v_end and min(eval_poly(c, t)[1] for t in ts) >= -1e-6
        if ok:
            return T
        T *= 1.05
    return T


def strike_profile(distance: float, v_c: float, a_brake: float, a_max: float,
                   v_max: float) -> tuple[float, np.ndarray] | None:
    """Rest-to-contact quintic over `distance` that arrives at `v_c` already decelerating at `a_brake`.

    Arriving while braking means the joint torques have begun to reverse before the blow: the FR3 may change
    its torque by at most 1000 Nm/s, so a swing that is still accelerating at contact keeps pushing the arm
    into the nail (and the tool through the grasp) for the ~100 ms it takes to reverse. Returns (T, coeffs)
    with the peak acceleration under `a_max`, the peak speed under `v_max` and the speed never negative, or
    None when no duration satisfies them (the caller lowers v_c).
    """
    if distance <= 0 or v_c <= 0:
        return None
    T = 0.8 * distance / v_c
    while T < 20.0 * distance / v_c:
        c = quintic_coeffs(0.0, 0.0, 0.0, distance, v_c, -a_brake, T)
        pva = np.array([eval_poly(c, t) for t in np.linspace(0.0, T, 200)])
        if pva[:, 1].min() < -1e-6:
            return None  # longer swings only dip further back
        if np.abs(pva[:, 2]).max() <= a_max and pva[:, 1].max() <= v_max:
            return T, c
        T *= 1.03
    return None


def brake_profile(v0: float, a0: float) -> tuple[float, float]:
    """Follow-through after the predicted contact: from speed v0 and deceleration a0 (> 0) to rest.

    Returns (T, distance) of a quintic (p: 0 -> d, v: v0 -> 0, a: -a0 -> 0) whose speed never reverses.
    """
    T = 1.5 * v0 / max(a0, 1e-6)
    for _ in range(60):
        for frac in np.linspace(0.40, 0.65, 26):
            d = frac * v0 * T
            c = quintic_coeffs(0.0, v0, -a0, d, 0.0, 0.0, T)
            vs = [eval_poly(c, t)[1] for t in np.linspace(0.0, T, 100)]
            if min(vs) >= -1e-6:
                return T, d
        T *= 1.05
    return T, 0.5 * v0 * T
