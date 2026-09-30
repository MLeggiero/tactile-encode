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
