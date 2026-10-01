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


def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


class StrikeGeometry:
    """The strike's path, parameterised by s = distance the hammer face travels; s = 0 is the hover pose and
    s = s_c0 (the hover clearance) the nominal contact, where the tool has its nominal orientation R_c.

    radius = 0: a straight line along the strike axis, orientation held.
    radius = L > 0: an arc. The tool turns about an axis through a pivot L behind the face along the handle,
    perpendicular to both the handle and the strike, so the face moves on a circle of radius L and meets the
    nail square, moving along the strike axis. The farther the pivot, the more the shoulder and elbow drive the
    swing (and the less the wrist does).
    """

    def __init__(self, contact_tcp: np.ndarray, R_c: np.ndarray, face_local: np.ndarray, s_c0: float,
                 radius: float = 0.0):
        self.x_c = np.asarray(contact_tcp, float)
        self.R_c = np.asarray(R_c, float)
        self.axis = -self.R_c[:, 1]
        self.s_c0 = float(s_c0)
        self.L = float(radius)
        if self.L > 0:
            face_c = self.x_c + self.R_c @ np.asarray(face_local, float)
            self.pivot = face_c + self.L * self.R_c[:, 0]
            self.a = self.R_c[:, 2]  # +rotation moves the face along the strike axis

    def shifted(self, dx: np.ndarray) -> StrikeGeometry:
        g = object.__new__(StrikeGeometry)
        g.__dict__.update(self.__dict__)
        g.x_c = self.x_c + dx
        if self.L > 0:
            g.pivot = self.pivot + dx
        return g

    def pose(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        if self.L <= 0:
            return self.x_c + self.axis * (s - self.s_c0), self.R_c
        Rt = _rot(self.a, (s - self.s_c0) / self.L)
        return self.pivot + Rt @ (self.x_c - self.pivot), Rt @ self.R_c

    def tangent(self, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """d(tcp)/ds, angular velocity per unit s, and d^2(tcp)/ds^2 at s."""
        if self.L <= 0:
            return self.axis, np.zeros(3), np.zeros(3)
        x, _ = self.pose(s)
        r = x - self.pivot
        w = self.a / self.L
        return np.cross(w, r), w, np.cross(w, np.cross(w, r))

    def s_of(self, x: np.ndarray) -> float:
        """Path parameter of the point on the path nearest the TCP position x."""
        if self.L <= 0:
            return self.s_c0 + float((x - self.x_c) @ self.axis)
        r0, r = self.x_c - self.pivot, x - self.pivot
        r = r - (r @ self.a) * self.a
        ang = np.arctan2(float(np.cross(r0, r) @ self.a), float(r0 @ r))
        return self.s_c0 + self.L * ang


class StrikePath:
    """A StrikeGeometry driven by 1-D segments in s; exposes the current orientation as `R`."""

    def __init__(self, geom: StrikeGeometry, segments: list[Segment]):
        self.geom = geom
        self.segments = segments
        self.R = geom.R_c

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
        p, _, _ = segs[-1].eval(segs[-1].t_end)
        return p, 0.0, 0.0

    def rotation(self, t: float) -> np.ndarray:
        return self.geom.pose(self.s(t)[0])[1]

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s, v, a = self.s(t)
        x, self.R = self.geom.pose(s)
        dx, w, ddx = self.geom.tangent(s)
        xd = np.concatenate([dx * v, w * v])
        xdd = np.concatenate([ddx * v * v + dx * a, w * a])
        return x, xd, xdd
