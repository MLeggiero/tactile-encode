"""Planned paths for the varied VLA demos (vla.variation): the windup curve of a strike and the arm's transit from a
start pose to the task.

CompositeStrikeGeometry: a StrikeGeometry's terminal stroke (the line or arc that meets the nail square), preceded by
a free cubic Hermite curve from a windup apex anywhere behind the hover pose. The curve joins the terminal stroke at
s_join (<= 0, on the hover side of the hover pose) with a continuous tangent, so the face still arrives along the
strike axis. It keeps StrikeGeometry's parameterisation (s = face distance along the path, s = 0 at the hover pose,
s_c0 at the nominal contact, the apex at negative s) and its interface (pose, tangent, s_of, shifted), so ScriptedSwing,
reference spreading and the residual's lookahead drive it unchanged. On an arc the TCP, nearer the pivot than the face,
moves k = |d(tcp)/ds| < 1 per unit s; the curve keeps that rate (its TCP arc length is k s), so the TCP's speed is
continuous at the junction and a curve along the stroke's own extension spans the same s as the stroke would. The
orientation turns uniformly along the curve from the apex's to the junction's. `straight` is False: the face moves
rigidly with the hand along a curve, never as a pure translation along the strike axis.

TransitPath: a time-parameterised move between two poses (a quadratic Bezier through a sideways via point, min-jerk
in time, the orientation turned about one axis), for the approach from a sampled start pose to the hover pose.
"""

from __future__ import annotations

import numpy as np

from tactile_sim.behaviors.trajectories import StrikeGeometry


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix."""
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    ang = np.arccos(c)
    if ang < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - ang < 1e-6:  # near pi: take the axis from the symmetric part
        B = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.maximum(np.diag(B), 0.0))
        k *= np.sign(w) + (w == 0)
        return ang * k / np.linalg.norm(k)
    return ang * w / (2.0 * np.sin(ang))


def rotmat(v: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector (Rodrigues)."""
    v = np.asarray(v, float)
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.eye(3)
    k = v / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def is_straight(geom) -> bool:
    """Whether the face moves as a pure translation along the strike axis on this geometry."""
    straight = getattr(geom, "straight", None)
    return geom.L <= 0 if straight is None else bool(straight)


class CompositeStrikeGeometry:
    straight = False

    def __init__(self, terminal: StrikeGeometry, apex: np.ndarray, R_apex: np.ndarray, s_join: float,
                 n_table: int = 400):
        if s_join > 1e-12:
            raise ValueError(f"the windup curve must join the terminal stroke at or before the hover pose (s_join "
                             f"{s_join:.3f} > 0)")
        self.terminal = terminal
        self.s_join = float(s_join)
        self.x_c, self.R_c, self.axis = terminal.x_c, terminal.R_c, terminal.axis
        self.s_c0, self.L = terminal.s_c0, terminal.L
        self.offset = np.zeros(3)
        p1, self.R_join = terminal.pose(self.s_join)
        t1, _, _ = terminal.tangent(self.s_join)
        p0 = np.asarray(apex, float)
        chord = p1 - p0
        d = float(np.linalg.norm(chord))
        if d < 1e-4:
            raise ValueError("the windup apex coincides with the junction")
        if float(chord @ t1) <= 0.0:
            raise ValueError("the windup apex is ahead of the junction: the curve would run backward into the stroke")
        # Hermite end tangents: leave the apex toward the junction, arrive along the stroke's tangent; both scaled by
        # the chord, so an apex on a straight stroke's backward extension gives the stroke itself
        self._p0, self._p1 = p0, p1
        self.k = float(np.linalg.norm(t1))  # the stroke's TCP travel per unit s at the junction (1 on a line)
        self._m0, self._m1 = chord, d * t1 / self.k
        self._phi = rotvec(self.R_join @ np.asarray(R_apex, float).T)  # world-frame turn from the apex to the junction
        self._R0 = np.asarray(R_apex, float)
        u = np.linspace(0.0, 1.0, n_table)
        pts = self._x(u)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self._u, self._len = u, np.concatenate([[0.0], np.cumsum(seg)])  # TCP arc length along the curve
        self.length = float(self._len[-1])
        self.s_apex = self.s_join - self.length / self.k
        self._pts = pts

    # -- the Hermite curve in u in [0, 1] (apex at 0, junction at 1) ------------------------------------------------
    def _x(self, u):
        u = np.asarray(u, float)[..., None]
        h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
        h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
        return h00 * self._p0 + h10 * self._m0 + h01 * self._p1 + h11 * self._m1

    def _dx(self, u: float) -> np.ndarray:
        return ((6 * u**2 - 6 * u) * self._p0 + (3 * u**2 - 4 * u + 1) * self._m0 + (-6 * u**2 + 6 * u) * self._p1
                + (3 * u**2 - 2 * u) * self._m1)

    def _ddx(self, u: float) -> np.ndarray:
        return (12 * u - 6) * self._p0 + (6 * u - 4) * self._m0 + (-12 * u + 6) * self._p1 + (6 * u - 2) * self._m1

    def _u_of(self, s: float) -> float:
        """Curve parameter at path parameter s (< s_join), interpolated in the arc-length table."""
        return float(np.interp((s - self.s_apex) * self.k, self._len, self._u))

    def _R(self, u: float) -> np.ndarray:
        return rotmat(u * self._phi) @ self._R0

    # -- StrikeGeometry's interface ---------------------------------------------------------------------------------
    def shifted(self, dx: np.ndarray) -> CompositeStrikeGeometry:
        g = object.__new__(CompositeStrikeGeometry)
        g.__dict__.update(self.__dict__)
        g.terminal = self.terminal.shifted(dx)
        g.x_c = g.terminal.x_c
        g.offset = self.offset + np.asarray(dx, float)
        return g

    def pose(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        if s >= self.s_join:
            return self.terminal.pose(s)
        u = self._u_of(s)
        return self._x(u) + self.offset, self._R(u)

    def tangent(self, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """d(tcp)/ds, angular velocity per unit s, and d^2(tcp)/ds^2 at s."""
        if s >= self.s_join:
            return self.terminal.tangent(s)
        u = self._u_of(s)
        d1, d2 = self._dx(u), self._ddx(u)
        sp = float(np.linalg.norm(d1))
        t = d1 / sp
        curv = (d2 - (d2 @ t) * t) / sp**2  # per unit TCP arc length; the TCP moves k per unit s
        return self.k * t, self._phi * self.k / sp, self.k**2 * curv  # du/ds = k / |dx/du|

    def s_of(self, x: np.ndarray) -> float:
        """Path parameter of the point on the path nearest the TCP position x."""
        x = np.asarray(x, float)
        s_t = max(self.terminal.s_of(x), self.s_join)
        d_t = float(np.linalg.norm(self.terminal.pose(s_t)[0] - x))
        dist = np.linalg.norm(self._pts + self.offset - x, axis=1)
        i = int(np.argmin(dist))
        if dist[i] < d_t:
            return float(self.s_apex + self._len[i] / self.k)
        return float(s_t)


class TransitPath:
    """From pose (p0, R0) at t0 to pose (p1, R1) at t0 + T: a quadratic Bezier through the control point (midpoint +
    `via`), min-jerk in time, the orientation turned about one world axis. Holds the end pose after T. Exposes the
    current orientation as `R`, as StrikePath does."""

    def __init__(self, t0: float, T: float, p0: np.ndarray, R0: np.ndarray, p1: np.ndarray, R1: np.ndarray,
                 via: np.ndarray | None = None):
        if T <= 0:
            raise ValueError("a transit needs a positive duration")
        self.t0, self.T = float(t0), float(T)
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        self.c = 0.5 * (self.p0 + self.p1) + (np.zeros(3) if via is None else np.asarray(via, float))
        self.R0 = np.asarray(R0, float)
        self.phi = rotvec(np.asarray(R1, float) @ self.R0.T)
        self.R = self.R0.copy()

    @property
    def t_end(self) -> float:
        return self.t0 + self.T

    def _u(self, t: float) -> tuple[float, float, float]:
        tau = min(max((t - self.t0) / self.T, 0.0), 1.0)
        u = tau**3 * (10 - 15 * tau + 6 * tau**2)
        if tau <= 0.0 or tau >= 1.0:
            return u, 0.0, 0.0
        ud = 30 * tau**2 * (1 - tau) ** 2 / self.T
        udd = 60 * tau * (1 - tau) * (1 - 2 * tau) / self.T**2
        return u, ud, udd

    def pose(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """The pose at t, without touching `R` (for lookahead)."""
        u = self._u(t)[0]
        x = (1 - u) ** 2 * self.p0 + 2 * (1 - u) * u * self.c + u**2 * self.p1
        return x, rotmat(u * self.phi) @ self.R0

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u, ud, udd = self._u(t)
        p0, c, p1 = self.p0, self.c, self.p1
        x = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * c + u**2 * p1
        dB = 2 * (1 - u) * (c - p0) + 2 * u * (p1 - c)
        ddB = 2 * (p1 - 2 * c + p0)
        self.R = rotmat(u * self.phi) @ self.R0
        xd = np.concatenate([dB * ud, self.phi * ud])
        xdd = np.concatenate([ddB * ud**2 + dB * udd, self.phi * udd])
        return x, xd, xdd
