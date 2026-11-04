"""Joint-space paths for the planned VLA demos (vla.variation): RRT-Connect around obstacles, shortcutting, a smooth
spline, and timing within the joint velocity limits.

The planner only sees a validity check, valid(q) -> bool (behaviors.feasibility.ArmProbe.valid in production: joint
margin and clearance to the table, board and robot body, the tool carried in the hand), so it is testable without
MuJoCo. Every edge is checked at EDGE_RES in joint space. The result is a JointPath: a cubic spline through the
waypoints, re-checked densely (spans that clip an obstacle are refitted through their midpoints, and as a last resort
the path falls back to the straight-segment polyline the planner proved valid), driven along its chord length by a
min-jerk progress. Its duration puts the peak joint speed at SPEED_MARGIN of the limit, slowed by a speed fraction.

A JointPath is the joint stream PositionL1 consumes (stream(t) -> (q, qd), holding its ends outside [t0, t0 + T]), so
the motion that is executed is the one that was checked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# the pre-roll, approach and windup are timed so no joint's target moves faster than SPEED_MARGIN of its velocity
# limit (the servos overshoot their targets; PositionL1 itself caps targets at 90 %): a sampled duration too short for
# its path is stretched to fit, up to MAX_STRETCH times
SPEED_MARGIN = 0.7
MAX_STRETCH = 2.5
MIN_JERK_PEAK = 1.875  # a min-jerk move's peak speed over its mean

STEP = 0.2  # rad, the longest tree extension
EDGE_RES = 0.02  # rad, the spacing of validity checks along an edge
MAX_ITERS = 4000
MAX_TIME = 30.0  # s of wall time
GOAL_BIAS = 0.15  # chance a sample is the other tree's root
SHORTCUTS = 120
REFITS = 4
MIN_DURATION = 0.4  # s


def edge_ok(a: np.ndarray, b: np.ndarray, valid, res: float = EDGE_RES) -> bool:
    """Whether every configuration on the straight segment a -> b (a itself excluded) is valid, checked every `res`."""
    n = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / res)))
    return all(valid(a + (b - a) * (k / n)) for k in range(1, n + 1))


@dataclass
class _Tree:
    nodes: list[np.ndarray]
    parent: list[int]

    def nearest(self, q: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(np.asarray(self.nodes) - q, axis=1)))

    def add(self, q: np.ndarray, parent: int) -> int:
        self.nodes.append(q)
        self.parent.append(parent)
        return len(self.nodes) - 1

    def branch(self, i: int) -> list[np.ndarray]:
        out = []
        while i >= 0:
            out.append(self.nodes[i])
            i = self.parent[i]
        return out[::-1]


@dataclass
class PlanResult:
    waypoints: list[np.ndarray] | None
    reason: str  # ok | start_invalid | goal_invalid | budget
    iterations: int = 0
    wall_s: float = 0.0
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.waypoints is not None


def rrt_connect(q_start, q_goal, valid, lo, hi, rng: np.random.Generator, step: float = STEP,
                res: float = EDGE_RES, max_iters: int = MAX_ITERS, max_time: float = MAX_TIME) -> PlanResult:
    """A collision-free polyline from q_start to q_goal (both included), or the reason there is none."""
    t0 = time.perf_counter()
    q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if not valid(q_start):
        return PlanResult(None, "start_invalid")
    if not valid(q_goal):
        return PlanResult(None, "goal_invalid")
    if edge_ok(q_start, q_goal, valid, res):
        return PlanResult([q_start, q_goal], "ok", 0, time.perf_counter() - t0)
    a, b = _Tree([q_start], [-1]), _Tree([q_goal], [-1])
    a_is_start = True

    def extend(tree: _Tree, target: np.ndarray) -> tuple[str, int]:
        i = tree.nearest(target)
        q = tree.nodes[i]
        d = target - q
        n = float(np.max(np.abs(d)))
        reached = n <= step
        q_new = target if reached else q + d * (step / n)
        if not edge_ok(q, q_new, valid, res):
            return "trapped", i
        return ("reached" if reached else "advanced"), tree.add(q_new, i)

    for it in range(1, max_iters + 1):
        if time.perf_counter() - t0 > max_time:
            return PlanResult(None, "budget", it, time.perf_counter() - t0)
        target = b.nodes[0] if rng.uniform() < GOAL_BIAS else rng.uniform(lo, hi)
        status, i_new = extend(a, target)
        if status != "trapped":
            q_new = a.nodes[i_new]
            while True:  # connect: b grows toward q_new until it reaches it or is blocked
                s_b, j = extend(b, q_new)
                if s_b != "advanced":
                    break
            if s_b == "reached":
                path_a, path_b = a.branch(i_new), b.branch(j)[::-1]
                path = path_a + path_b[1:]
                if not a_is_start:
                    path = path[::-1]
                return PlanResult(path, "ok", it, time.perf_counter() - t0,
                                  {"nodes": len(a.nodes) + len(b.nodes)})
        a, b = b, a
        a_is_start = not a_is_start
    return PlanResult(None, "budget", max_iters, time.perf_counter() - t0)


def shortcut(path: list[np.ndarray], valid, rng: np.random.Generator, iters: int = SHORTCUTS,
             res: float = EDGE_RES) -> list[np.ndarray]:
    """Random shortcutting: replace the stretch between two waypoints by a straight edge wherever it is valid. Never
    lengthens the path."""
    path = [np.asarray(q, float) for q in path]
    for _ in range(iters):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if edge_ok(path[i], path[j], valid, res):
            path = path[:i + 1] + path[j:]
    return path


def path_length(path) -> float:
    q = np.asarray(path, float)
    return float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1)))


class JointPath:
    """A joint-space path through `waypoints`, timed: stream(t) -> (q, qd). Between t0 and t0 + T a min-jerk progress
    drives the chord-length parameter of a cubic spline through the waypoints ("spline"), or of the polyline itself
    ("polyline"); before t0 it holds the first waypoint, after t0 + T the last."""

    def __init__(self, waypoints, T: float, t0: float = 0.0, shape: str = "spline"):
        self.waypoints = [np.asarray(q, float) for q in waypoints]
        if len(self.waypoints) < 2:
            self.waypoints = [self.waypoints[0], self.waypoints[0].copy()]
        if T <= 0:
            raise ValueError("a joint path needs a positive duration")
        self.T, self.t0, self.shape = float(T), float(t0), shape
        Q = np.asarray(self.waypoints)
        seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        keep = np.concatenate([[True], seg > 1e-9])
        Q = Q[keep]
        self._Q = Q
        self._u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
        self.length = float(self._u[-1])
        self._spline = None
        if shape == "spline" and len(Q) >= 3:
            from scipy.interpolate import CubicSpline

            self._spline = CubicSpline(self._u, Q, axis=0, bc_type="natural")

    @property
    def t_end(self) -> float:
        return self.t0 + self.T

    @property
    def q_start(self) -> np.ndarray:
        return self._Q[0].copy()

    @property
    def q_end(self) -> np.ndarray:
        return self._Q[-1].copy()

    def retimed(self, t0: float) -> JointPath:
        return JointPath(self.waypoints, self.T, t0, self.shape)

    def stretched(self, T: float) -> JointPath:
        return JointPath(self.waypoints, T, self.t0, self.shape)

    def geometry(self, u: float) -> tuple[np.ndarray, np.ndarray]:
        """(q, dq/du) at chord-length parameter u."""
        u = min(max(u, 0.0), self.length)
        if self.length <= 0.0:
            return self._Q[0].copy(), np.zeros(self._Q.shape[1])
        if self._spline is not None:
            return self._spline(u), self._spline(u, 1)
        k = int(np.clip(np.searchsorted(self._u, u, side="right") - 1, 0, len(self._Q) - 2))
        d = self._Q[k + 1] - self._Q[k]
        L = self._u[k + 1] - self._u[k]
        return self._Q[k] + d * ((u - self._u[k]) / L), d / L

    def _progress(self, t: float) -> tuple[float, float]:
        tau = min(max((t - self.t0) / self.T, 0.0), 1.0)
        s = tau**3 * (10 - 15 * tau + 6 * tau**2)
        sd = 30 * tau**2 * (1 - tau) ** 2 / self.T if 0.0 < tau < 1.0 else 0.0
        return s * self.length, sd * self.length

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        u, ud = self._progress(t)
        q, dq = self.geometry(u)
        return q, dq * ud

    def samples(self, n: int) -> np.ndarray:
        """n configurations evenly spaced in chord length (start and end included)."""
        return np.asarray([self.geometry(u)[0] for u in np.linspace(0.0, self.length, n)])

    def peak_speed_ratio(self, v_limit: np.ndarray, n: int = 200) -> float:
        """The fastest joint speed over the path as a fraction of its limit."""
        v_limit = np.asarray(v_limit, float)
        return max(float(np.max(np.abs(self(self.t0 + self.T * x)[1]) / v_limit)) for x in np.linspace(0, 1, n))

    def to_dict(self) -> dict:
        return {"waypoints": [q.tolist() for q in self.waypoints], "T": self.T, "shape": self.shape}

    @classmethod
    def from_dict(cls, d: dict, t0: float = 0.0) -> JointPath:
        return cls(d["waypoints"], d["T"], t0, d.get("shape", "spline"))


def timed(waypoints, v_limit, speed_frac: float = 1.0, shape: str = "spline", t_min: float = MIN_DURATION) -> JointPath:
    """The path through `waypoints`, as fast as SPEED_MARGIN of the joint velocity limits allows, slowed by
    1 / speed_frac (speed_frac in (0, 1]), and no shorter than t_min."""
    p = JointPath(waypoints, 1.0, 0.0, shape)
    ratio = p.peak_speed_ratio(v_limit)  # at T = 1 s; speed scales as 1 / T
    T = max(ratio / SPEED_MARGIN / max(min(speed_frac, 1.0), 1e-3), t_min)
    return p.stretched(T)


def smooth(path: list[np.ndarray], valid, check_n: int | None = None, res: float = EDGE_RES) -> tuple[list, str]:
    """A spline through the waypoints that stays valid when checked every `res` of chord length: spans that clip an
    obstacle get their midpoint added and the spline is refitted (up to REFITS times). Returns (waypoints, shape);
    shape "polyline" when the spline could not be made valid (the straight segments are, by construction)."""
    wps = [np.asarray(q, float) for q in path]
    if len(wps) <= 2:
        return wps, "spline"
    for _ in range(REFITS + 1):
        p = JointPath(wps, 1.0)
        n = check_n or max(2, int(np.ceil(p.length / res)) + 1)
        us = np.linspace(0.0, p.length, n)
        bad = [u for u in us if not valid(p.geometry(u)[0])]
        if not bad:
            return wps, "spline"
        spans = sorted({int(np.clip(np.searchsorted(p._u, u, side="right") - 1, 0, len(p._Q) - 2)) for u in bad})
        Q = list(p._Q)
        for k in reversed(spans):
            Q.insert(k + 1, 0.5 * (Q[k] + Q[k + 1]))
        wps = Q
    return [np.asarray(q, float) for q in path], "polyline"


def plan_path(q_start, q_goal, valid, lo, hi, v_limit, rng: np.random.Generator, speed_frac: float = 1.0,
              **rrt_kw) -> tuple[JointPath | None, PlanResult]:
    """RRT-Connect, shortcut, smooth and time: (path or None, the search's result)."""
    res = rrt_connect(q_start, q_goal, valid, lo, hi, rng, **rrt_kw)
    if not res.ok:
        return None, res
    wps = shortcut(res.waypoints, valid, rng)
    wps, shape = smooth(wps, valid)
    res.stats.update(raw_waypoints=len(res.waypoints), waypoints=len(wps), shape=shape)
    return timed(wps, v_limit, speed_frac, shape), res
