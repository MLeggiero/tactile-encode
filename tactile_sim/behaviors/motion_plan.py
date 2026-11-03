"""Per-episode motion plans for the varied VLA demos (vla.variation), and the pre-roll that puts the arm at a plan's
start pose before an episode is recorded.

A plan holds everything the sampler drew for one episode's arm motion. It lives outside SimConfig on purpose: the
hover pose, the grasp keyframe and the residual's training scene derive from SimConfig, and a plan changes none of
them. With no plan the bases (behaviors.swing.ScriptedSwing, behaviors.tool_tasks.PositionDrillFeed) run exactly as
before.

The pre-roll: the testbed settles the grasp at the hover pose as always (World.reset); the base's L2 tick is taken
off the scheduler, the arm follows a min-jerk joint stream to the plan's start configuration and holds it briefly, and
then the base begins the planned approach (`begin`) on the same tick the stream is released, so no command tick sees
a stale reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

PREROLL_HOLD = 0.2  # s: the arm holds the start pose this long before the base begins (the servos settle)


@dataclass(frozen=True)
class StartPose:
    """Where an episode starts: the controlled frame's position and orientation (rotation vector) and the arm's joint
    angles there (solved by the sampler's pre-check)."""

    pos: tuple[float, float, float]
    rotvec: tuple[float, float, float]
    q: tuple[float, ...]

    @property
    def R(self) -> np.ndarray:
        from tactile_sim.behaviors.strike_paths import rotmat

        return rotmat(np.asarray(self.rotvec, float))


@dataclass(frozen=True)
class HammerPlan:
    """A hammer episode's motion: the start pose, the approach to the hover pose (its duration and the sideways bulge
    of its path), the windup apex (an offset from the nominal apex `windup_height` back along the strike path), where
    the windup curve joins the terminal stroke (`junction` m before the hover pose), and the windup and settle
    durations. The strike speed stays the configuration's (vla.demos draws it per episode)."""

    start: StartPose
    approach_time: float
    approach_via: tuple[float, float, float]
    apex_offset: tuple[float, float, float]
    junction: float
    windup_time: float
    settle_time: float
    preroll_time: float = 1.0  # s, the unrecorded move from the hover pose to the start pose
    kind: str = field(default="hammer", init=False)


@dataclass(frozen=True)
class DrillPlan:
    """A drill episode's motion: the start pose and the approach to the feed's pre-contact pose (duration, bulge).
    The feed's own approach, press, drive, release and retract follow unchanged."""

    start: StartPose
    approach_time: float
    approach_via: tuple[float, float, float]
    preroll_time: float = 1.0
    kind: str = field(default="drill", init=False)


def plan_to_dict(plan) -> dict:
    return asdict(plan)


def plan_from_dict(d: dict):
    d = dict(d)
    kind = d.pop("kind")
    start = StartPose(**{k: tuple(v) for k, v in d.pop("start").items()})
    cls = HammerPlan if kind == "hammer" else DrillPlan
    return cls(start=start, **{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def min_jerk_joint_stream(t0: float, T: float, q0: np.ndarray, q1: np.ndarray):
    """stream(t) -> (q, qd): a min-jerk move from q0 at t0 to q1 at t0 + T, then holding q1."""
    q0, dq = np.asarray(q0, float), np.asarray(q1, float) - np.asarray(q0, float)

    def stream(t: float) -> tuple[np.ndarray, np.ndarray]:
        tau = min(max((t - t0) / T, 0.0), 1.0)
        s = tau**3 * (10 - 15 * tau + 6 * tau**2)
        sd = 30 * tau**2 * (1 - tau) ** 2 / T if 0.0 < tau < 1.0 else 0.0
        return q0 + dq * s, dq * sd

    return stream


def preroll(tb, q_start, T: float, hold: float = PREROLL_HOLD, base=None, plan=None, step_cb=None) -> None:
    """Move the arm to q_start in joint space (min-jerk over T, then `hold`) with the scripted L2 off, then release the
    joint stream and, when given, start `base` on the plan (`base.begin(t, plan)`) before the next step."""
    l2 = list(tb.l2_callbacks)
    tb.l2_callbacks[:] = []
    tb.l1.set_joint_stream(min_jerk_joint_stream(tb.t, T, tb.l1.q_ref.copy(), q_start))
    for _ in range(int(round((T + hold) / tb.world.dt))):
        tb.step()
        if step_cb is not None:
            step_cb(tb)
    tb.l1.set_joint_stream(None)
    tb.l2_callbacks[:] = l2
    if base is not None:
        base.begin(tb.t, plan)
