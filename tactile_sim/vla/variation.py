"""Varied motions for the VLA demos (vla.demos --generator planned): a declared parameter space per task, sampled per
episode and pre-checked before the episode is simulated.

Episode i of a dataset takes point i of a scrambled Sobol sequence seeded by the dataset's base seed (seed0), so a run
covers the space evenly and every episode is reproducible on its own. When the pre-check rejects a candidate, only the
group of parameters that failed is redrawn (default_rng((seed0, index, attempt))): the start for an unreachable or
colliding start or pre-roll, the start and the approach for a failing approach, the swing for a failing windup or
stroke. The episode's other coordinates keep their Sobol values, so a rejection does not unbalance the rest of the
space. The accepted draw is recorded (the last attempt's "u") for vla.coverage's intended-vs-achieved comparison. The
task's position on the table (a wider shift than vla.demos.TASK_SHIFT) is fixed by the Sobol point alone, since it
changes the scene the episode is built in.

What varies (HammerSpace, DrillSpace):
- the start: the controlled frame's position in a box around the hover / pre-contact pose, and a tilt;
- the approach: its duration and the sideways bulge of its path (behaviors.strike_paths.TransitPath);
- the hammer's strikes: the windup apex (an offset from the nominal apex in the strike frame: back along the path,
  along the handle, across both), where the windup curve joins the terminal stroke, and the windup and settle times.

The pre-check (behaviors.feasibility.ArmProbe) runs on a scratch copy of the episode's settled world: the start pose's
IK, the pre-roll's joint path, the approach, and the hammer's whole windup curve and stroke up to 3 cm before contact
(closer, the face is meant to meet the nail): reach, joint margin, IK continuity and clearance to the table, board,
torso and other arm; and for the hammer, that the swing's braking profile still fits the planned stroke at no less than
MIN_SPEED_RATIO of the scripted stroke's contact speed (behaviors.feasibility.strike_speed).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from tactile_sim.behaviors.feasibility import ArmProbe, Verdict, strike_speed
from tactile_sim.behaviors.motion_plan import DrillPlan, HammerPlan, StartPose
from tactile_sim.behaviors.strike_paths import CompositeStrikeGeometry, TransitPath, rotmat

MAX_ATTEMPTS = 20
PREROLL_TIME = 1.0  # s, the shortest pre-roll (longer when a joint must travel far: SPEED_MARGIN)
CONTACT_GUARD = 0.03  # m before the nominal contact: the pre-check's stroke ends here
MIN_SPEED_RATIO = 0.9  # a planned blow lands at least this fraction of the scripted stroke's speed
# the pre-roll, approach and windup are timed so no joint's target moves faster than SPEED_MARGIN of its velocity
# limit (the servos overshoot their targets; PositionL1 itself caps targets at 90 %): a sampled duration too short for
# its path is stretched to fit, up to MAX_STRETCH times, and the plan records the duration it got
SPEED_MARGIN = 0.7
MAX_STRETCH = 2.5
MIN_JERK_PEAK = 1.875  # a min-jerk move's peak speed over its mean


@dataclass(frozen=True)
class HammerSpace:
    # m: half-extents around the hover pose in world x and y, and the height above it (never below: the head hangs
    # 19 cm below the grip, a few cm over the board). The wrist's last joint sits 0.33 rad from its limit at the hover
    # pose, which caps the tilt.
    start_box: tuple[float, float, float] = (0.08, 0.08, 0.08)
    start_tilt: float = 8.0  # deg, each axis
    approach_time: tuple[float, float] = (0.8, 1.6)  # s
    approach_via: float = 0.06  # m, the bulge's half-extent on each axis
    # m: the windup apex's offset from the nominal one, back along the path (negative: a shorter windup; the swing's
    # single braking quintic cannot stretch a much longer stroke at this arm's speed cap), along the handle, across
    apex_back: tuple[float, float] = (-0.06, 0.01)
    apex_side: tuple[float, float] = (0.06, 0.06)
    junction: tuple[float, float] = (0.0, 0.06)  # m before the hover pose
    windup_time: tuple[float, float] = (0.35, 0.6)  # s
    settle_time: tuple[float, float] = (0.1, 0.3)  # s
    task_shift: float = 0.06  # m, each horizontal axis

    dims: int = 18


@dataclass(frozen=True)
class DrillSpace:
    start_box: tuple[float, float, float] = (0.08, 0.08, 0.08)
    start_tilt: float = 8.0
    approach_time: tuple[float, float] = (0.8, 1.6)
    approach_via: float = 0.05
    task_shift: float = 0.05

    dims: int = 12


SPACES = {"hammer": HammerSpace(), "drill": DrillSpace()}


def unit_point(task: str, seed0: int, index: int, attempt: int = 0) -> np.ndarray:
    """The episode's draw in [0, 1)^dims: Sobol point `index` on the first attempt, a seeded uniform draw after."""
    d = SPACES[task].dims
    if attempt == 0:
        from scipy.stats import qmc

        sob = qmc.Sobol(d, scramble=True, seed=seed0)
        if index:
            sob.fast_forward(index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # balance warnings for counts that are not powers of two
            return sob.random(1)[0]
    return np.random.default_rng((seed0, index, attempt)).random(d)


# the unit draw's coordinates by group (HammerSpace; DrillSpace has the first three)
SHIFT, START, APPROACH, SWING = slice(0, 2), slice(2, 8), slice(8, 12), slice(12, 18)
PARAM_NAMES = ("shift_x", "shift_y", "start_x", "start_y", "start_z", "tilt_x", "tilt_y", "tilt_z", "approach_time",
               "via_x", "via_y", "via_z", "apex_back", "apex_handle", "apex_across", "junction", "windup_time",
               "settle_time")


def redraw_groups(where: str) -> list[slice]:
    """The parameter groups to redraw after a rejection at `where` (Verdict.where)."""
    stage = where.split("[")[0].split(" ")[0]
    if stage in ("start", "preroll"):
        return [START]
    if stage == "approach":
        return [START, APPROACH]
    return [SWING]


def _sym(u: float, half: float) -> float:
    return (2.0 * u - 1.0) * half


def _lin(u: float, lohi: tuple[float, float]) -> float:
    return lohi[0] + u * (lohi[1] - lohi[0])


def task_shift(task: str, seed0: int, index: int) -> tuple[float, float]:
    """The task's horizontal shift on the table for this episode (from the Sobol point; retries keep it)."""
    u = unit_point(task, seed0, index, 0)
    h = SPACES[task].task_shift
    return _sym(u[0], h), _sym(u[1], h)


def _start(u: np.ndarray, space, p_hover: np.ndarray, R_hover: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bx, by, bz = space.start_box
    off = np.array([_sym(u[2], bx), _sym(u[3], by), u[4] * bz])  # the arm comes to the task from above or level
    tilt = np.radians([_sym(u[5], space.start_tilt), _sym(u[6], space.start_tilt), _sym(u[7], space.start_tilt)])
    return p_hover + off, rotmat(tilt) @ R_hover


def candidate(task: str, u: np.ndarray, p_hover: np.ndarray, R_hover: np.ndarray) -> dict:
    """The plan's parameters from a unit draw (the start's joint angles are added by the pre-check)."""
    sp = SPACES[task]
    p0, R0 = _start(u, sp, p_hover, R_hover)
    out = {"start_pos": p0, "start_R": R0, "approach_time": _lin(u[8], sp.approach_time),
           "approach_via": np.array([_sym(u[9], sp.approach_via), _sym(u[10], sp.approach_via),
                                     _sym(u[11], sp.approach_via)])}
    if task == "hammer":
        out.update(apex_offset=np.array([_lin(u[12], sp.apex_back), _sym(u[13], sp.apex_side[0]),
                                         _sym(u[14], sp.apex_side[1])]),
                   junction=_lin(u[15], sp.junction), windup_time=_lin(u[16], sp.windup_time),
                   settle_time=_lin(u[17], sp.settle_time))
    return out


def apex_world(world, cfg, offset_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The windup apex for an offset in the strike frame (back along the path, along the handle, across both)."""
    g = world.geom
    apex0, R_apex = g.pose(-cfg.swing.windup_height)
    t_back = -g.tangent(-cfg.swing.windup_height)[0]
    e_handle = world.tcp_R_nominal[:, 0]
    e_handle = e_handle - (e_handle @ t_back) * t_back
    e_handle /= np.linalg.norm(e_handle)
    e_across = np.cross(t_back, e_handle)
    o = np.asarray(offset_frame, float)
    return apex0 + o[0] * t_back + o[1] * e_handle + o[2] * e_across, R_apex


def _speed_ratio(qs: list[np.ndarray], T: float, v_limit: np.ndarray) -> float:
    """The fastest joint speed, as a fraction of its limit, of joint samples taken evenly over T seconds."""
    q = np.asarray(qs, float)
    return float(np.max(np.abs(np.diff(q, axis=0)) * (len(q) - 1) / T / v_limit))


def _timed(ratio: float, T: float, where: str) -> tuple[float, Verdict | None]:
    """A duration stretched so its peak joint speed is SPEED_MARGIN of the limit, or a too_fast rejection."""
    if ratio <= SPEED_MARGIN:
        return T, None
    if ratio / SPEED_MARGIN > MAX_STRETCH:
        return T, Verdict(False, "too_fast", f"{where} {ratio:.2f}x")
    return T * ratio / SPEED_MARGIN, None


def precheck(probe: ArmProbe, task: str, c: dict, q_hover: np.ndarray, target: tuple[np.ndarray, np.ndarray],
             cfg=None) -> tuple[Verdict, np.ndarray | None, dict]:
    """The candidate's start IK, pre-roll, approach and (hammer) windup curve and stroke, each motion timed within the
    joint velocity limits. Returns the verdict, the start's joint angles, and the plan's derived values: the durations
    it got (preroll_time, approach_time, windup_time) and, for the hammer, the world-frame apex offset."""
    v_lim = probe.w.arm_spec.velocity
    q_start, ep, er = probe.ik(c["start_pos"], c["start_R"], q_hover)
    if ep > 1e-3 or er > 1e-2:
        return Verdict(False, "unreachable", "start"), None, {}
    v = probe.check_joint_path(q_hover, q_start)
    if not v.ok:
        return v, None, {}
    peak = MIN_JERK_PEAK * float(np.max(np.abs(q_start - q_hover) / v_lim))
    extra: dict = {"preroll_time": max(PREROLL_TIME, peak / SPEED_MARGIN)}
    # the approach: evenly timed samples land on the same poses whatever the duration, so a stretch keeps the path
    T = c["approach_time"]
    tr = TransitPath(0.0, T, c["start_pos"], c["start_R"], target[0], target[1], c["approach_via"])
    v, qs = probe.check_poses([tr.pose(t) for t in np.linspace(0.0, T, 25)], q_start, "approach")
    if not v.ok:
        return v, None, {}
    extra["approach_time"], v = _timed(_speed_ratio(qs, T, v_lim), T, "approach")
    if v is not None:
        return v, None, {}
    if task == "hammer":
        w = probe.w
        apex, R_apex = apex_world(w, cfg, c["apex_offset"])
        try:
            g = CompositeStrikeGeometry(w.geom, apex, R_apex, -c["junction"])
        except ValueError:
            return Verdict(False, "bad_shape", "windup"), None, {}
        s = np.linspace(g.s_apex, g.s_c0 - CONTACT_GUARD, 14)
        v, _ = probe.check_poses([g.pose(x) for x in s], q_hover, "stroke")
        if not v.ok:
            return v, None, {}
        # the windup: min-jerk from the hover pose back to the apex over windup_time (ScriptedSwing._start_windup)
        Tw = c["windup_time"]
        tau = np.linspace(0.0, 1.0, 15)
        v, qs = probe.check_poses([g.pose(g.s_apex * x**3 * (10 - 15 * x + 6 * x**2)) for x in tau], q_hover,
                                  "windup")
        if not v.ok:
            return v, None, {}
        extra["windup_time"], v = _timed(_speed_ratio(qs, Tw, v_lim), Tw, "windup")
        if v is not None:
            return v, None, {}
        # the blow: no slower than MIN_SPEED_RATIO of the scripted stroke's under the same caps
        sw = cfg.swing
        v_h, a_h = probe.path_caps(w.geom, [0.0], q_hover)
        v_w, a_w = probe.path_caps(g, np.linspace(g.s_apex, 0.0, 8), q_hover)
        ref = strike_speed(g.s_c0 + sw.windup_height, sw.v_strike, float(v_h[0]), float(a_h[0]), sw)
        got = strike_speed(g.s_c0 - g.s_apex, sw.v_strike, min(float(v_h[0]), float(v_w.min())),
                           min(float(a_h[0]), float(a_w.min())), sw)
        if got < MIN_SPEED_RATIO * ref:
            return Verdict(False, "slow_strike", f"stroke {got:.2f} of {ref:.2f} m/s"), None, {}
        extra["strike_speed"] = got
        apex0, _ = w.geom.pose(-cfg.swing.windup_height)
        extra["apex_offset_world"] = apex - apex0
    return Verdict(True), q_start, extra


def sample_plan(task: str, tb, seed0: int, index: int, max_attempts: int = MAX_ATTEMPTS):
    """(plan or None, attempts) for episode `index` of a dataset seeded by seed0, on the testbed `tb` as it stands
    after reset (the arm at the hover pose, the grasp settled)."""
    w = tb.world
    probe = ArmProbe(w)
    q_hover = w.data.qpos[w.arm_qadr].copy()
    p_hover, R_hover = w.ctrl_pose()
    if task == "hammer":  # the swing's approach ends at the stroke's hover pose (ScriptedSwing.begin)
        target = (np.asarray(w.hover_tcp, float).copy(), np.asarray(w.hover_R, float).copy())
    else:  # the feed's transit ends at its pre-contact pose x0, where the grasp settled
        target = (p_hover.copy(), R_hover.copy())
    attempts = []
    u = unit_point(task, seed0, index, 0)
    for a in range(max_attempts):
        if a > 0:  # redraw only what failed; the rest keeps the episode's Sobol coordinates
            fresh = unit_point(task, seed0, index, a)
            for g in redraw_groups(attempts[-1]["where"]):
                u[g] = fresh[g]
        c = candidate(task, u, p_hover, R_hover)
        v, q_start, extra = precheck(probe, task, c, q_hover, target, tb.cfg)
        attempts.append({"attempt": a, "ok": v.ok, "reason": v.reason, "where": v.where})
        if not v.ok:
            continue
        attempts[-1]["u"] = u.tolist()
        start = StartPose(tuple(map(float, c["start_pos"])), tuple(map(float, _rotvec(c["start_R"]))),
                          tuple(map(float, q_start)))
        via = tuple(map(float, c["approach_via"]))
        timing = {"approach_time": float(extra["approach_time"]), "preroll_time": float(extra["preroll_time"])}
        if task == "hammer":
            return HammerPlan(start, approach_via=via, apex_offset=tuple(map(float, extra["apex_offset_world"])),
                              junction=float(c["junction"]), windup_time=float(extra["windup_time"]),
                              settle_time=float(c["settle_time"]), **timing), attempts
        return DrillPlan(start, approach_via=via, **timing), attempts
    return None, attempts


def _rotvec(R: np.ndarray) -> np.ndarray:
    from tactile_sim.behaviors.strike_paths import rotvec

    return rotvec(R)
