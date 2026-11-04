"""Varied motions for the VLA demos (vla.demos --generator planned): a declared parameter space per task, sampled per
episode and pre-checked before the episode is simulated.

Episode i of a dataset takes point i of a scrambled Sobol sequence seeded by the dataset's base seed (seed0), so a run
covers the space evenly and every episode is reproducible on its own. When the pre-check rejects a candidate, only the
group of parameters that failed is redrawn (default_rng((seed0, index, attempt))): the start for an unreachable or
colliding start, a pre-roll or an approach no path was found for; the swing (hammer) or the angled entry (drill) for a
failing windup, stroke or line-up. The episode's other coordinates keep their Sobol values, so a rejection does not
unbalance the rest of the space. The accepted draw is recorded (the last attempt's "u") for vla.coverage's
intended-vs-achieved comparison. The task's position on the table (a wider shift than vla.demos.TASK_SHIFT) is fixed by
the Sobol point alone, since it changes the scene the episode is built in; its range is where the arm can reach and
strike (HammerSpace.task_shift), and a hammer episode whose hover pose leaves no braking blow room is rejected before
any planning (hover_blow).

What varies (HammerSpace, DrillSpace):
- the start: a point on a shell around the hover / pre-contact pose (0-40 cm away, any azimuth, from a little below the
  hover height to well above it) with a tilt, or one of the arm's ANCHOR postures (tucked, lowered, raised)
  with a joint jitter;
- the approach: its speed (a fraction of the fastest the joint limits allow) and, for the curve, its sideways bulge;
- the hammer's strikes: the windup apex (an offset from the nominal apex in the strike frame: back along the path,
  along the handle, across both), where the windup curve joins the terminal stroke, and the windup and settle times;
  then each strike jitters the apex, the windup time and the strike speed around the episode's values (STRIKE_JITTER,
  from a per-episode generator; a strike whose jitter fails the pre-check keeps the episode's values);
- the drill's entry: the transit ends short of the pre-contact pose, turned up to 15 deg and up to 2 cm off the bit's
  axis, and the feed lines the bit up from there (behaviors.tool_tasks.PositionDrillFeed's align phase).

The pre-check (behaviors.feasibility.ArmProbe) runs on a scratch copy of the episode's settled world: the start's IK,
margin and clearance; the approach, first as the simple curve (behaviors.strike_paths.TransitPath, solved by IK into
joint waypoints that end at the hover posture's own solution) and, when that fails on clearance, margin, reach or
continuity, as a joint-space RRT-Connect path (behaviors.joint_planner) around the table, board and robot body; the
pre-roll from the hover pose to the start, straight or planned likewise; the hammer's whole windup curve and stroke up
to 3 cm before contact (closer, the face is meant to meet the nail) and that the swing's braking profile still fits it
at no less than MIN_SPEED_RATIO of the scripted stroke's contact speed (behaviors.feasibility.strike_speed); the drill's
line-up. The approach and pre-roll become timed joint paths (smoothed, re-checked densely, at most SPEED_MARGIN of the
joint velocity limits) that the bases execute as joint streams, so the motion run is the motion checked. The plan also
records which tenths of the approach have PUSH_REACH of room around the arm and tool, where vla.demos' DART pushes may
start.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from tactile_sim.behaviors import joint_planner as jp
from tactile_sim.behaviors.feasibility import ArmProbe, Verdict, strike_speed
from tactile_sim.behaviors.joint_planner import MAX_STRETCH, MIN_JERK_PEAK, SPEED_MARGIN  # noqa: F401 - re-exported
from tactile_sim.behaviors.motion_plan import DrillPlan, HammerPlan, StartPose
from tactile_sim.behaviors.strike_paths import CompositeStrikeGeometry, TransitPath, rotmat, rotvec

MAX_ATTEMPTS = 20
PREROLL_TIME = 1.0  # s, the shortest pre-roll (longer when a joint must travel far: SPEED_MARGIN)
CONTACT_GUARD = 0.03  # m before the nominal contact: the pre-check's stroke ends here
MIN_SPEED_RATIO = 0.9  # a planned blow lands at least this fraction of the scripted stroke's speed
CURVE_SAMPLES = 25  # IK samples along the simple curve
GOAL_MATCH = 0.05  # rad: the curve's last IK solution must be the hover posture's own (else the planner takes over)
PUSH_REACH = 0.05  # m: how far a DART push (vla.demos.PUSH_SCALE) can move the hand; a push needs this much room
CLEAR_WINDOWS = 10  # the approach's room for pushes is recorded per tenth
N_STRIKE_DRAWS = 20  # per-strike variations drawn (vla.demos.MAX_STRIKES); later strikes use the episode's values
# per-strike jitter around the episode's values: apex offset (m, each axis), windup time and strike speed (fractions)
STRIKE_JITTER = {"apex": 0.02, "windup_time": 0.15, "speed": 0.05}
STRIKE_TRIES = 3  # draws per strike before it falls back to the episode's values

# arm postures some episodes start from instead of the shell: joint offsets from the hover posture (rad), jittered by
# the space's anchor_jitter; a posture that is unreachable or collides in some scene is redrawn like any start
ANCHORS = {
    "tucked": (0.0, -0.6, 0.0, 0.9, 0.0, 0.0, 0.0),
    "lowered": (0.0, 0.35, 0.0, -0.35, 0.0, 0.3, 0.0),
    "raised": (0.6, 0.0, 0.3, 0.2, 0.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class HammerSpace:
    anchor_share: float = 0.2  # of the episodes, starting from an ANCHOR posture
    anchor_jitter: float = 0.12  # rad per joint
    # the shell: distance from the hover pose (m) and elevation above its horizontal plane (deg; the head hangs 19 cm
    # below the grip, so starts far below the hover height mostly put it in the table, and are redrawn)
    start_dist: tuple[float, float] = (0.0, 0.40)
    start_elev: tuple[float, float] = (-15.0, 70.0)
    start_tilt: float = 20.0  # deg, each axis
    speed_frac: tuple[float, float] = (0.45, 1.0)  # the approach's speed, of the fastest the joint limits allow
    approach_via: float = 0.06  # m, the curve's bulge, half-extent on each axis
    # m: the windup apex's offset from the nominal one, back along the path (negative: a shorter windup; the swing's
    # single braking quintic cannot stretch a much longer stroke at this arm's speed cap), along the handle, across
    apex_back: tuple[float, float] = (-0.06, 0.01)
    apex_side: tuple[float, float] = (0.06, 0.06)
    junction: tuple[float, float] = (0.0, 0.06)  # m before the hover pose
    windup_time: tuple[float, float] = (0.35, 0.6)  # s
    settle_time: tuple[float, float] = (0.1, 0.3)  # s
    # m, (x range, y range): where the task may sit on the table. Measured 2026-11-03 on the Vega U: past about +2 cm in
    # x the arm cannot reach the hover pose, and toward +x and +y its speed cap along the strike axis falls under
    # ~1 m/s, where no braking blow fits; inside this box the cap stays at or above 1.17 m/s
    task_shift: tuple = ((-0.06, 0.0), (-0.06, 0.03))

    dims: int = 20


@dataclass(frozen=True)
class DrillSpace:
    anchor_share: float = 0.2
    anchor_jitter: float = 0.12
    start_dist: tuple[float, float] = (0.0, 0.40)
    start_elev: tuple[float, float] = (-15.0, 70.0)
    start_tilt: float = 20.0
    speed_frac: tuple[float, float] = (0.45, 1.0)
    approach_via: float = 0.05
    # the angled entry: the transit ends turned up to align_tilt (deg) about an axis across the bit, align_back (m)
    # further back along the bit and up to align_across (m) off its axis; the line-up takes align_time (s)
    align_tilt: float = 15.0
    align_back: tuple[float, float] = (0.03, 0.06)
    align_across: float = 0.02
    align_time: tuple[float, float] = (0.5, 1.0)
    task_shift: tuple = ((-0.05, 0.05), (-0.05, 0.05))  # m, (x range, y range): the drill reaches all of it

    dims: int = 20


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


# the unit draw's coordinates by group; the last group is the hammer's swing or the drill's angled entry
SHIFT, START, APPROACH, SWING = slice(0, 2), slice(2, 10), slice(10, 14), slice(14, 20)
_COMMON = ("shift_x", "shift_y", "family", "start_dist", "start_azim", "start_elev", "tilt_x", "tilt_y", "tilt_z",
           "anchor", "speed_frac", "via_x", "via_y", "via_z")
PARAM_NAMES = {
    "hammer": _COMMON + ("apex_back", "apex_handle", "apex_across", "junction", "windup_time", "settle_time"),
    "drill": _COMMON + ("align_tilt", "align_tilt_dir", "align_back", "align_across", "align_across_dir",
                        "align_time"),
}


def redraw_groups(where: str) -> list[slice]:
    """The parameter groups to redraw after a rejection at `where` (Verdict.where)."""
    stage = where.split("[")[0].split(" ")[0]
    if stage in ("start", "preroll", "approach"):
        return [START]
    return [SWING]  # windup, stroke (hammer); align (drill)


def _sym(u: float, half: float) -> float:
    return (2.0 * u - 1.0) * half


def _lin(u: float, lohi: tuple[float, float]) -> float:
    return lohi[0] + u * (lohi[1] - lohi[0])


def task_shift(task: str, seed0: int, index: int) -> tuple[float, float]:
    """The task's horizontal shift on the table for this episode (from the Sobol point; retries keep it)."""
    u = unit_point(task, seed0, index, 0)
    (x_lo_hi, y_lo_hi) = SPACES[task].task_shift
    return _lin(u[0], x_lo_hi), _lin(u[1], y_lo_hi)


def _start(u: np.ndarray, space, p_hover: np.ndarray, R_hover: np.ndarray, q_hover: np.ndarray | None) -> dict:
    """The start from the draw: an anchor posture (joint angles) or a shell pose (position and orientation)."""
    if q_hover is not None and u[2] < space.anchor_share:
        name = sorted(ANCHORS)[min(int(u[9] * len(ANCHORS)), len(ANCHORS) - 1)]
        jit = np.random.default_rng([int(1e9 * u[9]), int(1e9 * u[3])]).uniform(-1.0, 1.0, len(q_hover))
        return {"family": "anchor", "anchor": name,
                "start_q": np.asarray(q_hover, float) + np.asarray(ANCHORS[name]) + space.anchor_jitter * jit}
    r = _lin(u[3], space.start_dist)
    az = np.pi * (2.0 * u[4] - 1.0)
    el = np.radians(_lin(u[5], space.start_elev))
    off = r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    tilt = np.radians([_sym(u[6], space.start_tilt), _sym(u[7], space.start_tilt), _sym(u[8], space.start_tilt)])
    return {"family": "shell", "start_pos": p_hover + off, "start_R": rotmat(tilt) @ R_hover}


def candidate(task: str, u: np.ndarray, p_hover: np.ndarray, R_hover: np.ndarray,
              q_hover: np.ndarray | None = None) -> dict:
    """The plan's parameters from a unit draw (an anchor start's pose and a shell start's joint angles are added by the
    pre-check). Without q_hover every start is a shell start."""
    sp = SPACES[task]
    out = _start(u, sp, p_hover, R_hover, q_hover)
    out.update(speed_frac=_lin(u[10], sp.speed_frac),
               approach_via=np.array([_sym(u[11], sp.approach_via), _sym(u[12], sp.approach_via),
                                      _sym(u[13], sp.approach_via)]))
    if task == "hammer":
        out.update(apex_offset=np.array([_lin(u[14], sp.apex_back), _sym(u[15], sp.apex_side[0]),
                                         _sym(u[16], sp.apex_side[1])]),
                   junction=_lin(u[17], sp.junction), windup_time=_lin(u[18], sp.windup_time),
                   settle_time=_lin(u[19], sp.settle_time))
    else:
        tilt_dir, across_dir = 2 * np.pi * u[15], 2 * np.pi * u[18]
        out.update(align_tilt=np.radians(u[14] * sp.align_tilt), align_tilt_dir=tilt_dir,
                   align_back=_lin(u[16], sp.align_back), align_across=u[17] * sp.align_across,
                   align_across_dir=across_dir, align_time=_lin(u[19], sp.align_time))
    return out


def bit_frame(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors across the bit axis a (a right-handed frame with it)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    e1 = np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


def align_pose(c: dict, x0: np.ndarray, R0: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The drill's angled-entry pose for candidate c: (position, orientation, the turn's rotation vector)."""
    e1, e2 = bit_frame(a)
    axis = np.cos(c["align_tilt_dir"]) * e1 + np.sin(c["align_tilt_dir"]) * e2
    rv = c["align_tilt"] * axis
    across = c["align_across"] * (np.cos(c["align_across_dir"]) * e1 + np.sin(c["align_across_dir"]) * e2)
    return np.asarray(x0, float) - c["align_back"] * np.asarray(a, float) + across, rotmat(rv) @ R0, rv


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


def _joint_path(probe: ArmProbe, q_from: np.ndarray, q_to: np.ndarray, pose_from, pose_to, via, speed_frac: float,
                rng: np.random.Generator, label: str) -> tuple[Verdict, jp.JointPath | None, str]:
    """A timed joint path from q_from to q_to: the simple curve between the two poses when it passes, else the
    straight joint move, then the planner. (verdict, path, "curve" | "joint" | "rrt" | "straight")."""
    v_lim = probe.w.arm_spec.velocity
    if pose_from is None:  # the pre-roll: a straight joint move when it is clear
        if jp.edge_ok(q_from, q_to, probe.valid, 0.05):
            return Verdict(True), jp.timed([q_from, q_to], v_lim, speed_frac, "spline"), "straight"
    else:
        tr = TransitPath(0.0, 1.0, pose_from[0], pose_from[1], pose_to[0], pose_to[1], via)
        v, qs = probe.check_poses([tr.pose(t) for t in np.linspace(0.0, 1.0, CURVE_SAMPLES)], q_from, label)
        if v.ok and float(np.max(np.abs(qs[-1] - q_to))) <= GOAL_MATCH:
            wps = [np.asarray(q_from, float)] + [np.asarray(q, float) for q in qs[1:-1]] + [np.asarray(q_to, float)]
            wps, shape = jp.smooth(wps, probe.valid, res=0.05)
            if shape == "spline":
                return Verdict(True), jp.timed(wps, v_lim, speed_frac, shape), "curve"
        if jp.edge_ok(q_from, q_to, probe.valid, 0.05):  # the curve's IK took another elbow: move in joint space
            return Verdict(True), jp.timed([q_from, q_to], v_lim, speed_frac, "spline"), "joint"
    path, res = jp.plan_path(q_from, q_to, probe.valid, probe.lo, probe.hi, v_lim, rng, speed_frac)
    if path is None:
        return Verdict(False, f"rrt_{res.reason}", label), None, "rrt"
    return Verdict(True), path, "rrt"


def _room(wide: ArmProbe, path: jp.JointPath, per_window: int = 4) -> tuple[bool, ...]:
    """Per tenth of the path: whether the arm and tool stay PUSH_REACH from every obstacle (a push may start there)."""
    qs = path.samples(CLEAR_WINDOWS * per_window)
    return tuple(bool(min(wide.clearance(q)[0] for q in qs[k * per_window:(k + 1) * per_window]) >= PUSH_REACH)
                 for k in range(CLEAR_WINDOWS))


def check_swing(probe: ArmProbe, cfg, apex_offset: np.ndarray, junction: float, windup_time: float,
                q_hover: np.ndarray) -> tuple[Verdict, dict]:
    """One strike's windup curve and stroke: reach, margin, continuity and clearance up to CONTACT_GUARD before the
    contact, the windup timed within the joint limits, and the blow at least MIN_SPEED_RATIO of the scripted one's
    speed. Returns the verdict and {"windup_time", "strike_speed", "apex_offset_world"}."""
    w = probe.w
    apex, R_apex = apex_world(w, cfg, apex_offset)
    try:
        g = CompositeStrikeGeometry(w.geom, apex, R_apex, -junction)
    except ValueError:
        return Verdict(False, "bad_shape", "windup"), {}
    s = np.linspace(g.s_apex, g.s_c0 - CONTACT_GUARD, 14)
    v, _ = probe.check_poses([g.pose(x) for x in s], q_hover, "stroke")
    if not v.ok:
        return v, {}
    # the windup: min-jerk from the hover pose back to the apex over windup_time (ScriptedSwing._start_windup)
    tau = np.linspace(0.0, 1.0, 15)
    v, qs = probe.check_poses([g.pose(g.s_apex * x**3 * (10 - 15 * x + 6 * x**2)) for x in tau], q_hover, "windup")
    if not v.ok:
        return v, {}
    Tw, v = _timed(_speed_ratio(qs, windup_time, w.arm_spec.velocity), windup_time, "windup")
    if v is not None:
        return v, {}
    sw = cfg.swing
    v_h, a_h = probe.path_caps(w.geom, [0.0], q_hover)
    v_w, a_w = probe.path_caps(g, np.linspace(g.s_apex, 0.0, 8), q_hover)
    ref = strike_speed(g.s_c0 + sw.windup_height, sw.v_strike, float(v_h[0]), float(a_h[0]), sw)
    got = strike_speed(g.s_c0 - g.s_apex, sw.v_strike, min(float(v_h[0]), float(v_w.min())),
                       min(float(a_h[0]), float(a_w.min())), sw)
    if ref <= 0.0 or got < MIN_SPEED_RATIO * ref:  # (ref 0: no blow fits even the scripted stroke; got 0 is no better)
        return Verdict(False, "slow_strike", f"stroke {got:.2f} of {ref:.2f} m/s"), {}
    apex0, _ = w.geom.pose(-sw.windup_height)
    return Verdict(True), {"windup_time": Tw, "strike_speed": got, "apex_offset_world": apex - apex0}


def strike_draws(probe: ArmProbe, cfg, c: dict, extra: dict, q_hover: np.ndarray, rng: np.random.Generator,
                 n: int = N_STRIKE_DRAWS) -> tuple[tuple, int]:
    """Per-strike variations around the episode's checked swing: (strikes, how many fell back to the episode's)."""
    out, fallbacks = [], 0
    for _ in range(n):
        for _try in range(STRIKE_TRIES):
            d_apex = rng.uniform(-1.0, 1.0, 3) * STRIKE_JITTER["apex"]
            f_w = 1.0 + rng.uniform(-1.0, 1.0) * STRIKE_JITTER["windup_time"]
            speed = 1.0 + rng.uniform(-1.0, 1.0) * STRIKE_JITTER["speed"]
            v, e = check_swing(probe, cfg, c["apex_offset"] + d_apex, c["junction"], c["windup_time"] * f_w, q_hover)
            if v.ok:
                break
        if not v.ok:
            fallbacks += 1
            e, speed = {"apex_offset_world": extra["apex_offset_world"], "windup_time": extra["windup_time"]}, 1.0
        out.append({"apex_offset": [float(x) for x in e["apex_offset_world"]],
                    "windup_time": float(e["windup_time"]), "speed": float(speed)})
    return tuple(out), fallbacks


def precheck(probe: ArmProbe, task: str, c: dict, q_hover: np.ndarray, target: tuple[np.ndarray, np.ndarray],
             cfg=None, rng: np.random.Generator | None = None, wide: ArmProbe | None = None,
             bit_axis: np.ndarray | None = None) -> tuple[Verdict, np.ndarray | None, dict]:
    """The candidate's start, approach (curve, else planner), pre-roll and, for the hammer, windup curve and stroke;
    for the drill, its angled entry and line-up. Returns the verdict, the start's joint angles, and the plan's derived
    values: the timed approach and pre-roll paths (JointPath), the planner kind, the start pose, the approach's room
    for pushes, and for the hammer the windup time and world-frame apex offset; for the drill the entry pose's turn."""
    rng = rng or np.random.default_rng(0)
    if c["family"] == "anchor":
        q_start = np.asarray(c["start_q"], float)
        if probe.margin(q_start) < 0.05:
            return Verdict(False, "joint_margin", "start"), None, {}
        probe.set_q(q_start)
        p0, R0 = probe.w.ctrl_pose(probe.d)
        c = dict(c, start_pos=np.array(p0, float), start_R=np.array(R0, float))
    else:
        q_start, ep, er = probe.ik(c["start_pos"], c["start_R"], q_hover)
        if ep > 1e-3 or er > 1e-2:
            return Verdict(False, "unreachable", "start"), None, {}
    if not probe.valid(q_start):
        clr, pair = probe.clearance(q_start)
        return Verdict(False, "collision" if clr < np.inf else "joint_margin", f"start {pair}"), None, {}
    extra: dict = {"start_pos": c["start_pos"], "start_R": c["start_R"]}
    # where the approach ends: the hover pose (hammer), the pre-contact pose or the angled entry short of it (drill),
    # in the hover posture's own IK solution so the handoff leaves PositionL1's null-space pull nothing to undo
    end = target
    if task == "drill":
        p_a, R_a, rv = align_pose(c, target[0], target[1], bit_axis)
        end, extra["align_rotvec"] = (p_a, R_a), rv
    q_end, ep, er = probe.ik(end[0], end[1], q_hover)
    if ep > 1e-3 or er > 1e-2 or not probe.valid(q_end):
        return Verdict(False, "unreachable" if ep > 1e-3 or er > 1e-2 else "collision",
                       "align" if task == "drill" else "approach"), None, {}
    v, path, kind = _joint_path(probe, q_start, q_end, (c["start_pos"], c["start_R"]), end, c["approach_via"],
                                c["speed_frac"], rng, "approach")
    if not v.ok:
        return v, None, {}
    extra.update(approach=path, planner=kind)
    v, pre, pre_kind = _joint_path(probe, q_hover, q_start, None, None, None, 1.0, rng, "preroll")
    if not v.ok:
        return v, None, {}
    extra.update(preroll=pre if pre_kind == "rrt" else None,
                 preroll_time=max(PREROLL_TIME, pre.T) if pre_kind != "rrt" else pre.T)
    if wide is not None:
        extra["room"] = _room(wide, path)
    if task == "drill":
        # the line-up from the entry pose to the pre-contact pose (PositionDrillFeed's align phase, task space)
        tr = TransitPath(0.0, 1.0, end[0], end[1], target[0], target[1])
        v, _ = probe.check_poses([tr.pose(t) for t in np.linspace(0.0, 1.0, 10)], q_end, "align")
        if not v.ok:
            return v, None, {}
        return Verdict(True), q_start, extra
    v, sw = check_swing(probe, cfg, c["apex_offset"], c["junction"], c["windup_time"], q_hover)
    if not v.ok:
        return v, None, {}
    extra.update(sw)
    return Verdict(True), q_start, extra


def hover_blow(probe: ArmProbe, cfg, q_hover: np.ndarray) -> float:
    """The contact speed the scripted stroke reaches from the hover pose under the arm's caps there: 0 when no braking
    blow fits at all (the hover pose sits where the arm is slow along the strike axis), in which case no planned windup
    can strike either, and the episode is rejected before anything is planned."""
    w, sw = probe.w, cfg.swing
    g = w.geom
    v_h, a_h = probe.path_caps(g, [0.0], q_hover)
    return strike_speed(g.s_c0 + sw.windup_height, sw.v_strike, float(v_h[0]), float(a_h[0]), sw)


def sample_plan(task: str, tb, seed0: int, index: int, max_attempts: int = MAX_ATTEMPTS,
                target: tuple[np.ndarray, np.ndarray] | None = None, strikes: bool = True):
    """(plan or None, attempts) for episode `index` of a dataset seeded by seed0, on the testbed `tb` as it stands
    after reset (the arm at the hover pose, the grasp settled). `target`: where the approach ends (the drill feed's
    aimed pre-contact pose x0, R; default: the hammer's stroke hover pose, or the drill's settled pose)."""
    w = tb.world
    probe = ArmProbe(w)
    wide = ArmProbe(w, margin=PUSH_REACH)
    q_hover = w.data.qpos[w.arm_qadr].copy()
    p_hover, R_hover = w.ctrl_pose()
    bit_axis = None
    if target is None:
        if task == "hammer":  # the swing's approach ends at the stroke's hover pose (ScriptedSwing.begin)
            target = (np.asarray(w.hover_tcp, float).copy(), np.asarray(w.hover_R, float).copy())
        else:  # the feed's transit ends at its pre-contact pose x0, where the grasp settled
            target = (p_hover.copy(), R_hover.copy())
    if task == "drill":
        R_t = np.asarray(target[1], float)
        bit_axis = -R_t[:, 1] if tb.cfg.drill.form == "pistol" else R_t[:, 2]
    attempts = []
    if task == "hammer":
        v_blow = hover_blow(probe, tb.cfg, q_hover)
        if v_blow <= 0.0:  # fixed by the episode's task position: no draw can help
            return None, [{"attempt": 0, "ok": False, "reason": "no_blow", "where": "hover", "family": ""}]
    u = unit_point(task, seed0, index, 0)
    for a in range(max_attempts):
        if a > 0:  # redraw only what failed; the rest keeps the episode's Sobol coordinates
            fresh = unit_point(task, seed0, index, a)
            for g in redraw_groups(attempts[-1]["where"]):
                u[g] = fresh[g]
        c = candidate(task, u, p_hover, R_hover, q_hover)
        rng = np.random.default_rng((seed0, index, a, 7))
        v, q_start, extra = precheck(probe, task, c, q_hover, target, tb.cfg, rng, wide, bit_axis)
        attempts.append({"attempt": a, "ok": v.ok, "reason": v.reason, "where": v.where, "family": c["family"]})
        if not v.ok:
            continue
        attempts[-1].update(u=u.tolist(), planner=extra["planner"])
        start = StartPose(tuple(map(float, extra["start_pos"])), tuple(map(float, rotvec(extra["start_R"]))),
                          tuple(map(float, q_start)))
        path = extra["approach"]
        common = {"approach_time": float(path.T), "approach_via": tuple(map(float, c["approach_via"])),
                  "preroll_time": float(extra["preroll_time"]), "approach_path": path.to_dict(),
                  "preroll_path": None if extra["preroll"] is None else extra["preroll"].to_dict(),
                  "planner": extra["planner"], "speed_frac": float(c["speed_frac"]), "room": extra.get("room", ())}
        if task == "hammer":
            draws, fallbacks = (), 0
            if strikes:
                draws, fallbacks = strike_draws(probe, tb.cfg, c, extra, q_hover,
                                                np.random.default_rng((seed0, index, 11)))
            attempts[-1]["strike_fallbacks"] = fallbacks
            return HammerPlan(start, apex_offset=tuple(map(float, extra["apex_offset_world"])),
                              junction=float(c["junction"]), windup_time=float(extra["windup_time"]),
                              settle_time=float(c["settle_time"]), strikes=draws, **common), attempts
        return DrillPlan(start, align_rotvec=tuple(map(float, extra["align_rotvec"])),
                         align_back=float(c["align_back"]),
                         align_across=float(c["align_across"]),
                         align_time=float(c["align_time"]), **common), attempts
    return None, attempts
