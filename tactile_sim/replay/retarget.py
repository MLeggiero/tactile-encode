"""Retarget a recorded tool motion onto the testbed: place its strike on our nail, express it as a TCP path for our
hammer and grasp, and slow it down until the arm can follow it within its limits.

Mapping. A task frame is built at the strike in both worlds from the strike direction (into the target) and the
handle's direction at contact made perpendicular to it (our scene's nominal handle direction on the target side). The
source's face path is carried into our scene by the rotation between the two frames, placed so that at the recorded
contact the face sits `engage` past our nail head (so the replayed blow drives the nail); with `aim_each` every later
contact is aimed at the nail too. The tool's orientation is carried by the same rotation; with `align_face` a constant
extra rotation turns the face's normal at contact onto our strike axis (Adroit's blows land ~45 deg off square). The
TCP follows from our hammer's face offset in the grasp frame, so the face path, not the hand path, is preserved
across the two tools.

Feasibility. The path is resampled at 1 kHz and slowed where it is too fast for the arm: a time warp stretches time
locally by s(t) = max(1, v/v_max, w/w_max, sqrt(a/a_max)) (smoothed), so a human's wrist snap between blows is slowed
while a blow that is already feasible keeps its speed. Within 40 ms of a recorded contact the tool was being stopped by
the nail; that deceleration is the plant's to produce, so it is not counted. The result is checked joint by joint with
damped least-squares IK along the path: reachability (position residual) and joint speed (90 % of the limits) slow it
further (uniformly) or reject it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from tactile_sim.config import SimConfig
from tactile_sim.model.builder import nail_head_target, strike_axis
from tactile_sim.model.tool_hammer import face_offset
from tactile_sim.replay.sources import ToolMotion


@dataclass
class ReplayPlan:
    t: np.ndarray  # (N,) s from the start of the motion, 1 kHz
    tcp: np.ndarray  # (N, 3)
    R: np.ndarray  # (N, 3, 3)
    face: np.ndarray  # (N, 3)
    t_contacts: list[float]  # recorded strike times on this time base
    time_scale: float  # >= 1: how much slower than recorded
    motion: str
    q: np.ndarray | None = None  # (M, 7) IK along the path (every ik_every samples)
    ik_residual: float = float("nan")  # worst position residual along the path (m)
    notes: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.t[-1])


def task_frame(axis: np.ndarray, handle: np.ndarray) -> np.ndarray:
    """Columns: the strike direction, the handle direction made perpendicular to it, and their cross product."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.asarray(handle, float) - (np.asarray(handle, float) @ a) * a
    if np.linalg.norm(h) < 0.2:
        raise ValueError("the handle is (nearly) along the strike direction")
    h /= np.linalg.norm(h)
    return np.column_stack([a, h, np.cross(a, h)])


def lowpass(t: np.ndarray, x: np.ndarray, cutoff: float) -> np.ndarray:
    """Zero-phase low-pass of a sampled path (marker / tracking noise); unchanged if too short or too slow."""
    from scipy.signal import butter, filtfilt

    fs = 1.0 / float(np.median(np.diff(t)))
    if cutoff <= 0 or cutoff >= 0.45 * fs or len(t) < 16:
        return x
    b, a = butter(2, cutoff / (0.5 * fs))
    return filtfilt(b, a, x, axis=0)


def _rot_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    c = float(np.clip(u @ v, -1.0, 1.0))
    ax = np.cross(u, v)
    s = np.linalg.norm(ax)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
        return Rotation.from_rotvec(np.pi * perp / np.linalg.norm(perp)).as_matrix()
    return Rotation.from_rotvec(ax / s * np.arctan2(s, c)).as_matrix()


def map_motion(motion: ToolMotion, cfg: SimConfig, engage: float = 0.004, align_face: bool = True,
               nail: np.ndarray | None = None, cutoff: float = 12.0, aim_each: bool = True
               ) -> tuple[np.ndarray, np.ndarray]:
    """The source face path and tool orientations carried into our scene: (face (N,3), R (N,3,3)).

    With `aim_each`, every recorded contact (not only the first) is moved onto the nail by a smooth offset
    interpolated between contacts: the source's target can be far larger than our 9 mm nail head (Adroit's is
    70 mm across), so the recorded scatter of blows would otherwise miss it. This is the aiming a robot would do."""
    from tactile_sim.model.builder import tcp_rotation

    k = motion.contact_index
    a_t = strike_axis(cfg)
    p_t = nail_head_target(cfg) if nail is None else np.asarray(nail, float)
    G = task_frame(a_t, tcp_rotation(cfg)[:, 0]) @ task_frame(motion.axis, motion.R[k, :, 0]).T
    face_src = lowpass(motion.t, motion.face, cutoff)
    face = p_t + engage * a_t + (face_src - face_src[k]) @ G.T
    if aim_each and len(motion.contacts) > 1:
        from scipy.interpolate import PchipInterpolator

        idx = sorted(set(motion.contacts))
        off = np.array([p_t + engage * a_t - face[i] for i in idx])
        tc = motion.t[idx]
        corr = PchipInterpolator(tc, off, axis=0, extrapolate=False)(motion.t)
        corr[motion.t < tc[0]] = off[0]
        corr[motion.t > tc[-1]] = off[-1]
        face = face + corr
    q = Rotation.from_matrix(motion.R).as_quat()
    q *= np.where(np.cumsum(np.r_[0, np.einsum("ij,ij->i", q[1:], q[:-1]) < 0]) % 2 == 1, -1.0, 1.0)[:, None]
    q = lowpass(motion.t, q, cutoff)  # sign-continuous quaternions, filtered like the positions
    R = np.einsum("ij,njk->nik", G, Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix())
    if align_face:
        C = _rot_between(-R[k, :, 1], a_t)
        R = np.einsum("ij,njk->nik", C, R)
    return face, R


def _resample(t_src: np.ndarray, face: np.ndarray, R: np.ndarray, W: np.ndarray, rate: float):
    """Sample the source path at `rate` on the new time base; W[i] is the new time of source sample i (monotone)."""
    t = np.arange(0.0, W[-1], 1.0 / rate)
    src = np.interp(t, W, t_src)  # the source time each output sample shows
    f = CubicSpline(t_src, face, axis=0)(src)
    Rr = Slerp(t_src, Rotation.from_matrix(R))(src).as_matrix()
    return t, f, Rr, src


def _rate_profile(t, tcp, R, src, t_contacts_src=(), guard=0.04, edge=0.05):
    """Per-sample TCP speed, angular speed and acceleration (length len(t)). Acceleration within `guard` (source
    time) of a recorded contact, and within `edge` of either end, is zeroed: the first is the nail stopping the tool,
    the second the spline's boundary."""
    dt = float(np.median(np.diff(t)))
    v = np.r_[0.0, np.linalg.norm(np.diff(tcp, axis=0), axis=1) / dt]
    rv = (Rotation.from_matrix(R[1:]) * Rotation.from_matrix(R[:-1]).inv()).as_rotvec()
    w = np.r_[0.0, np.linalg.norm(rv, axis=1) / dt]
    acc = np.r_[0.0, np.linalg.norm(np.diff(tcp, 2, axis=0), axis=1) / dt**2, 0.0]
    mask = (t < edge) | (t > t[-1] - edge)
    for tc in t_contacts_src:
        mask |= np.abs(src - tc) < guard
    acc[mask] = 0.0
    v[(t < edge) | (t > t[-1] - edge)] = 0.0
    return v, w, acc


def ik_along(world, tcp: np.ndarray, R: np.ndarray, every: int = 10, iters: int = 30,
             damping: float = 1e-3) -> tuple[np.ndarray, float]:
    """Damped least-squares IK along the path from the hover pose; returns joint path and worst position residual."""
    m = world.model
    d = mujoco.MjData(m)
    d.qpos[:] = world.data.qpos
    if world.q_hover is not None:
        d.qpos[world.arm_qadr] = world.q_hover
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    lo, hi = m.jnt_range[[m.dof_jntid[i] for i in world.arm_dofs]].T
    qs, worst = [], 0.0
    for i in range(0, len(tcp), every):
        for _ in range(iters if qs else 20 * iters):  # converge fully at the start, then track
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[world.tcp_site]
            Rc = d.site_xmat[world.tcp_site].reshape(3, 3)
            e = np.concatenate([tcp[i] - p, Rotation.from_matrix(R[i] @ Rc.T).as_rotvec()])
            if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-4:
                break
            mujoco.mj_jacSite(m, d, jp, jr, world.tcp_site)
            J = np.vstack([jp[:, world.arm_dofs], jr[:, world.arm_dofs]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            d.qpos[world.arm_qadr] = np.clip(d.qpos[world.arm_qadr] + dq, lo, hi)
        mujoco.mj_kinematics(m, d)
        worst = max(worst, float(np.linalg.norm(tcp[i] - d.site_xpos[world.tcp_site])))
        qs.append(d.qpos[world.arm_qadr].copy())
    return np.array(qs), worst


def plan_replay(motion: ToolMotion, world, engage: float = 0.004, align_face: bool = True, speed: float = 1.0,
                v_max: float = 1.5, w_max: float = 4.0, a_max: float = 25.0, rate: float = 1000.0,
                max_residual: float = 0.01, check_joints: bool = True, max_slowdown: float = 10.0) -> ReplayPlan:
    """Map a motion onto `world`'s nail and make it trackable: returns the 1 kHz TCP path and its time scale.

    `speed` > 1 plays the motion faster than recorded (e.g. DexToolBench's slow tracked swings) before the limits
    slow it again if needed."""
    cfg = world.cfg
    face_s, R_s = map_motion(motion, cfg, engage, align_face)
    fo = np.asarray(face_offset(cfg.hammer), float)
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    t_c_src = [float(motion.t[i]) for i in motion.contacts]
    W = motion.t / speed  # new time of each source sample
    for _ in range(8):
        t, face, R, src = _resample(motion.t, face_s, R_s, W, rate)
        tcp = face - np.einsum("nij,j->ni", R, fo)
        v, w, acc = _rate_profile(t, tcp, R, src, t_c_src)
        need = np.maximum.reduce([np.ones_like(v), v / v_max, w / w_max, np.sqrt(acc / a_max)])
        if need.max() <= 1.0 + 1e-3:
            break
        # stretch time locally; widen and smooth the stretch so the warp itself stays smooth
        n = max(1, int(0.08 * rate))
        need = uniform_filter1d(maximum_filter1d(need, 2 * n + 1), 2 * n + 1) * 1.03
        stretched = np.r_[0.0, np.cumsum(need[1:] * np.diff(t))]
        W = np.interp(W, t, stretched, right=stretched[-1] + (W[-1] - t[-1]) * need[-1])
    notes = {"v_peak": float(v.max()), "w_peak": float(w.max()), "a_peak": float(acc.max())}
    q, resid = None, float("nan")
    if check_joints:
        every = 10
        for _ in range(4):
            q, resid = ik_along(world, tcp, R, every=every)
            qd = np.abs(np.diff(q, axis=0)) / (every / rate)
            ratio = float(np.max(qd / (0.9 * world.arm_spec.velocity), initial=0.0))
            notes["joint_speed_ratio"] = ratio
            if ratio <= 1.0:
                break
            if _ == 3 or W[-1] * ratio / motion.t[-1] > max_slowdown:
                raise ValueError(f"{motion.name}: the arm cannot follow this path smoothly (joint speed "
                                 f"{ratio:.1f}x the limit after slowing {W[-1] / motion.t[-1]:.1f}x: a singularity "
                                 "or a wrist flip)")
            W = W * ratio * 1.05
            t, face, R, src = _resample(motion.t, face_s, R_s, W, rate)
            tcp = face - np.einsum("nij,j->ni", R, fo)
        if resid > max_residual:
            raise ValueError(f"{motion.name}: the retargeted path leaves the arm's reach ({resid * 1e3:.0f} mm off)")
    t_contacts = [float(np.interp(tc, src, t)) for tc in t_c_src]
    notes["slowdown_mean"] = float(t[-1] / max(motion.t[-1], 1e-9))
    return ReplayPlan(t, tcp, R, face, t_contacts, float(t[-1] / max(motion.t[-1], 1e-9)), motion.name, q, resid,
                      notes)
