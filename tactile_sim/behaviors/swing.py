"""Scripted swing: a 200 Hz state machine that plays the role of L2 plus a trivial L3.

Phases: approach -> windup -> swing (ante reference through the nail, detector armed 20 ms before the
predicted contact) -> post (hold, then recover to the hover pose) -> settle -> next strike or done.

The stroke is a straight line along the strike axis. The swing is a quintic from rest at the windup
point to the strike speed at the predicted contact, then continues at that speed for `overshoot`
past it (reference spreading's ante reference). Stiffness is high along the strike axis and moderate
across it during the swing, nominal otherwise.

Grip follows the human impact template (Johansson & Westling; White et al.): hold force until
t_c - 150 ms, ramp to a pre-load scaled by tool momentum by t_c - 50 ms, keep rising after the impact
flag to a peak ~60 ms later, then decay back to the hold force over 200 ms. Slip measured on one strike
raises the hold margin for the next one (feedback updates the next strike, not the current one).

Aiming: the swing line is shifted so the hammer face, not the TCP, passes through the nail. Before each
windup the face position at the settled hover pose is predicted from the TCP pose and the tool-in-hand
estimate (L0's "tool pose drift in the grasp"), so slip from the previous blow is re-aimed. The lateral
drift the swing itself adds (friction, dynamic tracking error) is learned across strikes with gain
`aim_gain` and pre-compensated (iterative re-aiming). The estimates default to simulation truth,
standing in for L0 and for vision of the nail.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tactile_sim.behaviors.trajectories import LinePath, Segment, min_jerk, quintic_coeffs, swing_duration
from tactile_sim.control.interface import L2Command, Mode
from tactile_sim.control.reference_spreading import PlainRef, ReferenceSpreader
from tactile_sim.model.builder import strike_axis


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (10 - 15 * x + 6 * x * x)


@dataclass
class StrikePlan:
    idx: int
    t_start: float  # swing start
    t_c_pred: float
    s_contact: float  # predicted contact, along the strike axis from the hover pose
    v_strike: float
    F_hold: float
    F_pre: float
    F_peak: float
    t_arm: float
    t_flag: float | None = None
    missed: bool = False


class ScriptedSwing:
    def __init__(self, tb, n_strikes: int | None = None, depth_estimate: Callable[[], float] | None = None,
                 on_swing_start=None, on_strike_end=None, tool_estimate=None, nail_estimate=None,
                 aim_gain: float = 0.7):
        self.tb = tb
        self.cfg = tb.cfg
        sw = self.cfg.swing
        self.n_strikes = sw.n_strikes if n_strikes is None else n_strikes
        self.axis = strike_axis()
        self.origin = tb.world.hover_tcp.copy()
        self.R = tb.world.tcp_R_nominal.copy()
        # strike frame for the stiffness: z along the strike axis
        x_k = np.array([1.0, 0.0, 0.0])
        z_k = self.axis
        self.R_K = np.column_stack([x_k, np.cross(z_k, x_k), z_k])
        c = self.cfg.controller
        self.K_nom = np.array(c.k_trans + c.k_rot, dtype=float)
        self.K_strike = np.array(tuple(sw.strike_k) + (250.0, 250.0, 250.0))
        self.depth_estimate = depth_estimate or (lambda: tb.world.plant.depth)
        self.tool_estimate = tool_estimate or tb.world.hammer_in_hand
        w = tb.world
        self.nail_estimate = nail_estimate or (lambda: w.data.site_xpos[w.site["nail_head_top"]].copy())
        from tactile_sim.model.tool_hammer import face_offset

        self.face_local = np.array(face_offset(self.cfg.hammer))
        self.aim_gain = aim_gain
        self.aim_offset = np.zeros(3)  # line origin shift from the nominal hover pose
        self.drift_per_v2 = np.zeros(3)  # learned lateral face drift per (strike speed)^2
        self.face_hover = None
        self.on_swing_start = on_swing_start
        self.on_strike_end = on_strike_end
        self.grip_margin = 0.0
        self.m_tool = tb.l1.payload_mass
        self.reset()

    # ------------------------------------------------------------------
    def face_estimate(self) -> np.ndarray:
        """Face position from the TCP pose and the tool-in-hand estimate."""
        p_t, R_t = self.tb.world.tcp_pose()
        p_rel, rv = self.tool_estimate()
        import mujoco

        q = np.zeros(4)
        ang = float(np.linalg.norm(rv))
        mujoco.mju_axisAngle2Quat(q, rv / ang if ang > 0 else np.array([1.0, 0, 0]), ang)
        R_rel = np.zeros(9)
        mujoco.mju_quat2Mat(R_rel, q)
        return p_t + R_t @ (p_rel + R_rel.reshape(3, 3) @ self.face_local)

    def _lateral(self, v: np.ndarray) -> np.ndarray:
        return v - (v @ self.axis) * self.axis

    def reset(self) -> None:
        self.phase = "approach"
        self.phase_t0 = self.tb.t
        self.k = 0
        self.plan: StrikePlan | None = None
        self.plans: list[StrikePlan] = []
        self.cmd = L2Command(self.tb.t, self.origin.copy(), self.R.copy(), K=self.K_nom.copy(),
                             F_grip=self.cfg.controller.grip_hold)
        x_now = self.tb.world.tcp_pose()[0]
        s0 = float((x_now - self.origin) @ self.axis)
        T = self.cfg.swing.approach_time
        self.line_origin = self.origin.copy()
        self.cmd.ref = PlainRef(LinePath(self.origin, self.axis, self.R, [Segment(self.tb.t, T, min_jerk(s0, 0.0, T))]))
        self.tb.l1.set_command(self.cmd)
        self.done = False

    @property
    def F_hold(self) -> float:
        return min(self.cfg.controller.grip_hold + self.grip_margin, 70.0)  # Franka Hand continuous rating

    def _set_phase(self, name: str, t: float) -> None:
        self.phase = name
        self.phase_t0 = t

    # ------------------------------------------------------------------
    def tick(self, t: float) -> None:
        sw = self.cfg.swing
        cmd = self.cmd
        cmd.t = t
        if self.done:
            return
        if self.tb.grip.dropped:
            self.done = True
            return
        if self.phase == "approach":
            if t >= self.phase_t0 + sw.approach_time + 0.1:
                self._start_windup(t)
        elif self.phase == "windup":
            if t >= self.phase_t0 + sw.windup_time + 0.05:
                self._start_swing(t)
        elif self.phase == "swing":
            p = self.plan
            if not self.tb.l1.detector.armed and p.t_flag is None and t >= p.t_arm and \
                    not getattr(self, "_armed_once", False):
                self.tb.l1.detector.arm(True)
                self._armed_once = True
            rs = cmd.ref
            if rs.t_switch is not None:
                p.t_flag = None if rs.missed else rs.t_switch
                p.missed = rs.missed
                self.tb.l1.detector.arm(False)
                self._set_phase("post", rs.t_switch)
                cmd.K = self.K_nom.copy()
        elif self.phase == "post":
            if t >= self.phase_t0 + 0.02 + sw.recover_time:
                self._set_phase("settle", t)
                cmd.ref = PlainRef(LinePath(self.line_origin, self.axis, self.R,
                                            [Segment(t, 0.001, min_jerk(0, 0, 0.001))]))
                cmd.R_K = np.eye(3)
        elif self.phase == "settle":
            if t >= self.phase_t0 + sw.settle_time:
                if self.on_strike_end is not None:
                    self.on_strike_end(self.plan)
                self.k += 1
                if self.k >= self.n_strikes or self.tb.world.plant.done():
                    self.done = True
                else:
                    self._start_windup(t)
        cmd.F_grip = self._grip(t)

    # ------------------------------------------------------------------
    def _start_windup(self, t: float) -> None:
        sw = self.cfg.swing
        self._set_phase("windup", t)
        self.cmd.K = self.K_nom.copy()
        self.cmd.R_K = np.eye(3)
        self.line_origin = self.origin + self.aim_offset
        x_now = self.tb.world.tcp_pose()[0]
        s0 = float((x_now - self.line_origin) @ self.axis)
        self.cmd.ref = PlainRef(LinePath(self.line_origin, self.axis, self.R,
                                         [Segment(t, sw.windup_time, min_jerk(s0, -sw.windup_height, sw.windup_time))]))

    def _start_swing(self, t: float) -> None:
        sw = self.cfg.swing
        self._set_phase("swing", t)
        # static aim at the windup pose: shift the line so the face, plus the drift the swing is expected
        # to add, lands on the nail. Tracking drift grows with the swing's acceleration, i.e. ~ v^2.
        v_next = sw.first_tap_speed if (self.k == 0 and sw.first_tap_speed > 0) else sw.v_strike
        self._v_swing = v_next
        self.face_hover = self.face_estimate()
        err = self._lateral(self.nail_estimate() - self.drift_per_v2 * v_next**2 - self.face_hover)
        self._aim_step = err
        self.aim_offset += err
        self.line_origin = self.origin + self.aim_offset
        # along-axis distance from the face (at the hover point of the line) to the nail head
        x_now = self.tb.world.tcp_pose()[0]
        s_now = float((x_now - self.line_origin) @ self.axis)
        s_c = float((self.nail_estimate() - self.face_estimate()) @ self.axis) + s_now
        dist = s_c - s_now
        v = self._v_swing
        T = swing_duration(dist, v, a_max=60.0)
        t_c = t + T
        seg_swing = Segment(t, T, quintic_coeffs(s_now, 0.0, 0.0, s_c, v, 0.0, T))
        T_over = sw.overshoot / v
        seg_over = Segment(t_c, T_over, quintic_coeffs(s_c, v, 0.0, s_c + sw.overshoot, v, 0.0, T_over))
        ante = LinePath(self.line_origin, self.axis, self.R, [seg_swing, seg_over])
        rs = ReferenceSpreader(ante, self._make_post, t_c, interim_lead=0.010, timeout=sw.strike_timeout)
        rs.R = self.R
        rs.t_armed = t_c - 0.020
        F_hold = self.F_hold
        F_pre = float(np.clip(F_hold + sw.grip_pre_gain * self.m_tool * v, F_hold, 120.0))
        self.plan = StrikePlan(self.k, t, t_c, s_c, v, F_hold, F_pre, min(1.2 * F_pre, 140.0), t_c - 0.020)
        self.plans.append(self.plan)
        self._armed_once = False
        self.cmd.ref = rs
        self.cmd.K = self.K_strike.copy()
        self.cmd.R_K = self.R_K.copy()
        self.cmd.t_c_pred = t_c
        if self.on_swing_start is not None:
            self.on_swing_start(self.plan)

    def _make_post(self, t_switch: float, x_now: np.ndarray):
        sw = self.cfg.swing
        # iterative re-aim: learn the lateral drift the swing adds between its start and contact (or timeout);
        # the face was shifted by the static aim at swing start, so measure from the aimed position
        if self.face_hover is not None:
            d_k = self._lateral(self.face_estimate() - (self.face_hover + self._aim_step))
            self.drift_per_v2 += self.aim_gain * (d_k / self._v_swing**2 - self.drift_per_v2)
        s_f = float((x_now - self.line_origin) @ self.axis)
        segs = [Segment(t_switch, 0.02, min_jerk(s_f, s_f, 0.02)),
                Segment(t_switch + 0.02, sw.recover_time, min_jerk(s_f, 0.0, sw.recover_time))]
        return LinePath(self.line_origin, self.axis, self.R, segs)

    # ------------------------------------------------------------------
    def _grip(self, t: float) -> float:
        sw = self.cfg.swing
        p = self.plan
        F_hold = self.F_hold
        if p is None or self.phase in ("approach", "windup") and p.idx != self.k:
            return F_hold
        if p.t_flag is None and not p.missed:
            t_ramp0 = p.t_c_pred - sw.grip_lead
            t_ramp1 = p.t_c_pred - sw.grip_ramp_end
            if t < t_ramp0:
                return p.F_hold
            return p.F_hold + (p.F_pre - p.F_hold) * _smoothstep((t - t_ramp0) / (t_ramp1 - t_ramp0))
        t_ref = p.t_flag if p.t_flag is not None else p.t_c_pred + sw.strike_timeout
        t_pk = t_ref + sw.grip_peak_delay
        if t < t_pk:
            return p.F_pre + (p.F_peak - p.F_pre) * _smoothstep((t - t_ref) / sw.grip_peak_delay)
        return p.F_peak + (F_hold - p.F_peak) * _smoothstep((t - t_pk) / sw.grip_decay)

    def register_slip(self, slip_trans: float, slip_rot: float) -> None:
        """Feedback for the next strike: raise the hold margin when this strike slipped."""
        if slip_trans > 0.002 or slip_rot > np.radians(0.5):
            self.grip_margin = min(self.grip_margin + self.cfg.swing.grip_margin_step, 40.0)

    @property
    def mode(self) -> Mode:
        return self.cmd.mode
