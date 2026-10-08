"""Scripted L2 stand-ins for the saw and driver tasks, emitting the same L2Command (pose + stiffness + feedforward
wrench + grip force) as the hammer's ScriptedSwing.

SawStroke: lower the teeth onto the board, then stroke sinusoidally along the blade with a steady push into the cut
and a soft spring along the cut direction, until the cut reaches its target; lift out.

DrillFeed: bring the bit to the screw head (or the board), press with a feedforward push and a soft spring along the
bit, pull the trigger once the wrist F/T reads the push, follow the bit in, and release when the tool reports the
screw seated (clutch slipping) or the bit through; back out. The stop uses the tool's own telemetry (what a driver's
motor current would show), not the plant's ground truth depth.
"""

from __future__ import annotations

import numpy as np

from tactile_sim.control.interface import L2Command


def _min_jerk(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x**3 * (10 - 15 * x + 6 * x**2)


class _ToolBehavior:
    def __init__(self, tb, t_max: float = 30.0):
        self.tb = tb
        self.cfg = tb.cfg
        self.w = tb.world
        self.t_max = t_max
        self.R = self.w.tcp_R_nominal.copy()
        self.reset()

    def reset(self) -> None:
        self.phase = "start"
        self.t0 = self.tb.t
        self.t_phase = self.tb.t
        self.x0 = self.w.ctrl_pose()[0].copy()
        c = self.cfg.controller
        self.cmd = L2Command(self.tb.t, self.x0.copy(), self.R.copy(), K=np.array(c.k_trans + c.k_rot, dtype=float),
                             R_K=self.R.copy(), F_grip=c.grip_hold)
        self.tb.l1.set_command(self.cmd)
        self.events: list[tuple[str, float]] = []

    def _set_phase(self, name: str, t: float) -> None:
        self.phase = name
        self.t_phase = t
        self.events.append((name, t))

    def grip(self, t: float) -> float:
        """Grip-force setpoint per pad. Constant here; the predictive load-based grip replaces this."""
        return self.cfg.controller.grip_hold

    @property
    def finished(self) -> bool:
        return self.phase == "done"

    def tick(self, t: float) -> None:
        self.cmd.t = t
        self._tick(t)
        self.cmd.F_grip = self.grip(t)
        self.tb.l1.grip_setpoint = self.cmd.F_grip

    def _tick(self, t: float) -> None:
        raise NotImplementedError


class SawStroke(_ToolBehavior):
    def _tick(self, t: float) -> None:
        s, cmd, R = self.cfg.saw, self.cmd, self.R
        u, f, n = R[:, 0], R[:, 1], R[:, 2]  # stroke, cut (the fingers' closing axis), across the blade
        c = self.cfg.controller
        K_stiff = np.array(c.k_trans + c.k_rot, dtype=float)
        x_now = self.w.ctrl_pose()[0]
        plant = self.w.plant
        if self.phase == "start":
            self._set_phase("approach", t)
        if self.phase == "approach":
            a = _min_jerk((t - self.t_phase) / s.approach_time)
            cmd.x_eq = self.x0 + f * (s.clearance + 0.002) * a
            cmd.xd_eq = np.zeros(6)
            cmd.K = K_stiff
            cmd.F_ff = np.concatenate([f * s.push_force * a, np.zeros(3)])
            if t - self.t_phase >= s.approach_time:
                self.base = x_now.copy()
                self._set_phase("stroke", t)
        if self.phase == "stroke":
            tau = t - self.t_phase
            ramp = min(tau * s.stroke_freq, 1.0)  # amplitude builds over the first stroke
            w = 2 * np.pi * s.stroke_freq
            A = s.stroke_amp * ramp
            cmd.x_eq = (self.base + u * A * np.sin(w * tau) + n * 0.0
                        + f * (float((x_now - self.base) @ f) + 0.01))  # the cut-direction spring follows the blade
            cmd.x_eq = cmd.x_eq - n * float((cmd.x_eq - self.base) @ n)
            cmd.xd_eq = np.concatenate([u * A * w * np.cos(w * tau), np.zeros(3)])
            cmd.xdd_ff = np.concatenate([-u * A * w * w * np.sin(w * tau), np.zeros(3)])
            ks, kn, kf = s.k_stroke
            cmd.K = np.array([ks, kf, kn] + list(c.k_rot), dtype=float)  # in R's (u, f, n) columns
            cmd.F_ff = np.concatenate([f * s.push_force, np.zeros(3)])
            if plant.done() or t - self.t0 > self.t_max:
                self.lift_from = x_now.copy()
                self._set_phase("lift", t)
        if self.phase == "lift":
            a = _min_jerk((t - self.t_phase) / 0.6)
            cmd.x_eq = self.lift_from + (self.x0 - self.lift_from) * a
            cmd.xd_eq = np.zeros(6)
            cmd.xdd_ff = np.zeros(6)
            cmd.K = K_stiff
            cmd.F_ff = np.zeros(6)
            if t - self.t_phase >= 0.6:
                self._set_phase("done", t)


class DrillFeed(_ToolBehavior):
    def reset(self) -> None:
        super().reset()
        self.ft_base = None
        self.t_trigger = float("nan")

    def _axial_ft(self) -> float | None:
        """Push measured by the wrist F/T along the bit (N, positive = pushing the tool into the work)."""
        sen = self.tb.sensors
        if "ft" not in sen or sen["ft"].latest().seq < 0:
            return None
        w = self.w
        f_world = w.data.site_xmat[w.site["ft_site"]].reshape(3, 3) @ sen["ft"].latest().value[:3]
        return float(f_world @ self.R[:, 2])

    def _tick(self, t: float) -> None:
        dc, cmd, R = self.cfg.drill, self.cmd, self.R
        a = R[:, 2]  # the bit (the hand's approach axis)
        c = self.cfg.controller
        K_stiff = np.array(c.k_trans + c.k_rot, dtype=float)
        x_now = self.w.ctrl_pose()[0]
        plant = self.w.plant
        if self.phase == "start":
            self._set_phase("approach", t)
        if self.phase == "approach":
            s = _min_jerk((t - self.t_phase) / dc.approach_time)
            cmd.x_eq = self.x0 + a * (dc.clearance - 0.002) * s
            cmd.K = K_stiff
            cmd.F_ff = np.zeros(6)
            if t - self.t_phase >= dc.approach_time:
                self.ft_base = self._axial_ft()
                self.lat0 = self.x0.copy()
                self._set_phase("press", t)
        if self.phase in ("press", "drive"):
            ramp = min((t - self.t_phase) / 0.2, 1.0) if self.phase == "press" else 1.0
            # soft along the bit (the push is feedforward), stiff across it to hold the bit on the axis
            along = float((x_now - self.lat0) @ a) + 0.005
            cmd.x_eq = self.lat0 + a * along
            k_along, k_across = dc.k_feed
            cmd.K = np.array([k_across, k_across, k_along] + list(c.k_rot), dtype=float)
            cmd.F_ff = np.concatenate([a * dc.push_force * ramp, np.zeros(3)])
            f_ax = self._axial_ft()
            if self.phase == "press" and f_ax is not None and self.ft_base is not None \
                    and abs(f_ax - self.ft_base) > dc.trigger_contact:
                plant.trigger = 1.0
                self.t_trigger = t
                self._set_phase("drive", t)
            if self.phase == "drive" and (plant.done() or plant.state == "stripped" or t - self.t0 > self.t_max):
                plant.trigger = 0.0
                self.back_from = x_now.copy()
                self._set_phase("retract", t)
            if self.phase == "press" and t - self.t_phase > 2.0:  # never touched: give up
                self.back_from = x_now.copy()
                self._set_phase("retract", t)
        if self.phase == "retract":
            s = _min_jerk((t - self.t_phase) / 0.6)
            cmd.x_eq = self.back_from + (self.x0 - self.back_from) * s
            cmd.K = K_stiff
            cmd.F_ff = np.zeros(6)
            if t - self.t_phase >= 0.6:
                self._set_phase("done", t)


class PositionDrillFeed(_ToolBehavior):
    """The driver task on a position-interface arm with no wrist F/T (the Vega U): bring the bit to the screw head
    (or the board), then command it press_depth past where it is so the servos push, ramped over press_time; pull
    the trigger trigger_delay later; keep the target press_depth ahead of the bit as the screw goes in (a hole: advance
    at feed_rate); release release_delay after the tool reports the screw seated (clutch slipping) or the bit
    through; back out. The bit is the pistol driver's -y axis or the inline driver's +z. With DrillCfg.aim_from_tool
    the hand's orientation and path are corrected once, at the start, for where the tool settled (aim_from_tool)."""

    def reset(self) -> None:
        self.R = self.w.tcp_R_nominal.copy()  # a reset aims afresh from the nominal frame
        super().reset()
        dc = self.cfg.drill
        self.a = -self.R[:, 1] if dc.form == "pistol" else self.R[:, 2]
        self.t_trigger = float("nan")
        self.t_done_seen = float("nan")
        self.trigger = 0.0  # the base's trigger command (0..1)
        self._t_pull: float | None = None
        self._pulled = False
        self._t_release: float | None = None
        self.aim_correction = (0.0, np.zeros(3))  # (rad turned, m shifted) at the start
        if dc.aim_from_tool:
            R_fix, shift = self.aim_from_tool()
            self.R = R_fix @ self.R
            self.cmd.R_eq = self.R.copy()
            self.cmd.R_K = self.R.copy()
            self.x0 = self.x0 - shift
            self.cmd.x_eq = self.x0.copy()
            self.aim_correction = (float(np.arccos(np.clip((np.trace(R_fix) - 1) / 2, -1, 1))), -shift)

    def aim_from_tool(self) -> tuple[np.ndarray, np.ndarray]:
        """One look at the tool where the grasp settled, before the task (as a robot would take with its head
        camera): the turn of the hand that lines the bit up with the work's axis, and how far across that axis the
        bit tip then sits from it. The base turns its orientation by the first and shifts its whole path by the
        second."""
        plant, d = self.w.plant, self.w.data
        a_bit = d.xmat[plant.tool].reshape(3, 3) @ plant.bit_local
        axis = np.cross(a_bit, plant.a)
        sin, cos = float(np.linalg.norm(axis)), float(a_bit @ plant.a)
        R_fix = np.eye(3)
        if sin > 1e-9:
            k = axis / sin
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R_fix = np.eye(3) + sin * K + (1 - cos) * K @ K  # Rodrigues: a_bit onto the work axis
        tcp = self.w.ctrl_pose()[0]
        tip = tcp + R_fix @ (d.site_xpos[plant.tip_site] - tcp)  # the tip once the hand has turned about the TCP
        e = tip - plant.head0
        return R_fix, e - (e @ plant.a) * plant.a

    def target(self, t: float) -> np.ndarray:
        """Where the base wants the grasp frame at time t (the residual's lookahead evaluates this ahead)."""
        dc = self.cfg.drill
        if self.phase in ("start", "approach"):
            s = _min_jerk((t - self.t_phase) / dc.approach_time) if self.phase == "approach" else 0.0
            return self.x0 + self.a * (dc.clearance - 0.002) * s
        if self.phase in ("press", "drive"):
            ramp = min((t - self.t_phase) / dc.press_time, 1.0) if self.phase == "press" else 1.0
            if dc.mode == "hole" and self.phase == "drive":  # feed-rate control
                return self.lat0 + self.a * (dc.press_depth + dc.feed_rate * (t - self.t_trigger))
            return self.lat0 + self.a * (self.along + dc.press_depth * ramp)
        if self.phase == "retract":
            s = _min_jerk((t - self.t_phase) / 0.6)
            return self.back_from + (self.x0 - self.back_from) * s
        return self.cmd.x_eq.copy()

    def _tick(self, t: float) -> None:
        dc, plant = self.cfg.drill, self.w.plant
        x_now = self.w.ctrl_pose()[0]
        if self.phase == "start":
            self._set_phase("approach", t)
        if self.phase == "approach" and t - self.t_phase >= dc.approach_time:
            self.lat0 = self.x0 + self.a * (dc.clearance - 0.002)
            self.along = 0.0
            self._set_phase("press", t)
        if self.phase in ("press", "drive"):
            # the target rides press_depth ahead of the bit as it goes in
            self.along = max(self.along, float((x_now - self.lat0) @ self.a))
            if self.phase == "press" and t - self.t_phase >= dc.press_time + dc.trigger_delay:
                self.trigger = dc.trigger_level
                self.t_trigger = t
                self._set_phase("drive", t)
            if self.phase == "drive" and plant.done() and np.isnan(self.t_done_seen):
                self.t_done_seen = t  # the clutch clicks; the trigger stays pulled release_delay longer
            released = not np.isnan(self.t_done_seen) and t - self.t_done_seen >= dc.release_delay
            if self.phase == "drive" and (released or plant.state == "stripped" or t - self.t0 > self.t_max):
                self.trigger = 0.0
                self.back_from = x_now.copy()
                self._set_phase("retract", t)
        if self.phase == "retract" and t - self.t_phase >= 0.6:
            self._set_phase("done", t)
        if getattr(plant, "trigger_qadr", -1) >= 0:
            # a trigger part: the index finger pulls it, its closing level ramped toward the command over
            # trigger_pull_time; the motor runs on the trigger's travel
            syn = self.w.hand.synergy
            goal = dc.trigger_rest + self.trigger * (dc.trigger_full - dc.trigger_rest)
            if self.trigger <= 0.0 and self._pulled:
                # letting go: the finger relaxes fully for trigger_release_time, then rests on the trigger again
                # (resting on a part-pulled trigger can hold it there)
                if self._t_release is None:
                    self._t_release = t
                if t - self._t_release < dc.trigger_release_time:
                    goal = 0.0
                else:
                    self._pulled = False
            if self.trigger > 0.0:
                self._pulled, self._t_release = True, None
            dt = t - self._t_pull if self._t_pull is not None else 0.0
            step = (dc.trigger_full - dc.trigger_rest) * dt / dc.trigger_pull_time
            syn.s_trigger += float(np.clip(goal - syn.s_trigger, -step, step))
            self._t_pull = t
        else:
            plant.trigger = self.trigger
        self.cmd.x_eq = self.target(t)


def make_behavior(tb, t_max: float = 30.0) -> _ToolBehavior:
    kind = tb.cfg.plant.kind
    if kind == "saw":
        return SawStroke(tb, t_max)
    if kind == "drill":
        return PositionDrillFeed(tb, t_max) if tb.world.position_arm else DrillFeed(tb, t_max)
    raise ValueError(f"no scripted tool behavior for plant kind {kind!r} (the hammer uses ScriptedSwing)")
