"""L1 for a position-interface arm (Dexmate Vega-1P through dexcontrol).

The arm accepts joint position targets with a velocity feedforward at `arm.command_rate` (100 Hz by default)
and tracks them with its own PD servos; it takes no torque commands. So the host side:

- runs impact detection at `controller.rate` (1 kHz) from the wrist F/T and the taxel-patch accelerometers,
  which are read independently of the arm's command stream (there is no joint-torque momentum observer);
- at each command tick evaluates the Cartesian reference one command period ahead, solves differential IK
  for the striking arm (orientation held, a weak null-space pull to the hover posture), adds the velocity
  feedforward, and offsets the target by the gravity droop g(q) / kp so the servo settles on it;
- limits every joint target's rate of change to 90 % of the joint's velocity limit;
- switches to the post-impact reference on the first command tick after the impact flag, which is up to one
  command period late: the arm cannot react between ticks.

Stiffness is whatever the servos have (factory gains x dexcontrol's P multiplier); the L2 command's
Cartesian stiffness is not applied.
"""

from __future__ import annotations

import mujoco
import numpy as np

from tactile_sim.config import SimConfig
from tactile_sim.control.interface import L2Command, Mode
from tactile_sim.control.kinematics import pose_error, site_jacobian, site_pose
from tactile_sim.control.l1 import L1Log
from tactile_sim.control.momentum_observer import ImpactDetector
from tactile_sim.control.supervisor import Supervisor
from tactile_sim.model.builder import strike_axis


class PositionL1:
    def __init__(self, world, cfg: SimConfig | None = None, sensors=None, tau_ext_holder=None, log: bool = True):
        self.world = world
        self.cfg = cfg or world.cfg
        c = self.cfg.controller
        self.dt = 1.0 / c.rate
        self.n_cmd = max(1, int(round(c.rate / self.cfg.arm.command_rate)))
        self.dt_cmd = self.n_cmd * self.dt
        self.sensors = sensors
        self.tau_ext_holder = tau_ext_holder
        self.axis = strike_axis(self.cfg)
        self.detector = ImpactDetector(self.axis, c.impact_force_thresh, c.impact_ft_thresh, c.impact_refractory,
                                       accel_thresh=c.impact_accel_thresh)
        self.supervisor = Supervisor(c)
        self.payload_mass, self.payload_com = world.hammer_payload()
        self.acc_name = world.hand.accel_names[0]
        self.n_acc = 1
        if sensors is not None and self.acc_name in sensors:
            self.n_acc = max(1, int(round(sensors[self.acc_name].spec.rate_hz / c.rate)))
        self._d = mujoco.MjData(world.model)  # scratch data for the IK
        self.do_log = log
        self.reset()

    def reset(self) -> None:
        w = self.world
        self.q_ref = w.data.qpos[w.arm_qadr].copy()
        self.q_null = w.q_hover.copy() if w.q_hover is not None else self.q_ref.copy()
        self.detector.reset()
        self.supervisor.reset()
        self.cmd: L2Command | None = None
        self.last_event = None
        self.gate_until = -np.inf
        self.grip_setpoint = self.cfg.controller.grip_hold
        self._k = 0
        self.f_ext = np.zeros(6)
        self.log = L1Log()

    @property
    def gated(self) -> bool:
        return False

    def set_command(self, cmd: L2Command) -> None:
        self.cmd = cmd

    def hold_here(self, t: float, F_grip: float | None = None) -> L2Command:
        p, R = self.world.tcp_pose()
        c = self.cfg.controller
        cmd = L2Command(t, p, R, K=np.array(c.k_trans + c.k_rot, dtype=float),
                        F_grip=c.grip_hold if F_grip is None else F_grip)
        self.set_command(cmd)
        return cmd

    # ------------------------------------------------------------------ IK
    def _fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w, d = self.world, self._d
        d.qpos[:] = w.data.qpos
        d.qpos[w.arm_qadr] = q
        mujoco.mj_kinematics(w.model, d)
        mujoco.mj_comPos(w.model, d)
        p, R = site_pose(d, w.tcp_site)
        return p, R, site_jacobian(w.model, d, w.tcp_site, w.arm_dofs)

    def _ik(self, x: np.ndarray, R: np.ndarray, xd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = self.q_ref.copy()
        lam = 1e-4
        for _ in range(3):
            p, Rq, J = self._fk(q)
            e = pose_error(x, R, p, Rq)
            JJt = J @ J.T + lam * np.eye(6)
            Jp = J.T @ np.linalg.solve(JJt, np.eye(6))
            N = np.eye(len(q)) - Jp @ J
            q = q + Jp @ e + N @ (0.02 * (self.q_null - q))
        _, _, J = self._fk(q)
        qd = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), xd)
        return q, qd

    # ------------------------------------------------------------------ ticks
    def tick(self, t: float) -> None:
        """Host loop at controller.rate: impact detection every tick, a position command every n_cmd ticks."""
        w = self.world
        ft = self._sensor("ft")
        f_ft_world = None if ft is None else w.data.site_xmat[w.site["ft_site"]].reshape(3, 3) @ ft[:3]
        t_ft = self.sensors["ft"].latest().t_sample if ft is not None else None
        acc = self.sensors[self.acc_name].window(self.n_acc) if self.sensors is not None and \
            self.acc_name in self.sensors else None
        ev = self.detector.update(t, None, f_ft_world, acc, t_ft)
        if ev is not None:
            self.last_event = ev
        if self._k % self.n_cmd == 0:
            self._command(t, ev is not None)
        elif self.do_log and ev is not None and self.log.impact_flag:
            self.log.impact_flag[-1] = True
        self._k += 1

    def _command(self, t: float, flagged: bool) -> None:
        w = self.world
        x_now, R_now = w.tcp_pose()
        cmd = self.supervisor.filter(self.cmd, t, x_now, R_now)
        if cmd.ref is not None and not self.supervisor.frozen:
            t_flag = None
            if self.last_event is not None and self.last_event.t_flag >= getattr(cmd.ref, "t_armed", -np.inf):
                t_flag = self.last_event.t_flag
            # one command period ahead: the target holds (ZOH) until the next tick
            x, xd, _, mode = cmd.ref.evaluate(t + self.dt_cmd, t_flag, x_now)
            cmd.x_eq, cmd.xd_eq, cmd.mode = x, xd, mode
            cmd.R_eq = getattr(cmd.ref, "R", cmd.R_eq)
        q, qd = self._ik(cmd.x_eq, cmd.R_eq, cmd.xd_eq)
        # never command a joint faster than 90 % of its velocity limit (a reference step, e.g. an aim
        # correction, would otherwise become a servo step)
        v_lim = 0.9 * w.arm_spec.velocity
        q = self.q_ref + np.clip(q - self.q_ref, -v_lim * self.dt_cmd, v_lim * self.dt_cmd)
        qd = np.clip(qd, -v_lim, v_lim)
        if self.supervisor.frozen or self.supervisor.stale:
            qd = np.zeros_like(qd)
        self.q_ref = q
        droop = w.gravity_torque(q) / w.servo_kp
        w.set_arm_position(q, qd, droop)
        w.hold_others()
        self.grip_setpoint = cmd.F_grip
        if self.do_log:
            L = self.log
            L.t.append(t)
            L.tau.append(w.arm_torque())
            L.q.append(w.data.qpos[w.arm_qadr].copy())
            L.qd.append(w.data.qvel[w.arm_dofs].copy())
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, R_now.reshape(-1))
            L.x.append(np.concatenate([x_now, quat]))
            mujoco.mju_mat2Quat(quat, cmd.R_eq.reshape(-1))
            L.x_eq.append(np.concatenate([cmd.x_eq, quat]))
            L.xd_eq.append(np.asarray(cmd.xd_eq, float).copy())
            L.K.append(np.concatenate([w.servo_kp[:3], w.servo_kp[3:6]]))
            L.F_ff.append(np.zeros(6))
            L.F_grip.append(cmd.F_grip)
            L.mode.append(int(cmd.mode) if not self.supervisor.stale else int(Mode.HOLD))
            L.impact_flag.append(flagged)
            L.gated.append(False)
            L.tau_ext.append(np.zeros(7))
            L.f_ext.append(np.zeros(6))

    def _sensor(self, name):
        if self.sensors is None or name not in self.sensors:
            return None
        s = self.sensors[name].latest()
        return None if s.seq < 0 else s.value
