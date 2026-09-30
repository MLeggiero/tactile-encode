"""L1 real-time controller: runs the impedance law, momentum observer, impact detector and gating at
1 kHz on top of the World, and hands the grasp-force setpoint to the grip loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tactile_sim.config import SimConfig
from tactile_sim.control.impedance import CartesianImpedance, ImpedanceOutput
from tactile_sim.control.interface import L2Command, Mode
from tactile_sim.control.momentum_observer import ImpactDetector, ImpactEvent, MomentumObserver, external_wrench
from tactile_sim.control.supervisor import Supervisor
from tactile_sim.model.builder import strike_axis


@dataclass
class L1Log:
    t: list = field(default_factory=list)
    tau: list = field(default_factory=list)
    q: list = field(default_factory=list)
    qd: list = field(default_factory=list)
    x: list = field(default_factory=list)  # tcp pos + quat (7)
    x_eq: list = field(default_factory=list)
    xd_eq: list = field(default_factory=list)
    K: list = field(default_factory=list)
    F_ff: list = field(default_factory=list)
    F_grip: list = field(default_factory=list)
    mode: list = field(default_factory=list)
    impact_flag: list = field(default_factory=list)
    gated: list = field(default_factory=list)
    tau_ext: list = field(default_factory=list)
    f_ext: list = field(default_factory=list)


class L1Controller:
    def __init__(self, world, cfg: SimConfig | None = None, sensors=None, tau_ext_holder=None, log: bool = True):
        self.world = world
        self.cfg = cfg or world.cfg
        c = self.cfg.controller
        self.dt = 1.0 / c.rate
        self.sensors = sensors
        self.tau_ext_holder = tau_ext_holder
        self.impedance = CartesianImpedance(c, self.dt, world.q_hover if world.q_hover is not None else
                                            np.asarray(self.cfg.arm.q_seed))
        self.observer = MomentumObserver(7, c.observer_gain, self.dt)
        self.friction_scale = 1.0  # observer's friction model relative to the truth
        self.axis = strike_axis()
        self.detector = ImpactDetector(self.axis, c.impact_force_thresh, c.impact_slope_thresh, c.impact_refractory,
                                       accel_thresh=c.impact_accel_thresh)
        self.supervisor = Supervisor(c)
        self.payload_mass, self.payload_com = world.hammer_payload()
        self.payload_inertia = world.hammer_inertia_tcp()
        n_acc = 1
        if sensors is not None and "pad_acc_L" in sensors:
            n_acc = max(1, int(round(sensors["pad_acc_L"].spec.rate_hz / c.rate)))
        self.n_acc = n_acc
        self.cmd: L2Command | None = None
        self.gate_until = -np.inf
        self.last: ImpedanceOutput | None = None
        self.last_event: ImpactEvent | None = None
        self.f_ext = np.zeros(6)
        self.do_log = log
        self.log = L1Log()

    def reset(self) -> None:
        self.impedance.q_null = self.world.q_hover.copy() if self.world.q_hover is not None else self.impedance.q_null
        self.impedance.reset()
        self.observer.reset()
        self.detector.reset()
        self.supervisor.reset()
        self.cmd = None
        self.gate_until = -np.inf
        self.last = None
        self.last_event = None
        self.log = L1Log()

    def set_command(self, cmd: L2Command) -> None:
        self.cmd = cmd

    def hold_here(self, t: float, F_grip: float | None = None) -> L2Command:
        p, R = self.world.tcp_pose()
        c = self.cfg.controller
        cmd = L2Command(t, p, R, K=np.array(c.k_trans + c.k_rot, dtype=float),
                        F_grip=c.grip_hold if F_grip is None else F_grip)
        self.set_command(cmd)
        return cmd

    @property
    def gated(self) -> bool:
        return self.world.t < self.gate_until

    def _sensor(self, name):
        if self.sensors is None or name not in self.sensors:
            return None
        s = self.sensors[name].latest()
        return None if s.seq < 0 else s.value

    def tick(self, t: float) -> None:
        w = self.world
        s = w.arm_state()
        payload = w.payload_torque(self.payload_mass, self.payload_com)
        # observer: the rigid model includes the nominal payload's gravity; known torques are the
        # commanded motor torque and the modeled joint friction
        M_eff = s.M + w.payload_mass_matrix(self.payload_mass, self.payload_com, self.payload_inertia)
        r = self.observer.update(M_eff, s.qd, w.data.qfrc_actuator[w.arm_dofs], s.bias + payload, s.passive,
                                 self.friction_scale * w.joint_friction())
        self.f_ext = external_wrench(s.J, r)
        if self.tau_ext_holder is not None:
            self.tau_ext_holder.value = r.copy()
        ft = self._sensor("ft")
        f_ft_world = None if ft is None else w.data.site_xmat[w.site["ft_site"]].reshape(3, 3) @ ft[:3]
        acc = None
        if self.sensors is not None and "pad_acc_L" in self.sensors:
            acc = self.sensors["pad_acc_L"].window(self.n_acc)
        ev = self.detector.update(t, self.f_ext, f_ft_world, acc)
        if ev is not None:
            self.last_event = ev
            self.gate_until = t + self.cfg.controller.gate_duration
        cmd = self.supervisor.filter(self.cmd, t, s.tcp_pos, s.tcp_R)
        if cmd.ref is not None and not self.supervisor.frozen:
            t_flag = None
            if self.last_event is not None and self.last_event.t_flag >= getattr(cmd.ref, "t_armed", -np.inf):
                t_flag = self.last_event.t_flag
            x, xd, xdd, mode = cmd.ref.evaluate(t, t_flag, s.tcp_pos)
            cmd.x_eq, cmd.xd_eq, cmd.xdd_ff, cmd.mode = x, xd, xdd, mode
            cmd.R_eq = getattr(cmd.ref, "R", cmd.R_eq)
        # while gated, joint-velocity feedback is not trusted: damp against the reference twist
        xd_used = cmd.xd_eq if self.gated else s.tcp_vel
        if cmd.payload_ff and np.any(cmd.xdd_ff[:3]):
            # the payload's inertia is not in the arm's task inertia: feed it forward explicitly
            saved = cmd.F_ff
            cmd.F_ff = saved + np.concatenate([self.payload_mass * cmd.xdd_ff[:3], np.zeros(3)])
            out = self.impedance.compute(s, cmd, xd_used, payload)
            cmd.F_ff = saved
        else:
            out = self.impedance.compute(s, cmd, xd_used, payload)
        tau = w.set_arm_torque(out.tau)
        self.last = out
        self.grip_setpoint = cmd.F_grip
        if self.do_log:
            L = self.log
            L.t.append(t)
            L.tau.append(tau)
            L.q.append(s.q)
            L.qd.append(s.qd)
            quat = np.zeros(4)
            import mujoco

            mujoco.mju_mat2Quat(quat, s.tcp_R.reshape(-1))
            L.x.append(np.concatenate([s.tcp_pos, quat]))
            mujoco.mju_mat2Quat(quat, cmd.R_eq.reshape(-1))
            L.x_eq.append(np.concatenate([cmd.x_eq, quat]))
            L.xd_eq.append(cmd.xd_eq.copy())
            L.K.append(out.K)
            L.F_ff.append(cmd.F_ff.copy())
            L.F_grip.append(cmd.F_grip)
            L.mode.append(int(cmd.mode) if not self.supervisor.stale else int(Mode.HOLD))
            L.impact_flag.append(ev is not None)
            L.gated.append(self.gated)
            L.tau_ext.append(r.copy())
            L.f_ext.append(self.f_ext.copy())
