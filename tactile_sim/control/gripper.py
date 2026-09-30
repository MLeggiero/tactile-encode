"""Grasp-force loop (500 Hz): PI + feedforward on the summed pad normal force from the pressure arrays,
plus drop detection (grip force collapses while the pad accelerometer spikes)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tactile_sim.config import SimConfig


@dataclass
class GripLog:
    t: list = field(default_factory=list)
    setpoint: list = field(default_factory=list)
    measured: list = field(default_factory=list)
    command: list = field(default_factory=list)


class GripForceLoop:
    def __init__(self, world, cfg: SimConfig, sensors=None, log: bool = True):
        self.world = world
        self.cfg = cfg
        self.sensors = sensors
        self.dt = 1.0 / cfg.controller.grip_rate
        self.do_log = log
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.dropped = False
        self.t_drop = float("nan")
        self._low_since: float | None = None
        self.measured = 0.0
        self.log = GripLog()

    def measure(self) -> float:
        """Grip force per pad seen by the pressure arrays (mean of the two pads)."""
        if self.sensors is None or "pressure_L" not in self.sensors:
            return float(np.mean(self.world.pad_normal_forces()))
        l, r = self.sensors["pressure_L"].latest(), self.sensors["pressure_R"].latest()
        if l.seq < 0:
            return float(np.mean(self.world.pad_normal_forces()))
        return 0.5 * float(np.sum(l.value) + np.sum(r.value))

    def tick(self, t: float, setpoint: float, l1=None) -> float:
        c = self.cfg.controller
        g = self.cfg.gripper
        f = self.measure()
        self.measured = f
        err = setpoint - f
        u_ff = setpoint
        u = u_ff + c.grip_kp * err + c.grip_ki * self.integral
        if -g.grip_force_max < u < g.grip_force_max:  # anti-windup: integrate only when unsaturated
            self.integral += err * self.dt
        u = float(np.clip(u, -g.grip_force_max, g.grip_force_max))
        self.world.set_grip_force(u)
        self._check_drop(t, setpoint, f, l1)
        if self.do_log:
            L = self.log
            L.t.append(t)
            L.setpoint.append(setpoint)
            L.measured.append(f)
            L.command.append(u)
        return u

    def _check_drop(self, t: float, setpoint: float, f: float, l1) -> None:
        c = self.cfg.controller
        if self.dropped or setpoint <= 0:
            return
        acc = 0.0
        if self.sensors is not None and "pad_acc_L" in self.sensors:
            a = self.sensors["pad_acc_L"].latest().value
            acc = float(np.linalg.norm(a))
        if f < c.drop_force_frac * setpoint:
            self._low_since = t if self._low_since is None else self._low_since
            if t - self._low_since >= 0.020 or acc > c.drop_accel:
                self.dropped = True
                self.t_drop = t
                if l1 is not None:
                    l1.supervisor.freeze()
        else:
            self._low_since = None
