"""Grasp-force loop (500 Hz): PI + feedforward on the summed pad normal force from the pressure arrays,
plus drop detection (grip force collapses while the pad accelerometer spikes)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tactile_sim.config import SimConfig
from tactile_sim.sensors.pressure import grip_force


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
        self._last_setpoint = None
        self._t_decrease = -np.inf
        self.measured = 0.0
        self.log = GripLog()

    def measure(self) -> float:
        """Grip force seen by the pressure arrays (Franka: mean per pad; dexterous hand: summed patches)."""
        hand = self.world.hand
        names = hand.pressure_names
        if self.sensors is None or names[0] not in self.sensors:
            return self.world.grip_truth()
        samples = [self.sensors[n].latest() for n in names]
        if samples[0].seq < 0:
            return self.world.grip_truth()
        floor = 2.0 * self.cfg.sensors.pressure_noise
        return hand.grip_from_patches(np.array([grip_force(s.value, floor) for s in samples]))

    def tick(self, t: float, setpoint: float, l1=None) -> float:
        c = self.cfg.controller
        f_max = self.world.hand.grip_force_max
        f = self.measure()
        self.measured = f
        err = setpoint - f
        u_ff = setpoint
        u = u_ff + c.grip_kp * err + c.grip_ki * self.integral
        lo = 0.0 if setpoint > 0 else -f_max
        if lo < u < f_max:  # anti-windup: integrate only when unsaturated
            self.integral += err * self.dt
        # while holding, the loop may relax the squeeze but never drive the fingers open
        u = float(np.clip(u, lo, f_max))
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
        if self._last_setpoint is None:  # first tick: compare with what the pads actually carry
            self._last_setpoint = max(setpoint, f)
        if setpoint < self._last_setpoint - 1e-9:
            self._t_decrease = t
        self._last_setpoint = setpoint
        # a commanded release unloads the pads for a moment; that is not a drop
        if self.dropped or setpoint <= 0 or t - self._t_decrease < 0.1:
            self._low_since = None
            return
        acc = 0.0
        acc_name = self.world.hand.accel_names[0]
        if self.sensors is not None and acc_name in self.sensors:
            a = self.sensors[acc_name].latest().value
            acc = float(np.linalg.norm(a))
        if c.drop_impact_holdoff > 0 and l1 is not None and l1.last_event is not None and \
                t - l1.last_event.t_flag < c.drop_impact_holdoff:
            acc = 0.0  # the blow itself, not a tool leaving the hand
        if f < max(c.drop_force_frac * setpoint, c.drop_force_abs):
            self._low_since = t if self._low_since is None else self._low_since
            if t - self._low_since >= 0.020 or acc > c.drop_accel:
                self.dropped = True
                self.t_drop = t
                if l1 is not None:
                    l1.supervisor.freeze()
        else:
            self._low_since = None
