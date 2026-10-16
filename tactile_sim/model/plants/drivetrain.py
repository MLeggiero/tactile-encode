"""A cordless drill/driver's drivetrain at the tool level, referred to the chuck: a brushed motor behind a planetary
gearbox, and a ball-detent clutch on the last stage's ring gear. Pure numpy and no MuJoCo, so the drill plant
(plants/drill.py) and the virtual dyno (tactile_sim.calibrate drill) step the same model.

- Motor: tau_m = stall * (u - w / w0) at the chuck. The trigger's PWM scales the voltage, so the speed line keeps its
  slope; with no soft start (a brushed drill's trigger) u follows the trigger, else it slews up at trigger_slew per
  second. Released, the motor is shorted (dynamic brake): the same line at u = 0, capped at brake_torque.
- Inertia: the rotor spins N times faster than the chuck, so it is reflected as J_r N^2. The spindle side carries
  J_spin = J_r N^2 + J_chuck, while the internals' angular momentum is only J_am w with J_am = J_r N + J_chuck. The
  housing reaction about the bit is tau_chuck + J_am dw/dt (the internals' angular momentum balance): a free
  spin-up kicks the hand with about the motor shaft's torque, not the chuck's.
- Clutch: below the breakaway (clutch_torque, and clutch_overshoot more to start it from rest: stiction) the clutch is
  rigid and the chuck turns with the spindle. A load that needs more holds the chuck while the ring gear turns, its
  balls climbing a ramp (the torque rises from clutch_low to 1 x clutch_torque over clutch_duty of a detent) and
  dropping into the next (nothing). The ring turns clutch_ring_ratio x the spindle speed, clutch_detents ratchets a
  turn, so the clicks' rate follows the motor's speed under the clutch's load. clutch_lock: drill mode (no clutch).
- The load is quasi-static (its inertia is in J_spin): a resistance `need` that holds the chuck if the drive cannot
  turn it (a screw, a bit in wood), or, with hold=False, an elastic one that can turn it back (a test joint).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DrivetrainParams:
    free_speed: float  # rad/s at the chuck, full trigger, no load
    stall_torque: float  # N m at the chuck, full trigger, quasi-static (ISO 5393 soft joint)
    gear_ratio: float  # motor to chuck
    spindle_inertia: float  # kg m^2 at the chuck: J_r N^2 + J_chuck
    chuck_inertia: float = 3e-5
    trigger_slew: float = 0.0  # 1/s, 0: none
    brake_torque: float = 2.0
    spindle_drag: float = 1e-3  # N m s
    clutch_torque: float = 1.5
    clutch_detents: int = 6
    clutch_ring_ratio: float = 1.25
    clutch_duty: float = 0.7
    clutch_low: float = 0.6
    clutch_overshoot: float = 0.1
    clutch_lock: bool = False

    @classmethod
    def from_cfg(cls, dc, **over) -> DrivetrainParams:
        names = cls.__dataclass_fields__
        kw = {k: getattr(dc, k) for k in names if hasattr(dc, k)}
        kw.update(over)
        return cls(**kw)

    @property
    def momentum_inertia(self) -> float:
        """J_am: the internals' angular momentum per unit chuck speed (J_r N + J_chuck)."""
        return (self.spindle_inertia - self.chuck_inertia) / self.gear_ratio + self.chuck_inertia

    def click_rate(self, w: float) -> float:
        """Clutch ratchets per second while it slips with the spindle at w."""
        return self.clutch_detents * self.clutch_ring_ratio * max(w, 0.0) / (2 * np.pi)

    def steady_speed(self, u: float, load: float) -> float:
        """The spindle speed the motor settles to at trigger u against a load torque."""
        return max(self.free_speed * (u - load / self.stall_torque), 0.0) if u > 0 else 0.0


class Drivetrain:
    def __init__(self, p: DrivetrainParams):
        self.p = p
        self.reset()

    def reset(self) -> None:
        self.w = 0.0  # spindle speed (chuck-referred)
        self.theta = 0.0  # spindle angle
        self.out_angle = 0.0  # chuck angle
        self.u = 0.0
        self.slipping = False
        self.phase = 0.0  # 0..1 through the current detent while the clutch slips
        self.clicks = 0
        self.tau_m = self.tau_chuck = self.tau_housing = self.dw_dt = 0.0

    def motor_torque(self, u: float, w: float) -> float:
        p = self.p
        if u <= 0.0:
            return -float(np.clip(p.stall_torque * w / p.free_speed, -p.brake_torque, p.brake_torque))
        return p.stall_torque * (u - w / p.free_speed)

    def clutch_capacity(self) -> float:
        """What the clutch passes now: rigid up to the breakaway; slipping, the ramp-and-drop of a detent."""
        p = self.p
        if p.clutch_lock:
            return np.inf
        if not self.slipping:
            return p.clutch_torque * (1.0 + p.clutch_overshoot)
        x = self.phase / p.clutch_duty
        return p.clutch_torque * (p.clutch_low + (1.0 - p.clutch_low) * x) if x < 1.0 else 0.0

    def step(self, trigger: float, need: float, dt: float, hold: bool = True) -> float:
        """Advance dt with the trigger at `trigger` (0..1) and the load needing `need` N m at the chuck. Returns the
        chuck's rotation over the step (rad, the screw's or bit's turn)."""
        p = self.p
        self.u = trigger if (trigger <= 0.0 or p.trigger_slew <= 0.0) else min(trigger, self.u + p.trigger_slew * dt)
        tau_m = self.motor_torque(self.u, self.w)
        drag = p.spindle_drag * self.w
        cap = self.clutch_capacity()
        if self.slipping and (need <= cap or need <= 0.0):
            self.slipping = False  # the load let go (or the ramp holds it): the balls settle, the chuck turns again
            self.phase = 0.0
        elif not self.slipping and need > cap and (hold or need > 0):
            self.slipping = True
            self.phase = 0.0
        if self.slipping:
            tau_c = self.clutch_capacity()
            w_new = self.w + (tau_m - tau_c - drag) / p.spindle_inertia * dt
            turn = 0.0
            self.phase += p.click_rate(0.5 * (self.w + max(w_new, 0.0))) * dt
            while self.phase >= 1.0:
                self.phase -= 1.0
                self.clicks += 1
            w_new = max(w_new, 0.0)  # the motor stalls against the ramp rather than running back
            tau_chuck = tau_c
        else:
            w_new = self.w + (tau_m - need - drag) / p.spindle_inertia * dt
            tau_chuck = need
            if hold and w_new < 0.0 and need >= 0.0:
                # the load holds a stalled spindle: it reacts what the motor applies, the spindle stays at rest
                w_new = 0.0
                tau_chuck = float(np.clip(tau_m - drag + p.spindle_inertia * self.w / dt, 0.0, need))
            turn = 0.5 * (self.w + w_new) * dt
        dw = w_new - self.w
        self.dw_dt = dw / dt
        self.theta += 0.5 * (self.w + w_new) * dt
        self.out_angle += turn
        self.w = w_new
        self.tau_m = tau_m
        self.tau_chuck = tau_chuck
        self.tau_housing = tau_chuck + p.momentum_inertia * self.dw_dt
        return turn
