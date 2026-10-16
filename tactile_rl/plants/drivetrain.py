"""The cordless drill/driver's drivetrain batched over environments: tactile_sim.model.plants.drivetrain.Drivetrain
step for step (float64), every parameter per env.

Motor tau_m = stall (u - w / w0) at the chuck (the shorted motor's brake, capped, with the trigger released); the
spindle carries J_spin; the ball-detent clutch is rigid below its breakaway and, slipping, passes the ramp-and-drop
of a detent while the ring turns clutch_ring_ratio x the spindle speed (clicks counted); a holding load stops a
stalled spindle; the housing reacts tau_chuck + J_am dw/dt.
"""

from __future__ import annotations

import math
from dataclasses import fields

import torch

from tactile_sim.model.plants.drivetrain import DrivetrainParams

PARAMS = tuple(f.name for f in fields(DrivetrainParams))


class TorchDrivetrain:
    def __init__(self, p: DrivetrainParams, num_envs: int, device: str | torch.device = "cpu"):
        self.n_env, self.dev = int(num_envs), torch.device(device)
        f64 = dict(dtype=torch.float64, device=self.dev)
        for k in PARAMS:
            v = getattr(p, k)
            if k == "clutch_lock":
                setattr(self, k, torch.full((self.n_env,), bool(v), dtype=torch.bool, device=self.dev))
            else:
                setattr(self, k, torch.full((self.n_env,), float(v), **f64))
        self.w = torch.zeros(self.n_env, **f64)
        self.theta = torch.zeros(self.n_env, **f64)
        self.out_angle = torch.zeros(self.n_env, **f64)
        self.u = torch.zeros(self.n_env, **f64)
        self.slipping = torch.zeros(self.n_env, dtype=torch.bool, device=self.dev)
        self.phase = torch.zeros(self.n_env, **f64)
        self.clicks = torch.zeros(self.n_env, dtype=torch.long, device=self.dev)
        self.tau_m = torch.zeros(self.n_env, **f64)
        self.tau_chuck = torch.zeros(self.n_env, **f64)
        self.tau_housing = torch.zeros(self.n_env, **f64)
        self.dw_dt = torch.zeros(self.n_env, **f64)

    def set_params(self, env_ids=None, **values) -> None:
        ids = slice(None) if env_ids is None else env_ids
        for k, v in values.items():
            if k not in PARAMS:
                raise KeyError(k)
            t = getattr(self, k)
            t[ids] = torch.as_tensor(v, dtype=t.dtype, device=self.dev)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        for t in (self.w, self.theta, self.out_angle, self.u, self.phase, self.tau_m, self.tau_chuck,
                  self.tau_housing, self.dw_dt):
            t[ids] = 0.0
        self.slipping[ids] = False
        self.clicks[ids] = 0

    @property
    def momentum_inertia(self) -> torch.Tensor:
        return (self.spindle_inertia - self.chuck_inertia) / self.gear_ratio + self.chuck_inertia

    def click_rate(self, w: torch.Tensor) -> torch.Tensor:
        return self.clutch_detents * self.clutch_ring_ratio * torch.clamp(w, min=0.0) / (2 * math.pi)

    def steady_speed(self, u: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
        s = torch.clamp(self.free_speed * (u - load / self.stall_torque), min=0.0)
        return torch.where(u > 0, s, torch.zeros_like(s))

    def motor_torque(self, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        brake = -torch.clamp(self.stall_torque * w / self.free_speed, -self.brake_torque, self.brake_torque)
        return torch.where(u <= 0.0, brake, self.stall_torque * (u - w / self.free_speed))

    def clutch_capacity(self, slipping: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        x = phase / self.clutch_duty
        ramp = torch.where(x < 1.0, self.clutch_torque * (self.clutch_low + (1.0 - self.clutch_low) * x),
                           torch.zeros_like(x))
        cap = torch.where(slipping, ramp, self.clutch_torque * (1.0 + self.clutch_overshoot))
        return torch.where(self.clutch_lock, torch.full_like(cap, math.inf), cap)

    def step(self, trigger: torch.Tensor, need: torch.Tensor, dt: float, hold: bool = True) -> torch.Tensor:
        """As Drivetrain.step for every env: trigger (N,) 0..1, need (N,) N m. Returns the chuck's turn (rad)."""
        trigger, need = trigger.double(), need.double()
        slew_on = (trigger > 0.0) & (self.trigger_slew > 0.0)
        self.u = torch.where(slew_on, torch.minimum(trigger, self.u + self.trigger_slew * dt), trigger)
        tau_m = self.motor_torque(self.u, self.w)
        drag = self.spindle_drag * self.w
        cap = self.clutch_capacity(self.slipping, self.phase)
        settle = self.slipping & ((need <= cap) | (need <= 0.0))
        holds = torch.ones_like(self.slipping) if hold else need > 0.0
        start = ~self.slipping & (need > cap) & holds
        slipping = (self.slipping & ~settle) | start
        phase = torch.where(settle | start, torch.zeros_like(self.phase), self.phase)
        # slipping: the clutch passes its ramp, the ring ratchets, the motor stalls rather than running back
        tau_c = self.clutch_capacity(slipping, phase)
        w_slip = self.w + (tau_m - tau_c - drag) / self.spindle_inertia * dt
        phase_s = phase + self.click_rate(0.5 * (self.w + torch.clamp(w_slip, min=0.0))) * dt
        n_clicks = torch.floor(phase_s)
        phase_s = phase_s - n_clicks
        w_slip = torch.clamp(w_slip, min=0.0)
        # rigid: the chuck turns with the spindle against the load; a holding load stops a stalled spindle
        w_rig = self.w + (tau_m - need - drag) / self.spindle_inertia * dt
        stall = (w_rig < 0.0) & (need >= 0.0) & hold
        tau_hold = torch.minimum(torch.clamp(tau_m - drag + self.spindle_inertia * self.w / dt, min=0.0), need)
        w_rig = torch.where(stall, torch.zeros_like(w_rig), w_rig)
        tau_rig = torch.where(stall, tau_hold, need)
        turn_rig = 0.5 * (self.w + w_rig) * dt
        w_new = torch.where(slipping, w_slip, w_rig)
        tau_chuck = torch.where(slipping, tau_c, tau_rig)
        turn = torch.where(slipping, torch.zeros_like(turn_rig), turn_rig)
        self.clicks = self.clicks + torch.where(slipping, n_clicks, torch.zeros_like(n_clicks)).long()
        self.phase = torch.where(slipping, phase_s, phase)
        self.slipping = slipping
        self.dw_dt = (w_new - self.w) / dt
        self.theta = self.theta + 0.5 * (self.w + w_new) * dt
        self.out_angle = self.out_angle + turn
        self.w = w_new
        self.tau_m = tau_m
        self.tau_chuck = tau_chuck
        self.tau_housing = tau_chuck + self.momentum_inertia * self.dw_dt
        return turn
