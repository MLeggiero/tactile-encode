"""WUJI Hand 2 joint drives in MIT mode, as the host sees them (GripperCfg.hand_control = "mit").

Every physics step, for every hand joint:

    host command (any rate) -> SDK cycle (sdk_rate): latch the latest command, optional first-order low-pass
        -> transport delay (cmd_latency + uniform jitter; commands arrive in order)
        -> driver servo law (every physics step, or at servo_rate): tau* = Kp (q* - q) + Kd (qd* - qd) + tau_ff
        -> current command i* = tau* / Kt, clamped to the effort limit (the SDK's current ceiling, at most the
           driver's peak current) and derated linearly with winding temperature from t_warn (full) to t_max
           (derate_floor), where the driver faults and stays off until reset; Kt is the nominal torque constant
           (tau_rated / i_rated) the SDK converts with
        -> current loop (first order, tau_i) -> joint torque kt_scale Kt i: the motor's true torque constant
           (WujiDriveCfg.kt_scale; the host reads back the current, not the torque)
        -> winding temperature: C_th dT/dt = i^2 R_wind - (T - T_amb) / R_th

Until its first command arrives a drive applies no torque. What the host reads back (encoder angle, current) goes
through the sensor pipeline (tactile_sim.sensors.hand_joints): sampled at sdk_rate, delayed by fb_latency,
quantized, with noise on the current.

This is the single reference implementation: the GPU training actuator and the robot-side checks mirror it.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from tactile_sim.config import WujiDriveCfg
from tactile_sim.control.interface import HandCommand

# per-group parameters in WujiDriveCfg follow this order
JOINT_GROUPS = ("mcp_flex", "mcp_abd", "pip_dip", "thumb_cmc", "thumb_mcp_ip")


def joint_group(name: str) -> str:
    """Drive group of a WUJI Hand 2 joint (vendor joint names, either hand)."""
    if "thumb" in name:
        return "thumb_cmc" if "_cmc_" in name else "thumb_mcp_ip"
    for suffix, group in (("_mcp_flex", "mcp_flex"), ("_mcp_abd", "mcp_abd"), ("_pip", "pip_dip"),
                          ("_dip", "pip_dip")):
        if name.endswith(suffix):
            return group
    raise ValueError(f"not a WUJI Hand 2 joint: {name!r}")


class WujiMitDrive:
    def __init__(self, cfg: WujiDriveCfg, joint_names: list[str], tau_rated: np.ndarray, physics_dt: float,
                 rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.n = len(joint_names)
        self.dt = physics_dt
        self.rng = rng or np.random.default_rng(0)
        gi = np.array([JOINT_GROUPS.index(joint_group(j)) for j in joint_names])
        self.group = gi
        self.tau_rated = np.asarray(tau_rated, dtype=float)
        self.i_rated = np.asarray(cfg.i_rated, dtype=float)[gi]
        self.kt = self.tau_rated / self.i_rated  # Nm/A, nominal: the SDK's torque -> current map
        self.kt_true = self.kt * float(cfg.kt_scale)  # Nm/A, the motors' (current -> torque)
        self.kp_nom = np.asarray(cfg.kp_nom, dtype=float)[gi]
        self.period = 1.0 / cfg.sdk_rate
        unknown = set(cfg.sdk_lpf_fields) - set(HandCommand.FIELDS)
        if unknown:
            raise ValueError(f"unknown sdk_lpf_fields {sorted(unknown)} (expected among {HandCommand.FIELDS})")
        self.lpf_a = 1.0 - np.exp(-2.0 * np.pi * cfg.sdk_lpf_hz / cfg.sdk_rate) if cfg.sdk_lpf_hz else 1.0
        self.lpf_mask = np.array([f in cfg.sdk_lpf_fields for f in HandCommand.FIELDS])[:, None]
        self.servo_period = None if cfg.servo_rate is None else 1.0 / cfg.servo_rate
        self.i_a = 1.0 - np.exp(-physics_dt / cfg.tau_i) if cfg.tau_i > 0 else 1.0
        if cfg.t_max <= cfg.t_warn:
            raise ValueError("t_max must exceed t_warn")
        self.reset()

    @property
    def i_limit(self) -> np.ndarray:
        """Current ceiling per joint before thermal derating (A)."""
        return self.i_rated * min(self.cfg.effort_limit_ratio, self.cfg.i_peak_ratio)

    def reset(self, t0: float = 0.0) -> None:
        n = self.n
        self.host: HandCommand | None = None
        self.filt: np.ndarray | None = None  # the SDK's (filtered) copy of the host command, (5, n)
        self.line: deque[tuple[float, np.ndarray]] = deque()  # commands in transit: (arrival time, (5, n))
        self.last_arrival = -np.inf
        self.active = np.zeros((5, n))  # the command each driver runs: zero gains, no torque until one arrives
        self.next_sdk = t0
        self.next_servo = t0
        self.tau_star = np.zeros(n)
        self.i = np.zeros(n)  # A
        self.T = np.full(n, float(self.cfg.t_amb))  # degC
        self.fault = np.zeros(n, dtype=bool)
        self.n_sdk = 0

    def shift_time(self, dt: float) -> None:
        """The simulation clock was moved by dt (World.reset restarts time at 0 after the grasp settles)."""
        self.next_sdk += dt
        self.next_servo += dt
        self.last_arrival += dt
        self.line = deque((t + dt, x) for t, x in self.line)

    def set_host_command(self, cmd: HandCommand) -> None:
        """The host's latest command; the SDK picks it up at its next cycle."""
        if cmd.q.shape != (self.n,):
            raise ValueError(f"hand command for {cmd.q.shape} joints, drive has {self.n}")
        self.host = cmd

    def _sdk_cycle(self, t_cycle: float) -> None:
        self.n_sdk += 1
        if self.host is None:
            return
        x = self.host.array()
        if self.filt is None or self.lpf_a >= 1.0:
            self.filt = x
        else:
            self.filt = np.where(self.lpf_mask, self.filt + self.lpf_a * (x - self.filt), x)
        c = self.cfg
        jitter = self.rng.uniform(-c.cmd_jitter, c.cmd_jitter) if c.cmd_jitter > 0 else 0.0
        arrival = max(t_cycle + c.cmd_latency + jitter, self.last_arrival)
        self.last_arrival = arrival
        self.line.append((arrival, self.filt.copy()))

    def update(self, t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Advance one physics step at time t with the joints at (q, qd); returns the joint torques (Nm)."""
        while self.next_sdk <= t + 1e-9:
            self._sdk_cycle(self.next_sdk)
            self.next_sdk += self.period
        while self.line and self.line[0][0] <= t + 0.5 * self.dt:
            self.active = self.line.popleft()[1]
        if self.servo_period is None or self.next_servo <= t + 1e-9:
            qs, qds, kp, kd, tau = self.active
            self.tau_star = kp * (qs - q) + kd * (qds - qd) + tau
            if self.servo_period is not None:
                while self.next_servo <= t + 1e-9:
                    self.next_servo += self.servo_period
        c = self.cfg
        self.fault |= self.T >= c.t_max
        frac = np.clip((self.T - c.t_warn) / (c.t_max - c.t_warn), 0.0, 1.0)
        derate = 1.0 - (1.0 - c.derate_floor) * frac
        lim = np.where(self.fault, 0.0, self.i_limit * derate)
        i_cmd = np.clip(self.tau_star / self.kt, -lim, lim)
        self.i = self.i + self.i_a * (i_cmd - self.i)
        self.T = self.T + self.dt / c.c_th * (self.i**2 * c.r_wind - (self.T - c.t_amb) / c.r_th)
        return self.kt_true * self.i

    @property
    def torque(self) -> np.ndarray:
        """Joint torque the drives apply now (Nm)."""
        return self.kt_true * self.i
