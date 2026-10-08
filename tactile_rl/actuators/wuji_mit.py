"""WUJI Hand 2 joint drives in MIT mode, batched over environments in torch: the GPU mirror of
tactile_sim.sim.wuji_drive.WujiMitDrive (the reference; tests/rl/test_rl_drive_parity.py holds the two together).

Per environment and physics step, exactly as the reference:

    host command -> SDK cycle (sdk_rate): latch the latest command, optional first-order low-pass on the fields in
        sdk_lpf_fields -> transport delay (cmd_latency + uniform jitter; arrivals never reorder) -> the driver runs
        the newest command that has arrived: tau* = Kp (q* - q) + Kd (qd* - qd) + tau_ff (every physics step, or at
        servo_rate) -> current command tau*/Kt clamped to min(effort_limit_ratio, i_peak_ratio) x i_rated, derated
        linearly from t_warn to derate_floor at t_max, where the driver faults (latched until reset) -> first-order
        current loop -> joint torque Kt i -> winding temperature C_th dT/dt = i^2 R - (T - T_amb) / R_th.

The gains travel with the command, so a stiffness change is delayed like a position change (mjlab's own actuator
delay does not delay gains). Every parameter the plan randomizes is a per-environment tensor (`set_params`). Times
are float64, as is the winding temperature (its per-step increments are ~1e-4 K, below float32 resolution near
70 degC); commands, currents and torques are float32 like the MuJoCo Warp state they feed.

`TorchWujiDrive.step` advances one physics step and must run exactly once per physics step (tactile_rl.mdp.actions
.SubstepHooks calls it); `WujiMitActuator.compute` only reads the result, so mjlab may call it any number of times.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.spec import create_motor_actuator

from tactile_sim.config import WujiDriveCfg
from tactile_sim.control.interface import HandCommand
from tactile_sim.sim.wuji_drive import JOINT_GROUPS, joint_group

if TYPE_CHECKING:
    from mjlab.entity import Entity

# per-environment parameters (set_params): (N,) float64 ...
PARAMS_F64 = ("cmd_latency", "cmd_jitter", "r_wind", "r_th", "c_th", "t_amb", "t_warn", "t_max", "derate_floor")
# ... (N,) float32 coefficients derived from sdk_lpf_hz and tau_i ...
PARAMS_F32 = ("lpf_a", "i_a")
# ... and (N, J) float32 per joint; i_limit = i_rated x min(effort_limit_ratio, i_peak_ratio)
PARAMS_JOINT = ("kt", "i_rated", "effort_limit_ratio", "i_peak_ratio", "i_limit")
STATE = ("t", "host", "host_valid", "filt", "filt_valid", "line_t", "line_cmd", "line_head", "line_n", "last_arrival",
         "active", "next_sdk", "next_servo", "tau_star", "i", "T", "fault", "n_sdk")


class TorchWujiDrive:
    """N environments x J WUJI joints (see module doc). Joint order is the caller's (`joint_names`)."""

    def __init__(self, cfg: WujiDriveCfg, joint_names: list[str], tau_rated, physics_dt: float, num_envs: int,
                 device: str | torch.device = "cpu", max_inflight: int = 8, generator: torch.Generator | None = None):
        self.cfg = cfg
        self.n_env = int(num_envs)
        self.n = len(joint_names)
        self.joint_names = list(joint_names)
        self.dt = float(physics_dt)
        self.device = torch.device(device)
        self.L = int(max_inflight)
        self.generator = generator
        self.sdk_period = 1.0 / cfg.sdk_rate
        self.servo_period = None if cfg.servo_rate is None else 1.0 / cfg.servo_rate
        if self.sdk_period < self.dt - 1e-12 or (self.servo_period is not None and self.servo_period < self.dt - 1e-12):
            raise ValueError("the SDK and servo rates must not exceed the physics rate (one cycle per step at most)")
        unknown = set(cfg.sdk_lpf_fields) - set(HandCommand.FIELDS)
        if unknown:
            raise ValueError(f"unknown sdk_lpf_fields {sorted(unknown)} (expected among {HandCommand.FIELDS})")
        if cfg.t_max <= cfg.t_warn:
            raise ValueError("t_max must exceed t_warn")
        self.lpf_mask = torch.tensor([f in cfg.sdk_lpf_fields for f in HandCommand.FIELDS], device=self.device)
        self.group = np.array([JOINT_GROUPS.index(joint_group(j)) for j in joint_names])
        self.tau_rated = torch.as_tensor(np.asarray(tau_rated, float), dtype=torch.float32, device=self.device)
        self._ar = torch.arange(self.n_env, device=self.device)
        self._slots = torch.arange(self.L, device=self.device)
        # replaced by tests to feed the reference's jitter draws: f(push mask (N,) bool) -> (N,) float64 seconds
        self.jitter_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
        self._alloc()
        self.set_params(cfg=cfg)
        self.reset()

    # ------------------------------------------------------------------ parameters
    def _alloc(self) -> None:
        N, J, L, dev = self.n_env, self.n, self.L, self.device
        f32, f64 = torch.float32, torch.float64
        for k in PARAMS_F64:
            setattr(self, k, torch.zeros(N, dtype=f64, device=dev))
        for k in PARAMS_F32:
            setattr(self, k, torch.zeros(N, dtype=f32, device=dev))
        for k in PARAMS_JOINT:
            setattr(self, k, torch.zeros(N, J, dtype=f32, device=dev))
        self.t = torch.zeros(N, dtype=f64, device=dev)
        self.host = torch.zeros(N, 5, J, dtype=f32, device=dev)
        self.host_valid = torch.zeros(N, dtype=torch.bool, device=dev)
        self.filt = torch.zeros(N, 5, J, dtype=f32, device=dev)
        self.filt_valid = torch.zeros(N, dtype=torch.bool, device=dev)
        self.line_t = torch.zeros(N, L, dtype=f64, device=dev)
        self.line_cmd = torch.zeros(N, L, 5, J, dtype=f32, device=dev)
        self.line_head = torch.zeros(N, dtype=torch.long, device=dev)
        self.line_n = torch.zeros(N, dtype=torch.long, device=dev)
        self.last_arrival = torch.zeros(N, dtype=f64, device=dev)
        self.active = torch.zeros(N, 5, J, dtype=f32, device=dev)
        self.next_sdk = torch.zeros(N, dtype=f64, device=dev)
        self.next_servo = torch.zeros(N, dtype=f64, device=dev)
        self.tau_star = torch.zeros(N, J, dtype=f32, device=dev)
        self.i = torch.zeros(N, J, dtype=f32, device=dev)
        self.T = torch.zeros(N, J, dtype=f64, device=dev)
        self.fault = torch.zeros(N, J, dtype=torch.bool, device=dev)
        self.n_sdk = torch.zeros(N, dtype=torch.long, device=dev)

    def _ids(self, env_ids) -> torch.Tensor | slice:
        return slice(None) if env_ids is None else env_ids

    def set_params(self, env_ids=None, cfg: WujiDriveCfg | None = None, **values) -> None:
        """Per-environment parameters, for domain randomization. With `cfg`, every parameter of the selected envs is
        first set from that WujiDriveCfg; `values` then set individual parameters (the others keep their values):
        PARAMS_F64 and `tau_i` (scalar, or (n,) per env), `sdk_lpf_hz` (None = off, a scalar, or (n,) with 0 = off),
        and `kt` / `i_rated` / `effort_limit_ratio` / `i_peak_ratio` (scalar, (J,) per joint, (n, 1) per env, or
        (n, J)). Setting `i_rated` without `kt` keeps the torque constant at tau_rated / i_rated."""
        ids = self._ids(env_ids)
        n = self.n_env if env_ids is None else len(env_ids)
        dev = self.device
        if cfg is not None:
            values = {**{k: getattr(cfg, k) for k in PARAMS_F64}, "sdk_lpf_hz": cfg.sdk_lpf_hz, "tau_i": cfg.tau_i,
                      "i_rated": np.asarray(cfg.i_rated, float)[self.group],
                      "effort_limit_ratio": cfg.effort_limit_ratio, "i_peak_ratio": cfg.i_peak_ratio, **values}

        def col(v, dtype) -> torch.Tensor:
            t = torch.as_tensor(v, dtype=dtype, device=dev)
            return t.expand(n).clone() if t.ndim == 0 else t

        def mat(v) -> torch.Tensor:
            t = torch.as_tensor(v, dtype=torch.float32, device=dev)
            if t.ndim == 1 and t.shape[0] != self.n:
                raise ValueError(f"a 1-D joint parameter has one entry per joint ({self.n}); per env use (n, 1)")
            return t.expand(n, self.n).clone()

        for k in PARAMS_F64:
            if k in values:
                getattr(self, k)[ids] = col(values.pop(k), torch.float64)
        if "sdk_lpf_hz" in values:
            lpf = values.pop("sdk_lpf_hz")
            hz = torch.zeros(n, dtype=torch.float64, device=dev) if lpf is None else col(lpf, torch.float64)
            sdk_rate = 1.0 / self.sdk_period
            self.lpf_a[ids] = torch.where(hz > 0, 1.0 - torch.exp(-2.0 * math.pi * hz / sdk_rate),
                                          torch.ones_like(hz)).float()
        if "tau_i" in values:
            tau_i = col(values.pop("tau_i"), torch.float64)
            self.i_a[ids] = torch.where(tau_i > 0, 1.0 - torch.exp(-self.dt / tau_i.clamp_min(1e-12)),
                                        torch.ones_like(tau_i)).float()
        for k in ("i_rated", "effort_limit_ratio", "i_peak_ratio", "kt"):
            if k in values:
                getattr(self, k)[ids] = mat(values.pop(k))
                if k == "i_rated" and "kt" not in values:
                    self.kt[ids] = self.tau_rated[None] / self.i_rated[ids]
        self.i_limit[ids] = self.i_rated[ids] * torch.minimum(self.effort_limit_ratio[ids], self.i_peak_ratio[ids])
        if values:
            raise KeyError(f"unknown drive parameters {sorted(values)}")
        self._check_line()

    def _check_line(self) -> None:
        worst = float((self.cmd_latency + self.cmd_jitter).max())
        if worst >= (self.L - 1) * self.sdk_period - 0.5 * self.dt:
            raise ValueError(f"latency + jitter {worst * 1e3:.2f} ms needs more than {self.L} commands in flight; "
                             "raise max_inflight")

    # ------------------------------------------------------------------ state
    def reset(self, env_ids=None, t0: float = 0.0) -> None:
        """As WujiMitDrive.reset: no command, zero gains (no torque until one arrives), windings at ambient."""
        ids = self._ids(env_ids)
        self.t[ids] = t0
        self.host[ids] = 0.0
        self.host_valid[ids] = False
        self.filt[ids] = 0.0
        self.filt_valid[ids] = False
        self.line_head[ids] = 0
        self.line_n[ids] = 0
        self.last_arrival[ids] = -math.inf
        self.active[ids] = 0.0
        self.next_sdk[ids] = t0
        self.next_servo[ids] = t0
        self.tau_star[ids] = 0.0
        self.i[ids] = 0.0
        self.T[ids] = self.t_amb[ids][:, None]
        self.fault[ids] = False
        self.n_sdk[ids] = 0

    def shift_time(self, delta: float | torch.Tensor, env_ids=None) -> None:
        """The clock moved by delta, scalar or per env (World.reset restarts time at 0 once the grasp has settled)."""
        ids = self._ids(env_ids)
        d = torch.as_tensor(delta, dtype=torch.float64, device=self.device)
        for k in ("t", "next_sdk", "next_servo", "last_arrival"):
            getattr(self, k)[ids] += d
        self.line_t[ids] += d[:, None] if d.ndim else d

    def get_state(self, env_ids=None) -> dict[str, torch.Tensor]:
        ids = self._ids(env_ids)
        return {k: getattr(self, k)[ids].clone() for k in STATE}

    def set_state(self, state: dict[str, torch.Tensor], env_ids=None) -> None:
        ids = self._ids(env_ids)
        for k in STATE:
            getattr(self, k)[ids] = state[k]

    # ------------------------------------------------------------------ commands
    def set_host(self, cmd: torch.Tensor, env_ids=None) -> None:
        """The host's latest MIT command (n, 5, J) rows q, qd, kp, kd, tau; the SDK picks it up at its next cycle."""
        ids = self._ids(env_ids)
        self.host[ids] = cmd.to(self.host.dtype)
        self.host_valid[ids] = True

    def _jitter(self, push: torch.Tensor) -> torch.Tensor:
        if self.jitter_fn is not None:
            return self.jitter_fn(push).to(torch.float64)
        u = torch.rand(self.n_env, dtype=torch.float64, device=self.device, generator=self.generator)
        return (2.0 * u - 1.0) * self.cmd_jitter

    @property
    def torque(self) -> torch.Tensor:
        """Joint torque the drives apply now (Nm), (N, J)."""
        return self.kt * self.i

    # ------------------------------------------------------------------ the physics step
    def step(self, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        """Advance one physics step with the joints at (q, qd) (N, J); returns the joint torques (N, J)."""
        t = self.t
        # SDK cycle at next_sdk: latch the host command, low-pass it, put it on the line
        due = self.next_sdk <= t + 1e-9
        self.n_sdk += due.long()
        push = due & self.host_valid
        x = self.host
        lp = (self.filt_valid & (self.lpf_a < 1.0))[:, None, None] & self.lpf_mask[None, :, None]
        filt = torch.where(lp, self.filt + self.lpf_a[:, None, None] * (x - self.filt), x)
        self.filt = torch.where(push[:, None, None], filt, self.filt)
        self.filt_valid = self.filt_valid | push
        arrival = torch.maximum(self.next_sdk + self.cmd_latency + self._jitter(push), self.last_arrival)
        self.last_arrival = torch.where(push, arrival, self.last_arrival)
        slot = (self.line_head + self.line_n) % self.L
        self.line_t[self._ar, slot] = torch.where(push, arrival, self.line_t[self._ar, slot])
        self.line_cmd[self._ar, slot] = torch.where(push[:, None, None], self.filt, self.line_cmd[self._ar, slot])
        self.line_n = self.line_n + push.long()
        self.next_sdk = torch.where(due, self.next_sdk + self.sdk_period, self.next_sdk)
        # the driver takes the newest command that has arrived (arrivals are monotone: the ready ones lead the line)
        order = (self.line_head[:, None] + self._slots[None]) % self.L
        ready = (self._slots[None] < self.line_n[:, None]) & \
            (self.line_t.gather(1, order) <= (t + 0.5 * self.dt)[:, None])
        n_pop = ready.sum(1)
        newest = (self.line_head + n_pop - 1) % self.L
        self.active = torch.where((n_pop > 0)[:, None, None], self.line_cmd[self._ar, newest], self.active)
        self.line_head = (self.line_head + n_pop) % self.L
        self.line_n = self.line_n - n_pop
        # servo law
        qs, qds, kp, kd, tau = self.active.unbind(1)
        law = kp * (qs - q) + kd * (qds - qd) + tau
        if self.servo_period is None:
            self.tau_star = law
        else:
            due_s = self.next_servo <= t + 1e-9
            self.tau_star = torch.where(due_s[:, None], law, self.tau_star)
            self.next_servo = torch.where(due_s, self.next_servo + self.servo_period, self.next_servo)
        # current clamp with thermal derating and the latched fault, current loop, windings
        self.fault = self.fault | (self.T >= self.t_max[:, None])
        frac = ((self.T - self.t_warn[:, None]) / (self.t_max - self.t_warn)[:, None]).clamp(0.0, 1.0)
        derate = 1.0 - (1.0 - self.derate_floor[:, None]) * frac
        lim = torch.where(self.fault, 0.0, self.i_limit.double() * derate).float()
        i_cmd = torch.minimum(torch.maximum(self.tau_star / self.kt, -lim), lim)
        self.i = self.i + self.i_a[:, None] * (i_cmd - self.i)
        heat = self.i.double() ** 2 * self.r_wind[:, None] - (self.T - self.t_amb[:, None]) / self.r_th[:, None]
        self.T = self.T + (self.dt / self.c_th)[:, None] * heat
        self.t = t + self.dt
        return self.kt * self.i


# ---------------------------------------------------------------------- mjlab actuator


@dataclass(kw_only=True)
class WujiMitActuatorCfg(ActuatorCfg):
    """WUJI MIT-mode drives on the hand joints (`target_names_expr`). The joints' own <motor> actuators (the testbed's
    m_<joint>) carry the torque; one is added where a joint has none."""

    drive: WujiDriveCfg = field(default_factory=WujiDriveCfg)
    tau_rated: tuple[float, ...] = ()  # Nm per target joint, in the entity's joint order; () = from the scene export
    scene_key: str = ""  # tactile_sim.sim.training_parts.config_key of the scene (for the rated torques)
    physics_dt: float = 0.00025
    max_inflight: int = 8

    def build(self, entity: Entity, target_ids: list[int], target_names: list[str]) -> WujiMitActuator:
        return WujiMitActuator(self, entity, target_ids, target_names)


class WujiMitActuator(Actuator):
    """The drive as an mjlab actuator: `compute` writes the drives' present torque to the motors' ctrl. The drive itself
    is stepped by the env's substep hook (`TorchWujiDrive.step`), never here: mjlab also calls compute outside the
    physics loop (after resets)."""

    cfg: WujiMitActuatorCfg

    def __init__(self, cfg: WujiMitActuatorCfg, entity: Entity, target_ids: list[int], target_names: list[str]):
        super().__init__(cfg, entity, target_ids, target_names)
        tau = tuple(cfg.tau_rated)
        if not tau and cfg.scene_key:
            from tactile_sim.sim.training_parts import training_parts

            parts = training_parts(cfg.scene_key)
            rated = dict(zip(parts.hand_joints, parts.hand_tau_rated, strict=True))
            tau = tuple(float(rated[n]) for n in target_names)
        if len(tau) != len(target_names):
            raise ValueError(f"tau_rated has {len(tau)} entries for {len(target_names)} joints")
        self.tau_rated = tau
        self.drive: TorchWujiDrive | None = None

    def edit_spec(self, spec, target_names: list[str]) -> None:
        for name, lim in zip(target_names, self.tau_rated, strict=True):
            act = next((a for a in spec.actuators if a.target == name), None)
            if act is None:
                act = create_motor_actuator(spec, name, effort_limit=2.0 * lim)
            self._mjs_actuators.append(act)

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        self.drive = TorchWujiDrive(self.cfg.drive, list(self.target_names), self.tau_rated, self.cfg.physics_dt,
                                    data.nworld, device, self.cfg.max_inflight)

    def get_command(self, data) -> ActuatorCmd:
        # the drive keeps its own command line; mjlab's joint targets are unused (no per-substep gathers)
        z = self.drive.torque
        return ActuatorCmd(position_target=z, velocity_target=z, effort_target=z, pos=z, vel=z)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        del cmd
        return self.drive.torque

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if self.drive is not None:
            self.drive.reset(env_ids)
