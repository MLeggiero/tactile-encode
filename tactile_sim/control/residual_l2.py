"""The residual hand policy (L2) on the testbed: observations from the robot's sensors, a policy, safety limits,
and composition with the base hand command into the MIT commands the WUJI drives take (obs_spec).

Two scheduler tasks drive it (Testbed registers them when ResidualCfg.enabled):

- `sample` at 1 kHz pushes the fast inputs (taxel codes and patch features from the TaxelScan fast profile, board
  status, hand encoder angles and currents) into a ring `stack` deep, so a slower policy still sees every
  millisecond since its previous step;
- `tick` at ResidualCfg.rate builds the observation, runs the policy, bounds the result and sends the command.

Safety, applied every tick in this order: a non-finite base command is replaced by the last finite one (none yet:
nothing is sent) and counts as a stale input; non-finite observation entries read zero (counted), so they never
reach the policy's recurrent state, and a policy whose state turns non-finite anyway restarts from zero (counted);
non-finite actions become zero; the action is clipped to [-1, 1]; on the drill task, offsets on the trigger joints
are capped, and may only open the finger while the base disables the trigger (the interlock exists only where there
is a trigger: task "drill"); if any input is stale the offsets decay to zero instead; the offsets' slew is limited per
second. The GPU training env (tactile_rl) mirrors this.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import mujoco
import numpy as np

from tactile_sim.control.interface import HandCommand
from tactile_sim.control.obs_spec import (
    ACTION_DIM,
    BOARDS,
    FORECAST_DIM,
    LOOKAHEAD,
    N_ARM,
    N_JOINTS,
    TASKS,
    ActionSpec,
    JointParams,
    arm_lookahead,
    assert_deployable,
    base_tau_obs,
    compose,
    obs_dim,
    obs_slices,
    offsets,
    slew,
    spec_hash,
    stack_for,
)
from tactile_sim.sim.world import full_inertia

# what a base source returns at time t: the base MIT command (5, 20), the lookahead (hand 20 then arm 7, per LOOKAHEAD
# time; the observation keeps the arm's part), trigger enable, forecast
BaseFn = Callable[[float], tuple[np.ndarray, np.ndarray, bool, np.ndarray]]


class ZeroPolicy:
    """No residual: the base command passes through unchanged."""

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(ACTION_DIM)


def trigger_indices(cfg, joint_names: list[str]) -> np.ndarray:
    """Hand joints under the drill-trigger interlock (ResidualCfg.trigger_joints): only on the drill task, the only
    one with a trigger; none elsewhere. Shared with the GPU training env, so both apply the same interlock."""
    if cfg.task != "drill":
        return np.zeros(0, dtype=int)
    return np.array([joint_names.index(j) for j in cfg.trigger_joints if j in joint_names], dtype=int)


def make_policy(name: str):
    """"zero", or the path of an exported policy (.onnx with its sidecar)."""
    if name == "zero":
        return ZeroPolicy()
    if name.endswith(".onnx"):
        from tactile_sim.control.policy_io import OnnxPolicy

        return OnnxPolicy(name)
    raise ValueError(f"unknown residual policy {name!r}; pass a policy object to ResidualHandL2 instead")


class ResidualHandL2:
    def __init__(self, tb, cfg=None, policy=None, base_fn: BaseFn | None = None):
        self.tb = tb
        self.cfg = cfg or tb.cfg.residual
        c = self.cfg
        hand = tb.world.hand
        if getattr(hand, "drive", None) is None:
            raise ValueError("the residual commands MIT-mode drives: set GripperCfg.hand_control = 'mit'")
        if tb.cfg.sensors.ts_model != "rev3":
            raise ValueError("the residual reads the TaxelScan fast profile: set SensorsCfg.ts_model = 'rev3'")
        if c.task not in TASKS:
            raise ValueError(f"unknown task {c.task!r} (expected one of {TASKS})")
        self.hand = hand
        self.spec = ActionSpec()
        self.jp = JointParams.for_joints(hand.joint_names, hand.tau_max, tb.cfg.wuji.kp_nom, self._i_eff())
        self.stack = stack_for(c.rate)
        self.dt = 1.0 / c.rate
        self.policy = policy if policy is not None else make_policy(c.policy)
        self.check_policy(self.policy)
        self.base_fn = base_fn or self.synergy_base
        sens = tb.sensors
        assert_deployable(list(hand.pressure_names) + ["hand_q", "hand_current"])
        self.patch_sensors = [sens[n] for n in hand.pressure_names]
        by_board: dict[int, object] = {}
        for s in self.patch_sensors:
            by_board.setdefault(s.board, s)
        if len(by_board) != BOARDS:
            raise ValueError(f"expected {BOARDS} TaxelScan boards, the hand has {len(by_board)}")
        self.board_sensors = [by_board[b] for b in sorted(by_board)]
        self.q_sensor, self.i_sensor = sens["hand_q"], sens["hand_current"]
        self.i_rated = hand.drive.i_rated
        self.trig = trigger_indices(c, hand.joint_names)
        self.close_dir = hand.synergy.close_dir
        self.task_vec = np.eye(len(TASKS))[TASKS.index(c.task)]
        slices = obs_slices(self.stack)
        unknown = set(c.mask_blocks) - set(slices)
        if unknown:
            raise ValueError(f"unknown observation blocks to mask: {sorted(unknown)}")
        self.masked = [slices[b] for b in c.mask_blocks]
        self.reset()

    def check_policy(self, policy) -> None:
        """An exported policy runs only on the interface and rate it was trained for."""
        meta = getattr(policy, "meta", None)
        if meta is None:
            return
        want = spec_hash(self.spec, self.stack)
        if meta.spec_hash != want:
            raise ValueError(f"policy {meta.name!r} was trained on interface {meta.spec_hash}, this loop is {want}")
        if abs(meta.rate - self.cfg.rate) > 1e-9:
            raise ValueError(f"policy {meta.name!r} decides at {meta.rate} Hz, the loop runs at {self.cfg.rate} Hz")
        mask = tuple(meta.extra.get("mask_blocks", ()))
        if set(mask) != set(self.cfg.mask_blocks):
            raise ValueError(f"policy {meta.name!r} was trained blind to {sorted(mask)}, the loop masks "
                             f"{sorted(self.cfg.mask_blocks)}")

    def _i_eff(self) -> np.ndarray:
        """Each hand joint's effective inertia (diagonal of the joint-space inertia) at the current pose."""
        w = self.tb.world
        m, d = w.model, w.data
        mujoco.mj_forward(m, d)
        M = full_inertia(m, d, np.zeros((m.nv, m.nv)))
        return np.diag(M)[self.hand.dofs].copy()

    def reset(self) -> None:
        self.fast_dim = sum(d for d in (448, 66, 2 * BOARDS, N_JOINTS, N_JOINTS))
        self.ring: deque[np.ndarray] = deque([np.zeros(self.fast_dim)] * self.stack, maxlen=self.stack)
        self.last_a = np.zeros(ACTION_DIM)
        self.off = np.zeros((2, N_JOINTS))
        self.last_seq = [-1] * BOARDS
        self.policy.reset()
        self.n_ticks = self.n_nonfinite = self.n_stale = 0
        self.n_obs_nonfinite = self.n_base_nonfinite = self.n_state_reset = 0
        self.good_base: np.ndarray | None = None
        self.last_obs: np.ndarray | None = None
        self.last_cmd: HandCommand | None = None
        self.last_base: np.ndarray | None = None

    # ------------------------------------------------------------------ base
    def synergy_base(self, t: float) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
        """The grip synergy as the base command, from the joint angles the host last read; no planned motion, no
        forecast. Stands in until a scripted, replayed or VLA base source is attached."""
        q = self.q_sensor.latest().value
        base = np.stack(self.hand.synergy.mit_command(self.hand.s, q))
        return base, np.zeros((N_JOINTS + N_ARM) * len(LOOKAHEAD)), False, np.zeros(FORECAST_DIM)

    # ------------------------------------------------------------------ observation
    def fast_sample(self, t: float) -> np.ndarray:
        codes, feats = [], []
        for s in self.patch_sensors:
            codes.append(s.codes.astype(float) / (s.codec.levels - 1))
            f = s.features
            rows, cols = s.grid
            loaded = f[0] > 0
            feats.append([f[0] / 100.0, f[1] / (256.0 * rows) if loaded else -1.0,
                          f[2] / (256.0 * cols) if loaded else -1.0, f[3] / (rows * cols), f[4] / 1000.0,
                          f[5] / 100.0])
        status = []
        for b, s in enumerate(self.board_sensors):
            smp = s.latest()
            age = (t - smp.t_avail) / 0.005 if smp.seq >= 0 else 1.0
            new = smp.seq - self.last_seq[b] if smp.seq >= 0 else 0
            self.last_seq[b] = smp.seq
            status += [age, float(new)]
        q = self.q_sensor.latest().value
        cur = self.i_sensor.latest().value / self.i_rated
        return np.concatenate([np.concatenate(codes), np.ravel(feats), status, q, cur])

    def sample(self, t: float) -> None:
        """1 kHz: the newest fast inputs, kept `stack` deep."""
        self.ring.appendleft(self.fast_sample(t))

    def observe(self, t: float, base: np.ndarray, look: np.ndarray, trigger: bool, forecast: np.ndarray) -> np.ndarray:
        fast = np.stack(self.ring)  # (stack, fast_dim), newest first
        sl = np.cumsum([0, 448, 66, 2 * BOARDS, N_JOINTS, N_JOINTS])
        parts = [fast[:, sl[i]:sl[i + 1]].ravel() for i in range(5)]  # each fast block holds its `stack` samples
        obs = np.concatenate(parts + [self.last_a, base[0], base_tau_obs(base, self.jp.tau_rated), arm_lookahead(look),
                                      np.asarray(forecast, float), self.task_vec, [1.0 if trigger else 0.0]])
        if obs.shape != (obs_dim(self.stack),):
            raise ValueError(f"observation has {obs.shape[0]} entries, the spec {obs_dim(self.stack)}")
        for sl in self.masked:
            obs[sl] = 0.0
        return obs

    def stale(self, t: float) -> bool:
        c = self.cfg
        ages = [t - s.latest().t_avail if s.latest().seq >= 0 else np.inf for s in self.board_sensors]
        q = self.q_sensor.latest()
        return max(ages) > c.stale_taxel or q.seq < 0 or t - q.t_avail > c.stale_joint

    # ------------------------------------------------------------------ control
    def _trigger_guard(self, off: np.ndarray, trigger: bool) -> np.ndarray:
        if not len(self.trig):
            return off
        c = self.cfg
        out = off.copy()
        j = self.trig
        d = self.close_dir[j]
        dq = np.clip(out[0, j], -c.trigger_dq_max, c.trigger_dq_max)
        tau_close = np.clip(d * out[1, j], c.trigger_tau[0], c.trigger_tau[1])
        if not trigger:  # the trigger is disabled: the residual may only open the finger
            dq = np.where(d * dq > 0, 0.0, dq)
            tau_close = np.minimum(tau_close, 0.0)
        out[0, j] = dq
        out[1, j] = d * tau_close
        return out

    def tick(self, t: float) -> None:
        base, look, trigger, forecast = self.base_fn(t)
        base = np.asarray(base, dtype=float)
        bad_base = not np.all(np.isfinite(base))
        if bad_base:  # never pass a non-finite command on: hold the last finite base, or send nothing yet
            self.n_base_nonfinite += 1
            if self.good_base is None:
                return
            base = self.good_base
        else:
            self.good_base = base
        obs = self.observe(t, base, look, trigger, forecast)
        bad = ~np.isfinite(obs)
        if bad.any():  # a non-finite input would stay in the policy's recurrent state until reset
            obs = np.where(bad, 0.0, obs)
            self.n_obs_nonfinite += 1
        a = np.asarray(self.policy(obs), dtype=float).reshape(ACTION_DIM)
        h = getattr(self.policy, "h", None)
        if h is not None and not np.all(np.isfinite(h)):
            self.policy.reset()
            self.n_state_reset += 1
        if not np.all(np.isfinite(a)):
            a = np.zeros_like(a)
            self.n_nonfinite += 1
        a = np.clip(a, -1.0, 1.0)
        target = self._trigger_guard(offsets(self.spec, self.jp, a), trigger)
        if bad_base or self.stale(t):
            target = self.off * np.exp(-self.dt / self.cfg.decay_tau)
            self.n_stale += 1
        self.off = self._trigger_guard(slew(self.spec, self.jp, self.off, target, self.dt), trigger)
        cmd = HandCommand.from_array(t, compose(self.spec, self.jp, base, self.off), src="residual")
        self.hand.set_hand_command(cmd)
        self.last_a, self.last_obs, self.last_cmd, self.last_base = a, obs, cmd, base
        self.n_ticks += 1
