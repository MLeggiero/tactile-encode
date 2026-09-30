"""Gymnasium environment exposing the L2 -> L1 interface for Task A.

One env step is one L2 tick (200 Hz by default, 5 ms): the action sets the L1 command, then the physics
runs until the next tick with L1 at 1 kHz and the grip loop at 500 Hz underneath.

Action (Box, 6):  [dx, dy, dz]  target TCP offset from the hover pose (m, world, clipped to +-0.25)
                  k_scale       stiffness multiplier on the nominal impedance (0.25 .. 2)
                  f_ff          feedforward force along the strike axis (N, -60 .. 60)
                  f_grip        grasp-force setpoint per pad (N, 0 .. 120)
Observation (Dict): every sensor sample that became available during the step (fixed-length windows),
plus TCP pose, the impact flag raised during the step and the grip loop's force reading.
Reward: nail advance in mm during the step, minus 0.05 per degree of tool rotation in the grasp and a
small penalty for torque saturation. Terminates when the nail is driven or the tool is dropped.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tactile_sim.config import SimConfig, fast_config
from tactile_sim.control.interface import L2Command
from tactile_sim.logging.strike_metrics import rotvec_diff
from tactile_sim.sim.testbed import Testbed


class TactileHammerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: SimConfig | None = None, max_time: float = 6.0, auto_arm_gap: float = 0.02):
        super().__init__()
        self.base_cfg = cfg or fast_config()
        self.max_time = max_time
        self.auto_arm_gap = auto_arm_gap
        self.tb: Testbed | None = None
        self.dr = None
        c = self.base_cfg
        self.dt_env = 1.0 / c.swing.rate
        self.n_ft = int(round(c.sensors.ft_rate * self.dt_env))
        self.n_acc = int(round(c.sensors.accel_rate * self.dt_env))
        self.action_space = spaces.Box(
            low=np.array([-0.25, -0.25, -0.25, 0.25, -60.0, 0.0], dtype=np.float32),
            high=np.array([0.25, 0.25, 0.25, 2.0, 60.0, 120.0], dtype=np.float32))
        inf = np.inf
        self.observation_space = spaces.Dict({
            "ft": spaces.Box(-inf, inf, (self.n_ft, 6), np.float32),
            "pad_acc": spaces.Box(-inf, inf, (self.n_acc, 6), np.float32),
            "pressure": spaces.Box(0.0, inf, (2, 16), np.float32),
            "joint_pos": spaces.Box(-inf, inf, (7,), np.float32),
            "joint_vel": spaces.Box(-inf, inf, (7,), np.float32),
            "tau_ext": spaces.Box(-inf, inf, (7,), np.float32),
            "tcp": spaces.Box(-inf, inf, (7,), np.float32),
            "impact": spaces.Box(0.0, 1.0, (1,), np.float32),
            "grip": spaces.Box(-inf, inf, (1,), np.float32),
        })

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg = self.base_cfg
        if cfg.dr.enabled:
            from tactile_sim.sim.randomize import apply_dr, apply_runtime_dr, sample_dr

            self.dr = sample_dr(cfg, self.np_random)
            cfg = apply_dr(cfg, self.dr)
            self.tb = Testbed(cfg, seed=int(self.np_random.integers(2**31)))
            apply_runtime_dr(self.tb, self.dr)
        elif self.tb is None:
            self.tb = Testbed(cfg, seed=seed)
        else:
            self.tb.reset(seed)
        self.cfg = cfg
        tb = self.tb
        tb.l2_callbacks[:] = []
        w = tb.world
        c = cfg.controller
        self.K_nom = np.array(c.k_trans + c.k_rot, dtype=float)
        self.cmd = L2Command(tb.t, w.hover_tcp.copy(), w.tcp_R_nominal.copy(), K=self.K_nom.copy(),
                             F_grip=c.grip_hold)
        tb.l1.set_command(self.cmd)
        self._rv0 = w.hammer_in_hand()[1]
        self._depth = w.plant.depth
        self._n_events = 0
        return self._obs(False), self._info()

    def step(self, action):
        tb, w = self.tb, self.tb.world
        a = np.clip(np.asarray(action, dtype=float), self.action_space.low, self.action_space.high)
        cmd = self.cmd
        cmd.x_eq = w.hover_tcp + a[:3]
        cmd.xd_eq = np.zeros(6)
        cmd.K = self.K_nom * a[3]
        cmd.F_ff = np.concatenate([a[4] * tb.l1.axis, np.zeros(3)])
        cmd.F_grip = float(a[5])
        n = int(round(self.dt_env / w.dt))
        sat = 0
        for _ in range(n):
            cmd.t = tb.t
            if not tb.l1.detector.armed and 0 < w.plant.gap() < self.auto_arm_gap:
                tb.l1.detector.arm(True)
            tb.step()
            sat += int(np.any(np.abs(w.data.ctrl[w.arm_act]) >= w.tau_limit - 1e-6))
        impact = len(tb.l1.detector.events) > self._n_events
        self._n_events = len(tb.l1.detector.events)
        depth = w.plant.depth
        rot = rotvec_diff(self._rv0, w.hammer_in_hand()[1])
        reward = 1e3 * (depth - self._depth) - 0.05 * np.degrees(rot) * self.dt_env - 0.01 * sat / n
        self._depth = depth
        terminated = bool(w.plant.done() or tb.grip.dropped)
        truncated = bool(tb.t >= self.max_time)
        return self._obs(impact), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------
    def _window(self, name: str, n: int, dim: int) -> np.ndarray:
        v = self.tb.sensors[name].window(n)
        out = np.zeros((n, dim), dtype=np.float32)
        if len(v):
            out[-len(v):] = v[-n:]
        return out

    def _obs(self, impact: bool) -> dict:
        tb, w = self.tb, self.tb.world
        p, R = w.tcp_pose()
        import mujoco

        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.reshape(-1))
        acc = np.concatenate([self._window("pad_acc_L", self.n_acc, 3), self._window("pad_acc_R", self.n_acc, 3)], 1)
        return {
            "ft": self._window("ft", self.n_ft, 6),
            "pad_acc": acc,
            "pressure": np.stack([tb.sensors.latest("pressure_L"), tb.sensors.latest("pressure_R")]).astype(np.float32),
            "joint_pos": tb.sensors.latest("joint_pos").astype(np.float32),
            "joint_vel": tb.sensors.latest("joint_vel").astype(np.float32),
            "tau_ext": tb.sensors.latest("tau_ext").astype(np.float32),
            "tcp": np.concatenate([p, q]).astype(np.float32),
            "impact": np.array([float(impact)], dtype=np.float32),
            "grip": np.array([tb.grip.measured], dtype=np.float32),
        }

    def _info(self) -> dict:
        w = self.tb.world
        p, rv = w.hammer_in_hand()
        info = {"t": self.tb.t, "nail_depth": w.plant.depth, "face_gap": w.plant.gap(),
                "tool_in_hand_pos": p, "tool_in_hand_rotvec": rv, "dropped": self.tb.grip.dropped}
        if self.dr is not None:
            info["dr"] = self.dr.as_dict()
        return info


def make_env(**kw) -> TactileHammerEnv:
    return TactileHammerEnv(**kw)


gym.register(id="TactileHammer-v0", entry_point="tactile_sim.env.gym_env:TactileHammerEnv") \
    if "TactileHammer-v0" not in gym.registry else None
