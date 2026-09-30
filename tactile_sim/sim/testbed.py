"""Testbed: World + sensor suite + L1 controller + grip loop, stepped by the multi-rate scheduler.

Per physics step: scheduled callbacks (L2 source at 200 Hz, L1 at 1 kHz, grip loop at 500 Hz) run
first and write actuator commands, then MuJoCo steps, then every sensor samples the new state.
"""

from __future__ import annotations

import numpy as np

from tactile_sim.config import SimConfig
from tactile_sim.control.gripper import GripForceLoop
from tactile_sim.control.l1 import L1Controller
from tactile_sim.sensors import Holder, build_sensor_suite
from tactile_sim.sim.scheduler import RateScheduler
from tactile_sim.sim.world import World


class Testbed:
    __test__ = False  # not a pytest test class
    def __init__(self, cfg: SimConfig | None = None, seed: int | None = 0, log: bool = True):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(seed)
        self.world = World(self.cfg)
        self.world.reset()
        self.tau_ext = Holder(7)
        self.sensors = build_sensor_suite(self.world, self.cfg, self.rng, self.tau_ext)
        self.l1 = L1Controller(self.world, self.cfg, self.sensors, self.tau_ext, log=log)
        self.grip = GripForceLoop(self.world, self.cfg, self.sensors, log=log)
        self.sched = RateScheduler(self.world.dt)
        self.l2_callbacks = []
        self._build_schedule()
        self.reset(seed)

    def _build_schedule(self) -> None:
        c = self.cfg.controller
        self.sched.tasks.clear()
        self.sched.register("l2", self.cfg.swing.rate, self._l2_tick)
        self.sched.register("l1", c.rate, self.l1.tick)
        self.sched.register("grip", c.grip_rate, self._grip_tick)
        if self.world.hand.kind != "franka":
            # the hand's own joint controller (MIT mode), after the grip loop has set this tick's synergy
            self.sched.register("hand", self.cfg.gripper.hand_rate, self.world.hand.tick)

    def _l2_tick(self, t: float) -> None:
        for fn in self.l2_callbacks:
            fn(t)

    def _grip_tick(self, t: float) -> None:
        self.grip.tick(t, getattr(self.l1, "grip_setpoint", self.cfg.controller.grip_hold), self.l1)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            for s in self.sensors.sensors.values():
                s.rng = self.rng
        self.world.reset(seed)
        self.sensors.reset(self.world.t)
        self.l1.reset()
        self.grip.reset()
        self.sched.reset()
        self.l1.hold_here(self.world.t)
        self.l1.grip_setpoint = self.cfg.controller.grip_hold

    @property
    def t(self) -> float:
        return self.world.t

    def step(self, n: int = 1) -> None:
        w, sch, sen = self.world, self.sched, self.sensors
        for _ in range(n):
            sch.tick(w.t)
            w.step()
            sen.step(w.t)

    def run_for(self, duration: float, callback=None) -> None:
        """Advance `duration` seconds; `callback(tb)` runs after every physics step if given."""
        n = int(round(duration / self.world.dt))
        for _ in range(n):
            self.step()
            if callback is not None:
                callback(self)
