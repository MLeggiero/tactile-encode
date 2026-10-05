"""Safety supervisor: stale-command fallback and the hold pose. Torque saturation lives in World."""

from __future__ import annotations

import numpy as np

from tactile_sim.config import ControllerCfg
from tactile_sim.control.interface import L2Command, Mode


class Supervisor:
    def __init__(self, cfg: ControllerCfg):
        self.cfg = cfg
        self.hold_cmd: L2Command | None = None
        self.stale = False
        self.frozen = False

    def reset(self) -> None:
        self.hold_cmd = None
        self.stale = False
        self.frozen = False

    def freeze(self) -> None:
        """Drop detected (or other fault): hold the current pose until reset."""
        self.frozen = True

    def filter(self, cmd: L2Command | None, t: float, tcp_pos: np.ndarray, tcp_R: np.ndarray) -> L2Command:
        stale = cmd is None or (t - cmd.t) > self.cfg.stale_timeout
        if stale or self.frozen:
            if self.hold_cmd is None:
                grip = self.cfg.grip_hold if cmd is None else max(cmd.F_grip, self.cfg.grip_hold)
                K = np.array(self.cfg.k_trans + self.cfg.k_rot, dtype=float)
                self.hold_cmd = L2Command(t, tcp_pos.copy(), tcp_R.copy(), K=K, F_grip=grip, mode=Mode.HOLD)
            self.stale = stale
            return self.hold_cmd
        self.hold_cmd = None
        self.stale = False
        return cmd
