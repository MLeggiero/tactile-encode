"""The residual's safety layer (HandSafety): what bounds the policy's output on the robot, whatever the policy observes.

ResidualHandL2.tick applies, in order: a non-finite base held at the last finite one; non-finite inputs zeroed and a
non-finite recurrent state restarted; non-finite actions zeroed; the action clipped to [-1, 1]; the drill-trigger
interlock (`trigger_guard`); stale inputs decaying the offsets; the offsets' slew limit. The interlock lives here,
outside the policy: on the drill task the residual's offsets on the trigger finger's joints are capped, and may only
open the finger while the base disables the trigger (the base's trigger enable is an observation for the policy, but
the guard never depends on what the policy makes of it). The GPU training env applies the same guard
(tactile_rl.mdp.action_map.TorchActionMap.trigger_guard).
"""

from __future__ import annotations

import numpy as np


class HandSafety:
    def __init__(self, cfg, trig: np.ndarray, close_dir: np.ndarray):
        """`cfg`: the ResidualCfg (trigger_dq_max, trigger_tau); `trig`: the trigger finger's joints
        (residual_l2.trigger_indices: none off the drill task); `close_dir`: the synergy's closing direction."""
        self.cfg = cfg
        self.trig = np.asarray(trig, dtype=int)
        self.close_dir = np.asarray(close_dir, dtype=float)

    def trigger_guard(self, off: np.ndarray, trigger: bool) -> np.ndarray:
        """The offsets (2, 20) [dq, dtau] with the interlock applied (unchanged where there is no trigger)."""
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
