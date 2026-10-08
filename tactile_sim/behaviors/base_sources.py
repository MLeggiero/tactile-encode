"""Base sources for the residual policy: what drives the robot under it, in the form ResidualHandL2 takes
(base_fn(t) -> base MIT hand command (5, 20), lookahead block, trigger enable, forecast block).

SwingBase wraps the scripted hammer swing (behaviors.swing.ScriptedSwing on a position-interface arm), DrillBase
the scripted drill feed (behaviors.tool_tasks.PositionDrillFeed). SwingBase:

- base hand command: the grip synergy as MIT commands, from the angles the host last read; the synergy's grip
  scalar follows the swing's grip setpoint through the grip loop, as in the scripted runs;
- lookahead: the arm's joint targets at the LOOKAHEAD times, relative to the current target, from the swing's
  path evaluated ahead (the ante path through the predicted contact, or the post path once it has switched; the
  reference spreader itself is never evaluated ahead, since that would declare a miss), solved with the arm's own
  IK and refreshed at the arm's command rate; the hand's targets do not move ahead of time under this base;
- forecast: the swing's predicted contact time from the windup on (during the windup, the end of the windup plus
  the previous swing's duration), an impulse regime, and an amplitude from the head's momentum, through a
  ForecastModel that adds a VLA forecast head's errors and masks whole episodes.
"""

from __future__ import annotations

import numpy as np

from tactile_sim.behaviors.forecast import ForecastModel, ForecastTruth
from tactile_sim.control.obs_spec import LOOKAHEAD, N_ARM, N_JOINTS

PULSE = 0.003  # s, the blow's duration used for the amplitude estimate (m v / pulse)


def _ahead_path(ref):
    """The pure path a reference follows from now on, safe to evaluate at future times."""
    if ref is None:
        return None
    if hasattr(ref, "ante"):  # ReferenceSpreader
        return ref.post if ref.post is not None else ref.ante
    return getattr(ref, "path", None)


class SwingBase:
    def __init__(self, tb, swing, forecast: ForecastModel | None = None, rng: np.random.Generator | None = None):
        self.tb = tb
        self.swing = swing
        self.forecast = forecast or ForecastModel(tb.cfg.forecast, rng)
        self.l1 = tb.l1
        self.has_ik = hasattr(self.l1, "_ik") and hasattr(self.l1, "q_ref")
        self.reset()

    def reset(self) -> None:
        self.forecast.reset()
        self.arm_look = np.zeros((len(LOOKAHEAD), N_ARM))
        self.next_look = -np.inf
        self.t_swing_prev = 0.3  # s, a swing's duration before the first one is known
        self.windup_t0: float | None = None

    # ------------------------------------------------------------------ pieces
    def _update_arm_lookahead(self, t: float) -> None:
        if not self.has_ik:
            return
        cmd = self.swing.cmd
        path = _ahead_path(cmd.ref)
        if path is None:
            self.arm_look[:] = 0.0
            return
        R = getattr(cmd.ref, "R", None)
        R = cmd.R_eq if R is None else R
        R_path = getattr(path, "R", None)
        R = R if R_path is None else R_path
        for k, dt in enumerate(LOOKAHEAD):
            x, xd, _ = path.evaluate(t + dt)
            q, _ = self.l1._ik(np.asarray(x, float), R, np.asarray(xd, float))
            self.arm_look[k] = q - self.l1.q_ref

    def truth(self, t: float) -> ForecastTruth:
        sw = self.swing
        m_head = self.tb.cfg.hammer.head_mass
        if sw.phase == "windup":
            if self.windup_t0 is None:
                self.windup_t0 = sw.phase_t0
            t_c = self.windup_t0 + self.tb.cfg.swing.windup_time + 0.05 + self.t_swing_prev
            v = self.tb.cfg.swing.v_strike
            return ForecastTruth(True, t_c, "impulse", 0.0, m_head * v / PULSE)
        self.windup_t0 = None
        p = getattr(sw, "plan", None)
        if sw.phase == "swing" and p is not None:
            self.t_swing_prev = p.t_c_pred - p.t_start
            return ForecastTruth(True, p.t_c_pred, "impulse", 0.0, m_head * p.v_strike / PULSE)
        if sw.phase == "post" and p is not None and t <= p.t_c_pred + 0.06:
            return ForecastTruth(True, p.t_c_pred, "impulse", 0.0, m_head * p.v_strike / PULSE)
        return ForecastTruth(False)

    # ------------------------------------------------------------------ the base_fn
    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
        hand = self.tb.world.hand
        q = self.tb.sensors["hand_q"].latest().value
        base = np.stack(hand.synergy.mit_command(hand.s, q))
        if t >= self.next_look - 1e-12:
            self._update_arm_lookahead(t)
            rate = getattr(self.tb.cfg.arm, "command_rate", 100.0)
            self.next_look = (np.floor(t * rate + 1e-9) + 1.0) / rate
        look = np.concatenate([np.concatenate([np.zeros(N_JOINTS), self.arm_look[k]]) for k in range(len(LOOKAHEAD))])
        return base, look, False, self.forecast(t, self.truth(t))


class DrillBase:
    """The scripted drill feed (behaviors.tool_tasks.PositionDrillFeed) as the residual's base: the synergy as MIT
    hand commands, the arm's joint targets at the lookahead times from the feed's own target(t) (solved with the
    arm's IK at its command rate), the trigger enabled while the base pulls it, and the plant's next load event
    (DrillPlant.event_truth: seating and clutch ripple, breakthrough) through a ForecastModel."""

    def __init__(self, tb, feed, forecast: ForecastModel | None = None, rng: np.random.Generator | None = None):
        self.tb = tb
        self.feed = feed
        self.forecast = forecast or ForecastModel(tb.cfg.forecast, rng)
        self.l1 = tb.l1
        self.has_ik = hasattr(self.l1, "_ik") and hasattr(self.l1, "q_ref")
        self.reset()

    def reset(self) -> None:
        self.forecast.reset()
        self.arm_look = np.zeros((len(LOOKAHEAD), N_ARM))
        self.next_look = -np.inf

    def _update_arm_lookahead(self, t: float) -> None:
        if not self.has_ik or self.feed.phase in ("start", "done"):
            self.arm_look[:] = 0.0
            return
        R = self.feed.R
        for k, dt in enumerate(LOOKAHEAD):
            q, _ = self.l1._ik(np.asarray(self.feed.target(t + dt), float), R, np.zeros(6))
            self.arm_look[k] = q - self.l1.q_ref

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
        hand = self.tb.world.hand
        q = self.tb.sensors["hand_q"].latest().value
        base = np.stack(hand.synergy.mit_command(hand.s, q))
        if t >= self.next_look - 1e-12:
            self._update_arm_lookahead(t)
            rate = getattr(self.tb.cfg.arm, "command_rate", 100.0)
            self.next_look = (np.floor(t * rate + 1e-9) + 1.0) / rate
        look = np.concatenate([np.concatenate([np.zeros(N_JOINTS), self.arm_look[k]]) for k in range(len(LOOKAHEAD))])
        trigger = self.feed.trigger > 0
        return base, look, trigger, self.forecast(t, self.tb.world.plant.event_truth(t))
