"""Reference spreading for the strike (van Steen et al.): an ante-impact reference that continues
through the expected contact, a post-impact reference that starts where the tool actually is when
the impact is flagged, and an interim mode around the predicted contact. The switch to the post
reference happens on the impact flag, never on the clock; a clock timeout only declares a miss."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tactile_sim.control.interface import Mode


class ReferenceSpreader:
    def __init__(self, ante, make_post: Callable[[float, np.ndarray], object], t_c_pred: float,
                 interim_lead: float = 0.010, timeout: float = 0.100):
        self.ante = ante
        self.make_post = make_post
        self.t_c_pred = t_c_pred
        self.interim_lead = interim_lead
        self.timeout = timeout
        self.post = None
        self.t_switch: float | None = None
        self.missed = False

    def evaluate(self, t: float, t_flag: float | None, x_now: np.ndarray):
        if self.post is None:
            if t_flag is not None:
                self.t_switch = t_flag
                self.post = self.make_post(t_flag, x_now)
            elif t > self.t_c_pred + self.timeout:
                self.missed = True
                self.t_switch = t
                self.post = self.make_post(t, x_now)
        if self.post is not None:
            x, xd, xdd = self.post.evaluate(t)
            return x, xd, xdd, Mode.POST
        x, xd, xdd = self.ante.evaluate(t)
        if t >= self.t_c_pred - self.interim_lead:
            # interim: keep the ante position/velocity; drop an accelerating feedforward (it would push through
            # the contact) but keep a braking one (its torque reversal is already under way)
            if float(xdd[:3] @ xd[:3]) >= 0.0:
                xdd = np.zeros(6)
            return x, xd, xdd, Mode.INTERIM
        return x, xd, xdd, Mode.ANTE


class PlainRef:
    """A path reference without impact handling (approach, windup, recover)."""

    def __init__(self, path, mode: Mode = Mode.FREE):
        self.path = path
        self.mode = mode

    def evaluate(self, t: float, t_flag: float | None, x_now: np.ndarray):
        x, xd, xdd = self.path.evaluate(t)
        return x, xd, xdd, self.mode
