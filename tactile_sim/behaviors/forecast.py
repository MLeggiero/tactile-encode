"""The load-regime forecast a base hands the residual, made from the truth with the errors a VLA's forecast head
would have (ForecastCfg): a per-episode mask, a per-episode bias and per-update jitter on the event time (growing
with the lead time, `horizon_err`), regime confusion (a per-episode share of the probability on one wrong regime,
which is named above 0.5), log-normal errors on frequency and amplitude, dropped updates, and a latency between the
state a forecast reflects and when the residual sees it. Truth comes from whoever knows the plan: a scripted swing's
predicted contact time, a plant's state, or the load generator's schedule.

The uncertainties the forecast reports (the sigma fields of forecast_vector) are a forecaster's claim, not its true
error: they come from `claimed` (ForecastCfg's nominal values), whatever error model the episode runs with, so a
policy cannot read an episode's forecast reliability off them."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from tactile_sim.config import ForecastCfg
from tactile_sim.control.obs_spec import FORECAST_DIM, REGIMES, forecast_vector


@dataclass
class ForecastTruth:
    valid: bool
    t_event: float = 0.0  # absolute time of the next event (s)
    regime: str = "impulse"
    f_dom: float = 0.0  # Hz
    amp: float = 0.0  # N


class ForecastModel:
    def __init__(self, cfg: ForecastCfg | None = None, rng: np.random.Generator | None = None,
                 claimed: ForecastCfg | None = None):
        self.cfg = cfg or ForecastCfg()
        self.claimed = claimed or ForecastCfg()
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        c, r = self.cfg, self.rng
        self.masked = bool(r.uniform() < c.mask_prob)
        self.bias = float(r.normal(0.0, c.bias_sigma))
        self.confusion = float(r.uniform(0.0, c.class_confusion))
        # the wrong regime the confusion goes to, among the other ones (0 .. R - 2, skipping the true one)
        self.wrong = min(int(r.uniform() * (len(REGIMES) - 1)), len(REGIMES) - 2)
        self.next_update = -np.inf
        self.pending: deque[tuple[float, np.ndarray, float]] = deque()  # (t_issue, regime probs + f + amp, t_event)
        self.visible: tuple[float, np.ndarray, float] | None = None

    def _issue(self, t: float, truth: ForecastTruth) -> None:
        c, r = self.cfg, self.rng
        if not truth.valid or r.uniform() < c.dropout:
            if not truth.valid:
                self.pending.append((t, np.zeros(0), np.nan))  # an explicit "no event ahead"
            return
        k = REGIMES.index(truth.regime)
        p = np.zeros(len(REGIMES))
        p[k] = 1.0 - self.confusion
        p[self.wrong + (self.wrong >= k)] = self.confusion
        f = truth.f_dom * float(np.exp(r.normal(0.0, c.freq_err)))
        amp = truth.amp * float(np.exp(r.normal(0.0, c.amp_err)))
        sigma = float(np.hypot(c.jitter_sigma, c.horizon_err * max(truth.t_event - t, 0.0)))
        t_event = truth.t_event + self.bias + float(r.normal(0.0, 1.0)) * sigma
        self.pending.append((t, np.concatenate([p, [f, amp]]), t_event))

    def __call__(self, t: float, truth: ForecastTruth) -> np.ndarray:
        """The 13-D forecast block the residual sees at time t, given the truth as of now."""
        c = self.cfg
        if self.masked:
            return np.zeros(FORECAST_DIM)
        if t >= self.next_update - 1e-12:
            self._issue(t, truth)
            self.next_update = (np.floor(t * c.update_rate + 1e-9) + 1.0) / c.update_rate  # next grid point
        while self.pending and self.pending[0][0] + c.latency <= t + 1e-12:
            self.visible = self.pending.popleft()
        if self.visible is None or np.isnan(self.visible[2]):
            return np.zeros(FORECAST_DIM)
        t_issue, v, t_event = self.visible
        n = len(REGIMES)
        cl = self.claimed
        s_t = float(np.hypot(cl.bias_sigma, cl.jitter_sigma))
        return forecast_vector(True, t_event - t, s_t, v[:n], v[n], cl.freq_err * v[n], v[n + 1],
                               cl.amp_err * v[n + 1], age=t - t_issue)
