"""Synthetic loads at the tool's working point: the load-generator stage of the residual's training.

An episode's schedule is a Poisson stream of events (LoadGenCfg.rate), each one of the regimes the residual must
handle, drawn with LoadGenCfg.p_kind:

- impulse: a half-sine force pulse of 0.5-5 ms along the strike axis (within a cone), like a hammer blow;
- periodic: a force along the feed or a torque about the long axis at 1-100 Hz (a saw stroke to clutch ripple),
  its amplitude drifting as a random walk;
- step: a steady push that disappears (drill breakthrough) or appears;
- bind: a 20-80 ms torque spike about the long axis (a bit grabbing);
- push: a steady force along the feed.

Forces are in the tool frame (x along the handle or bit, z the strike or feed direction), torques about the tool
x axis; `wrench(t)` sums the active events. `forecast_truth(t)` gives the next event as a forecast truth: the
schedule is the ground truth a VLA's forecast head would be trained to predict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tactile_sim.behaviors.forecast import ForecastTruth
from tactile_sim.config import LoadGenCfg

KINDS = ("impulse", "periodic", "step", "bind", "push")
REGIME_OF = {"impulse": "impulse", "periodic": "periodic", "step": "step", "bind": "bind", "push": "steady"}


@dataclass
class LoadEvent:
    kind: str
    t0: float
    duration: float
    amp: float  # N, or Nm for a torque
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))  # unit force direction
    torque: bool = False  # about the tool x axis instead of a force
    freq: float = 0.0  # Hz
    phase: float = 0.0
    sign: float = 1.0  # step: +1 the push vanishes at t0 (breakthrough), -1 it appears
    drift: np.ndarray | None = None  # periodic: amplitude multiplier samples, one per 10 ms

    @property
    def t1(self) -> float:
        return self.t0 + self.duration

    def value(self, t: float) -> float:
        """Signed load magnitude along `direction` (or about x) at time t."""
        tau = t - self.t0
        if self.kind == "step":
            hold = self.duration
            if self.sign > 0:  # a push held before t0, gone after
                return self.amp if -hold <= tau < 0 else 0.0
            return self.amp if 0 <= tau < hold else 0.0
        if tau < 0 or tau >= self.duration:
            return 0.0
        if self.kind == "impulse":
            return self.amp * np.sin(np.pi * tau / self.duration)
        if self.kind == "bind":
            return self.amp * np.sin(np.pi * tau / self.duration) ** 2
        if self.kind == "periodic":
            k = min(int(tau / 0.01), len(self.drift) - 1) if self.drift is not None else 0
            m = self.drift[k] if self.drift is not None else 1.0
            return self.amp * m * np.sin(2 * np.pi * self.freq * tau + self.phase)
        return self.amp  # push


def _u(rng, lohi):
    return float(rng.uniform(*lohi))


def _cone(rng, half_angle: float) -> np.ndarray:
    """A unit vector within half_angle of +z, uniform over the cap."""
    c = rng.uniform(np.cos(half_angle), 1.0)
    phi = rng.uniform(0, 2 * np.pi)
    s = np.sqrt(1 - c * c)
    return np.array([s * np.cos(phi), s * np.sin(phi), c])


def sample_schedule(cfg: LoadGenCfg, rng: np.random.Generator, t_end: float, t_start: float = 0.2) -> list[LoadEvent]:
    """An episode's events from t_start to t_end, non-overlapping within each kind."""
    p = np.asarray(cfg.p_kind, dtype=float)
    p = p / p.sum()
    events, t = [], t_start
    while True:
        t += float(rng.exponential(1.0 / cfg.rate))
        if t >= t_end:
            break
        kind = KINDS[int(rng.choice(len(KINDS), p=p))]
        if kind == "impulse":
            ev = LoadEvent(kind, t, _u(rng, cfg.impulse_ms) * 1e-3, _u(rng, cfg.impulse_n),
                           direction=_cone(rng, np.radians(cfg.cone_deg)))
        elif kind == "periodic":
            dur = _u(rng, cfg.periodic_s)
            torque = bool(rng.uniform() < 0.5)
            amp = _u(rng, cfg.periodic_nm if torque else cfg.periodic_n)
            n = int(np.ceil(dur / 0.01)) + 1
            walk = np.cumsum(rng.normal(0.0, cfg.drift * np.sqrt(0.01), n))
            ev = LoadEvent(kind, t, dur, amp, torque=torque, freq=float(np.exp(rng.uniform(*np.log(cfg.periodic_hz)))),
                           phase=float(rng.uniform(0, 2 * np.pi)), drift=np.clip(1.0 + walk, 0.2, 2.0))
        elif kind == "step":
            hold = _u(rng, cfg.step_hold_s)
            sign = 1.0 if rng.uniform() < 0.7 else -1.0
            # breakthrough (+1): the push holds from now and vanishes at t0; appearance (-1): it starts at t0
            ev = LoadEvent(kind, t + hold if sign > 0 else t, hold, _u(rng, cfg.step_n), sign=sign)
        elif kind == "bind":
            ev = LoadEvent(kind, t, _u(rng, cfg.bind_ms) * 1e-3, _u(rng, cfg.bind_nm), torque=True)
        else:
            ev = LoadEvent(kind, t, _u(rng, cfg.push_s), _u(rng, cfg.push_n))
        events.append(ev)
        t = max(t, _end(ev))  # the next event starts after this one is over
    return events


def _end(ev: LoadEvent) -> float:
    return ev.t0 if (ev.kind == "step" and ev.sign > 0) else ev.t1


def wrench(events: list[LoadEvent], t: float) -> tuple[np.ndarray, np.ndarray]:
    """Force (N) and torque (Nm) on the tool at time t, tool frame."""
    f = np.zeros(3)
    tq = np.zeros(3)
    for ev in events:
        v = ev.value(t)
        if v == 0.0:
            continue
        if ev.torque:
            tq[0] += v
        else:
            f += v * ev.direction
    return f, tq


def forecast_truth(events: list[LoadEvent], t: float) -> ForecastTruth:
    """The next event, or the one under way: when it happens (a breakthrough step: when the push vanishes), its
    regime, frequency and amplitude."""
    for ev in events:
        if _end(ev) > t:
            return ForecastTruth(True, ev.t0, REGIME_OF[ev.kind], ev.freq, ev.amp)
    return ForecastTruth(False)
