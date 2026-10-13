"""Synthetic loads at the tool's working point: the load-generator stage of the residual's training.

An episode's schedule is a Poisson stream of events (LoadGenCfg.rate), each one of the regimes the residual must
handle, drawn with LoadGenCfg.p_kind:

- impulse: a half-sine force pulse of 0.5-5 ms along the strike axis (within a cone), like a hammer blow;
- periodic: a force along the feed or a torque about the long axis at 1-100 Hz (a saw stroke to clutch ripple),
  its amplitude drifting as a random walk;
- step: a steady push that disappears (drill breakthrough) or appears;
- bind: a 20-80 ms torque spike about the long axis (a bit grabbing), or about any tool axis (LoadGenCfg.bind_axes);
- push: a steady force along the feed;
- torque_impulse: a half-sine torque pulse about a tool axis, either sign (a glancing blow's twist; drawn only when
  LoadGenCfg.p_kind has a sixth entry).

Forces are in the tool frame (x along the handle or bit, z the strike or feed direction), torques about the tool
x axis unless the event names its axis (`axis`, a signed unit vector); `wrench(t)` sums the active events.
`forecast_truth(t)` gives the next event as a forecast truth: the schedule is the ground truth a VLA's forecast head
would be trained to predict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tactile_sim.behaviors.forecast import ForecastTruth
from tactile_sim.config import LoadGenCfg

KINDS = ("impulse", "periodic", "step", "bind", "push", "torque_impulse")
REGIME_OF = {"impulse": "impulse", "periodic": "periodic", "step": "step", "bind": "bind", "push": "steady",
             "torque_impulse": "impulse"}
AXES = np.eye(3)


def kind_probs(cfg: LoadGenCfg) -> np.ndarray:
    """The kinds' probabilities in KINDS order (a five-entry p_kind draws no torque impulses)."""
    p = np.zeros(len(KINDS))
    q = np.asarray(cfg.p_kind, dtype=float)
    if len(q) not in (5, 6):
        raise ValueError(f"p_kind has {len(q)} entries (5 or 6: {KINDS})")
    p[:len(q)] = q
    return p / p.sum()


@dataclass
class LoadEvent:
    kind: str
    t0: float
    duration: float
    amp: float  # N, or Nm for a torque
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))  # unit force direction
    torque: bool = False  # a torque (about `axis`) instead of a force
    axis: np.ndarray | None = None  # a torque's signed unit axis, tool frame (None: +x, the long axis)
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
        if self.kind in ("impulse", "torque_impulse"):
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


def _axis(rng) -> np.ndarray:
    """A tool axis, either sign."""
    return AXES[int(rng.integers(3))] * (1.0 if rng.uniform() < 0.5 else -1.0)


def sample_schedule(cfg: LoadGenCfg, rng: np.random.Generator, t_end: float, t_start: float = 0.2) -> list[LoadEvent]:
    """An episode's events from t_start to t_end, non-overlapping within each kind. A configuration without the sixth
    kind and with binds about the long axis draws exactly the schedules it drew before torque impulses existed."""
    p = kind_probs(cfg)
    if len(cfg.p_kind) == 5:
        p = p[:5]  # the same draws as before the sixth kind
    events, t = [], t_start
    while True:
        t += float(rng.exponential(1.0 / cfg.rate))
        if t >= t_end:
            break
        kind = KINDS[int(rng.choice(len(p), p=p))]
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
            if cfg.bind_axes == "xyz":
                ev.axis = _axis(rng)
            elif cfg.bind_axes != "x":
                raise ValueError(f"bind_axes {cfg.bind_axes!r} (expected 'x' or 'xyz')")
        elif kind == "torque_impulse":
            ev = LoadEvent(kind, t, _u(rng, cfg.torque_impulse_ms) * 1e-3, _u(rng, cfg.torque_impulse_nm), torque=True)
            ev.axis = _axis(rng)
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
            if ev.axis is None:
                tq[0] += v
            else:
                tq += v * np.asarray(ev.axis, float)
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
