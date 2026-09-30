"""Rate-limited sensor model.

Every physics step each sensor reads its raw signal. The pipeline is:

    raw (physics rate) -> band-limit (Butterworth, at the physics rate)
        -> decimate to the sensor rate ("mean" boxcar or "zoh" last value)
        -> latency (sample taken at t_sample becomes visible at t_avail = t_sample + latency)
        -> + per-episode bias + white noise -> quantize -> saturate

Consumers read `latest()`, which is zero-order-held between samples and already delayed, so a
controller sees exactly what a real one would. Loggers read the full history with both timestamps.
Filtering is done per block at emission time (one scipy call per sample, not per physics step).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class SensorSpec:
    name: str
    rate_hz: float
    dim: int
    bandwidth_hz: float | None = None
    filter_order: int = 2
    latency_s: float = 0.0
    noise_std: float | tuple[float, ...] = 0.0
    bias_std: float | tuple[float, ...] = 0.0
    saturation: float | tuple[float, float] | None = None  # symmetric limit or (lo, hi)
    quant_step: float | None = None
    decimation: str = "mean"  # "mean" or "zoh"


@dataclass
class SensorSample:
    t_sample: float
    t_avail: float
    value: np.ndarray
    seq: int


class RateLimitedSensor:
    def __init__(self, spec: SensorSpec, physics_dt: float, read_fn: Callable[[], np.ndarray],
                 rng: np.random.Generator | None = None, keep_history: bool = True):
        if spec.rate_hz <= 0:
            raise ValueError("rate must be positive")
        if spec.decimation not in ("mean", "zoh"):
            raise ValueError(f"unknown decimation {spec.decimation!r}")
        self.spec = spec
        self.dt = physics_dt
        self.read_fn = read_fn
        self.rng = rng or np.random.default_rng()
        self.keep_history = keep_history
        self.period = 1.0 / spec.rate_hz
        f_nyq = 0.5 / physics_dt
        self.sos = None
        if spec.bandwidth_hz is not None:
            if spec.bandwidth_hz >= f_nyq:
                raise ValueError(f"{spec.name}: bandwidth {spec.bandwidth_hz} Hz >= physics Nyquist {f_nyq} Hz")
            self.sos = signal.butter(spec.filter_order, spec.bandwidth_hz, fs=1.0 / physics_dt, output="sos")
        self.noise = np.broadcast_to(np.asarray(spec.noise_std, dtype=float), (spec.dim,)).copy()
        self.bias_std = np.broadcast_to(np.asarray(spec.bias_std, dtype=float), (spec.dim,)).copy()
        if spec.saturation is None:
            self.lo, self.hi = -np.inf, np.inf
        elif np.isscalar(spec.saturation):
            self.lo, self.hi = -float(spec.saturation), float(spec.saturation)
        else:
            self.lo, self.hi = float(spec.saturation[0]), float(spec.saturation[1])
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, t0: float = 0.0) -> None:
        d = self.spec.dim
        self.bias = self.rng.normal(0.0, 1.0, d) * self.bias_std
        self.zi = None
        self.block: list[np.ndarray] = []
        self.next_t = t0  # first sample at t0
        self.pending: deque[tuple[float, np.ndarray]] = deque()
        self.seq = 0
        self._latest = SensorSample(t0, t0, np.zeros(d), -1)
        self.t_sample: list[float] = []
        self.t_avail: list[float] = []
        self.values: list[np.ndarray] = []
        self.noise_scale = 1.0

    def _filter(self, block: np.ndarray) -> np.ndarray:
        if self.sos is None:
            return block
        if self.zi is None:
            # start the filter at steady state on the first value to avoid a startup transient
            zi0 = signal.sosfilt_zi(self.sos)  # (n_sections, 2)
            self.zi = zi0[:, :, None] * block[0][None, None, :]
        out, self.zi = signal.sosfilt(self.sos, block, axis=0, zi=self.zi)
        return out

    def step(self, t: float) -> SensorSample | None:
        """Call once per physics step, after mj_step, with the current sim time."""
        self.block.append(np.array(self.read_fn(), dtype=float).reshape(self.spec.dim))
        emitted = None
        if t >= self.next_t - 1e-9:
            blk = self._filter(np.stack(self.block))
            clean = blk.mean(axis=0) if self.spec.decimation == "mean" else blk[-1]
            self.block = []
            self.pending.append((t, clean))
            self.next_t += self.period
            while self.next_t <= t + 1e-9:  # physics slower than the sensor: skip missed slots
                self.next_t += self.period
        while self.pending and self.pending[0][0] + self.spec.latency_s <= t + 1e-9:
            ts, clean = self.pending.popleft()
            emitted = self._emit(ts, clean)
        return emitted

    def _emit(self, ts: float, clean: np.ndarray) -> SensorSample:
        v = clean + self.bias
        if np.any(self.noise > 0):
            v = v + self.rng.normal(0.0, 1.0, self.spec.dim) * self.noise * self.noise_scale
        q = self.spec.quant_step
        if q:
            v = np.round(v / q) * q
        v = np.clip(v, self.lo, self.hi)
        s = SensorSample(ts, ts + self.spec.latency_s, v, self.seq)
        self.seq += 1
        self._latest = s
        if self.keep_history:
            self.t_sample.append(s.t_sample)
            self.t_avail.append(s.t_avail)
            self.values.append(v)
        return s

    # ------------------------------------------------------------------
    def latest(self) -> SensorSample:
        """Most recent available sample (zero-order hold)."""
        return self._latest

    def window(self, n: int) -> np.ndarray:
        """Last n available samples, oldest first (fewer if not yet available)."""
        if not self.values:
            return np.zeros((0, self.spec.dim))
        return np.stack(self.values[-n:])

    def history(self) -> dict[str, np.ndarray]:
        d = self.spec.dim
        return {
            "t_sample": np.asarray(self.t_sample, dtype=float),
            "t_avail": np.asarray(self.t_avail, dtype=float),
            "value": np.stack(self.values) if self.values else np.zeros((0, d)),
            "seq": np.arange(len(self.values), dtype=np.int64),
        }


class SensorSuite:
    """A named collection of sensors stepped together."""

    def __init__(self, sensors: list[RateLimitedSensor] | None = None):
        self.sensors: dict[str, RateLimitedSensor] = {}
        for s in sensors or []:
            self.add(s)

    def add(self, s: RateLimitedSensor) -> None:
        if s.spec.name in self.sensors:
            raise ValueError(f"duplicate sensor {s.spec.name!r}")
        self.sensors[s.spec.name] = s

    def __getitem__(self, name: str) -> RateLimitedSensor:
        return self.sensors[name]

    def __contains__(self, name: str) -> bool:
        return name in self.sensors

    def names(self) -> list[str]:
        return list(self.sensors)

    def reset(self, t0: float = 0.0) -> None:
        for s in self.sensors.values():
            s.reset(t0)

    def step(self, t: float) -> None:
        for s in self.sensors.values():
            s.step(t)

    def latest(self, name: str) -> np.ndarray:
        return self.sensors[name].latest().value

    def set_noise_scale(self, k: float) -> None:
        for s in self.sensors.values():
            s.noise_scale = k
