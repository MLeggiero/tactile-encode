"""TaxelScan Rev3 readout: a piezoresistive taxel skin scanned by an RP2350's on-chip ADC, one frame per 1 ms.

Per board (one per patch by default, or one for the whole hand), every frame:

    force per taxel (physics rate) -> elastomer skin low-pass (first order, ts_bandwidth)
        -> sequential scan: taxel i is sampled at t_frame + offset + i * t_taxel,
           t_taxel = settle + oversample / adc_rate (linear interpolation between physics steps)
        -> divider: x = gain_i * F / (F + f_half)  (a force-sensitive resistor against a fixed resistor)
        -> 12-bit SAR ADC: counts = round(x * 4095 + offset_i + noise), noise from the ENOB, /sqrt(oversample)
        -> on-board calibration back to newtons with the nominal curve: F = f_half * x / (1 - x)
        -> frame sent when the scan ends; visible to the host ts_latency later (USB full-speed polling)

A sample's `t_sample` is the frame's start; each taxel was in fact sampled up to one scan length later. The
per-taxel gain and offset errors are what calibration and taring leave behind; they are drawn once per sensor,
like a physical skin's, and kept across resets.
"""

from __future__ import annotations

import numpy as np

from tactile_sim.sensors.base import RateLimitedSensor, SensorSample, SensorSpec


def taxel_time(s) -> float:
    """Time one taxel takes on a TaxelScan board (s): mux settling plus its conversions."""
    return s.ts_settle + s.ts_oversample / s.ts_adc_rate


class TaxelScanSensor(RateLimitedSensor):
    def __init__(self, name: str, n: int, physics_dt: float, read_fn, rng, s, offset: float = 0.0):
        spec = SensorSpec(name, s.ts_rate, n, latency_s=s.ts_latency, saturation=(0.0, np.inf), decimation="zoh")
        self.s = s
        self.t_tax = taxel_time(s)
        self.offset = offset
        if offset + n * self.t_tax > 1.0 / s.ts_rate + 1e-12:
            raise ValueError(f"{name}: scanning {n} taxels takes {1e3 * (offset + n * self.t_tax):.3f} ms, longer "
                             f"than the {1e3 / s.ts_rate:.3f} ms frame")
        self.a = 1.0 - np.exp(-2 * np.pi * s.ts_bandwidth * physics_dt)
        self.full = 2**s.ts_adc_bits - 1
        rng = rng or np.random.default_rng()
        self.gain = 1.0 + s.ts_gain_mismatch * rng.normal(size=n)
        self.offs = s.ts_offset_lsb * rng.normal(size=n)
        self.noise_lsb = 2.0 ** (s.ts_adc_bits - s.ts_enob) / np.sqrt(12.0) / np.sqrt(s.ts_oversample)
        # an unloaded taxel reads below this (N): three sigma of residual offset plus ADC noise near zero force
        self.floor = s.ts_f_half * 3.0 * np.hypot(s.ts_offset_lsb, self.noise_lsb) / self.full
        super().__init__(spec, physics_dt, read_fn, rng)

    def reset(self, t0: float = 0.0) -> None:
        super().reset(t0)
        self.xf = None
        self.prev = None  # (t, filtered force) at the previous physics step
        self.frame_t = t0 + self.offset
        self.idx = 0
        self.buf = np.zeros(self.spec.dim)
        self.frames: list[tuple[float, float, np.ndarray]] = []  # (t_frame, t_done, force)
        self.counts = np.zeros(self.spec.dim, dtype=int)

    def step(self, t: float) -> SensorSample | None:
        raw = np.asarray(self.read_fn(), dtype=float).reshape(self.spec.dim)
        self.xf = raw.copy() if self.xf is None else self.xf + self.a * (raw - self.xf)
        tp, xp = self.prev if self.prev is not None else (t - self.dt, self.xf)
        n = self.spec.dim
        while True:
            # taxels whose sampling instant falls in (tp, t]
            hi = min(n, int(np.floor((t - self.frame_t) / self.t_tax + 1e-9)) + 1)
            if hi > self.idx:
                i = np.arange(self.idx, hi)
                al = np.clip((self.frame_t + i * self.t_tax - tp) / (t - tp), 0.0, 1.0)
                self.buf[i] = xp[i] + al * (self.xf[i] - xp[i])
                self.idx = hi
            if self.idx < n:
                break
            self.frames.append((self.frame_t - self.offset, self.frame_t + n * self.t_tax, self.buf.copy()))
            self.frame_t += self.period
            self.idx = 0
        self.prev = (t, self.xf)
        emitted = None
        while self.frames and self.frames[0][1] + self.spec.latency_s <= t + 0.5 * self.dt:
            t0, t_done, f = self.frames.pop(0)
            emitted = self._emit_frame(t0, t_done, f)
        return emitted

    def adc(self, f: np.ndarray) -> np.ndarray:
        """Counts the board reads for taxel forces `f` (N)."""
        f = np.maximum(f, 0.0)
        x = self.gain * f / (f + self.s.ts_f_half)
        c = x * self.full + self.offs + self.rng.normal(size=f.shape) * self.noise_lsb * self.noise_scale
        return np.clip(np.round(c), 0, self.full).astype(int)

    def calibrate(self, counts: np.ndarray) -> np.ndarray:
        """Board firmware: counts -> N with the nominal divider curve."""
        x = np.minimum(counts / self.full, 0.999)
        return self.s.ts_f_half * x / (1.0 - x)

    def _emit_frame(self, t0: float, t_done: float, f: np.ndarray) -> SensorSample:
        self.counts = self.adc(f)
        v = self.calibrate(self.counts)
        s = SensorSample(t0, t_done + self.spec.latency_s, v, self.seq)
        self.seq += 1
        self._latest = s
        if self.keep_history:
            self.t_sample.append(s.t_sample)
            self.t_avail.append(s.t_avail)
            self.values.append(v)
        return s


def make_taxelscan(world, cfg, rng) -> list[TaxelScanSensor]:
    """One sensor per conforming patch. `ts_boards="hand"`: one board scans the patches back to back, so each
    patch's scan starts where the previous one's ended."""
    s = cfg.sensors
    hand = world.hand
    out, off = [], 0.0
    for k, name in enumerate(hand.pressure_names):
        n = int(np.prod(hand.grid(k)))
        out.append(TaxelScanSensor(name, n, world.dt, (lambda k=k: world.pad_taxels(k)), rng, s,
                                   offset=off if s.ts_boards == "hand" else 0.0))
        off += n * taxel_time(s)
    if s.ts_boards not in ("hand", "per_patch"):
        raise ValueError(f"unknown ts_boards {s.ts_boards!r}")
    return out
