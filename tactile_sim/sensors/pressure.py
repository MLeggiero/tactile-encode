"""Pressure arrays: an 8 x 8 grid (64 taxels) per taxel patch at 1 kHz by default, normal force per taxel.
The Franka Hand has one patch per pad (L, R); a dexterous hand has the patches of its layout. TaxelScan skins
(conforming patches) are read by tactile_sim.sensors.taxelscan instead."""

from __future__ import annotations

import numpy as np

from tactile_sim.sensors.base import RateLimitedSensor, SensorSpec


def make_pressure(world, cfg, rng) -> list[RateLimitedSensor]:
    s = cfg.sensors
    if any(p.taxel_pos is not None for p in world.hand.patches):
        from tactile_sim.sensors.taxelscan import make_taxelscan

        return make_taxelscan(world, cfg, rng)
    out = []
    for side, name in enumerate(world.hand.pressure_names):
        nr, nc = world.hand.grid(side)
        spec = SensorSpec(name, s.pressure_rate, nr * nc, bandwidth_hz=s.pressure_bandwidth,
                          latency_s=s.pressure_latency, noise_std=s.pressure_noise, saturation=(0.0, s.pressure_range),
                          decimation="mean")
        out.append(RateLimitedSensor(spec, world.dt, (lambda side=side: world.pad_taxels(side)), rng))
    return out


def grip_force(taxels: np.ndarray, floor: float = 0.0) -> float:
    """Summed normal force. Taxels at or below `floor` are treated as unloaded: noise on an unloaded
    taxel is clipped at zero, so summing it raw half-rectifies the noise into a positive offset."""
    t = np.asarray(taxels, dtype=float)
    return float(np.sum(t[t > floor]))


def center_of_pressure(taxels: np.ndarray, pad_half: tuple[float, float, float],
                       grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CoP on the pad face (x along the handle, z across), metres from the pad centre."""
    nr, nc = grid
    f = np.asarray(taxels, dtype=float).reshape(nr, nc)
    tot = f.sum()
    if tot <= 1e-9:
        return np.zeros(2)
    hx, _, hz = pad_half
    xs = -hx + (2 * np.arange(nr) + 1) * hx / nr
    zs = -hz + (2 * np.arange(nc) + 1) * hz / nc
    return np.array([(f.sum(axis=1) * xs).sum() / tot, (f.sum(axis=0) * zs).sum() / tot])
