"""Pad pressure arrays (Xela uSkin / PapillArray class): 4 x 4 taxels per pad, normal force per taxel."""

from __future__ import annotations

import numpy as np

from tactile_sim import names
from tactile_sim.model.gripper import taxel_site
from tactile_sim.sensors.base import RateLimitedSensor, SensorSpec


def make_pressure(world, cfg, rng) -> list[RateLimitedSensor]:
    s = cfg.sensors
    sd = world.data.sensordata
    nr, nc = names.TAXEL_GRID
    out = []
    for side, tag in enumerate("LR"):
        idx = np.array([world.sensor_slices[taxel_site(side, r, c)].start for r in range(nr) for c in range(nc)])
        spec = SensorSpec(f"pressure_{tag}", s.pressure_rate, nr * nc, bandwidth_hz=s.pressure_bandwidth,
                          latency_s=s.pressure_latency, noise_std=s.pressure_noise, saturation=(0.0, s.pressure_range),
                          decimation="mean")
        out.append(RateLimitedSensor(spec, world.dt, (lambda idx=idx: sd[idx]), rng))
    return out


def grip_force(taxels: np.ndarray) -> float:
    return float(np.sum(taxels))


def center_of_pressure(taxels: np.ndarray, pad_half: tuple[float, float, float]) -> np.ndarray:
    """CoP on the pad face (x along the handle, z across), metres from the pad centre."""
    nr, nc = names.TAXEL_GRID
    f = np.asarray(taxels, dtype=float).reshape(nr, nc)
    tot = f.sum()
    if tot <= 1e-9:
        return np.zeros(2)
    hx, _, hz = pad_half
    xs = -hx + (2 * np.arange(nr) + 1) * hx / nr
    zs = -hz + (2 * np.arange(nc) + 1) * hz / nc
    return np.array([(f.sum(axis=1) * xs).sum() / tot, (f.sum(axis=0) * zs).sum() / tot])
