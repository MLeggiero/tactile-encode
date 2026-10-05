"""Wrist 6-axis force/torque sensor (Bota SensONE / ATI Net F/T class), in the sensor frame."""

from __future__ import annotations

import numpy as np

from tactile_sim.sensors.base import RateLimitedSensor, SensorSpec


def make_ft(world, cfg, rng) -> RateLimitedSensor:
    s = cfg.sensors
    fs, ts = world.sensor_slices["ft_force"], world.sensor_slices["ft_torque"]
    sd = world.data.sensordata

    def read():
        return np.concatenate([sd[fs], sd[ts]])

    spec = SensorSpec("ft", s.ft_rate, 6, bandwidth_hz=s.ft_bandwidth, latency_s=s.ft_latency,
                      noise_std=(s.ft_noise_f,) * 3 + (s.ft_noise_t,) * 3, bias_std=(0.5,) * 3 + (0.02,) * 3,
                      saturation=None, decimation="mean")
    sensor = RateLimitedSensor(spec, world.dt, read, rng)
    lim = np.array((s.ft_range_f,) * 3 + (s.ft_range_t,) * 3)
    sensor.lo, sensor.hi = -lim, lim
    return sensor
