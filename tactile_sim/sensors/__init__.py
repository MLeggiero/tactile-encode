"""Sensor models (rate, bandwidth, latency, noise, saturation) and the default testbed suite."""

from __future__ import annotations

import numpy as np

from tactile_sim.sensors.accel import make_accels
from tactile_sim.sensors.base import RateLimitedSensor, SensorSample, SensorSpec, SensorSuite
from tactile_sim.sensors.camera import make_camera
from tactile_sim.sensors.ft import make_ft
from tactile_sim.sensors.joints import Holder, make_joint_sensors
from tactile_sim.sensors.pressure import make_pressure


def build_sensor_suite(world, cfg=None, rng: np.random.Generator | None = None,
                       tau_ext: Holder | None = None) -> SensorSuite:
    cfg = cfg or world.cfg
    rng = rng or np.random.default_rng()
    suite = SensorSuite()
    suite.add(make_ft(world, cfg, rng))
    for s in make_accels(world, cfg, rng) + make_pressure(world, cfg, rng) + \
            make_joint_sensors(world, cfg, rng, tau_ext):
        suite.add(s)
    if cfg.sensors.cameras:
        cam = make_camera(world, cfg, rng)
        if cam is not None:
            suite.add(cam)
    return suite


__all__ = ["RateLimitedSensor", "SensorSample", "SensorSpec", "SensorSuite", "Holder", "build_sensor_suite"]
