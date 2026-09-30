"""3-axis MEMS accelerometers: one under each taxel patch (8 kHz, +-16 g) and one in the hammer head."""

from __future__ import annotations

from tactile_sim.sensors.base import RateLimitedSensor, SensorSpec


def make_accels(world, cfg, rng) -> list[RateLimitedSensor]:
    s = cfg.sensors
    sd = world.data.sensordata
    out = []
    for name in [*world.hand.accel_names, "hammer_acc"]:
        mj = name
        sl = world.sensor_slices[mj]
        # the hammer-head IMU is a force-truth rig instrument: wide range, same bandwidth
        rng_lim = s.accel_range if name != "hammer_acc" else 200 * 9.80665
        spec = SensorSpec(name, s.accel_rate, 3, bandwidth_hz=s.accel_bandwidth, latency_s=s.accel_latency,
                          noise_std=s.accel_noise, bias_std=0.05, saturation=rng_lim, decimation="mean")
        out.append(RateLimitedSensor(spec, world.dt, (lambda sl=sl: sd[sl]), rng))
    return out
