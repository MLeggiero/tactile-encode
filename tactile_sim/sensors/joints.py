"""Arm joint encoders, velocities and link-side torques at 1 kHz, plus the observer's external torque."""

from __future__ import annotations

import numpy as np

from tactile_sim.sensors.base import RateLimitedSensor, SensorSpec


class Holder:
    """A value another component writes (e.g. the momentum observer); a sensor then samples it."""

    def __init__(self, dim: int):
        self.value = np.zeros(dim)


def make_joint_sensors(world, cfg, rng, tau_ext: Holder | None = None) -> list[RateLimitedSensor]:
    s = cfg.sensors
    d = world.data
    qa, dofs = world.arm_qadr, world.arm_dofs
    out = [
        RateLimitedSensor(SensorSpec("joint_pos", s.joint_rate, 7, latency_s=s.joint_latency,
                                     quant_step=s.joint_pos_quant, decimation="zoh"),
                          world.dt, lambda: d.qpos[qa], rng),
        RateLimitedSensor(SensorSpec("joint_vel", s.joint_rate, 7, latency_s=s.joint_latency,
                                     noise_std=s.joint_vel_noise, decimation="zoh"),
                          world.dt, lambda: d.qvel[dofs], rng),
        # link-side torque: motor torque minus what the joint's own damping/friction absorbs
        RateLimitedSensor(SensorSpec("joint_tau", s.joint_rate, 7, latency_s=s.joint_latency,
                                     noise_std=s.joint_tau_noise, bias_std=0.05, decimation="mean"),
                          world.dt, lambda: d.qfrc_actuator[dofs] + d.qfrc_passive[dofs], rng),
    ]
    if tau_ext is not None:
        out.append(RateLimitedSensor(SensorSpec("tau_ext", s.tau_ext_rate, 7, decimation="zoh"),
                                     world.dt, lambda: tau_ext.value, rng))
    return out
