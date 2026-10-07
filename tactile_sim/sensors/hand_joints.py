"""Hand joint readback through the WUJI SDK (MIT-mode drives only): encoder angle and motor current per joint,
published once per SDK cycle and delayed by the feedback latency (WujiDriveCfg)."""

from __future__ import annotations

import numpy as np

from tactile_sim.sensors.base import RateLimitedSensor, SensorSpec


def make_hand_joint_sensors(world, cfg, rng) -> list[RateLimitedSensor]:
    hand = world.hand
    drive = getattr(hand, "drive", None)
    if drive is None:
        return []
    w = cfg.wuji
    d = world.data
    n = len(hand.qadr)
    return [
        RateLimitedSensor(SensorSpec("hand_q", w.sdk_rate, n, latency_s=w.fb_latency,
                                     quant_step=2.0 * np.pi / 2**w.enc_bits, decimation="zoh"),
                          world.dt, lambda: d.qpos[hand.qadr], rng),
        RateLimitedSensor(SensorSpec("hand_current", w.sdk_rate, n, latency_s=w.fb_latency, noise_std=w.i_noise,
                                     quant_step=w.i_quant, decimation="zoh"),
                          world.dt, lambda: drive.i, rng),
    ]
