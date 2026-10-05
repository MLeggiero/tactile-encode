"""Optional 30 Hz RGB cameras. Offscreen rendering needs EGL or OSMesa; without it cameras are disabled."""

from __future__ import annotations

import os
import warnings

import numpy as np

from tactile_sim.sensors.base import RateLimitedSensor, SensorSpec


def try_make_renderer(model, width: int = 128, height: int = 96):
    """Return a mujoco.Renderer or None (with a warning) when no GL backend is available."""
    try:
        import mujoco

        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        return mujoco.Renderer(model, height=height, width=width)
    except Exception as e:  # noqa: BLE001 - any GL failure means "no camera"
        warnings.warn(f"camera disabled: offscreen rendering unavailable ({type(e).__name__}: {e})",
                      RuntimeWarning, stacklevel=2)
        return None


def make_camera(world, cfg, rng, cam: str = "scene_cam", width: int = 128, height: int = 96):
    r = try_make_renderer(world.model, width, height)
    if r is None:
        return None
    d = world.data

    def read():
        r.update_scene(d, camera=cam)
        return r.render().astype(np.float32).reshape(-1)

    spec = SensorSpec(f"camera_{cam}", cfg.sensors.camera_rate, width * height * 3, latency_s=0.03,
                      decimation="zoh")
    s = RateLimitedSensor(spec, world.dt, read, rng)
    # rendering every physics step would be wasteful: only read when a frame is due
    s.step = _frame_step(s)
    return s


def _frame_step(s: RateLimitedSensor):
    base_read = s.read_fn
    last = {"frame": None}

    def step(t):
        due = (s.k % s.every == 0) if s.every is not None else (t >= s.next_t - 1e-9)
        if due or last["frame"] is None:
            last["frame"] = base_read()
        s.read_fn = lambda: last["frame"]
        return RateLimitedSensor.step(s, t)

    return step
