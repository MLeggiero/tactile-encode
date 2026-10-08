"""The slow policy (L3) in simulation: the Vega U head camera, the VLA's state and action layout, scripted demos
recorded as LeRobot datasets, and the closed-loop client that runs a fine-tuned VLA in the CPU testbed.

Offscreen rendering needs EGL (or OSMesa), which MuJoCo picks when it is first imported; importing this package
first selects EGL unless MUJOCO_GL is already set."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
