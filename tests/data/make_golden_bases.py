"""Record the golden traces of the default (unplanned) hammer and drill bases: the arm's 100 Hz targets and the task's
progress over a short fast-physics episode, built exactly as vla.demos builds a scripted demo. tests/test_motion_plan.py
compares a fresh run against them, so the planned-motion code paths provably leave the defaults unchanged.

    MUJOCO_GL=glfw python tests/data/make_golden_bases.py
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("golden_bases.json")
SEED = 100_003
HAMMER_STRIKES = 2
DRILL_T = 1.2  # s: approach, press and the trigger pull


def run_base(task: str) -> dict:
    """The task's scripted episode for SEED on the fast physics, with the residual's base attached (vla.demos)."""
    from tactile_sim.vla.demos import EPISODES, demo_config

    cfg, draw = demo_config(task, SEED, fast=True)
    tb, base, _ = EPISODES[task](cfg, SEED)
    tb.sensors.set_noise_scale(draw["dr"]["_runtime"]["sensor_noise_scale"])
    tb.residual.base_fn = base
    if task == "hammer":
        _hammer_run(tb, base)()
        progress = float(tb.world.plant.depth)
    else:
        n = int(round(DRILL_T / tb.world.dt))
        tb.step(n)
        progress = float(tb.world.plant.progress)
    log = tb.l1.log
    return {"t": np.asarray(log.t).tolist(), "x_eq": np.asarray(log.x_eq).tolist(),
            "q": np.asarray(log.q).tolist(), "progress": progress}


def _hammer_run(tb, base):
    swing = base.swing

    def run():
        for _ in range(int(round(6.0 / tb.world.dt))):
            tb.step()
            if swing.k >= HAMMER_STRIKES or swing.done:
                break

    return run


def fingerprint() -> dict:
    import mujoco

    return {"platform": platform.platform(), "machine": platform.machine(), "numpy": np.__version__,
            "mujoco": mujoco.__version__}


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    out = {"fingerprint": fingerprint(), "seed": SEED, "hammer": run_base("hammer"), "drill": run_base("drill")}
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT}: hammer {len(out['hammer']['t'])} ticks (depth {out['hammer']['progress']:.4f} m), "
          f"drill {len(out['drill']['t'])} ticks (progress {out['drill']['progress']:.3f})")


if __name__ == "__main__":
    main()
