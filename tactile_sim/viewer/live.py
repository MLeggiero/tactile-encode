"""Watch an episode live in MuJoCo's native viewer (needs a display; not usable in a headless container).

    python -m tactile_sim.viewer.live --n 5 --slowdown 20

The viewer shows the full-resolution Menagerie FR3 and Franka Hand meshes. `--slowdown` stretches wall time
relative to simulated time (20 = a 5 ms blow takes 0.1 s on screen). Close the window to stop.
"""

from __future__ import annotations

import argparse
import time

from tactile_sim.config import SimConfig, fast_config
from tactile_sim.episode import Episode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="4 kHz physics")
    ap.add_argument("--slowdown", type=float, default=10.0, help="wall seconds per simulated second")
    args = ap.parse_args(argv)
    import mujoco.viewer

    cfg = fast_config() if args.fast else SimConfig()
    ep = Episode(cfg, seed=args.seed)
    ep.swing.n_strikes = args.n
    tb = ep.tb
    sync_every = max(1, int(round(1.0 / 60.0 / args.slowdown / tb.world.dt)))  # ~60 fps on screen
    with mujoco.viewer.launch_passive(tb.world.model, tb.world.data) as v:
        v.cam.lookat[:] = tb.world.hover_tcp
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.1, 135.0, -20.0
        k = 0
        t_wall0, t_sim0 = time.perf_counter(), tb.t
        while v.is_running() and not ep.swing.done:
            tb.step()
            ep._step_cb(tb)  # noqa: SLF001 - keep the strike records
            k += 1
            if k % sync_every == 0:
                v.sync()
                lag = (tb.t - t_sim0) * args.slowdown - (time.perf_counter() - t_wall0)
                if lag > 0:
                    time.sleep(lag)
    for r in ep.records:
        print(f"strike {r.idx + 1}: {'hit' if r.hit else 'miss'}, advance {r.depth_inc * 1e3:.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
