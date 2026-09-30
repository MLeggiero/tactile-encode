"""Run a Task A episode and write the force-truth log.

    python -m tactile_sim.run_strikes --n 10 --out runs/demo.h5 --seed 0
    python -m tactile_sim.run_strikes --n 10 --fast --flex-mode rigid --out runs/rigid.h5
    python -m tactile_sim.run_strikes --n 10 --dr --seed 3 --out runs/dr3.h5
    python -m tactile_sim.run_strikes --n 10 --hand wuji2 --out runs/wuji.h5
    python -m tactile_sim.run_strikes --n 10 --hand wuji2 --self-locking --out runs/wuji_lock.h5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tactile_sim.config import SimConfig, fast_config, hand_config
from tactile_sim.episode import Episode
from tactile_sim.logging.writer import write_episode


def build_config(args) -> SimConfig:
    cfg = hand_config(args.hand, fast_config() if args.fast else SimConfig())
    over: dict = {"arm": {"source": args.arm}}
    if args.self_locking:
        over["gripper"] = {"lock_mode": "self_locking"}
    if args.dt:
        over["physics"] = {"timestep": args.dt}
    if args.flex_mode:
        over["arm"]["flex_mode"] = args.flex_mode
    if args.v_strike:
        over["swing"] = {"v_strike": args.v_strike}
    if args.dr:
        over["dr"] = {"enabled": True}
    return cfg.replace(**over)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="number of strikes")
    ap.add_argument("--out", type=Path, default=Path("runs/demo.h5"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=None, help="physics timestep (s)")
    ap.add_argument("--fast", action="store_true", help="4 kHz physics (test settings)")
    ap.add_argument("--arm", choices=["auto", "fr3", "fallback"], default="auto")
    ap.add_argument("--flex-mode", choices=["wrist", "rigid"], default=None)
    ap.add_argument("--v-strike", type=float, default=None, help="strike speed (m/s)")
    ap.add_argument("--dr", action="store_true", help="enable domain randomization")
    ap.add_argument("--hand", choices=["franka", "wuji2"], default="franka")
    ap.add_argument("--self-locking", action="store_true", help="WUJI: joints that do not backdrive")
    args = ap.parse_args(argv)
    cfg = build_config(args)
    t0 = time.time()
    ep = Episode(cfg, seed=args.seed)
    res = ep.run(args.n)
    path = write_episode(res, args.out, cfg, args.seed)
    s = res.summary
    print(f"{len(res.strikes)} strikes in {ep.tb.t:.2f} s sim ({time.time() - t0:.1f} s wall), "
          f"arm={ep.tb.world.arm_source}, dt={ep.tb.world.dt * 1e3:.3f} ms")
    for r in res.strikes:
        lat = f"{r.flag_latency * 1e3:.2f} ms ({r.flag_source})" if np.isfinite(r.flag_latency) else "none"
        print(f"  #{r.idx + 1:2d} {'hit ' if r.hit else 'MISS'} v={r.v_strike_actual:4.2f} m/s "
              f"peak={r.peak_force_truth:6.0f} N width={r.pulse_width * 1e3:4.1f} ms "
              f"advance={r.depth_inc * 1e3:5.2f} mm depth={r.depth_after * 1e3:5.2f} mm "
              f"slip={r.slip_trans * 1e3:4.2f} mm/{np.degrees(r.slip_rot):4.2f} deg flag={lat}")
    stt = s.get("strikes_to_target")
    print(f"total {s.get('total_depth', 0) * 1e3:.1f} mm, hit rate {s.get('hit_rate', 0):.2f}, "
          f"strikes to {cfg.plant.drive_target * 1e3:.0f} mm: {'not reached' if not np.isfinite(stt) else int(stt)}, "
          f"drops {s.get('drops', 0)}")
    lim = {k[6:]: v for k, v in s.items() if k.startswith("limit_") and isinstance(v, float)}
    print("hardware limits (worst ratio, 1 = at the limit): " +
          ", ".join(f"{k} {v:.2f} ({ep.limits.where.get(k, '')})".replace(" ()", "") for k, v in lim.items()))
    print(f"limits exceeded: {s['limit_violations'] or 'none'}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
