"""The hammer's base at reduced grip: every base-library swing replayed on the GPU (unperturbed, its episode's
randomization, zero residual) with the hand base at a hold level (a fraction of the library's grip): hits, nail
advance, slip in the hand per strike (swing start to the entry's end) and lost grasps (the tool 2 cm or 30 degrees from
its start in the hand at any point).

    python -m tactile_rl.bench.hammer_hold --lib runs/base_library/hammer_v2.npz [--holds 0.3,0.5,0.8,1.0] \
        [--entries 0:400] [--out hold.json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def sweep(library: str, entries: list[int], holds) -> dict[float, dict]:
    from tactile_rl.parity.strikes import gpu_replay

    out = {}
    for h in holds:
        g = gpu_replay(library, entries, hold_level=float(h))
        rows = g["rows"]
        lost = np.array([r["lost"] for r in rows], bool)
        hit = np.array([r["hit"] for r in rows], bool)
        keep = ~lost
        st = np.array([r["slip_trans"] for r in rows], float) * 1e3
        sr = np.degrees(np.array([r["slip_rot"] for r in rows], float))
        adv = np.array([r["advance"] for r in rows], float) * 1e3
        out[float(h)] = {"strikes": len(rows), "hit": float(hit.mean()), "lost": float(lost.mean()),
                         "slip_mm": float(st[keep].mean()) if keep.any() else float("nan"),
                         "slip_deg": float(sr[keep].mean()) if keep.any() else float("nan"),
                         "slip_mm_p90": float(np.percentile(st[keep], 90)) if keep.any() else float("nan"),
                         "slip_deg_p90": float(np.percentile(sr[keep], 90)) if keep.any() else float("nan"),
                         "advance_mm": float(adv[hit & keep].mean()) if (hit & keep).any() else float("nan")}
        print(h, out[float(h)], flush=True)
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - report script
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--holds", default="0.3,0.5,0.8,1.0")
    ap.add_argument("--entries", default=None, help="a:b range (default: every entry)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from tactile_sim.behaviors.base_library import load_library

    n = len(load_library(a.lib)["e_t0"])
    lo, hi = (int(x) for x in a.entries.split(":")) if a.entries else (0, n)
    res = sweep(a.lib, list(range(lo, min(hi, n))), [float(x) for x in a.holds.split(",")])
    print("| hold level | strikes | hits | lost grasps | slip per strike (mm / deg) | p90 (mm / deg) | advance (mm) |")
    print("|---|---|---|---|---|---|---|")
    for h, r in res.items():
        print(f"| {h:g} | {r['strikes']} | {100 * r['hit']:.1f} % | {100 * r['lost']:.1f} % | {r['slip_mm']:.2f} / "
              f"{r['slip_deg']:.2f} | {r['slip_mm_p90']:.2f} / {r['slip_deg_p90']:.2f} | {r['advance_mm']:.2f} |")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({str(k): v for k, v in res.items()}, fh, indent=1)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
