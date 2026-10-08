"""pi0.5 norm stats for a Vega U + WUJI LeRobot dataset from its parquet files alone, with no video decoding. It
computes what openpi's scripts/compute_norm_stats.py does for the pi05_vega_wuji config: the state, and the 16-step
action chunks padded with the episode's last frame, with the arm's 7 as deltas from the frame's state. Then it widens
the ranges as openpi_vega_wuji.widen_norm_stats does (repeated here so this runs without openpi).

On dataset v1 it matches openpi's output to within 0.4 % of each dimension's q01-q99 range (mean and quantiles). The
state std differs where openpi's float32 sums lose precision on near-constant joints; pi0.5 normalizes by quantiles.
openpi's own pass decodes every video frame, which took 11-17 s per batch on Skynet's shared storage (a day for
dataset v2); this takes seconds.

    python openpi_norm_stats.py <dataset> <norm_stats.json> [--compare <openpi's norm_stats.json>]"""
import glob
import json
import sys

import numpy as np
import pandas as pd

HORIZON, N_ARM, MIN_WIDTH = 16, 7, 0.02
ds, out = sys.argv[1], sys.argv[2]
states, chunks = [], []
for f in sorted(glob.glob(f"{ds}/data/chunk-*/episode_*.parquet")):
    d = pd.read_parquet(f, columns=["observation.state", "action"])
    s, a = np.stack(d["observation.state"]).astype(np.float64), np.stack(d["action"]).astype(np.float64)
    idx = np.minimum(np.arange(len(a))[:, None] + np.arange(HORIZON)[None, :], len(a) - 1)
    c = a[idx]  # (T, H, 27)
    c[..., :N_ARM] -= s[:, None, :N_ARM]
    states.append(s)
    chunks.append(c.reshape(-1, c.shape[-1]))


def stats(x):
    lo, hi = np.quantile(x, 0.01, axis=0), np.quantile(x, 0.99, axis=0)
    mid, half = 0.5 * (lo + hi), 0.5 * np.maximum(hi - lo, MIN_WIDTH)
    return {"mean": x.mean(0).tolist(), "std": np.maximum(x.std(0), 0.25 * MIN_WIDTH).tolist(),
            "q01": (mid - half).tolist(), "q99": (mid + half).tolist()}


res = {"norm_stats": {"state": stats(np.concatenate(states)), "actions": stats(np.concatenate(chunks))}}
json.dump(res, open(out, "w"), indent=2)
print(f"{sum(len(s) for s in states)} frames -> {out}")
if "--compare" in sys.argv:
    ref = json.load(open(sys.argv[sys.argv.index("--compare") + 1]))["norm_stats"]
    for key in ("state", "actions"):
        for st in ("mean", "std", "q01", "q99"):
            a, b = np.asarray(res["norm_stats"][key][st]), np.asarray(ref[key][st])
            rng = np.asarray(ref[key]["q99"]) - np.asarray(ref[key]["q01"])
            err = np.abs(a - b) / np.maximum(rng, 1e-6)
            print(f"  {key:7s} {st:4s}: max |diff| {np.abs(a - b).max():.2e}, "
                  f"max relative to the q01-q99 range {err.max():.2e}")
