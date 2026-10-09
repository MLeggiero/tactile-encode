"""E6's first number: GPU-trained policies in the CPU testbed against the GPU env, per hold level (milestone 5).

The CPU side is residual_experiment's E1hold:
- the 8 kHz validator with noslip 5, the student as an ONNX graph in ResidualHandL2;
- whole episodes of up to 10 strikes, with the base at the hold level throughout;
- a drop ends the episode.

The GPU side is residual_eval's hammer runs, scored in the testbed's strike window:
- each env plays one library swing;
- it starts from the CPU's full-hold state at the swing's windup, with the base at the hold level from there.

Per policy and hold level, on both sides:
- lost grasps, per strike and on the strikes that open an episode;
- net slip and slip path per kept strike;
- I^2 t from the swing's start to the strike's end;
- the nail's advance.

Then each sim's own effect: the policy against that sim's zero residual, at the same hold and at the full hold.
The transfer ratio is the CPU's effect over the GPU's; 1 means the policy does in the testbed what it did where it
was trained.

    python -m tactile_rl.bench.transfer --cpu-root RESULTS --gpu eval_1k.json eval_250.json [--out transfer.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

KEYS = ("lost", "lost_first", "net_mm", "net_deg", "path_mm", "path_deg", "i2t", "advance_mm")
CPU_KEYS = dict(zip(KEYS, ("lost", "lost_first", "net_mm", "net_deg", "path_mm", "path_deg", "i2t_w", "advance_mm"),
                    strict=True))
GPU_KEYS = dict(zip(KEYS, ("lost_sw", "lost_first", "net_sw_mm", "net_sw_deg", "path_sw_mm", "path_sw_deg", "i2t_sw",
                           "advance_sw_mm"), strict=True))


def pair(cpu_rows: list[dict], gpu_rows: list[dict], zero: str = "zero@1000") -> dict[tuple[str, float], dict]:
    """(policy, hold) -> {"cpu": metrics, "gpu": metrics} in KEYS. The CPU's base is the GPU's `zero`; a CPU policy
    "residual:NAME" is the GPU's NAME."""
    out: dict[tuple[str, float], dict] = {}
    for r in cpu_rows:
        name = "zero" if r["policy"] == "base" else r["policy"].removeprefix("residual:")
        out.setdefault((name, float(r["hold"])), {})["cpu"] = {k: r.get(v, math.nan) for k, v in CPU_KEYS.items()}
    for r in gpu_rows:
        if r.get("task") != "hammer" or "net_sw_mm" not in r:
            continue
        name = "zero" if r["policy"] == zero else r["policy"]
        if name.startswith("zero@"):
            continue  # another rate's zero residual
        out.setdefault((name, float(r["hold"])), {})["gpu"] = {k: r.get(v, math.nan) for k, v in GPU_KEYS.items()}
    return out


def _ratio(a: float, b: float) -> float:
    return a / b if b and math.isfinite(a) and math.isfinite(b) else math.nan


def effects(pairs: dict[tuple[str, float], dict]) -> list[dict]:
    """Per policy and hold (both sims present): each sim's ratios to its own zero residual at the same hold and at
    the full hold, and the transfer ratio (CPU over GPU) of each."""
    rows = []
    for (name, hold), p in sorted(pairs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if name == "zero" or "cpu" not in p or "gpu" not in p:
            continue
        r = {"policy": name, "hold": hold}
        for ref_name, ref_hold in (("same", hold), ("full", 1.0)):
            z = pairs.get(("zero", ref_hold), {})
            if "cpu" not in z or "gpu" not in z:
                continue
            for k in ("net_mm", "net_deg", "path_mm", "path_deg", "i2t", "advance_mm"):
                c, g = _ratio(p["cpu"][k], z["cpu"][k]), _ratio(p["gpu"][k], z["gpu"][k])
                r[f"{k}_{ref_name}_cpu"], r[f"{k}_{ref_name}_gpu"] = c, g
                r[f"{k}_{ref_name}_transfer"] = _ratio(c, g)
        rows.append(r)
    return rows


def tables(pairs: dict[tuple[str, float], dict], eff: list[dict]) -> str:
    def cell(p, k, fmt, scale=1.0):
        c, g = (p.get(s, {}).get(k, math.nan) * scale for s in ("cpu", "gpu"))
        return f"{c:{fmt}} / {g:{fmt}}"

    out = ["| policy | hold | lost per strike | first strikes lost | net slip mm | net slip deg | slip path mm | "
           "slip path deg | I2t (swing to end) | advance mm |", "|---" * 10 + "|"]
    for (name, hold), p in sorted(pairs.items(), key=lambda kv: (kv[0][1], kv[0][0] != "zero", kv[0][0])):
        out.append(f"| {name} | {hold:g} | {cell(p, 'lost', '.1f', 100)} % | {cell(p, 'lost_first', '.0f', 100)} % | "
                   f"{cell(p, 'net_mm', '.3f')} | {cell(p, 'net_deg', '.3f')} | {cell(p, 'path_mm', '.2f')} | "
                   f"{cell(p, 'path_deg', '.2f')} | {cell(p, 'i2t', '.3f')} | {cell(p, 'advance_mm', '.2f')} |")
    out += ["", "each sim against its own zero residual (CPU / GPU, then the transfer ratio CPU over GPU)", "",
            "| policy | hold | net slip mm vs zero, same hold | net slip deg vs zero, same hold | net slip mm vs zero, "
            "full hold | net slip deg vs zero, full hold | slip path deg vs zero, full hold | I2t vs zero, full hold |",
            "|---" * 8 + "|"]
    for r in eff:
        cells = []
        for k, ref in (("net_mm", "same"), ("net_deg", "same"), ("net_mm", "full"), ("net_deg", "full"),
                       ("path_deg", "full"), ("i2t", "full")):
            c, g, t = (r.get(f"{k}_{ref}_{s}", math.nan) for s in ("cpu", "gpu", "transfer"))
            cells.append(f"{c:.2f} / {g:.2f} ({t:.2f})")
        out.append(f"| {r['policy']} | {r['hold']:g} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - report script
    from tactile_sim.experiments.residual_experiment import hold_table

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-root", required=True, help="residual_experiment's results root (with E1hold/)")
    ap.add_argument("--gpu", nargs="+", required=True, help="residual_eval's --out files")
    ap.add_argument("--zero", default="zero@1000", help="the GPU's zero residual standing for the CPU's base")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    _, cpu_rows = hold_table("E1hold", Path(a.cpu_root))
    gpu_rows = [r for f in a.gpu for r in json.loads(Path(f).read_text())]
    pairs = pair(cpu_rows, gpu_rows, a.zero)
    eff = effects(pairs)
    print(tables(pairs, eff))
    if a.out:
        Path(a.out).write_text(json.dumps({"pairs": [{"policy": n, "hold": h, **p} for (n, h), p in pairs.items()],
                                           "effects": eff}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
