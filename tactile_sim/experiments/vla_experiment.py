"""E2: the VLA alone vs the VLA with the residual, in closed loop in the CPU testbed.

The VLA is served (GR00T: gr00t/eval/run_gr00t_server.py over ZeroMQ; pi0.5: openpi's scripts/serve_policy.py over a
websocket). Every episode connects to it, draws its seed's physics randomization and task shift exactly as the demos
do (vla.demos.demo_config), and runs the task with the VLA driving the arm and giving the residual its base
(vla.client.run_vla_episode). Conditions share each seed's draws: "vla" (the residual at zero) and one
"vla+residual:<name>" per exported policy. One JSON per episode, and the VLA's chunks beside it (.npz: the material
for residual bases from VLA runs).

    python -m tactile_sim.experiments.vla_experiment run --task hammer --server groot://localhost:5555 \\
        --seeds 100 --workers 6 [--policy NAME=PATH ...] --out runs/e2/groot_hammer
    python -m tactile_sim.experiments.vla_experiment report --task hammer --out runs/e2/groot_hammer
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tactile_sim.experiments.residual_experiment import paired_bootstrap

MAX_TIME = {"hammer": 40.0, "drill": 12.0}  # s per episode
METRICS = {"hammer": ("done", "strikes", "advance_mm", "slip_rot_deg", "drops", "t_done"),
           "drill": ("done", "cam_outs", "slip_rot_deg", "slip_trans_mm", "drops", "t_done", "aim_mm", "trigger")}


@dataclass(frozen=True)
class Condition:
    name: str
    policy: str = "zero"  # the residual: "zero" or an exported policy (.onnx with its sidecar)
    overrides: dict = field(default_factory=dict)


def conditions(policies: dict[str, str] | None = None) -> list[Condition]:
    out = [Condition("vla")]
    for name, path in (policies or {}).items():
        from tactile_sim.control.policy_io import PolicyMeta

        rate = PolicyMeta.read(path).rate  # the residual runs at its own rate
        out.append(Condition(f"vla+residual:{name}", path, {"residual": {"rate": rate}}))
    return out


def make_vla_policy(server: str):
    """A client for "groot://host:port" or "openpi://host:port"."""
    from tactile_sim.vla.client import GrootPolicy, OpenPiPolicy

    kind, _, addr = server.partition("://")
    host, _, port = addr.rpartition(":")
    if kind == "groot":
        return GrootPolicy(host or "localhost", int(port or 5555))
    if kind == "openpi":
        return OpenPiPolicy(host or "localhost", int(port or 8000))
    raise ValueError(f"unknown server {server!r} (expected groot://host:port or openpi://host:port)")


@dataclass(frozen=True)
class Job:
    task: str
    condition: Condition
    seed: int
    server: str
    latency: float = 0.1
    exec_steps: int = 8
    blend: float = 0.0
    smooth: int = 0
    hand_tables: str | None = None  # a LeRobot dataset (or its sidecar): the hand tables it was recorded under

    def path(self, root: Path) -> Path:
        safe = self.condition.name.replace(":", "_").replace("+", "__")
        return root / safe / f"seed{self.seed}.json"


def run_job(job: Job, root: str) -> dict:
    from tactile_sim.control.residual_l2 import make_policy
    from tactile_sim.vla.client import run_vla_episode
    from tactile_sim.vla.demos import demo_config

    out = job.path(Path(root))
    if out.exists():
        return {"status": "skip", "path": str(out)}
    t0 = time.time()
    cfg, draw = demo_config(job.task, job.seed)
    if job.condition.overrides:
        cfg = cfg.replace(**job.condition.overrides)
    residual = make_policy(job.condition.policy)
    r = run_vla_episode(cfg, make_vla_policy(job.server), job.seed, latency=job.latency, exec_steps=job.exec_steps,
                        max_time=MAX_TIME[job.task], residual_policy=residual, blend=job.blend,
                        smooth=job.smooth, table=dataset_table(job.hand_tables, job.task))
    chunks = r.pop("chunks")
    payload = {"job": {"task": job.task, "condition": job.condition.name, "policy": job.condition.policy,
                       "seed": job.seed, "server": job.server, "latency": job.latency, "exec_steps": job.exec_steps,
                       "blend": job.blend, "smooth": job.smooth, "hand_tables": job.hand_tables},
               **r, "draw": draw, "wall_s": time.time() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".chunks.npz"), **chunks)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, default=_jsonable))
    tmp.rename(out)
    return {"status": "ok", "path": str(out), "wall_s": payload["wall_s"]}


def dataset_table(path: str | None, task: str):
    """The hand table a dataset recorded the task under (its LeRobot sidecar, meta/vega_wuji.json), or None (the
    task's own, as the demos build it now)."""
    if path is None:
        return None
    from tactile_sim.vla.action import HandTable
    from tactile_sim.vla.demos import INSTRUCTIONS

    f = Path(path)
    if f.is_dir():
        f = f / "meta" / "vega_wuji.json"
    tables = json.loads(f.read_text()).get("hand_tables")
    if tables is None:
        raise ValueError(f"{f} keeps one hand table for all tasks (exported before 4d0b3a3): omit --hand-tables")
    return HandTable.from_dict(tables[INSTRUCTIONS[task]])


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def run_all(jobs: list[Job], root: Path, workers: int) -> None:
    from tactile_sim.vla.demos import demo_config, prewarm

    todo = [j for j in jobs if not j.path(root).exists()]
    print(f"{len(jobs)} episodes, {len(jobs) - len(todo)} done, {len(todo)} to run on {workers} workers", flush=True)
    if not todo:
        return
    prewarm([demo_config(j.task, j.seed)[0] for j in todo])
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
        futs = {ex.submit(run_job, j, str(root)): j for j in todo}
        for k, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                msg = f"ok {f.result().get('wall_s', 0):.0f}s"
            except Exception:  # noqa: BLE001 - report and keep going
                msg = "ERROR\n" + traceback.format_exc()
            print(f"[{k}/{len(todo)}] {j.condition.name}/seed{j.seed} {msg} "
                  f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)


def per_seed(results: list[dict]) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for r in results:
        m = {"done": float(r["done"]), "drops": float(r["dropped"]), "t_done": float(r["t_done"])}
        if r["task"] == "drill":
            m.update(cam_outs=float(r["cam_outs"]), slip_rot_deg=r["max_slip_rot_deg"],
                     slip_trans_mm=r["max_slip_trans_mm"], aim_mm=r.get("closest_aim_mm", math.nan),
                     trigger=r.get("max_trigger", math.nan))
        else:
            st = r["strikes"]
            m.update(strikes=float(len(st)),
                     advance_mm=1e3 * float(np.mean([s["advance"] for s in st])) if st else math.nan,
                     slip_rot_deg=float(np.mean([s["slip_rot_deg"] for s in st])) if st else math.nan)
        out.setdefault(r["job"]["condition"], {})[r["job"]["seed"]] = m
    return out


def report(task: str, root: Path, ref: str = "vla") -> str:
    data = per_seed([json.loads(p.read_text()) for p in sorted(root.glob("*/seed*.json"))])
    if ref not in data:
        return f"no '{ref}' results under {root}"
    keys = METRICS[task]
    lines = ["| condition | n | " + " | ".join(keys) + " |", "|---" * (len(keys) + 2) + "|"]
    for cond, seeds in sorted(data.items()):
        cells = []
        for k in keys:
            vals = [seeds[s][k] for s in seeds if np.isfinite(seeds[s][k])]
            mean = float(np.mean(vals)) if vals else math.nan
            common = [s for s in sorted(set(seeds) & set(data[ref])) if np.isfinite(seeds[s][k]) and
                      np.isfinite(data[ref][s][k])]
            if cond == ref or len(common) < 2:
                cells.append(f"{mean:.3g}")
            else:
                d, lo, hi = paired_bootstrap([data[ref][s][k] for s in common], [seeds[s][k] for s in common])
                cells.append(f"{mean:.3g} ({d:+.2g} [{lo:+.2g}, {hi:+.2g}])")
        lines.append(f"| {cond} | {len(seeds)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("list", "run", "report"))
    ap.add_argument("--task", default="hammer", choices=sorted(MAX_TIME))
    ap.add_argument("--server", default="groot://localhost:5555")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=200_000, help="away from the demos' seeds")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--latency", type=float, default=0.1, help="s from observation to the chunk taking effect")
    ap.add_argument("--exec-steps", type=int, default=8, help="frames executed per chunk")
    ap.add_argument("--blend", type=float, default=0.0, help="s over which a new chunk fades in over the last")
    ap.add_argument("--smooth", type=int, default=0, help="passes of a [1, 2, 1]/4 filter along each chunk")
    ap.add_argument("--hand-tables", default=None,
                    help="the VLA's training dataset (or its meta/vega_wuji.json): run its hands under the tables it "
                         "was recorded with (default: the task's own)")
    ap.add_argument("--policy", action="append", default=[], help="NAME=PATH of an exported residual policy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    root = Path(a.out)
    conds = conditions(dict(p.split("=", 1) for p in a.policy))
    jobs = [Job(a.task, c, a.seed0 + s, a.server, a.latency, a.exec_steps, a.blend, a.smooth, a.hand_tables)
            for c in conds for s in range(a.seeds)]
    if a.cmd == "list":
        for j in jobs:
            print(j.condition.name, j.seed, "done" if j.path(root).exists() else "")
    elif a.cmd == "run":
        run_all(jobs, root, a.workers)
    else:
        print(report(a.task, root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
