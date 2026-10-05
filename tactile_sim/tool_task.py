"""Run a saw or driver task with the scripted behavior and collect force-truth.

    python -m tactile_sim.tool_task --task saw
    python -m tactile_sim.tool_task --task drill --mode hole --fast
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from tactile_sim.behaviors.tool_tasks import make_behavior
from tactile_sim.config import SimConfig, fast_config, task_config
from tactile_sim.limits import LimitMonitor
from tactile_sim.logging.strike_metrics import rotvec_diff
from tactile_sim.sim.testbed import Testbed


@dataclass
class ToolTaskResult:
    summary: dict
    truth: dict[str, np.ndarray]
    events: list[tuple[str, float]] = field(default_factory=list)


class ToolTask:
    def __init__(self, cfg: SimConfig, seed: int = 0, t_max: float = 30.0, record_dt: float = 0.001):
        self.cfg = cfg
        self.tb = Testbed(cfg, seed=seed)
        self.behavior = make_behavior(self.tb, t_max)
        self.tb.l2_callbacks.append(self.behavior.tick)
        self.limits = LimitMonitor(self.tb.world)
        self.record_every = max(1, int(round(record_dt / self.tb.world.dt)))

    def run(self, t_max: float | None = None) -> ToolTaskResult:
        tb, w = self.tb, self.tb.world
        t_end = (t_max if t_max is not None else self.behavior.t_max) + 2.0
        p0, r0 = w.hammer_in_hand()
        rec: dict[str, list] = {"t": [], "grip": [], "slip_trans": [], "slip_rot": [], "ft": []}
        k = 0
        t_done = float("nan")
        while not self.behavior.finished and tb.t < t_end:
            tb.step()
            self.limits.step()
            if np.isnan(t_done) and w.plant.done():
                t_done = tb.t
            k += 1
            if k % self.record_every:
                continue
            p, r = w.hammer_in_hand()
            rec["t"].append(tb.t)
            rec["grip"].append(tb.grip.measured)
            rec["slip_trans"].append(float(np.linalg.norm(p - p0)))
            rec["slip_rot"].append(rotvec_diff(r0, r))
            ft = tb.sensors["ft"].latest()
            rec["ft"].append(ft.value.copy() if ft.seq >= 0 else np.zeros(6))
            for key, v in w.plant.truth().items():
                rec.setdefault(key, []).append(v)
        truth = {key: np.asarray(v, dtype=float) for key, v in rec.items()}
        s = {"task": self.cfg.plant.kind, "done": bool(w.plant.done()), "t_done": t_done,
             "progress": float(w.plant.progress), "max_slip_trans": float(np.max(truth["slip_trans"], initial=0.0)),
             "max_slip_rot": float(np.max(truth["slip_rot"], initial=0.0)),  # includes the pads' elastic twist
             "net_slip_trans": float(truth["slip_trans"][-1]) if len(truth["t"]) else 0.0,
             "net_slip_rot": float(truth["slip_rot"][-1]) if len(truth["t"]) else 0.0,
             "mean_grip": float(np.mean(truth["grip"])) if len(truth["grip"]) else float("nan")}
        if self.cfg.plant.kind == "saw":
            s.update(strokes=int(w.plant.strokes), peak_normal=float(np.max(truth["saw_normal_force"], initial=0.0)),
                     peak_lateral=float(np.max(truth["saw_lateral_force"], initial=0.0)))
        else:
            s.update(mode=self.cfg.drill.mode, state=w.plant.state, cam_outs=int(w.plant.cam_outs),
                     peak_torque=float(np.max(truth["drill_torque"], initial=0.0)),
                     peak_axial=float(np.max(truth["drill_axial_force"], initial=0.0)))
        s.update({f"limit_{k}": v for k, v in self.limits.summary().items()})
        viol = self.limits.violations()
        s["limit_violations"] = ",".join(sorted(viol))
        return ToolTaskResult(s, truth, list(self.behavior.events))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["saw", "drill"], required=True)
    ap.add_argument("--mode", choices=["screw", "hole"], default="screw", help="driver: drive a screw or drill a hole")
    ap.add_argument("--fast", action="store_true", help="4 kHz physics instead of 8 kHz")
    ap.add_argument("--t-max", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    base = fast_config() if args.fast else SimConfig()
    over = {"drill": {"mode": args.mode}} if args.task == "drill" else {}
    cfg = task_config(args.task, base, **over)
    res = ToolTask(cfg, seed=args.seed, t_max=args.t_max).run()
    for k, v in res.summary.items():
        print(f"{k:24s} {v:.4g}" if isinstance(v, float) else f"{k:24s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
