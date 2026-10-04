"""Replay a retargeted tool motion on the testbed and record what the sensors feel.

The plan's TCP path becomes the L1 reference (a chunk the 1 kHz controller evaluates itself): a min-jerk move from
the hover pose to the path's start, the path, then a hold. The nail is the testbed's plant, so the blows are produced
by our contact model; the recorded motion only shapes how the hammer arrives. The impact detector is armed around
each recorded contact (as the swing layer arms it ahead of a predicted one). When it flags a blow,
the reference pauses for `hold` (the blow's force comes from the plant, not from the arm chasing a reference into the
nail) and then resumes. Everything else (grip loop, limits, sensor models) is the testbed as configured.

    python -m tactile_sim.replay --source adroit --demo 0 --fast --out runs/replay_adroit0.npz
    python -m tactile_sim.replay --source dextoolbench --task hammer/claw_hammer/swing_side --speed 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tactile_sim import names
from tactile_sim.config import SimConfig, fast_config, hand_config
from tactile_sim.control.interface import L2Command, Mode
from tactile_sim.limits import LimitMonitor
from tactile_sim.logging.strike_metrics import rotvec_diff
from tactile_sim.replay.retarget import ReplayPlan, plan_replay
from tactile_sim.replay.sources import load
from tactile_sim.sim.testbed import Testbed
from tactile_sim.sim.truth import pair_force


def _min_jerk(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10 - 15 * x + 6 * x**2), (30 * x**2 - 60 * x**3 + 30 * x**4)


class ReplayRef:
    """L1 reference for a replay: lead-in from the current pose, the plan, then hold the last pose."""

    t_armed = -np.inf

    def __init__(self, plan: ReplayPlan, t0: float, x_start: np.ndarray, R_start: np.ndarray, lead_in: float,
                 hold: float = 0.05):
        self.plan = plan
        self.t0 = t0
        self.lead_in = lead_in
        self.hold = hold
        self.x_start, self.R_start = x_start, R_start
        self._lead = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R_start, plan.R[0]])))
        self.R = R_start.copy()
        self.paused = 0.0  # accumulated pause time
        self._pause_until = -np.inf
        self._last_flag = -np.inf
        self.tau = 0.0  # time along the plan

    @property
    def finished_at(self) -> float:
        return self.t0 + self.lead_in + self.plan.duration + self.paused

    def evaluate(self, t, t_flag, x_now):
        if t_flag is not None and t_flag > self._last_flag and t - self.t0 > self.lead_in:
            self._last_flag = t_flag
            self._pause_until = t_flag + self.hold
        rel = t - self.t0
        zero = np.zeros(6)
        if rel < self.lead_in:
            s, ds = _min_jerk(rel / self.lead_in)
            x = self.x_start + (self.plan.tcp[0] - self.x_start) * s
            xd = np.concatenate([(self.plan.tcp[0] - self.x_start) * ds / self.lead_in, np.zeros(3)])
            self.R = self._lead([s]).as_matrix()[0]
            return x, xd, zero, Mode.FREE
        if t < self._pause_until:
            self.paused += t - getattr(self, "_t_last", t)
            self._t_last = t
            i = min(int(self.tau * 1000.0), len(self.plan.t) - 1)
            self.R = self.plan.R[i]
            return self.plan.tcp[i], zero, zero, Mode.POST
        self._t_last = t
        self.tau = rel - self.lead_in - self.paused
        p = self.plan
        i = int(np.clip(np.searchsorted(p.t, self.tau), 1, len(p.t) - 1))
        dt = p.t[i] - p.t[i - 1]
        x = p.tcp[i]
        xd = np.concatenate([(p.tcp[i] - p.tcp[i - 1]) / dt, np.zeros(3)]) if self.tau < p.t[-1] else zero
        self.R = p.R[i]
        return x, xd, zero, Mode.FREE


@dataclass
class ReplayResult:
    summary: dict
    truth: dict[str, np.ndarray]
    sensors: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    strikes: list = field(default_factory=list)  # StrikeRecord per landed blow (for the replay viewer)
    testbed: Testbed | None = None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        flat = {f"truth/{k}": v for k, v in self.truth.items()}
        for name, h in self.sensors.items():
            for k, v in h.items():
                flat[f"sensors/{name}/{k}"] = v
        flat["summary"] = np.array(repr(self.summary))
        np.savez_compressed(path, **flat)


class ReplayEpisode:
    def __init__(self, cfg: SimConfig, plan: ReplayPlan | None = None, motion=None, seed: int = 0,
                 lead_in: float = 1.5, hold: float = 0.05, frame_hz: float | None = None, **plan_kw):
        self.cfg = cfg
        self.tb = Testbed(cfg, seed=seed)
        self.plan = plan if plan is not None else plan_replay(motion, self.tb.world, **plan_kw)
        self.lead_in = lead_in
        self.hold = hold
        w = self.tb.world
        self.frame_decim = None if not frame_hz else max(1, int(round(1.0 / (frame_hz * w.dt))))
        self.frames_t: list[float] = []
        self.frames_pos: list[np.ndarray] = []
        self.frames_quat: list[np.ndarray] = []

    def run(self, tail: float = 0.4) -> ReplayResult:
        tb, w = self.tb, self.tb.world
        x0, R0 = w.tcp_pose()
        ref = ReplayRef(self.plan, tb.t, x0, R0, self.lead_in, self.hold)
        c = self.cfg.controller
        cmd = L2Command(tb.t, x0.copy(), R0.copy(), K=np.array(c.k_trans + c.k_rot, dtype=float), F_grip=c.grip_hold,
                        ref=ref)
        tb.l1.set_command(cmd)

        armed_for: set[int] = set()

        def keep_alive(t):
            cmd.t = t
            tb.l1.grip_setpoint = c.grip_hold
            # arm the impact detector around each recorded contact, as the swing layer does ahead of a predicted one
            if t - ref.t0 > ref.lead_in:
                for j, tc in enumerate(self.plan.t_contacts):
                    if j not in armed_for and tc - 0.05 <= ref.tau <= tc + 0.15:
                        tb.l1.detector.arm(True)
                        armed_for.add(j)

        tb.l2_callbacks.append(keep_alive)
        limits = LimitMonitor(w)
        g_face = w.model.geom(names.HAMMER_FACE_GEOM).id
        g_nail = w.model.geom(names.NAIL_HEAD_GEOM).id
        p0, r0 = w.hammer_in_hand()
        T: dict[str, list] = {k: [] for k in ("t", "contact_force", "nail_depth", "slip_trans", "slip_rot", "grip",
                                               "tcp", "tcp_ref", "face_speed", "pad_forces",
                                               "hammer_in_hand_rotvec", "face_pos")}
        every = max(1, int(round(0.001 / w.dt)))
        k = 0
        flags: list[tuple[float, str]] = []
        while tb.t < ref.finished_at + tail:
            tb.step()
            limits.step()
            ev = tb.l1.last_event
            if ev is not None and (not flags or ev.t_flag > flags[-1][0]):
                flags.append((ev.t_flag, ev.source))
            k += 1
            if self.frame_decim and k % self.frame_decim == 0:
                self.frames_t.append(tb.t)
                self.frames_pos.append(w.data.xpos.copy())
                self.frames_quat.append(w.data.xquat.copy())
            if k % every:
                continue
            f, _ = pair_force(w.model, w.data, g_face, g_nail)
            p, r = w.hammer_in_hand()
            T["t"].append(tb.t)
            T["contact_force"].append(f)
            T["nail_depth"].append(max(w.plant.depth, 0.0))  # the nail rebounds by microns at its stop
            T["slip_trans"].append(float(np.linalg.norm(p - p0)))
            T["slip_rot"].append(rotvec_diff(r0, r))
            T["grip"].append(tb.grip.measured)
            T["tcp"].append(w.tcp_pose()[0])
            T["tcp_ref"].append(tb.l1.cmd.x_eq.copy() if tb.l1.cmd is not None else np.full(3, np.nan))
            T["face_speed"].append(float(np.linalg.norm(w.plant.face_velocity())))
            T["pad_forces"].append(np.asarray(w.pad_normal_forces(), dtype=float))
            T["hammer_in_hand_rotvec"].append(r.copy())
            T["face_pos"].append(w.data.site_xpos[w.site[names.HAMMER_FACE_SITE]].copy())
        truth = {k: np.asarray(v, dtype=float) for k, v in T.items()}
        f = truth["contact_force"]
        on = np.flatnonzero((f > 5.0) & ~np.r_[False, f[:-1] > 5.0])
        blows = [i for j, i in enumerate(on) if j == 0 or truth["t"][i] - truth["t"][on[j - 1]] > 0.05]
        track = np.linalg.norm(truth["tcp"] - truth["tcp_ref"], axis=1)
        s = {"motion": self.plan.motion, "time_scale": self.plan.time_scale, "duration": self.plan.duration,
             "recorded_contacts": len(self.plan.t_contacts), "blows": len(blows),
             "peak_force": float(f.max(initial=0.0)),
             "impact_speeds": [round(float(truth["face_speed"][max(i - 2, 0)]), 3) for i in blows],
             "nail_depth": float(truth["nail_depth"][-1]) if len(f) else 0.0,
             "max_slip_trans": float(truth["slip_trans"].max(initial=0)),
             "net_slip_rot": float(truth["slip_rot"][-1]) if len(f) else 0.0,
             "tracking_rms": float(np.sqrt(np.nanmean(track**2))), "tracking_max": float(np.nanmax(track))}
        s.update({f"limit_{k}": v for k, v in limits.summary().items()})
        s["limit_violations"] = ",".join(sorted(limits.violations()))
        sensors = {name: tb.sensors[name].history() for name in tb.sensors.names()}
        return ReplayResult(s, truth, sensors, self._strikes(truth, blows, flags), tb)

    def _strikes(self, T: dict, blows: list[int], flags: list[tuple[float, str]]) -> list:
        """One StrikeRecord per landed blow, with the fields the replay viewer shows."""
        from tactile_sim.logging.strike_metrics import StrikeRecord
        from tactile_sim.sim.truth import pulse_stats

        t, f = T["t"], T["contact_force"]
        recs = []
        for j, i in enumerate(blows):
            t_c = float(t[i])
            nxt = blows[j + 1] if j + 1 < len(blows) else len(t)
            w0, w1 = max(i - 5, 0), min(i + 40, nxt)
            st = pulse_stats(t[w0:w1], f[w0:w1])
            before = max(i - 60, 0)
            after = min(nxt - 1, i + 200)
            fl = [(tf, src) for tf, src in flags if t_c - 0.002 <= tf <= t_c + 0.02]
            r = StrikeRecord(idx=j, t_swing_start=float(t[before]))
            r.t_c_pred = t_c
            r.t_contact_truth = t_c
            r.hit = True
            r.peak_force_truth = float(st["peak"])
            r.impulse = float(st["impulse"])
            r.pulse_width = float(st["width"])
            r.depth_before = float(T["nail_depth"][before])
            r.depth_after = float(T["nail_depth"][after])
            r.v_strike_actual = float(T["face_speed"][max(i - 2, 0)])
            r.slip_trans = float(abs(T["slip_trans"][after] - T["slip_trans"][before]))
            r.slip_rot = float(abs(T["slip_rot"][after] - T["slip_rot"][before]))
            r.grip_at_contact = float(T["grip"][i])
            if fl:
                r.t_flag, r.flag_source = float(fl[0][0]), fl[0][1]
            recs.append(r)
        return recs


def robot_config(robot: str = "fr3", hand: str | None = None, fast: bool = False) -> SimConfig:
    """Testbed config for a replay: the FR3 with the Franka Hand (or a WUJI hand), or the Vega U with WUJI hands."""
    base = fast_config() if fast else SimConfig()
    hand = hand or ("wuji2" if robot.startswith("vega") else "franka")
    return hand_config(hand, base, robot=robot)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["adroit", "dextoolbench"], required=True)
    ap.add_argument("--demo", type=int, default=0, help="adroit: demonstration index (0-24)")
    ap.add_argument("--task", default="hammer/claw_hammer/swing_side", help="dextoolbench: category/object/task")
    ap.add_argument("--speed", type=float, default=1.0, help="play faster than recorded before the limits apply")
    ap.add_argument("--engage", type=float, default=0.004, help="m the recorded contact reaches past the nail head")
    ap.add_argument("--no-align", action="store_true", help="keep the recorded face angle at contact")
    ap.add_argument("--robot", choices=["fr3", "vega_1u"], default="fr3",
                    help="fr3 (Franka Hand unless --hand) or vega_1u (WUJI Hand 2, position-only arm)")
    ap.add_argument("--hand", choices=["franka", "wuji2"], default=None)
    ap.add_argument("--fast", action="store_true", help="4 kHz physics instead of 8 kHz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="save truth and sensor streams (.npz)")
    args = ap.parse_args(argv)
    motion = load(args.source, **({"demo": args.demo} if args.source == "adroit" else {"task": args.task}))
    cfg = robot_config(args.robot, args.hand, args.fast)
    ep = ReplayEpisode(cfg, motion=motion, seed=args.seed, speed=args.speed, engage=args.engage,
                       align_face=not args.no_align)
    res = ep.run()
    for k, v in res.summary.items():
        print(f"{k:24s} {v:.4g}" if isinstance(v, float) else f"{k:24s} {v}")
    if args.out:
        res.save(args.out)
        print(f"wrote {args.out}")
    return 0
