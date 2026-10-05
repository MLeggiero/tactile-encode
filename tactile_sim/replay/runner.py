"""Replay a retargeted tool motion on the testbed and record what the sensors feel.

The plan's TCP path becomes the L1 reference (a chunk the 1 kHz controller evaluates itself): a min-jerk move from
the hover pose to the path's start, the path, then a hold. The nail is the testbed's plant, so the blows are produced
by our contact model; the recorded motion only shapes how the hammer arrives. The impact detector is armed around
each recorded contact (as the swing layer arms it ahead of a predicted one). After a flag the reference simply
continues with the recorded rebound (optionally pausing for `hold`); between blows the retargeted path stands off
the nail, so the hammer does not rest on it.

Every face-on-nail contact is classified: a *blow* is short (<= 20 ms above 5 N) and arrives along the nail axis
(>= 0.05 m/s); anything else is a *press*. A blow is *off-centre* if the nail head is not wholly under the face
(lateral offset > face radius - head radius). The hammer is also checked for passing *through* the nail: the hammer
head and the nail shank have no contact pair (only the face strikes the nail), so their distance is monitored.
Everything else (grip loop, limits, sensor models) is the testbed as configured.

    python -m tactile_sim.replay --source adroit --demo 0 --fast --out runs/replay_adroit0.npz
    python -m tactile_sim.replay --source dextoolbench --task hammer/claw_hammer/swing_side --speed 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tactile_sim import names
from tactile_sim.config import SimConfig, fast_config, hand_config
from tactile_sim.control.interface import L2Command, Mode
from tactile_sim.limits import LimitMonitor
from tactile_sim.logging.strike_metrics import rotvec_diff
from tactile_sim.model.builder import strike_axis
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
                 hold: float = 0.0):
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
        # reference twist and acceleration along the plan, for the controller's feedforward (tracks the strikes
        # instead of lagging them by the impedance's spring stretch)
        from tactile_sim.replay.retarget import lowpass

        t = plan.t
        v = np.gradient(plan.tcp, t, axis=0)
        rv = (Rotation.from_matrix(plan.R[1:]) * Rotation.from_matrix(plan.R[:-1]).inv()).as_rotvec()
        w = np.vstack([rv, rv[-1:]]) / np.r_[np.diff(t), np.diff(t)[-1:]][:, None]
        self.twist = np.hstack([lowpass(t, v, 40.0), lowpass(t, w, 40.0)])
        acc = np.gradient(self.twist, t, axis=0)
        self.accel = lowpass(t, acc, 30.0)

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
        del dt
        x = p.tcp[i]
        moving = self.tau < p.t[-1]
        xd = self.twist[i] if moving else zero
        xdd = self.accel[i] if moving else zero
        self.R = p.R[i]
        return x, xd, xdd, Mode.FREE


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
                 lead_in: float = 1.5, hold: float = 0.0, frame_hz: float | None = None, **plan_kw):
        self.cfg = cfg
        self.tb = Testbed(cfg, seed=seed)
        if plan is None:
            # a position-interface arm (Vega) lags its 100 Hz targets more than the FR3's torque loop follows its
            # path, so each recorded contact is aimed deeper past the nail head (else ~half its blows fall short)
            plan_kw.setdefault("engage", 0.008 if self.tb.world.arm_spec.interface == "position" else 0.004)
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
                                               "hammer_in_hand_rotvec", "face_pos", "v_axis", "face_offset",
                                               "head_shank_gap")}
        a_ax = strike_axis(self.cfg)
        head_site = w.model.site(names.NAIL_HEAD_SITE).id
        g_head = w.model.geom(names.HAMMER_HEAD_GEOM).id
        nail_b = w.model.body(names.NAIL_BODY).id
        g_shank = [g for g in range(w.model.ngeom) if w.model.geom_bodyid[g] == nail_b and g != g_nail][0]
        fromto = np.zeros(6)
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
            fp = w.data.site_xpos[w.site[names.HAMMER_FACE_SITE]].copy()
            T["face_pos"].append(fp)
            T["v_axis"].append(float(w.plant.face_velocity() @ a_ax))
            rel = fp - w.data.site_xpos[head_site]
            T["face_offset"].append(float(np.linalg.norm(rel - (rel @ a_ax) * a_ax)))
            T["head_shank_gap"].append(float(mujoco.mj_geomDistance(w.model, w.data, g_head, g_shank, 0.05, fromto)))
        truth = {k: np.asarray(v, dtype=float) for k, v in T.items()}
        f = truth["contact_force"]
        blows, presses, off_centre = self._classify(truth)
        # the shank capsule's cap reaches shank_radius - head_half_h past the head's face: ignore that overlap
        gap = truth["head_shank_gap"] + max(self.cfg.plant.shank_radius - self.cfg.plant.head_half_h, 0.0) + 5e-4
        through = int(np.sum((gap < 0) & ~np.r_[False, gap[:-1] < 0]))
        track = np.linalg.norm(truth["tcp"] - truth["tcp_ref"], axis=1)
        s = {"motion": self.plan.motion, "time_scale": self.plan.time_scale, "duration": self.plan.duration,
             "recorded_contacts": len(self.plan.t_contacts), "blows": len(blows), "presses": presses,
             "off_centre_blows": off_centre, "through_nail": through,
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

    def _classify(self, T: dict, thresh: float = 5.0, max_dur: float = 0.020, v_min: float = 0.05
                  ) -> tuple[list[int], int, int]:
        """Contact events on the nail head -> (blow onset indices, number of presses, number of off-centre blows)."""
        from tactile_sim.model.tool_hammer import hammer_geometry

        t, f = T["t"], T["contact_force"]
        on = f > thresh
        starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
        ends = np.flatnonzero(on & ~np.r_[on[1:], False])
        geo = hammer_geometry(self.cfg.hammer)
        allow = max(geo.face_radius - self.cfg.plant.head_radius, 0.0)
        blows, presses, off = [], 0, 0
        for a, b in zip(starts, ends, strict=True):
            dur = t[b] - t[a]
            v = T["v_axis"][max(a - 2, 0)]
            if dur <= max_dur and v >= v_min:
                if blows and t[a] - t[blows[-1]] < 0.03:
                    continue  # chatter within one blow
                blows.append(int(a))
                off += int(T["face_offset"][a] > allow)
            else:
                presses += 1
        return blows, presses, off

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
    ap.add_argument("--source", choices=["adroit", "dextoolbench", "grab"], required=True)
    ap.add_argument("--demo", type=int, default=0, help="adroit: demonstration index (0-24)")
    ap.add_argument("--task", default="hammer/claw_hammer/swing_side", help="dextoolbench: category/object/task")
    ap.add_argument("--seq", default="s1/hammer_use_1",
                    help="grab: subject/sequence (manual download, $TACTILE_SIM_GRAB)")
    ap.add_argument("--speed", type=float, default=1.0, help="play faster than recorded before the limits apply")
    ap.add_argument("--engage", type=float, default=None,
                    help="m the recorded contact reaches past the nail head (default 4 mm, 8 mm on a position arm)")
    ap.add_argument("--no-align", action="store_true", help="keep the recorded face angle at contact")
    ap.add_argument("--robot", choices=["fr3", "vega_1u"], default="fr3",
                    help="fr3 (Franka Hand unless --hand) or vega_1u (WUJI Hand 2, position-only arm)")
    ap.add_argument("--hand", choices=["franka", "wuji2"], default=None)
    ap.add_argument("--fast", action="store_true", help="4 kHz physics instead of 8 kHz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="save truth and sensor streams (.npz)")
    args = ap.parse_args(argv)
    kw = {"adroit": {"demo": args.demo}, "dextoolbench": {"task": args.task}, "grab": {"seq": args.seq}}[args.source]
    motion = load(args.source, **kw)
    cfg = robot_config(args.robot, args.hand, args.fast)
    ep = ReplayEpisode(cfg, motion=motion, seed=args.seed, speed=args.speed,
                       **({} if args.engage is None else {"engage": args.engage}),
                       align_face=not args.no_align)
    res = ep.run()
    for k, v in res.summary.items():
        print(f"{k:24s} {v:.4g}" if isinstance(v, float) else f"{k:24s} {v}")
    if args.out:
        res.save(args.out)
        print(f"wrote {args.out}")
    return 0
