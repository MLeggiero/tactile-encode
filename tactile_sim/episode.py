"""One Task A episode: settle the grasp, run scripted strikes, and collect force-truth records.

Every physics step the recorder samples ground truth the sensors never see (hammer-nail contact force,
nail depth, tool pose in the hand). Per strike it builds a StrikeRecord: impact pulse, flag latency by
source, nail advance, slip in the grasp, grip timing relative to the impact, and post-impact ringing on
the wrist F/T. Slip feeds back into the next strike's grip margin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from tactile_sim import names
from tactile_sim.behaviors.swing import ScriptedSwing, StrikePlan
from tactile_sim.config import SimConfig
from tactile_sim.logging.strike_metrics import StrikeRecord, grasp_slip
from tactile_sim.sim.testbed import Testbed
from tactile_sim.sim.truth import pair_force, pulse_stats


@dataclass
class TruthLog:
    t: list = field(default_factory=list)
    face_pos: list = field(default_factory=list)
    face_vel: list = field(default_factory=list)
    nail_depth: list = field(default_factory=list)
    contact_force: list = field(default_factory=list)
    pad_forces: list = field(default_factory=list)
    hammer_in_hand_pos: list = field(default_factory=list)
    hammer_in_hand_rotvec: list = field(default_factory=list)
    qfrc_constraint_arm: list = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=float) for k, v in self.__dict__.items()}


@dataclass
class EpisodeResult:
    strikes: list[StrikeRecord]
    summary: dict
    truth: dict[str, np.ndarray]
    events: list[tuple[str, float]]
    testbed: Testbed


def ringing_energy(t: np.ndarray, f_axis: np.ndarray, t_c: float, band=(30.0, 300.0),
                   post=(0.005, 0.050), pre=(-0.050, -0.005)) -> tuple[float, float]:
    """Mean-square band-passed strike-axis force after the impact (ringing) and before it."""
    if len(t) < 32 or not np.isfinite(t_c):
        return float("nan"), float("nan")
    fs = 1.0 / float(np.median(np.diff(t)))
    hi = min(band[1], 0.45 * fs)
    sos = signal.butter(2, [band[0], hi], btype="band", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, f_axis - f_axis.mean())
    m_post = (t >= t_c + post[0]) & (t < t_c + post[1])
    m_pre = (t >= t_c + pre[0]) & (t < t_c + pre[1])
    if not m_post.any() or not m_pre.any():
        return float("nan"), float("nan")
    return float(np.mean(y[m_post] ** 2)), float(np.mean(y[m_pre] ** 2))


class Episode:
    def __init__(self, cfg: SimConfig | None = None, seed: int | None = 0, record_truth: bool = True,
                 testbed: Testbed | None = None, frame_hz: float | None = None):
        self.cfg = cfg or SimConfig()
        self.tb = testbed or Testbed(self.cfg, seed=seed)
        if testbed is not None:
            self.tb.reset(seed)
        self.record_truth = record_truth
        w = self.tb.world
        self.g_face = w.model.geom(names.HAMMER_HEAD_GEOM).id
        self.g_nail = w.model.geom(names.NAIL_HEAD_GEOM).id
        self.g_board = w.model.geom(names.BOARD_GEOM).id
        self.face_site = w.site[names.HAMMER_FACE_SITE]
        self.truth = TruthLog()
        self.records: list[StrikeRecord] = []
        self.events: list[tuple[str, float]] = []
        self._cur: StrikeRecord | None = None
        self._series_t: list[float] = []
        self._series_f: list[float] = []
        self._board_hit = False
        self._k = 0
        self.frame_decim = None if not frame_hz else max(1, int(round(1.0 / (frame_hz * w.dt))))
        self.frames_t: list[float] = []
        self.frames_pos: list[np.ndarray] = []
        self.frames_quat: list[np.ndarray] = []
        self.swing = ScriptedSwing(self.tb, on_swing_start=self._on_swing_start, on_strike_end=self._on_strike_end)
        self.tb.l2_callbacks[:] = [self.swing.tick]

    # ---------------------------------------------------------------- hooks
    def _on_swing_start(self, plan: StrikePlan) -> None:
        w = self.tb.world
        p, r = w.hammer_in_hand()
        rec = StrikeRecord(idx=plan.idx, t_swing_start=plan.t_start, t_c_pred=plan.t_c_pred,
                           depth_before=w.plant.depth)
        rec._p0, rec._r0 = p, r  # noqa: SLF001 - private scratch
        self._cur = rec
        self._series_t, self._series_f = [], []
        self._board_hit = False
        self._peak_tau = np.zeros(7)
        self._grip_samples: list[tuple[float, float]] = []
        self.events.append((f"swing_start_{plan.idx}", plan.t_start))

    def _on_strike_end(self, plan: StrikePlan) -> None:
        rec = self._cur
        if rec is None:
            return
        tb, w = self.tb, self.tb.world
        rec.depth_after = w.plant.depth
        p, r = w.hammer_in_hand()
        rec.slip_trans, rec.slip_rot = grasp_slip(rec._p0, rec._r0, p, r)  # noqa: SLF001
        rec.drop = tb.grip.dropped
        rec.peak_joint_torque = self._peak_tau.copy()
        ts, fs = np.asarray(self._series_t), np.asarray(self._series_f)
        if len(ts) and fs.max() > 0:
            st = pulse_stats(ts, fs)
            rec.hit = True
            rec.peak_force_truth, rec.impulse, rec.pulse_width = st["peak"], st["impulse"], st["width"]
            t_pk_load = st["t_peak"]
        else:
            rec.hit = False
            t_pk_load = float("nan")
        ev = [e for e in tb.l1.detector.events if e.t_flag >= rec.t_swing_start]
        if ev:
            rec.t_flag, rec.flag_source = ev[0].t_flag, ev[0].source
        # measured F/T along the strike axis around the contact
        ft = tb.sensors["ft"].history()
        if len(ft["t_sample"]) and np.isfinite(rec.t_contact_truth):
            Rft = w.data.site_xmat[w.site[names.FT_SITE]].reshape(3, 3)
            f_ax = ft["value"][:, :3] @ Rft.T @ tb.l1.axis
            m = (ft["t_sample"] >= rec.t_contact_truth - 0.005) & (ft["t_sample"] < rec.t_contact_truth + 0.05)
            rec.peak_ft_meas = float(np.max(np.abs(f_ax[m] - f_ax[m][0]))) if m.any() else 0.0
            win = (ft["t_sample"] >= rec.t_contact_truth - 0.25) & (ft["t_sample"] < rec.t_contact_truth + 0.1)
            rec.ringing_energy, rec.pre_energy = ringing_energy(ft["t_sample"][win], f_ax[win], rec.t_contact_truth)
        # grip timing relative to the impact (truth pad force)
        if self._grip_samples and np.isfinite(rec.t_contact_truth):
            gt = np.array(self._grip_samples)
            after = gt[(gt[:, 0] >= rec.t_contact_truth) & (gt[:, 0] <= rec.t_contact_truth + 0.3)]
            if len(after):
                k = int(np.argmax(after[:, 1]))
                rec.grip_peak = float(after[k, 1])
                rec.t_grip_peak_rel = float(after[k, 0] - t_pk_load)
            rec.t_grip_ramp_start_rel = float((plan.t_c_pred - self.cfg.swing.grip_lead) - rec.t_contact_truth)
        self.records.append(rec)
        self.swing.register_slip(rec.slip_trans, rec.slip_rot)
        self.events.append((f"strike_end_{rec.idx}", tb.t))
        self._cur = None

    # ---------------------------------------------------------------- per step
    def _step_cb(self, tb: Testbed) -> None:
        w = tb.world
        d = w.data
        t = w.t
        rec = self._cur
        f_nail = 0.0
        if rec is not None or self.record_truth:
            f_nail, _ = pair_force(w.model, d, self.g_face, self.g_nail)
        if rec is not None:
            if f_nail > 0:
                if not np.isfinite(rec.t_contact_truth):
                    rec.t_contact_truth = t
                    v = w.site_velocity(self.face_site)[:3] @ tb.l1.axis
                    rec.v_strike_actual = float(v)
                    rec.grip_at_contact = float(np.mean(w.pad_normal_forces()))
                self._series_t.append(t)
                self._series_f.append(f_nail)
            elif np.isfinite(rec.t_contact_truth) and t - rec.t_contact_truth < 0.05:
                self._series_t.append(t)
                self._series_f.append(0.0)
            np.maximum(self._peak_tau, np.abs(d.ctrl[w.arm_act]), out=self._peak_tau)
            if self._k % 4 == 0:
                self._grip_samples.append((t, float(np.mean(w.pad_normal_forces()))))
        if self.record_truth and self._k % self.cfg.logging.truth_decim == 0:
            T = self.truth
            T.t.append(t)
            T.face_pos.append(d.site_xpos[self.face_site].copy())
            T.face_vel.append(w.site_velocity(self.face_site)[:3])
            T.nail_depth.append(w.plant.depth)
            T.contact_force.append(f_nail)
            T.pad_forces.append(w.pad_normal_forces())
            p, r = w.hammer_in_hand()
            T.hammer_in_hand_pos.append(p)
            T.hammer_in_hand_rotvec.append(r)
            T.qfrc_constraint_arm.append(d.qfrc_constraint[w.arm_dofs].copy())
        if self.frame_decim and self._k % self.frame_decim == 0:
            self.frames_t.append(t)
            self.frames_pos.append(d.xpos.copy())
            self.frames_quat.append(d.xquat.copy())
        self._k += 1

    # ---------------------------------------------------------------- run
    def run(self, n_strikes: int | None = None, max_time: float | None = None) -> EpisodeResult:
        sw = self.swing
        if n_strikes is not None:
            sw.n_strikes = n_strikes
        per_strike = self.cfg.swing.windup_time + 0.6 + self.cfg.swing.recover_time + self.cfg.swing.settle_time + 0.3
        max_time = max_time or (self.cfg.swing.approach_time + 0.5 + sw.n_strikes * per_strike)
        tb = self.tb
        n_max = int(round(max_time / tb.world.dt))
        for _ in range(n_max):
            tb.step()
            self._step_cb(tb)
            if sw.done:
                break
        if tb.grip.dropped:
            self.events.append(("drop", tb.grip.t_drop))
            if self._cur is not None:
                self._cur.drop = True
                self.records.append(self._cur)
                self._cur = None
        return EpisodeResult(self.records, summarize(self.records, self.cfg), self.truth.arrays(), self.events, tb)


def summarize(records: list[StrikeRecord], cfg: SimConfig) -> dict:
    n = len(records)
    if n == 0:
        return {"n_strikes": 0}
    depth = [r.depth_after for r in records]
    reached = [i for i, dd in enumerate(depth) if dd >= cfg.plant.drive_target - 1e-4]
    lat = np.array([r.flag_latency for r in records if r.hit and np.isfinite(r.t_flag)])
    return {
        "n_strikes": n,
        "total_depth": float(depth[-1] - records[0].depth_before),
        "hit_rate": float(np.mean([r.hit for r in records])),
        "strikes_to_target": float(reached[0] + 1) if reached else float("nan"),
        "mean_depth_inc": float(np.mean([r.depth_inc for r in records])),
        "max_slip_trans": float(max(r.slip_trans for r in records)),
        "max_slip_rot": float(max(r.slip_rot for r in records)),
        "drops": int(sum(r.drop for r in records)),
        "mean_peak_force": float(np.mean([r.peak_force_truth for r in records if r.hit] or [0.0])),
        "mean_pulse_width": float(np.mean([r.pulse_width for r in records if r.hit] or [0.0])),
        "mean_v_strike": float(np.mean([r.v_strike_actual for r in records if r.hit] or [0.0])),
        "mean_flag_latency": float(lat.mean()) if len(lat) else float("nan"),
        "max_flag_latency": float(lat.max()) if len(lat) else float("nan"),
        "flag_rate": float(np.mean([np.isfinite(r.t_flag) for r in records])),
    }
