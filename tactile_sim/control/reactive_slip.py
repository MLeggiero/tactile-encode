"""Reactive grip (1 kHz): detect the tool slipping in the hand and answer with the hand's maximum grip.

Detectors (ReactiveCfg.mode):

- "oracle": the tool's motion in the TCP frame from simulation truth, with no sensor and no latency, so it bounds what
  any reactive grip loop can achieve. Fast: its rate over one tick above `omega_thresh` or `v_thresh` (a blow, a
  jerk). Slow: the pose, low-passed, drifting more than `drift_rot_thresh` or `drift_trans_thresh` from its
  reference (creep under a periodic or steady load, which is slower than the grasp's own elastic flexing).
- "taxel": what the hand's own sensors see. Fast: per patch carrying more than `load_floor`, the centre of pressure
  moving faster than `cop_speed_thresh` over `window`, or the patch losing more than `load_drop_thresh` of its load
  within `window` (the unloading that precedes slip on piezoresistive skins, Teshigawara et al. 2009); and any pad
  accelerometer deviating from its 5 ms baseline by more than `accel_thresh`. Slow: a patch's low-passed centre of
  pressure drifting more than `cop_drift_thresh` from its reference, or its pressure pattern turning by more than
  `pattern_rot_thresh` (principal axis of the low-passed second moments; a handle turning about the pad normal
  barely moves the centre of pressure but turns the elongated pattern it presses).
- "log": compute both detectors and record them, never act (threshold calibration); slow references never reset.

The slow signals pass two cascaded first-order low-passes of `drift_tau`, and their references reset at every
detection, so continued creep keeps re-triggering. A detection (fast ones sustained for `confirm` ticks) raises the
grip command at once, bypassing the grip loop's own rate, to the hand's pre-impact ceiling (`boost_to` = "pre", the
cap the anticipatory ramp uses) or its maximum; the boost holds for `hold_time` after the last detection, then ramps
back to the scheduled grip over `release_time`, or, latched (`latch`), decays toward it with `latch_tau`, keeping a
raised margin as a hand does after a slip. For `hold_time` + `release_time` after a detection the slow detectors are
blind (the squeeze itself moves the tool and the pressure pattern); then they re-reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tactile_sim.config import ReactiveCfg
from tactile_sim.logging.strike_metrics import rotvec_diff

MODES = ("off", "oracle", "taxel", "log")
TRACE_KEYS = ("t", "omega", "v", "drift_rot", "drift_trans", "cop_speed", "load_drop", "accel", "cop_drift",
              "pattern_rot", "boost")


@dataclass
class Detection:
    t: float  # first tick of a confirmed slip episode
    source: str  # the indicators over threshold, joined by "+"


class _LowPass2:
    """Two cascaded first-order low-passes."""

    def __init__(self, a: float):
        self.a = a
        self.y1 = self.y2 = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.y1 is None:
            self.y1, self.y2 = x.copy(), x.copy()
        else:
            self.y1 = self.y1 + self.a * (x - self.y1)
            self.y2 = self.y2 + self.a * (self.y1 - self.y2)
        return self.y2


class ReactiveSlip:
    def __init__(self, tb, cfg: ReactiveCfg | None = None, record: bool = False):
        self.tb = tb
        self.cfg = cfg or tb.cfg.reactive
        if self.cfg.mode not in MODES:
            raise ValueError(f"unknown reactive mode {self.cfg.mode!r} (expected one of {MODES})")
        self.dt = 1.0 / self.cfg.rate
        self.hand = tb.world.hand
        if self.cfg.boost_to not in ("pre", "max"):
            raise ValueError(f"unknown boost_to {self.cfg.boost_to!r} (expected 'pre' or 'max')")
        self.f_max = float(self.hand.grip_pre_max if self.cfg.boost_to == "pre" else self.hand.grip_force_max)
        self.acting = self.cfg.mode in ("oracle", "taxel")
        self.record = record or self.cfg.mode == "log"
        self.n_win = max(1, int(round(self.cfg.window / self.dt)))
        self.a_lp = 1.0 - np.exp(-self.dt / self.cfg.drift_tau)
        self._coords = [self._taxel_coords(k) for k in range(len(self.hand.patches))]
        self.reset()

    def _taxel_coords(self, k: int) -> np.ndarray:
        """Taxel centres in the patch body's frame, (n, 3)."""
        p = self.hand.patches[k]
        if p.taxel_pos is not None:
            return np.asarray(p.taxel_pos, dtype=float)
        nr, nc = self.hand.grid(k)
        hx, hz = p.half
        xs = -hx + (2 * np.arange(nr) + 1) * hx / nr
        zs = -hz + (2 * np.arange(nc) + 1) * hz / nc
        X, Z = np.meshgrid(xs, zs, indexing="ij")  # rows run along x, as in sensors.pressure.center_of_pressure
        return np.column_stack([X.ravel(), np.zeros(nr * nc), Z.ravel()])

    def reset(self) -> None:
        self.t_last = -np.inf  # last tick of a confirmed detection
        self.count = 0  # consecutive ticks a fast detector was over threshold
        self.boost = 0.0
        self._boosting = False
        self.ticks_fired = 0
        self.detections: list[Detection] = []
        self._prev_pose = None
        self._lp_p = _LowPass2(self.a_lp)
        self._lp_r = _LowPass2(self.a_lp)
        self._ref_pose = None
        self._hist: list[tuple[list[float], list[np.ndarray | None]]] = []
        self._lp_cop = [_LowPass2(self.a_lp) for _ in self.hand.patches]
        self._ref_cop: list[np.ndarray | None] = [None] * len(self.hand.patches)
        self._lp_mom = [_LowPass2(self.a_lp) for _ in self.hand.patches]
        self._ref_axis: list[np.ndarray | None] = [None] * len(self.hand.patches)
        self._acc_base: dict[str, np.ndarray] = {}
        self._acc_seen: dict[str, int] = {}
        self.trace = {k: [] for k in TRACE_KEYS} if self.record else None

    # ------------------------------------------------------------------ detectors
    def _oracle(self) -> tuple[float, float, float, float]:
        """(rate rad/s, rate m/s, slow drift rad, slow drift m)."""
        p, r = self.tb.world.hammer_in_hand()
        prev, self._prev_pose = self._prev_pose, (p, r)
        pf, rf = self._lp_p(p), self._lp_r(r)
        if self._ref_pose is None:
            self._ref_pose = (pf.copy(), rf.copy())
        drift_rot = rotvec_diff(self._ref_pose[1], rf)
        drift_trans = float(np.linalg.norm(pf - self._ref_pose[0]))
        if prev is None:
            return 0.0, 0.0, drift_rot, drift_trans
        return (rotvec_diff(prev[1], r) / self.dt, float(np.linalg.norm(p - prev[0])) / self.dt, drift_rot,
                drift_trans)

    def _axis(self, mom: np.ndarray) -> np.ndarray | None:
        """Principal axis of a pressure pattern's second moments, if the pattern is elongated enough to have one."""
        w, v = np.linalg.eigh(mom)
        return v[:, 2] if w[1] > 0 and w[2] > self.cfg.pattern_aniso * w[1] else None

    def _taxel(self) -> tuple[float, float, float, float, float]:
        """(centre-of-pressure speed m/s, relative load drop, accelerometer deviation m/s^2, slow CoP drift m,
        slow pattern rotation rad)."""
        sen = self.tb.sensors
        loads: list[float] = []
        cops: list[np.ndarray | None] = []
        cop_drift = pattern_rot = 0.0
        for k, name in enumerate(self.hand.pressure_names):
            s = sen[name].latest()
            if s.seq < 0:
                loads.append(0.0)
                cops.append(None)
                continue
            f = np.clip(np.asarray(s.value, dtype=float), 0.0, None)
            load = float(f.sum())
            loads.append(load)
            cop = f @ self._coords[k] / load if load > self.cfg.load_floor else None
            cops.append(cop)
            if cop is not None:
                cf = self._lp_cop[k](cop)
                if self._ref_cop[k] is None:
                    self._ref_cop[k] = cf.copy()
                cop_drift = max(cop_drift, float(np.linalg.norm(cf - self._ref_cop[k])))
                dx = self._coords[k] - cop
                mom = self._lp_mom[k](((dx * f[:, None]).T @ dx / load).ravel()).reshape(3, 3)
                ax = self._axis(mom)
                if ax is not None:
                    if self._ref_axis[k] is None:
                        self._ref_axis[k] = ax
                    pattern_rot = max(pattern_rot, float(np.arccos(np.clip(abs(ax @ self._ref_axis[k]), 0.0, 1.0))))
        self._hist.append((loads, cops))
        if len(self._hist) > self.n_win + 1:
            self._hist.pop(0)
        cop_speed = load_drop = 0.0
        if len(self._hist) > self.n_win:
            loads0, cops0 = self._hist[0]
            span = self.n_win * self.dt
            for k in range(len(loads)):
                if cops[k] is not None and cops0[k] is not None:
                    cop_speed = max(cop_speed, float(np.linalg.norm(cops[k] - cops0[k])) / span)
                if loads0[k] > self.cfg.load_floor:
                    load_drop = max(load_drop, (loads0[k] - loads[k]) / loads0[k])
        acc_dev = 0.0
        for name in self.hand.accel_names:
            if name not in sen:
                continue
            sensor = sen[name]
            n_new = len(sensor.values) - self._acc_seen.get(name, 0)
            self._acc_seen[name] = len(sensor.values)
            if n_new <= 0:
                continue
            win = np.asarray(sensor.values[-n_new:], dtype=float)
            base = self._acc_base.get(name)
            if base is None:
                base = win[0].copy()
            acc_dev = max(acc_dev, float(np.max(np.linalg.norm(win - base, axis=1))))
            a = 1.0 - np.exp(-sensor.period / 0.005)  # 5 ms baseline: swing accelerations pass, jolts do not
            for row in win:
                base = base + a * (row - base)
            self._acc_base[name] = base
        return cop_speed, load_drop, acc_dev, cop_drift, pattern_rot

    def _rebase(self) -> None:
        """A detection: the slow detectors measure drift from here on."""
        if self._lp_p.y2 is not None:
            self._ref_pose = (self._lp_p.y2.copy(), self._lp_r.y2.copy())
        self._ref_cop = [None if lp.y2 is None else lp.y2.copy() for lp in self._lp_cop]
        self._ref_axis = [None] * len(self._ref_axis)  # re-acquired from the filtered pattern at the next tick

    # ------------------------------------------------------------------ 1 kHz tick
    def tick(self, t: float) -> None:
        c = self.cfg
        if c.mode == "off":
            return
        om = v = dr = dtr = cs = ld = ad = cd = pr = 0.0
        if c.mode in ("oracle", "log"):
            om, v, dr, dtr = self._oracle()
        if c.mode in ("taxel", "log"):
            cs, ld, ad, cd, pr = self._taxel()
        if c.mode == "oracle":
            fast = [("rate", om > c.omega_thresh or v > c.v_thresh)]
            slow = [("drift", dr > c.drift_rot_thresh or dtr > c.drift_trans_thresh)]
        elif c.mode == "taxel":
            fast = [("cop", cs > c.cop_speed_thresh), ("load", ld > c.load_drop_thresh),
                    ("accel", ad > c.accel_thresh)]
            slow = [("cop-drift", cd > c.cop_drift_thresh), ("pattern-rot", pr > c.pattern_rot_thresh)]
        else:
            fast, slow = [], []
        if t - self.t_last <= c.hold_time + c.release_time:
            # the loop's own squeeze moves the tool and the pressure pattern: the slow detectors are blind while it
            # steps up and settles, and measure from where the grasp settled afterwards
            slow = []
            self._boosting = True
        elif self._boosting:
            self._boosting = False
            self._rebase()
        self.count = self.count + 1 if any(on for _, on in fast) else 0
        src = [n for n, on in fast if on] if self.count >= c.confirm else []
        src += [n for n, on in slow if on]
        if src:
            self.ticks_fired += 1
            if t - self.t_last > c.hold_time:
                self.detections.append(Detection(t, "+".join(src)))
            self.t_last = t
            if any(on for _, on in slow):
                self._rebase()
        since = t - self.t_last
        if since <= c.hold_time:
            self.boost = self.f_max
        elif c.latch:
            self.boost = self.f_max * float(np.exp(-(since - c.hold_time) / c.latch_tau))
        elif since <= c.hold_time + c.release_time:
            self.boost = self.f_max * (1.0 - (since - c.hold_time) / c.release_time)
        else:
            self.boost = 0.0
        if self.acting:
            self.tb.grip_boost = self.boost
            if self.boost > getattr(self.tb.l1, "grip_setpoint", 0.0):
                self.tb.world.set_grip_force(self.boost)  # act now, not at the grip loop's next tick
        if self.trace is not None:
            for key, val in zip(TRACE_KEYS, (t, om, v, dr, dtr, cs, ld, ad, cd, pr, self.boost), strict=True):
                self.trace[key].append(val)

    def detections_between(self, t0: float, t1: float) -> list[Detection]:
        return [d for d in self.detections if t0 <= d.t < t1]
