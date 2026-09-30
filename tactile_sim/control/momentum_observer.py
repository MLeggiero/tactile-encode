"""Generalized-momentum observer and impact detector.

Observer (De Luca et al.), in discrete time at the L1 rate:
    p = M(q) qd,   pdot = tau_m + tau_passive + tau_fric - (qfrc_bias - Mdot qd) + tau_ext
    r_k = K_O [ p_k - p_0 - sum_j ( tau_m + tau_passive + tau_fric - qfrc_bias + Mdot qd + r_{j-1} ) dt ]
r converges to tau_ext with a first-order lag of time constant 1/K_O. MuJoCo's qfrc_bias is
gravity + Coriolis, and C + C^T = Mdot gives the (qfrc_bias - Mdot qd) term. Joint dry friction is
a MuJoCo constraint (not a passive force), so the friction model (its generalized force) is passed in
explicitly; scaling it away from the truth is how model mismatch is randomized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MomentumObserver:
    def __init__(self, n: int, gain: float, dt: float):
        self.n = n
        self.K = np.full(n, float(gain))
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self.r = np.zeros(self.n)
        self.integral = np.zeros(self.n)
        self.p0: np.ndarray | None = None
        self.M_prev: np.ndarray | None = None

    def update(self, M: np.ndarray, qd: np.ndarray, tau_m: np.ndarray, bias: np.ndarray,
               passive: np.ndarray, friction: np.ndarray) -> np.ndarray:
        p = M @ qd
        if self.p0 is None:
            self.p0 = p.copy()
            self.M_prev = M.copy()
            return self.r
        Mdot = (M - self.M_prev) / self.dt
        self.M_prev = M.copy()
        self.integral += (tau_m + passive + friction - bias + Mdot @ qd + self.r) * self.dt
        self.r = self.K * (p - self.p0 - self.integral)
        return self.r


def external_wrench(J: np.ndarray, r: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    """Least-squares wrench at the TCP consistent with tau_ext = J^T F."""
    JJt = J @ J.T
    return np.linalg.solve(JJt + damping * np.eye(6), J @ r)


@dataclass
class ImpactEvent:
    t_flag: float
    source: str
    peak_load: float = 0.0


class ImpactDetector:
    """Flags an impact from any of three channels and latches the peak load.

    - "observer": strike-axis component of the observer's external force exceeds `force_thresh`
    - "ft": the wrist F/T force along the strike axis departs from its running baseline (time constant
      `ft_tau`, 10 ms) by more than `ft_thresh`. With compliant pads the tool rings in the grasp at ~65 Hz
      during ordinary swings, so a slope test cannot separate swings from blows; a blow moves the force
      60-80 N within ~2 ms, a swing or the ring well under 30 N.
    - "accel": any pad-accelerometer sample since the last tick deviates from a slow (~5 ms) baseline
      by more than `accel_thresh`; swing accelerations change slowly, impacts do not
    A refractory period keeps post-impact ringing from re-triggering. The detector only fires while
    armed (the swing layer arms it ahead of the predicted contact).
    """

    def __init__(self, strike_axis: np.ndarray, force_thresh: float, ft_thresh: float, refractory: float,
                 accel_thresh: float = 60.0, sources: tuple[str, ...] = ("observer", "ft", "accel"),
                 accel_alpha: float = 0.2, ft_tau: float = 0.010):
        self.axis = np.asarray(strike_axis, dtype=float)
        self.force_thresh = force_thresh
        self.ft_thresh = ft_thresh
        self.accel_thresh = accel_thresh
        self.accel_alpha = accel_alpha
        self.ft_tau = ft_tau
        self.refractory = refractory
        self.sources = sources
        self.reset()

    def reset(self) -> None:
        self.armed = False
        self.events: list[ImpactEvent] = []
        self.last_flag_t = -np.inf
        self.t_prev: float | None = None
        self.ft_base: float | None = None
        self.acc_base: np.ndarray | None = None
        self.current: ImpactEvent | None = None

    def arm(self, armed: bool = True) -> None:
        self.armed = armed

    def update(self, t: float, f_ext_obs: np.ndarray | None, f_ft_world: np.ndarray | None,
               pad_acc: np.ndarray | None, t_ft: float | None = None) -> ImpactEvent | None:
        """`t_ft` is the F/T sample's own timestamp; the slope uses sample times, not controller ticks."""
        fired: str | None = None
        load = 0.0
        if f_ext_obs is not None:
            # the tool pushes on the nail along +axis; the reaction on the robot is along -axis
            load = max(load, float(-f_ext_obs[:3] @ self.axis))
            if "observer" in self.sources and load > self.force_thresh:
                fired = "observer"
        if f_ft_world is not None:
            f_ax = float(f_ft_world @ self.axis)
            ts = t if t_ft is None else t_ft
            if self.t_prev is None or ts > self.t_prev:  # only on a new sample
                if self.ft_base is None:
                    self.ft_base = f_ax
                dev = abs(f_ax - self.ft_base)
                if fired is None and "ft" in self.sources and dev > self.ft_thresh:
                    fired = "ft"
                dt = ts - self.t_prev if self.t_prev is not None else 0.0
                self.ft_base += (1.0 - np.exp(-dt / self.ft_tau)) * (f_ax - self.ft_base)
                self.t_prev = ts
        if pad_acc is not None and len(pad_acc):
            win = np.atleast_2d(pad_acc)
            if self.acc_base is None:
                self.acc_base = win.mean(axis=0)
            dev = float(np.max(np.linalg.norm(win - self.acc_base, axis=1)))
            if fired is None and "accel" in self.sources and dev > self.accel_thresh:
                fired = "accel"
            self.acc_base = (1 - self.accel_alpha) * self.acc_base + self.accel_alpha * win.mean(axis=0)
        if self.current is not None:
            self.current.peak_load = max(self.current.peak_load, load)
        if fired and self.armed and t - self.last_flag_t > self.refractory:
            ev = ImpactEvent(t, fired, load)
            self.events.append(ev)
            self.current = ev
            self.last_flag_t = t
            self.armed = False
            return ev
        return None
