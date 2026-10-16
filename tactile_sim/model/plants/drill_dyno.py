"""A virtual dyno for the drill's drivetrain (plants/drivetrain.py): the bench tests a drill's published ratings come
from, run on the model alone, so its nominal values can be checked against (and its spindle inertia fitted to) the
ratings, without a robot or MuJoCo.

- no-load speed at a full trigger;
- ISO 5393 joints: the tool runs free, then tightens a joint whose torque rises from 10 % to 100 % of the test
  torque over 27 deg (hard) or 650 deg (soft); the test torque is the tool's own peak (iterated to a fixed point).
  A drill's "hard" and "soft" ratings are these peaks with the clutch locked (drill mode);
- the clutch on a locked chuck: its mean and peak torque, the spindle speed it slips at, and the click rate;
- the brake: the time to stop from no-load speed when the trigger is released;
- the housing's angular momentum balance;
- ISO 5349-1 Wh frequency weighting for hand-transmitted vibration (the a_h a drill's manual declares).

    python -m tactile_sim.calibrate drill
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tactile_sim.model.plants.drivetrain import Drivetrain, DrivetrainParams

DT = 1e-4


def no_load_speed(p: DrivetrainParams, t: float = 1.0, dt: float = DT) -> float:
    dr = Drivetrain(p)
    for _ in range(int(t / dt)):
        dr.step(1.0, 0.0, dt)
    return dr.w


def joint_peak(p: DrivetrainParams, test_torque: float, angle_deg: float, dt: float = DT, t_max: float = 20.0) -> float:
    """Peak chuck torque on an ISO 5393 joint: 10 % to 100 % of test_torque over angle_deg, the tool at no-load speed
    when it engages (a light run-down)."""
    k = 0.9 * test_torque / np.radians(angle_deg)
    th0 = 0.1 * test_torque / k  # the joint starts at 10 % of the test torque once snug
    dr = Drivetrain(p)
    dr.w = p.free_speed
    dr.u = 1.0
    peak = 0.0
    for _ in range(int(t_max / dt)):
        dr.step(1.0, k * (th0 + dr.out_angle), dt)
        peak = max(peak, dr.tau_chuck)
        # stopped (a hard joint) or settled on the stall (a soft one), or the clutch ratchets
        if dr.w <= 1e-3 * p.free_speed or (dr.slipping and dr.clicks > 3):
            break
    return peak


def iso5393(p: DrivetrainParams, angle_deg: float, iters: int = 6) -> float:
    """The tool's ISO 5393 torque on a joint of this rate (its own peak as the test torque)."""
    T = p.stall_torque
    for _ in range(iters):
        T = joint_peak(p, T, angle_deg)
    return T


def hard_torque(p: DrivetrainParams) -> float:
    return iso5393(replace(p, clutch_lock=True), 27.0)


def soft_torque(p: DrivetrainParams) -> float:
    return iso5393(replace(p, clutch_lock=True), 650.0)


def fit_spindle_inertia(p: DrivetrainParams, hard: float, lo: float = 2e-4, hi: float = 5e-2) -> float:
    """The spindle inertia that gives this hard-joint torque (the soft joint does not depend on it)."""
    from scipy.optimize import brentq

    return float(brentq(lambda J: hard_torque(replace(p, spindle_inertia=J)) - hard, lo, hi, xtol=1e-6))


def clutch_test(p: DrivetrainParams, t: float = 1.0, dt: float = DT) -> dict[str, float]:
    """A full trigger into a locked chuck: the clutch ratchets."""
    dr = Drivetrain(p)
    tau, w = [], []
    n = int(t / dt)
    clicks0 = None
    for i in range(n):
        dr.step(1.0, 1e3, dt)
        if i >= n // 2:  # past the spin-up
            clicks0 = dr.clicks if clicks0 is None else clicks0
            tau.append(dr.tau_chuck)
            w.append(dr.w)
    t_half = (n - n // 2) * dt
    return {"mean_torque": float(np.mean(tau)), "peak_torque": float(np.max(tau)), "slip_speed": float(np.mean(w)),
            "click_rate": (dr.clicks - clicks0) / t_half, "click_rate_model": p.click_rate(float(np.mean(w)))}


def brake_stop_time(p: DrivetrainParams, dt: float = DT) -> float:
    dr = Drivetrain(p)
    dr.w = p.free_speed
    for i in range(int(5.0 / dt)):
        dr.step(0.0, 0.0, dt)
        if dr.w < 0.01 * p.free_speed:
            return (i + 1) * dt
    return np.inf


def momentum_residual(p: DrivetrainParams, seed: int = 0, t: float = 2.0, dt: float = DT) -> float:
    """|int (tau_housing - tau_chuck) dt - J_am (w_end - w_0)| over a random trigger and load sequence (should be 0)."""
    rng = np.random.default_rng(seed)
    dr = Drivetrain(p)
    acc = 0.0
    trig, need = 0.0, 0.0
    for i in range(int(t / dt)):
        if i % 500 == 0:
            trig = float(rng.choice([0.0, 0.4, 1.0]))
            need = float(rng.uniform(0.0, 2.0 * p.clutch_torque))
        dr.step(trig, need, dt)
        acc += (dr.tau_housing - dr.tau_chuck) * dt
    return abs(acc - p.momentum_inertia * dr.w)


def wh_filter(fs: float):
    """ISO 5349-1 / ISO 8041 Wh (hand-arm) weighting as a digital filter (b, a) at sample rate fs: band limits
    6.31 Hz - 1258.9 Hz, then the acceleration-velocity transition at 15.915 Hz (Q 0.64)."""
    from scipy.signal import bilinear

    w1, w2, w3, w4 = (2 * np.pi * f for f in (6.310, 1258.9, 15.915, 15.915))
    q = 1 / np.sqrt(2)
    hp = (np.array([1.0, 0.0, 0.0]), np.array([1.0, w1 / q, w1 ** 2]))
    lp = (np.array([w2 ** 2]), np.array([1.0, w2 / q, w2 ** 2]))
    tr = (np.array([w4 ** 2 / w3, w4 ** 2]), np.array([1.0, w4 / 0.64, w4 ** 2]))
    b = np.polymul(np.polymul(hp[0], lp[0]), tr[0])
    a = np.polymul(np.polymul(hp[1], lp[1]), tr[1])
    return bilinear(b, a, fs)


def wh_rms(acc: np.ndarray, fs: float, skip: float = 0.2) -> float:
    """The vibration total value a_hv (m/s^2): the root sum of squares of the Wh-weighted RMS of each axis of
    `acc` (n x 3, m/s^2, gravity included: the weighting removes it), past the filter's first `skip` seconds."""
    from scipy.signal import lfilter

    b, a = wh_filter(fs)
    y = lfilter(b, a, np.asarray(acc, float) - np.asarray(acc, float)[0], axis=0)[int(skip * fs):]
    return float(np.sqrt(np.sum(np.mean(y ** 2, axis=0))))


def report(p: DrivetrainParams, hard_spec: float | None = None, soft_spec: float | None = None) -> list[str]:
    ct = clutch_test(p)
    rows = [f"no-load speed      {no_load_speed(p):6.2f} rad/s  ({no_load_speed(p) * 60 / (2 * np.pi):5.0f} rpm)",
            f"ISO 5393 soft      {soft_torque(p):6.2f} Nm" + (f"   (rating {soft_spec:g})" if soft_spec else ""),
            f"ISO 5393 hard      {hard_torque(p):6.2f} Nm" + (f"   (rating {hard_spec:g})" if hard_spec else ""),
            f"clutch {p.clutch_torque:4.2f} Nm    mean {ct['mean_torque']:5.2f}  peak {ct['peak_torque']:5.2f} Nm  "
            f"slips at {ct['slip_speed']:5.1f} rad/s  clicks {ct['click_rate']:5.1f}/s "
            f"(model {ct['click_rate_model']:5.1f})",
            f"brake stop         {brake_stop_time(p) * 1e3:6.0f} ms",
            f"J_spin {p.spindle_inertia:.2e}  J_am {p.momentum_inertia:.2e} kg m^2  "
            f"momentum residual {momentum_residual(p):.1e} N m s"]
    return rows
