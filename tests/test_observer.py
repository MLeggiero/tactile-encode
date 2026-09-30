"""M4: momentum observer and impact detection."""

import numpy as np
import pytest

from tactile_sim.config import SimConfig, fast_config
from tactile_sim.control.momentum_observer import ImpactDetector
from tactile_sim.sim.testbed import Testbed


def test_observer_quiet_at_rest():
    tb = Testbed(fast_config())
    tb.l2_callbacks.append(lambda t: setattr(tb.l1.cmd, "t", t))
    tb.run_for(0.3)
    assert np.linalg.norm(tb.l1.f_ext[:3]) < 1.0


def test_observer_matches_applied_push():
    tb = Testbed(fast_config())
    tb.l2_callbacks.append(lambda t: setattr(tb.l1.cmd, "t", t))
    tb.run_for(0.1)
    tb.world.data.xfrc_applied[tb.world.hand_body, :3] = [0.0, 20.0, 0.0]
    tb.run_for(0.4)
    assert tb.l1.f_ext[1] == pytest.approx(20.0, rel=0.15)
    assert abs(tb.l1.f_ext[0]) < 2 and abs(tb.l1.f_ext[2]) < 2


def _approach_strike(cfg, sources, v=1.0):
    tb = Testbed(cfg)
    c = cfg.controller
    tb.l1.detector = ImpactDetector(tb.l1.axis, c.impact_force_thresh, c.impact_ft_thresh, c.impact_refractory,
                                    accel_thresh=c.impact_accel_thresh, sources=sources)
    cmd = tb.l1.cmd
    cmd.K = np.array([1500.0, 2500, 1500, 60, 60, 60])
    p0 = cmd.x_eq.copy()
    t0 = tb.t + 0.02
    truth = {"t_c": None}

    def l2(t):
        cmd.t = t
        if t > t0:  # smooth start (constant jerk-free ramp over 20 ms), then constant speed
            s = t - t0
            cmd.x_eq = p0 + np.array([0, v * s**2 / 0.04 if s < 0.02 else v * (s - 0.01), 0])
            cmd.xd_eq = np.array([0, v * min(1.0, s / 0.02), 0, 0, 0, 0])
        if t > t0 + 0.04:
            tb.l1.detector.arm(True) if not tb.l1.detector.events else None

    def watch(b):
        if truth["t_c"] is None and b.world.plant.contact_active():
            truth["t_c"] = b.t

    tb.l2_callbacks.append(l2)
    tb.run_for(0.2, watch)
    ev = tb.l1.detector.events
    return truth["t_c"], (ev[0] if ev else None), tb


@pytest.mark.parametrize("cfg", [fast_config(), SimConfig()], ids=["4kHz", "8kHz"])
def test_impact_flag_within_2ms(cfg):
    t_c, ev, tb = _approach_strike(cfg, ("observer", "ft", "accel"))
    assert t_c is not None and ev is not None
    assert 0.0 <= ev.t_flag - t_c <= 0.002, (ev.source, ev.t_flag - t_c)
    assert ev.source == "accel"
    # velocity feedback is gated for the configured window after the flag
    L = tb.l1.log
    t = np.array(L.t)
    gated = np.array(L.gated)
    win = (t >= ev.t_flag) & (t < ev.t_flag + cfg.controller.gate_duration - 0.0015)
    assert gated[win].all() and not gated[t > ev.t_flag + cfg.controller.gate_duration + 0.0015].any()


def test_joint_side_observer_lags_through_compliant_grasp():
    """The joint-torque observer alone sees the blow only after the pads and wrist pass it on."""
    t_c, ev, _ = _approach_strike(fast_config(), ("observer",))
    assert ev is not None and ev.source == "observer"
    assert 0.001 < ev.t_flag - t_c < 0.008


def _shake_events(sources, smooth):
    cfg = fast_config()
    tb = Testbed(cfg)
    c = cfg.controller
    tb.l1.detector = ImpactDetector(tb.l1.axis, c.impact_force_thresh, c.impact_ft_thresh, c.impact_refractory,
                                    accel_thresh=c.impact_accel_thresh, sources=sources)
    tb.l1.detector.arm(True)
    cmd = tb.l1.cmd
    p0 = cmd.x_eq.copy()
    w = 2 * np.pi * 5

    def l2(t):  # 3 cm back-and-forth at 5 Hz, never reaching the nail
        cmd.t = t
        a = min(1.0, t / 0.1) if smooth else 1.0  # smooth: amplitude ramps in over 100 ms
        cmd.x_eq = p0 + np.array([0, -0.03 * a * np.sin(w * t), 0])
        cmd.xd_eq = np.array([0, -0.03 * a * w * np.cos(w * t), 0, 0, 0, 0])

    tb.l2_callbacks.append(l2)
    tb.run_for(0.4)
    return tb.l1.detector.events


def test_detector_ignores_swing_accelerations():
    assert _shake_events(("observer", "ft", "accel"), smooth=True) == []


def test_accel_and_observer_ignore_an_abrupt_start():
    """A velocity step rattles the tool in the pads; the F/T slope can read that as a small blow, which is
    why the detector is only armed just before the predicted contact. The other channels stay quiet."""
    assert _shake_events(("observer", "accel"), smooth=False) == []
