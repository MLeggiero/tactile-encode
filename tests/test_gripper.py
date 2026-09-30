"""M5: grasp-force loop, slip and drop detection."""

import numpy as np
import pytest

from tactile_sim.config import fast_config
from tactile_sim.logging.strike_metrics import grasp_slip, rotvec_diff
from tactile_sim.sensors.pressure import grip_force
from tactile_sim.sim.testbed import Testbed


def test_noise_floor_removes_rectified_offset():
    rng = np.random.default_rng(0)
    tax = np.clip(np.r_[20.0, 20.0, np.zeros(14)] + rng.normal(0, 0.3, (2000, 16)), 0, None)
    raw = tax.sum(axis=1).mean()
    floored = np.mean([grip_force(t, 0.9) for t in tax])
    assert raw - 40 > 1.0  # half-rectified noise biases the raw sum upward
    assert abs(floored - 40) < 0.2


def test_rotvec_diff():
    assert rotvec_diff(np.zeros(3), np.array([0, 0, 0.3])) == pytest.approx(0.3)
    assert rotvec_diff(np.array([0.1, 0, 0]), np.array([0.1, 0, 0])) == pytest.approx(0.0, abs=1e-9)
    t, r = grasp_slip(np.zeros(3), np.zeros(3), np.array([0.003, 0.004, 0]), np.array([0, 0.02, 0]))
    assert t == pytest.approx(0.005) and r == pytest.approx(0.02)


def test_grip_holds_setpoint_and_settles():
    tb = Testbed(fast_config())
    tb.l2_callbacks.append(lambda t: setattr(tb.l1.cmd, "t", t))
    tb.run_for(0.2)
    truth = tb.world.pad_normal_forces()
    assert np.allclose(truth, tb.cfg.controller.grip_hold, atol=2.0)
    assert tb.grip.measured == pytest.approx(truth.mean(), abs=1.5)
    tb.l1.cmd.F_grip = 80.0
    f = []
    tb.run_for(0.1, lambda b: f.append(b.world.pad_normal_forces().mean()))
    f = np.array(f)
    t = np.arange(1, len(f) + 1) * tb.world.dt
    outside = np.nonzero(np.abs(f - 80.0) > 0.05 * 80.0)[0]
    t_settle = t[outside[-1]] if len(outside) else 0.0
    assert t_settle < 0.050


def _shake(grip, T=1.0):
    tb = Testbed(fast_config())
    cmd = tb.l1.cmd
    p0 = cmd.x_eq.copy()
    cmd.F_grip = grip
    cmd.K = np.array([3000.0, 3000, 3000, 100, 100, 100])
    w = 2 * np.pi * 5.0
    amp = 0.02  # 2 cm at 5 Hz: ~2 g peak on the hand, along gravity (tangential to the pads)

    def l2(t):
        cmd.t = t
        cmd.x_eq = p0 + np.array([0, 0, amp * np.sin(w * t)])
        cmd.xd_eq = np.array([0, 0, amp * w * np.cos(w * t), 0, 0, 0])
        cmd.xdd_ff = np.array([0, 0, -amp * w * w * np.sin(w * t), 0, 0, 0])

    tb.l2_callbacks.append(l2)
    tb.run_for(0.3)
    p_a, r_a = tb.world.hammer_in_hand()
    tb.run_for(T)
    p_b, r_b = tb.world.hammer_in_hand()
    return grasp_slip(p_a, r_a, p_b, r_b), tb


def test_firm_grip_survives_shake():
    """The default hold force (55 N, under the Franka Hand's 70 N continuous rating) survives a 2 g shake.
    At 40 N the real hand's 17 mm pads let the hammer twist ~7 deg."""
    (slip_t, slip_r), tb = _shake(fast_config().controller.grip_hold)
    assert slip_t < 0.007 and np.degrees(slip_r) < 1.4
    assert not tb.grip.dropped


def test_weak_grip_slips():
    (slip_t, slip_r), tb = _shake(10.0)
    assert slip_t > 0.007 or np.degrees(slip_r) > 1.4
    assert not tb.grip.dropped


def test_drop_detected_and_arm_frozen():
    (_, _), tb = _shake(2.0)
    assert tb.grip.dropped
    assert tb.l1.supervisor.frozen
    assert np.all(np.isfinite(tb.world.data.qpos[tb.world.arm_qadr]))


def test_commanded_release_is_not_a_drop():
    tb = Testbed(fast_config())
    tb.l2_callbacks.append(lambda t: setattr(tb.l1.cmd, "t", t))
    tb.run_for(0.1)
    tb.l1.cmd.F_grip = 20.0
    tb.run_for(0.3)
    assert not tb.grip.dropped
    assert tb.world.pad_normal_forces().mean() == pytest.approx(20.0, abs=2.0)
