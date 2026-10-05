"""M4: L1 Cartesian impedance, reference limiter, stiffness slew, supervisor."""

import numpy as np
import pytest

from tactile_sim.config import fast_config
from tactile_sim.control.interface import Mode
from tactile_sim.sim.testbed import Testbed


def keep_fresh(tb):
    tb.l2_callbacks.append(lambda t: setattr(tb.l1.cmd, "t", t))


@pytest.fixture(scope="module")
def tb():
    return Testbed(fast_config())


def test_holds_pose(tb):
    tb.reset(0)
    keep_fresh(tb)
    x0 = tb.l1.cmd.x_eq.copy()
    tb.run_for(1.0)
    assert np.linalg.norm(tb.world.tcp_pose()[0] - x0) < 0.001
    tb.l2_callbacks.clear()


def _step_response(fric_scale, dist=0.05):
    tb = Testbed(fast_config())
    tb.world.model.dof_frictionloss[tb.world.arm_dofs] *= fric_scale
    tb.reset(0)
    keep_fresh(tb)
    tb.run_for(0.2)
    p0 = tb.world.tcp_pose()[0].copy()
    tb.l1.cmd.x_eq = p0 + np.array([dist, 0, 0])
    xs = []
    tb.run_for(1.0, lambda b: xs.append(b.world.tcp_pose()[0][0] - p0[0]))
    return np.array(xs)


def test_step_response_without_friction():
    xs = _step_response(0.0)
    assert xs[-1] == pytest.approx(0.05, rel=0.02)
    assert xs.max() <= 0.05 * 1.10  # overshoot < 10 %


def test_step_response_with_joint_friction_stays_in_friction_band():
    xs = _step_response(1.0)
    assert 0.80 * 0.05 < xs[-1] <= 0.05 * 1.10


def test_reference_limiter_bounds_spring_force(tb):
    tb.reset(0)
    keep_fresh(tb)
    p0 = tb.world.tcp_pose()[0].copy()
    tb.l1.cmd.x_eq = p0 + np.array([0.0, 0.5, 0.0])  # 50 cm away
    tb.run_for(0.01)
    c = tb.cfg.controller
    fs = tb.l1.last.F_spring[:3]
    assert np.linalg.norm(fs) <= max(c.k_trans) * c.delta_max_pos * 1.001
    tb.l2_callbacks.clear()


def test_stiffness_slew_rate(tb):
    tb.reset(0)
    keep_fresh(tb)
    tb.run_for(0.05)
    k0 = tb.l1.impedance.K_t[:3].copy()
    tb.l1.cmd.K = np.array([6000.0, 6000, 6000, 60, 60, 60])
    tb.run_for(0.02)
    rate = (tb.l1.impedance.K_t[0] - k0[0]) / 0.02
    assert rate == pytest.approx(tb.cfg.controller.k_dot_max, rel=0.1)
    tb.l2_callbacks.clear()


def test_stale_command_falls_back_to_hold(tb):
    tb.reset(0)
    p0 = tb.world.tcp_pose()[0].copy()
    tb.l1.cmd.x_eq = p0 + np.array([0.0, 0.1, 0.0])  # never refreshed -> goes stale after 20 ms
    tb.run_for(0.2)
    assert tb.l1.supervisor.stale
    assert tb.l1.log.mode[-1] == Mode.HOLD
    assert np.linalg.norm(tb.world.tcp_pose()[0] - p0) < 0.02


def test_torque_saturation(tb):
    w = tb.world
    tau = w.set_arm_torque(np.full(7, 1e4))
    assert np.allclose(tau, w.tau_limit)
