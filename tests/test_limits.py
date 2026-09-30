"""Hardware limits are part of the test: the FR3's torque, torque-rate, velocity and range limits and the
payload rating must hold through a strike episode (see tactile_sim.limits)."""

import numpy as np
import pytest

from tactile_sim.behaviors.trajectories import eval_poly, strike_profile
from tactile_sim.config import fast_config
from tactile_sim.episode import Episode
from tactile_sim.limits import FR3_TORQUE_RATE, FR3_VELOCITY, check_payload, rate_limit


def test_rate_limit_clips_each_joint():
    prev = np.zeros(7)
    tau = np.array([5.0, -5.0, 0.5, 0, 0, 0, 2.0])
    out = rate_limit(tau, prev, 1000.0, 0.001)
    assert np.allclose(out, [1.0, -1.0, 0.5, 0, 0, 0, 1.0])


def test_strike_profile_arrives_braking_within_its_caps():
    T, c = strike_profile(0.21, 1.7, 12.0, 20.0, 2.1)
    p, v, a = eval_poly(c, T)
    assert p == pytest.approx(0.21) and v == pytest.approx(1.7) and a == pytest.approx(-12.0)
    ts = np.linspace(0, T, 400)
    pva = np.array([eval_poly(c, t) for t in ts])
    assert pva[:, 1].min() >= -1e-6 and pva[:, 1].max() <= 2.1 + 1e-6 and np.abs(pva[:, 2]).max() <= 20.0 + 1e-6
    assert strike_profile(0.21, 2.1, 12.0, 20.0, 2.1) is None  # braking into contact needs a peak above v_c


@pytest.fixture(scope="module")
def franka_episode():
    ep = Episode(fast_config(), seed=0)
    return ep, ep.run(3)


def test_franka_episode_holds_every_fr3_limit(franka_episode):
    ep, res = franka_episode
    s = res.summary
    assert s["hit_rate"] == 1.0 and s["drops"] == 0
    assert s["limit_violations"] == "", ep.limits.violations()
    for key in ("limit_arm_torque", "limit_arm_torque_rate", "limit_arm_velocity", "limit_arm_range"):
        assert s[key] <= 1.0 + 1e-6, key
    assert s["limit_payload"] < 1.0


def test_commanded_torque_never_steps_faster_than_the_fr3_allows(franka_episode):
    _, res = franka_episode
    tau = np.asarray(res.testbed.l1.log.tau)
    rate = np.abs(np.diff(tau, axis=0)) * res.testbed.cfg.controller.rate
    assert rate.max() <= FR3_TORQUE_RATE * (1 + 1e-6)


def test_measured_joint_speed_stays_under_the_velocity_reflex(franka_episode):
    _, res = franka_episode
    qd = np.asarray(res.testbed.l1.log.qd)
    assert np.all(np.abs(qd) <= FR3_VELOCITY)


def test_swing_speed_is_capped_by_the_joint_velocity_limits(franka_episode):
    ep, res = franka_episode
    sw, cfg = ep.swing, ep.cfg.swing
    assert 1.5 < sw.v_cap < 4.0
    for plan in sw.plans:
        assert plan.v_strike <= cfg.v_margin * sw.v_cap + 1e-9


def test_payload_rating(franka_episode):
    _, res = franka_episode
    assert 0.3 < check_payload(res.testbed.world) < 1.0  # F/T body + Franka Hand + 665 g hammer vs 3 kg
