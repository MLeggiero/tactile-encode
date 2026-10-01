"""Dexmate Vega U with WUJI Hand 2 on both arms: model, the position-command interface, the arc strike, and
the hardware limits through a strike episode (and that the Vega-1P variant still builds)."""

import numpy as np
import pytest

from tactile_sim.assets.fetch_hands import hand_available
from tactile_sim.behaviors.trajectories import StrikeGeometry
from tactile_sim.config import fast_config, hand_config

requires_vega = pytest.mark.skipif(not (hand_available("vega_1u") and hand_available("wuji2")
                                        and hand_available("wuji2_left")), reason="Vega U / WUJI not cached")


def vega_cfg(**over):
    return hand_config("wuji2", fast_config(), robot="vega_1u", **over)


def test_arc_geometry_is_consistent():
    R = np.diag([1.0, -1.0, -1.0])
    g = StrikeGeometry(np.array([0.4, 0.0, 0.8]), R, np.array([-0.19, -0.067, 0.0]), 0.06, radius=0.5)
    x_c, R_c = g.pose(0.06)
    assert np.allclose(x_c, [0.4, 0.0, 0.8]) and np.allclose(R_c, R)  # nominal pose at contact
    face = lambda s: g.pose(s)[0] + g.pose(s)[1] @ np.array([-0.19, -0.067, 0.0])  # noqa: E731
    ds = 1e-5
    v_face = (face(0.06 + ds) - face(0.06 - ds)) / (2 * ds)
    assert np.allclose(v_face, g.axis, atol=1e-6)  # the face meets the nail square, at unit speed per unit s
    for s in (-0.15, 0.0, 0.03):
        dx, w, _ = g.tangent(s)
        assert np.allclose(dx, (g.pose(s + ds)[0] - g.pose(s - ds)[0]) / (2 * ds), atol=1e-6)
        assert g.s_of(g.pose(s)[0]) == pytest.approx(s, abs=1e-9)
    line = StrikeGeometry(np.array([0.4, 0.0, 0.8]), R, np.zeros(3), 0.06)
    assert np.allclose(line.pose(0.0)[0], [0.4, -0.06, 0.8])  # straight strike: hover = contact - clearance


@pytest.fixture(scope="module")
def vega_tb():
    from tactile_sim.sim.testbed import Testbed

    return Testbed(vega_cfg())


@requires_vega
def test_vega_model_ratings_and_interface(vega_tb):
    w = vega_tb.world
    m = w.model
    spec = w.arm_spec
    assert spec.robot == "vega_1u" and spec.interface == "position" and spec.torque_rate is None
    assert np.allclose(spec.torque, [150, 150, 80, 80, 25, 25, 25])
    assert np.allclose(spec.velocity, [2.4, 2.4, 2.7, 2.7, 2.7, 2.7, 2.7])
    for a, lim in zip(w.arm_act, spec.torque, strict=True):
        assert m.actuator_forcerange[a] == pytest.approx([-lim, lim])
    assert m.body("hand").id > 0 and m.body("hand_left").id > 0  # a WUJI hand on each arm
    assert len(w.left_hand_act) == 20 and len(w.hand.joint_names) == 20
    assert vega_tb.l1.n_cmd == 10  # 100 Hz position targets under a 1 kHz host loop
    # Vega U: fixed pedestal; lift and torso flip set before the run, not driven
    names = [m.joint(i).name for i in range(m.njnt)]
    assert "Lift" not in names and "torso_flip" not in names and "torso_j1" not in names
    assert not any("wheel" in n for n in names)
    assert w.data.body("arm_center").xpos[2] == pytest.approx(1.343, abs=0.01)  # lift at 0.1 m


@requires_vega
def test_vega_u_setup_axes_are_bounded():
    from tactile_sim.model.robots import load_vega_tree

    with pytest.raises(ValueError):
        load_vega_tree(vega_cfg(arm={"vega_lift": 0.5}).arm)
    with pytest.raises(ValueError):
        load_vega_tree(vega_cfg(arm={"vega_flip": 1.2}).arm)


@pytest.mark.skipif(not hand_available("vega_1p"), reason="Vega-1P not cached")
def test_vega_1p_variant_builds():
    from tactile_sim.sim.world import World

    w = World(hand_config("wuji2", fast_config(), robot="vega_1p"))
    assert w.arm_spec.robot == "vega_1p" and "torso_j1" in w.held


@requires_vega
def test_p_multiplier_is_bounded_like_dexcontrol():
    from tactile_sim.model.robots import servo_gains

    with pytest.raises(ValueError):
        servo_gains(vega_cfg(arm={"servo_p_mult": 5.0}).arm)


@requires_vega
def test_position_targets_change_only_at_the_command_rate(vega_tb):
    tb = vega_tb
    tb.reset(0)
    w = tb.world
    seen = []
    for _ in range(int(0.1 / w.dt)):
        tb.step()
        seen.append(w.data.ctrl[w.arm_act].copy())
    seen = np.array(seen)
    changes = np.flatnonzero(np.any(np.abs(np.diff(seen, axis=0)) > 0, axis=1))
    assert len(changes) <= 11
    assert np.all(np.diff(changes) >= int(round(0.01 / w.dt)) - 1)


@requires_vega
def test_vega_holds_the_hover_pose_and_the_hammer(vega_tb):
    tb = vega_tb
    tb.reset(0)
    p0, r0 = tb.world.hammer_in_hand()
    tb.run_for(0.4)
    x, _ = tb.world.tcp_pose()
    assert np.linalg.norm(x - tb.world.hover_tcp) < 3e-3  # gravity droop compensated through the targets
    p1, r1 = tb.world.hammer_in_hand()
    assert np.linalg.norm(p1 - p0) < 0.2e-3


@pytest.fixture(scope="module")
def vega_episode():
    from tactile_sim.episode import Episode

    ep = Episode(vega_cfg(), seed=0)
    return ep, ep.run(3)


@requires_vega
def test_vega_strikes_within_the_arm_limits(vega_episode):
    ep, res = vega_episode
    s = res.summary
    assert s["hit_rate"] >= 2 / 3 and s["drops"] == 0
    assert s["total_depth"] > 0.5e-3
    for key in ("limit_arm_torque", "limit_arm_velocity", "limit_arm_range", "limit_hand_torque"):
        assert s[key] <= 1.0 + 1e-6, (key, ep.limits.where.get(key[6:]))
    assert "limit_arm_torque_rate" not in s  # Dexmate specifies none
    assert s["limit_payload"] < 1.0
