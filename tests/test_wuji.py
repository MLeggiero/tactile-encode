"""WUJI Hand 2 on the FR3: model import, grasp synthesis, taxel patches, joint-level control, and the
hardware limits through a strike episode."""

import numpy as np
import pytest

from tactile_sim.assets.fetch_hands import hand_available, hand_xml
from tactile_sim.config import fast_config, hand_config
from tactile_sim.limits import FR3_VELOCITY
from tactile_sim.sim.testbed import Testbed

requires_wuji = pytest.mark.skipif(not hand_available("wuji2"), reason="WUJI Hand 2 not cached")

# vendor actuatorfrcrange per joint type (hand2_beta2 right_with_mount.xml)
RATED = {"mcp_flex": 2.0, "mcp_abd": 0.2, "pip": 0.3, "dip": 0.3, "thumb_cmc_flex": 0.6, "thumb_cmc_abd": 0.6,
         "thumb_mcp": 0.3, "thumb_ip": 0.3}


def _rated(name: str) -> float:
    for k in sorted(RATED, key=len, reverse=True):
        if name.endswith(k):
            return RATED[k]
    raise KeyError(name)


def wuji_cfg(**over):
    return hand_config("wuji2", fast_config(), **over)


@pytest.fixture(scope="module")
def wuji_tb():
    return Testbed(wuji_cfg())


@requires_wuji
def test_hand_model_keeps_vendor_torque_ratings(wuji_tb):
    w = wuji_tb.world
    m = w.model
    hand = w.hand
    assert len(hand.joint_names) == 20
    for j, act, lim in zip(hand.joint_names, hand.act, hand.tau_max, strict=True):
        assert lim == pytest.approx(_rated(j)), j
        assert m.actuator_ctrlrange[act] == pytest.approx([-lim, lim]), j
    # joint law output is clipped to the rating whatever the synergy asks
    tau = hand.synergy.torque(5.0, np.zeros(20), np.full(20, -100.0))
    assert np.all(np.abs(tau) <= hand.tau_max + 1e-12)


@requires_wuji
def test_grasp_keyframe_is_a_seated_wrap(wuji_tb):
    g = wuji_tb.world.hand.grasp
    assert g.hold_drift < 0.2e-3  # the hammer no longer moves once seated
    assert g.contact_force.get("r_wrist", 0) > 10  # the handle bears on the palm
    fingers = [b for b, f in g.contact_force.items() if f > 1 and "thumb" not in b and b != "r_wrist"]
    assert len(fingers) >= 3
    assert any("thumb" in b and f > 5 for b, f in g.contact_force.items())


@requires_wuji
def test_taxelscan_skins_load_where_the_grasp_loads_the_hand(wuji_tb):
    w = wuji_tb.world
    names = [p.name for p in w.hand.patches]
    assert names[0] == "palm" and len(names) == 11
    assert w.hand.accel_names == ["pad_acc_palm"]
    forces = w.pad_normal_forces()
    for k, p in enumerate(w.hand.patches):
        tax = w.pad_taxels(k)
        assert tax.shape == (128 if p.name == "palm" else 32,)
        assert tax.sum() == pytest.approx(forces[k], rel=1e-6)
    # the handle bears on the palm and the fingertips
    assert forces[names.index("palm")] > 10
    assert sum(forces[names.index(f"{f}_distal")] > 1 for f in ("index", "middle", "ring", "pinky")) >= 3
    assert "pad_acc_palm" in wuji_tb.sensors and "pressure_index_distal" in wuji_tb.sensors


@requires_wuji
def test_flat_patch_layout_still_available():
    tb = Testbed(wuji_cfg(gripper={"patch_layout": "palm_thumb"}))
    w = tb.world
    assert [p.name for p in w.hand.patches] == ["palm", "thumb"]
    for k in range(2):
        tax = w.pad_taxels(k)
        assert tax.shape == (64,)
        assert tax.sum() == pytest.approx(w.pad_normal_forces()[k], rel=1e-6)
        assert tax.sum() > 2.0


@requires_wuji
def test_static_hold_and_grip_loop(wuji_tb):
    tb = wuji_tb
    tb.reset(0)
    p0, r0 = tb.world.hammer_in_hand()
    tb.run_for(0.5)
    p1, r1 = tb.world.hammer_in_hand()
    assert np.linalg.norm(p1 - p0) < 0.1e-3 and np.degrees(np.linalg.norm(r1 - r0)) < 0.2
    assert tb.grip.measured == pytest.approx(tb.cfg.controller.grip_hold, rel=0.15)
    assert tb.sched.rate("hand") == 1000.0


@requires_wuji
def test_a_tool_leaving_the_wrap_is_a_drop(wuji_tb):
    """Patches alone cannot tell a lost tool from a handle pressed onto the bare fingers; the fingers closing
    into the space the tool vacated can."""
    tb = wuji_tb
    tb.reset(0)
    tb.run_for(0.2)
    assert not tb.grip.dropped
    w = tb.world
    w.data.qpos[w.hammer_qadr:w.hammer_qadr + 3] += np.array([0.0, 0.0, -1.0])  # pull the hammer out
    w.data.qvel[w.hammer_dofadr:w.hammer_dofadr + 6] = 0.0
    tb.run_for(0.3)
    assert tb.grip.dropped and tb.l1.supervisor.frozen
    assert w.hand.closed_further() > 0.1


@requires_wuji
def test_self_locking_drive_does_not_backdrive():
    tb = Testbed(wuji_cfg(gripper={"lock_mode": "self_locking"}))
    w, hand = tb.world, tb.world.hand
    assert hand.locked
    k = hand.joint_names.index("r_index_finger_mcp_flex")
    q0 = w.data.qpos[hand.qadr[k]]
    for _ in range(400):  # 0.1 s of a 5 Nm opening load, 2.5x the joint's rating
        tb.step()
        w.data.qfrc_applied[hand.dofs[k]] = -5.0
    w.data.qfrc_applied[:] = 0
    assert w.data.qpos[hand.qadr[k]] >= q0 - 2e-3


@pytest.fixture(scope="module")
def wuji_episode():
    from tactile_sim.episode import Episode

    ep = Episode(wuji_cfg(), seed=0)
    return ep, ep.run(3)


@requires_wuji
def test_wuji_strikes_within_the_arm_limits(wuji_episode):
    ep, res = wuji_episode
    s = res.summary
    assert s["hit_rate"] >= 2 / 3 and s["drops"] == 0
    assert s["total_depth"] > 1e-3
    for key in ("limit_arm_torque", "limit_arm_torque_rate", "limit_arm_velocity", "limit_arm_range",
                "limit_hand_torque"):
        assert s[key] <= 1.0 + 1e-6, (key, ep.limits.where.get(key[6:]))
    assert s["limit_payload"] < 1.0
    qd = np.asarray(res.testbed.l1.log.qd)
    assert np.all(np.abs(qd) <= FR3_VELOCITY)


@requires_wuji
@pytest.mark.xfail(strict=True, reason="known: blows drive the thumb's CMC joints into their hard stops at "
                   "several times the joints' rated torque (the stop rating is not published)")
def test_wuji_hard_stops_stay_within_rating(wuji_episode):
    _, res = wuji_episode
    assert res.summary["limit_hand_stop_load"] <= 1.0


def test_hand_config_rejects_unknown_hand():
    with pytest.raises(ValueError):
        hand_config("shadow")


@requires_wuji
def test_hand_asset_is_pinned():
    p = hand_xml("wuji2")
    assert p is not None and p.name == "right_with_mount.xml"
