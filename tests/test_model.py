import mujoco
import numpy as np
import pytest

from tactile_sim import names
from tactile_sim.config import SimConfig, fast_config
from tactile_sim.model import build_scene, strike_axis, tcp_rotation
from tactile_sim.sim.world import World


@pytest.mark.parametrize("flex", ["rigid", "wrist"])
@pytest.mark.parametrize("pad_mode", ["explicit", "soft_contact"])
def test_scene_compiles_with_expected_names(flex, pad_mode):
    cfg = fast_config(arm={"flex_mode": flex}, gripper={"pad_mode": pad_mode})
    m = mujoco.MjModel.from_xml_string(build_scene(cfg).xml)
    for n in names.ARM_JOINTS + names.FINGER_JOINTS + [names.NAIL_JOINT]:
        assert m.joint(n).id >= 0
    for n in [names.FT_SITE, names.TCP_SITE, names.HAMMER_FACE_SITE, names.HAMMER_REF_SITE, names.NAIL_HEAD_SITE]:
        assert m.site(n).id >= 0
    n_wrist = 3 if flex == "wrist" else 0
    n_pad = 6 if pad_mode == "explicit" else 0
    assert m.nv == 7 + n_wrist + 2 + n_pad + 6 + 1
    assert m.nu == 8
    assert m.npair == 4
    # every automatic collision is off: contacts only come from the explicit pairs
    assert np.all(m.geom_contype == 0) and np.all(m.geom_conaffinity == 0)


def test_unknown_modes_rejected():
    with pytest.raises(ValueError):
        build_scene(fast_config(arm={"flex_mode": "sea"}))
    with pytest.raises(ValueError):
        build_scene(fast_config(gripper={"pad_mode": "gel"}))


def test_drill_stub_raises():
    with pytest.raises(NotImplementedError):
        build_scene(fast_config(plant={"kind": "drill"}))


def test_config_roundtrip():
    cfg = SimConfig().replace(arm={"flex_mode": "rigid"}, swing={"v_strike": 1.7})
    cfg2 = SimConfig.from_dict(cfg.to_dict())
    assert cfg2 == cfg
    with pytest.raises(KeyError):
        cfg.replace(arm={"no_such_field": 1})


def test_strike_geometry():
    R = tcp_rotation()
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(strike_axis(), [0, 1, 0])


def test_reset_places_grasp(settled_world):
    w = settled_world
    p_tcp, R_tcp = w.tcp_pose()
    assert np.linalg.norm(p_tcp - w.hover_tcp) < 0.005
    assert np.allclose(R_tcp, w.tcp_R_nominal, atol=0.02)
    p_rel, rv = w.hammer_in_hand()
    assert np.linalg.norm(p_rel) < 0.002
    assert np.degrees(np.linalg.norm(rv)) < 1.0
    f = w.pad_normal_forces()
    assert np.allclose(f, w.cfg.controller.grip_hold, rtol=0.1)
    assert w.plant.depth == pytest.approx(0.0, abs=1e-6)
    assert w.plant.gap() == pytest.approx(w.cfg.scene.hover_clearance, abs=0.005)
    assert w.t == 0.0


@pytest.mark.parametrize("dt", [0.000125, 0.00025])
def test_hold_is_stable(dt):
    w = World(fast_config(physics={"timestep": dt}))
    w.reset()
    p0, r0 = w.hammer_in_hand()
    for _ in range(int(2.0 / dt)):
        w.set_grip_force(w.cfg.controller.grip_hold)
        w.set_arm_torque(w.hold_torque(w.q_hover))
        w.step()
    d = w.data
    assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
    p1, r1 = w.hammer_in_hand()
    assert np.linalg.norm(p1 - p0) < 0.002  # stays in the grasp (< 2 mm)
    assert np.degrees(np.linalg.norm(r1 - r0)) < 0.5
    assert np.linalg.norm(d.qvel) < 1e-2  # at rest
    assert w.plant.depth < 1e-6  # nail untouched


def test_ft_sees_hand_and_tool_weight(settled_world):
    w = settled_world
    fz = w.sensor("ft_force")[2]
    # MuJoCo's force sensor carries the whole subtree of the sensor body, sensor mass included,
    # plus the hammer that hangs in the pads (a separate free body)
    m_below = w.model.body_subtreemass[w.model.body(names.FT_BODY).id] + w.model.body_subtreemass[w.hammer_body]
    assert abs(abs(fz) - m_below * 9.81) < 0.3


def test_taxels_register_grip(settled_world):
    w = settled_world
    for side in "LR":
        tot = sum(w.sensor(f"taxel_{side}_{r}{c}")[0] for r in range(4) for c in range(4))
        assert tot == pytest.approx(w.cfg.controller.grip_hold, rel=0.15)
