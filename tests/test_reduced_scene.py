"""The reduced training scene (SceneCfg.reduced): the Vega U's right arm with the right WUJI Hand 2 (MIT-mode drives),
the hammer and the nail plant, without the left arm, the left hand, the head or visual-only meshes. The CPU testbed
runs it, and its export for the GPU training env (tactile_sim.sim.training_parts) carries what the GPU needs."""

import mujoco
import numpy as np
import pytest

from tactile_sim.assets.fetch_hands import hand_available
from tactile_sim.config import fast_config, hand_config

requires_vega = pytest.mark.skipif(not (hand_available("vega_1u") and hand_available("wuji2")),
                                   reason="Vega U / WUJI not cached")


def reduced_cfg(**over):
    return hand_config("wuji2", fast_config(), robot="vega_1u", gripper={"hand_control": "mit"},
                       scene={"reduced": True}, **over)


@pytest.fixture(scope="module")
def reduced_tb():
    from tactile_sim.sim.testbed import Testbed

    return Testbed(reduced_cfg())


@requires_vega
def test_reduced_scene_holds_only_the_striking_side(reduced_tb):
    w = reduced_tb.world
    m = w.model
    joints = [m.joint(i).name for i in range(m.njnt)]
    assert joints[0] == "hammer_free"  # the tool's free joint leads qpos (mjlab's entity root)
    assert joints[1:8] == [f"R_arm_j{i}" for i in range(1, 8)] and joints[-1] == "nail_slide"
    assert len(w.hand.joint_names) == 20 and joints[8:28] == w.hand.joint_names
    assert not any(j.startswith(("L_arm", "head", "l_")) for j in joints)
    assert m.nu == 27 and len(w.held) == 0 and len(w.left_hand_act) == 0
    # no visual-only meshes; every mesh left is a collision hull of a contact pair
    paired = set(m.pair_geom1) | set(m.pair_geom2)
    meshes = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    assert meshes and all(g in paired or m.geom_group[g] == 3 for g in meshes)
    assert m.nsensor > 0  # the testbed's F/T and accelerometers stay


@requires_vega
def test_reduced_scene_keeps_the_striking_side_dynamics():
    from tactile_sim.model.builder import build_scene

    full = mujoco.MjModel.from_xml_string(build_scene(hand_config("wuji2", fast_config(), robot="vega_1u")).xml)
    red = mujoco.MjModel.from_xml_string(build_scene(reduced_cfg()).xml)
    for b in range(1, red.nbody):
        name = red.body(b).name
        fb = full.body(name)
        assert red.body_mass[b] == fb.mass[0], name
        assert np.allclose(red.body_inertia[b], fb.inertia) and np.allclose(red.body_ipos[b], fb.ipos), name
    for j in range(red.njnt):
        fj = full.joint(red.joint(j).name)
        assert np.allclose(red.jnt_range[j], fj.range)
        d, fd = red.jnt_dofadr[j], fj.dofadr[0]
        assert red.dof_armature[d] == full.dof_armature[fd] and red.dof_damping[d] == full.dof_damping[fd]
    assert red.npair == full.npair


@requires_vega
def test_reduced_scene_settles_and_holds_the_hammer(reduced_tb):
    tb = reduced_tb
    tb.reset(0)
    w = tb.world
    p0, r0 = w.hammer_in_hand()
    tb.run_for(0.3)
    p1, r1 = w.hammer_in_hand()
    assert np.linalg.norm(p1 - p0) < 0.2e-3
    assert np.degrees(np.linalg.norm(r1 - r0)) < 0.2
    x, _ = w.tcp_pose()
    assert np.linalg.norm(x - w.hover_tcp) < 3e-3  # the arm servos hold the hover pose (droop compensated)


def test_reduced_scene_is_the_vega_u_wuji_hammer_only():
    from tactile_sim.model.builder import build_reduced_scene

    with pytest.raises(ValueError):
        build_reduced_scene(hand_config("wuji2", fast_config()))  # FR3
    with pytest.raises(ValueError):
        build_reduced_scene(fast_config())


@requires_vega
def test_training_export_carries_the_reset_and_the_model_edits():
    from tactile_sim.sim.training_parts import export_training_parts, training_config

    parts = export_training_parts(training_config())
    m = mujoco.MjModel.from_xml_string(parts.xml)
    assert m.nsensor == 0 and not any(m.site(i).name.startswith("taxel_") for i in range(m.nsite))
    assert m.opt.noslip_iterations == 0 and m.opt.timestep == 0.00025
    assert m.nq == parts.qpos_start.shape[0] == parts.qpos_settled.shape[0]
    for n, r in zip(parts.hand_actuators, parts.hand_ctrlrange, strict=True):
        assert np.allclose(m.actuator(n).ctrlrange, r)  # the MIT drives clamp; the motors never do
    assert np.allclose(parts.hand_ctrlrange[:, 1], 2.0 * parts.hand_tau_rated)
    assert np.all(m.pair(parts.face_pair).solref < 0)  # the nail plant's direct-form restitution
    assert np.all(np.abs(parts.arm_target - parts.q_hover) < 0.02)  # hover pose + gravity droop
    assert 0.0 < parts.s_hold < 1.0 and parts.s_settle(0, 600) == 0.0
    assert parts.weld_eq_data[10] == 1.0 and np.isclose(np.linalg.norm(parts.weld_eq_data[6:10]), 1.0)
    assert np.linalg.norm(parts.hammer_in_hand[0]) < 1e-3 and np.all(parts.i_eff > 0)
