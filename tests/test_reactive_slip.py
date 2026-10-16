"""Reactive slip loop, the slip-ablation knobs and the grasp-wrench truth."""

import numpy as np
import pytest

from tactile_sim.config import ReactiveCfg, SimConfig, fast_config
from tactile_sim.experiments import slip_ablation as sa
from tactile_sim.logging.strike_metrics import grasp_slip
from tactile_sim.sim.testbed import Testbed


def _shake(cfg, grip, T=1.0):
    """2 g, 5 Hz shake along gravity (tangential to the Franka pads) at a constant grip setpoint."""
    tb = Testbed(cfg)
    cmd = tb.l1.cmd
    p0 = cmd.x_eq.copy()
    cmd.F_grip = grip
    cmd.K = np.array([3000.0, 3000, 3000, 100, 100, 100])
    w = 2 * np.pi * 5.0
    amp = 0.02

    def l2(t):
        cmd.t = t
        cmd.x_eq = p0 + np.array([0, 0, amp * np.sin(w * t)])
        cmd.xd_eq = np.array([0, 0, amp * w * np.cos(w * t), 0, 0, 0])
        cmd.xdd_ff = np.array([0, 0, -amp * w * w * np.sin(w * t), 0, 0, 0])

    tb.l2_callbacks.append(l2)
    tb.run_for(0.3)
    p_a, r_a = tb.world.hammer_in_hand()
    boost = []
    tb.run_for(T, lambda b: boost.append(b.grip_boost))
    p_b, r_b = tb.world.hammer_in_hand()
    return grasp_slip(p_a, r_a, p_b, r_b), tb, np.array(boost)


def test_reactive_off_by_default():
    tb = Testbed(fast_config())
    assert tb.reactive is None
    assert all(task.name != "reactive" for task in tb.sched.tasks)
    assert tb.grip_boost == 0.0


def test_unknown_reactive_mode_rejected():
    with pytest.raises(ValueError):
        Testbed(fast_config(reactive={"mode": "psychic"}))


def test_log_mode_never_acts():
    (_, _), tb, boost = _shake(fast_config(reactive={"mode": "log"}), 20.0, T=0.3)
    assert np.all(boost == 0.0)
    tr = tb.reactive.trace
    assert len(set(len(v) for v in tr.values())) == 1 and len(tr["t"]) > 100
    assert max(tr["omega"]) > 0.0 and max(tr["cop_speed"]) > 0.0


def test_oracle_catches_a_slipping_grasp():
    """At 20 N per pad the hammer slips in a 2 g shake; the oracle loop sees it and clamps at the hand's maximum."""
    (st0, sr0), _, _ = _shake(fast_config(), 20.0)
    (st1, sr1), tb, boost = _shake(fast_config(reactive={"mode": "oracle"}), 20.0)
    assert tb.reactive.detections, "no slip detected"
    assert boost.max() == pytest.approx(tb.world.hand.grip_pre_max)
    assert st1 <= st0 and sr1 <= sr0
    assert st1 < 0.5 * st0 or sr1 < 0.5 * sr0


def test_taxel_coordinates_span_the_franka_pad():
    tb = Testbed(fast_config(reactive={"mode": "log"}))
    c = tb.reactive._coords[0]  # noqa: SLF001
    g = tb.cfg.gripper
    assert c.shape == (64, 3)
    assert np.abs(c[:, 0]).max() < g.pad_half[0] and np.abs(c[:, 2]).max() < g.pad_half[2]


def test_new_knobs_default_to_the_old_behavior():
    cfg = SimConfig()
    assert cfg.swing.strike_k_rot == 250.0 and cfg.swing.grip_profile == "template"
    assert cfg.arm.wrist_range == pytest.approx(0.1) and cfg.reactive.mode == "off"
    tb = Testbed(fast_config())
    m = tb.world.model
    j = m.joint("wrist_flex_x").id
    assert np.allclose(m.jnt_range[j], (-0.1, 0.1))


def test_grasp_wrench_balances_gravity():
    from tactile_sim.episode import Episode
    from tactile_sim.sim.truth import contact_wrench

    ep = Episode(fast_config(), seed=0, record_truth=False)
    tb, w = ep.tb, ep.tb.world
    tb.run_for(0.4)
    d = w.data
    F, M = contact_wrench(w.model, d, ep.tool_geoms, ep.hand_geoms, d.site_xpos[ep.ref_site])
    weight = w.model.body_subtreemass[w.hammer_body] * 9.81
    assert F[2] == pytest.approx(weight, rel=0.03)
    assert abs(F[0]) < 0.05 * weight and abs(F[1]) < 0.05 * weight


@pytest.mark.parametrize("platform,task", [(pf, t) for pf, ts in sa.TASKS.items() for t in ts])
def test_every_condition_builds_a_config(platform, task):
    conds = sa.conditions(platform, task)
    names = [c.name for c in conds]
    assert len(names) == len(set(names)) and names[0] == "B"
    for c in conds:
        for stage in sa.STAGES:
            sa.Job(stage, platform, task, c, 0).config()  # unknown sections or fields raise KeyError


def test_thresholds_name_reactive_fields():
    fields = set(ReactiveCfg.__dataclass_fields__)
    for hand, th in sa.THRESHOLDS.items():
        assert set(th) <= fields, hand


def test_baseline_hammer_turns_off_anticipation_and_adaptation():
    cfg = sa.base_config("fr3_franka", "hammer")
    assert cfg.swing.grip_profile == "constant" and cfg.swing.grip_margin_step == 0.0


@pytest.mark.slow
def test_constant_profile_holds_the_grip_through_a_strike():
    from tactile_sim.episode import Episode

    cfg = fast_config(swing={"grip_profile": "constant", "grip_margin_step": 0.0})
    res = Episode(cfg, seed=0, record_truth=False).run(1)
    r = res.strikes[0]
    hold = cfg.controller.grip_hold
    assert r.extra["grip_mean"] == pytest.approx(hold, rel=0.1)
    assert r.extra["grasp_peak_pitch"] > 0.0 and np.isfinite(r.extra["slip_impact_rot"])


def test_flange_control_ignores_wrist_deflection():
    """wrist_control = "flange": the controlled frame rides on the F/T body, so bending the wrist moves the TCP but
    not the controlled frame; at zero deflection the two coincide."""
    import mujoco

    from tactile_sim.sim.world import World

    w = World(fast_config(arm={"wrist_control": "flange"}))
    m, d = w.model, w.data
    qadr = [m.joint(n).qposadr[0] for n in ("wrist_flex_x", "wrist_flex_y")]
    d.qpos[qadr] = 0.0
    mujoco.mj_forward(m, d)
    (pc, Rc), (pt, Rt) = w.ctrl_pose(), w.tcp_pose()
    assert np.allclose(pc, pt, atol=1e-9) and np.allclose(Rc, Rt, atol=1e-9)
    d.qpos[qadr[0]] = 0.05
    mujoco.mj_forward(m, d)
    pc2, _ = w.ctrl_pose()
    pt2, _ = w.tcp_pose()
    assert np.allclose(pc2, pc, atol=1e-9) and np.linalg.norm(pt2 - pt) > 1e-3
    J = w.ctrl_jacobian(w.arm_dofs)
    assert J.shape == (6, len(w.arm_dofs))


def test_effort_is_the_commanded_grip():
    tb = Testbed(fast_config())
    tb.world.set_grip_force(37.0)
    assert tb.world.hand.command() == pytest.approx(37.0)


def test_boost_ceiling():
    tb = Testbed(fast_config(reactive={"mode": "oracle"}))
    assert tb.reactive.f_max == pytest.approx(tb.world.hand.grip_pre_max)
    tb = Testbed(fast_config(reactive={"mode": "oracle", "boost_to": "max"}))
    assert tb.reactive.f_max == pytest.approx(tb.world.hand.grip_force_max)
    with pytest.raises(ValueError):
        Testbed(fast_config(reactive={"mode": "oracle", "boost_to": "most"}))


def test_grip_lag_is_first_order():
    lag = 0.010
    tb = Testbed(fast_config(gripper={"grip_lag": lag}))
    hand, w = tb.world.hand, tb.world
    f0 = hand.command()
    w.set_grip_force(f0 + 50.0)
    assert hand.command() == pytest.approx(f0)  # nothing applied until the drives step
    n = int(round(lag / w.dt))
    for _ in range(n):
        hand.actuate()
    assert hand.command() == pytest.approx(f0 + 50.0 * (1 - np.exp(-1.0)), rel=1e-6)


def test_no_lag_applies_at_once():
    tb = Testbed(fast_config())
    tb.world.set_grip_force(33.0)
    assert tb.world.hand.command() == pytest.approx(33.0)


def test_open_grip_loop_commands_the_setpoint():
    tb = Testbed(fast_config(controller={"grip_loop": "open"}))
    tb.l2_callbacks.append(lambda t: setattr(tb.l1.cmd, "t", t))
    tb.l1.cmd.F_grip = 47.0
    tb.run_for(0.05)
    assert tb.world.hand.command() == pytest.approx(47.0)
    with pytest.raises(ValueError):
        Testbed(fast_config(controller={"grip_loop": "ajar"}))


def test_latched_boost_decays_slowly():
    tb = Testbed(fast_config(reactive={"mode": "oracle", "latch": True, "latch_tau": 2.0}))
    rx = tb.reactive
    rx.t_last = 0.0  # as if a slip was confirmed at t = 0
    rx.cfg.omega_thresh = rx.cfg.v_thresh = 1e9  # no new detections
    rx.cfg.drift_rot_thresh = rx.cfg.drift_trans_thresh = 1e9
    rx.tick(1.2)
    expect = rx.f_max * np.exp(-(1.2 - rx.cfg.hold_time) / 2.0)
    assert rx.boost == pytest.approx(expect)
    assert tb.grip_boost == pytest.approx(expect)


def test_ray_cull_keeps_the_skin_load_shape_identical():
    """The bounding-sphere cut in the TaxelScan elastic-foundation model must not change a single taxel."""
    from tactile_sim.assets.fetch_hands import hand_available
    from tactile_sim.config import hand_config
    from tactile_sim.sim.hands import HandIO

    if not hand_available("wuji2"):
        pytest.skip("WUJI Hand 2 not cached")
    tb = Testbed(hand_config("wuji2", fast_config()))
    tb.run_for(0.1)
    hand = tb.world.hand
    compared = 0
    for k, p in enumerate(hand.patches):
        if p.taxel_pos is None:
            continue
        centre = [p.taxel_pos.mean(axis=0)]  # a contact at the patch centre: every nearby taxel casts its rays
        try:
            HandIO._ray_cull = True
            a = hand._foundation(k, centre, hand.tool_geoms)  # noqa: SLF001
            HandIO._ray_cull = False
            b = hand._foundation(k, centre, hand.tool_geoms)  # noqa: SLF001
        finally:
            HandIO._ray_cull = True
        if a is None or b is None:
            assert a is None and b is None
            continue
        assert np.array_equal(a, b)
        compared += 1
    assert compared > 0


def test_runtime_randomization_without_an_observer():
    """The Vega's position-interface L1 has no momentum observer; runtime randomization must not need one."""
    from types import SimpleNamespace

    from tactile_sim.sim.randomize import apply_runtime_dr, sample_dr

    s = sample_dr(SimConfig(), np.random.default_rng(0))
    scaled = []
    tb = SimpleNamespace(sensors=SimpleNamespace(set_noise_scale=scaled.append), l1=SimpleNamespace())
    apply_runtime_dr(tb, s)
    assert scaled == [s.sensor_noise_scale]
