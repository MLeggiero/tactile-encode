"""Planned motion for the varied VLA demos: the composite strike path, the transit, the feasibility probe, and that the
default (unplanned) hammer and drill bases are unchanged against their golden traces (tests/data/golden_bases.json)."""

import json
import platform
from pathlib import Path

import numpy as np
import pytest

from tactile_sim.behaviors.strike_paths import CompositeStrikeGeometry, TransitPath, is_straight, rotmat, rotvec
from tactile_sim.behaviors.trajectories import StrikeGeometry

GOLDEN = Path(__file__).parent / "data" / "golden_bases.json"


def _terminal(radius: float) -> StrikeGeometry:
    R = rotmat(np.array([0.3, -0.2, 0.1]))
    return StrikeGeometry(np.array([0.5, -0.2, 0.9]), R, np.array([0.0, -0.03, 0.19]), 0.06, radius)


@pytest.mark.parametrize("radius", [0.0, 0.6])
def test_the_composite_meets_the_nail_square_and_joins_smoothly(radius):
    g = _terminal(radius)
    apex, R_apex = g.pose(-0.15)
    c = CompositeStrikeGeometry(g, apex + np.array([0.03, -0.04, 0.02]), R_apex, -0.04)
    for s in (0.0, g.s_c0, g.s_c0 + 0.02):  # the terminal stroke, contact and overshoot are the configured stroke's
        p1, R1 = c.pose(s)
        p2, R2 = g.pose(s)
        assert np.allclose(p1, p2) and np.allclose(R1, R2)
    eps = 1e-6
    pa, _ = c.pose(c.s_join - eps)
    pb, _ = c.pose(c.s_join + eps)
    assert np.linalg.norm(pa - pb) < 1e-4  # continuous position across the junction
    ta, wa, _ = c.tangent(c.s_join - 1e-4)
    tb, wb, _ = c.tangent(c.s_join + 1e-4)
    assert np.linalg.norm(ta - tb) < 0.01 * np.linalg.norm(tb)  # the TCP's velocity per unit s is continuous
    # s keeps the stroke's meaning (face distance): the TCP moves the stroke's k per unit s along the whole curve
    k = np.linalg.norm(g.tangent(0.0)[0])
    s_mid = 0.5 * (c.s_apex + c.s_join)
    d = (c.pose(s_mid + 1e-4)[0] - c.pose(s_mid - 1e-4)[0]) / 2e-4
    assert abs(np.linalg.norm(d) - k) < 0.01 * k
    if radius > 0:  # a curve along the arc's own chord spans about the arc's s and turns at about the arc's rate
        assert c.s_apex == pytest.approx(-0.15, abs=0.02)
        assert np.linalg.norm(wa) == pytest.approx(np.linalg.norm(wb), rel=0.3)
    assert not is_straight(c) and is_straight(_terminal(0.0)) and not is_straight(_terminal(0.6))


def test_the_composite_round_trips_s_and_shifts():
    g = _terminal(0.6)
    apex, R_apex = g.pose(-0.15)
    c = CompositeStrikeGeometry(g, apex + np.array([0.02, 0.05, 0.0]), R_apex, -0.03)
    for s in np.linspace(c.s_apex + 0.005, g.s_c0, 9):
        assert abs(c.s_of(c.pose(s)[0]) - s) < 2e-3
    dx = np.array([0.01, -0.02, 0.005])
    cs = c.shifted(dx)
    for s in (c.s_apex + 0.01, c.s_join - 0.01, 0.0, g.s_c0):
        assert np.allclose(cs.pose(s)[0], c.pose(s)[0] + dx)
        assert np.allclose(cs.pose(s)[1], c.pose(s)[1])


def test_an_apex_on_the_strokes_extension_reproduces_a_straight_stroke():
    g = _terminal(0.0)
    apex, R = g.pose(-0.15)
    c = CompositeStrikeGeometry(g, apex, R, -0.05)
    for s in np.linspace(-0.149, -0.051, 7):
        assert np.allclose(c.pose(s)[0], g.pose(s)[0], atol=2e-4)


def test_an_apex_ahead_of_the_junction_is_rejected():
    g = _terminal(0.0)
    ahead, R = g.pose(0.02)
    with pytest.raises(ValueError):
        CompositeStrikeGeometry(g, ahead, R, -0.03)


def test_a_transit_starts_and_ends_at_rest_on_its_poses():
    R0, R1 = rotmat(np.array([0.2, 0.0, 0.0])), rotmat(np.array([0.0, 0.3, -0.1]))
    p0, p1 = np.zeros(3), np.array([0.1, 0.05, -0.08])
    tr = TransitPath(1.0, 0.8, p0, R0, p1, R1, via=np.array([0.0, 0.03, 0.0]))
    x, xd, _ = tr.evaluate(1.0)
    assert np.allclose(x, p0) and np.allclose(xd, 0) and np.allclose(tr.R, R0)
    x, xd, _ = tr.evaluate(1.8)
    assert np.allclose(x, p1) and np.allclose(xd, 0) and np.allclose(tr.R, R1)
    assert np.allclose(tr.pose(1.4)[0], tr.evaluate(1.4)[0])
    assert np.allclose(rotvec(rotmat(np.array([0.1, -2.0, 0.4]))), [0.1, -2.0, 0.4])


# ---------------------------------------------------------------------- golden traces of the default bases
def _compare(fresh: dict, gold: dict, exact: bool) -> None:
    assert len(fresh["t"]) == len(gold["t"])
    for key in ("x_eq", "q"):
        a, b = np.asarray(fresh[key]), np.asarray(gold[key])
        if exact:
            assert np.array_equal(a, b), key
        else:
            np.testing.assert_allclose(a, b, atol=1e-9, err_msg=key)
    assert abs(fresh["progress"] - gold["progress"]) <= (0.0 if exact else 1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("task", ["hammer", "drill"])
def test_the_default_bases_match_their_golden_traces(task, monkeypatch):
    """Bit for bit on the machine that recorded them; to 1e-9 elsewhere (BLAS and MuJoCo builds differ)."""
    pytest.importorskip("h5py")
    from tactile_sim.assets.fetch_hands import hand_xml

    try:
        hand_xml("wuji2")
    except Exception:  # noqa: BLE001
        pytest.skip("needs the WUJI Hand 2 cache (make assets)")
    import importlib.util
    import sys

    import mujoco

    spec = importlib.util.spec_from_file_location("make_golden_bases", GOLDEN.with_name("make_golden_bases.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_golden_bases"] = mod
    spec.loader.exec_module(mod)
    gold = json.loads(GOLDEN.read_text())
    fp = gold["fingerprint"]
    same = (fp["platform"], fp["numpy"], fp["mujoco"]) == (platform.platform(), np.__version__, mujoco.__version__)
    _compare(mod.run_base(task), gold[task], exact=same)


def _planned(task: str, index: int, want=None):
    """A fast-physics episode of `task` with its base attached and a planned motion sampled (the first of indices
    index, index + 1, ... whose plan satisfies `want`)."""
    from tactile_sim.assets.fetch_hands import hand_available

    if not (hand_available("wuji2") and hand_available("vega_1u")):
        pytest.skip("WUJI Hand 2 or the Vega U not cached")
    import tactile_sim.vla  # noqa: F401 - selects the GL backend before mujoco's first import
    from tactile_sim.vla.demos import EPISODES, demo_config
    from tactile_sim.vla.variation import sample_plan

    for i in range(index, index + 12):
        cfg, draw = demo_config(task, 100_000 + i, fast=True)
        tb, base, _ = EPISODES[task](cfg, 100_000 + i)
        tb.residual.base_fn = base
        target = (base.feed.x0.copy(), base.feed.R.copy()) if task == "drill" else None
        plan, _ = sample_plan(task, tb, 7, i, target=target, strikes=task == "hammer")
        if plan is not None and (want is None or want(plan)):
            return tb, base, plan
    pytest.skip("no plan of the wanted kind in 12 episodes")


@pytest.mark.slow
def test_the_hammer_approach_streams_the_checked_path_and_hands_over_without_a_step():
    from tactile_sim.behaviors.joint_planner import JointPath
    from tactile_sim.behaviors.motion_plan import preroll
    from tactile_sim.control.obs_spec import LOOKAHEAD

    tb, base, plan = _planned("hammer", 0, lambda p: p.approach_time > 0.6)
    sw = base.swing
    preroll(tb, np.asarray(plan.start.q), plan.preroll_time, base=sw, plan=plan, path=plan.preroll_path)
    assert np.abs(tb.l1.q_ref - np.asarray(plan.start.q)).max() < 1e-3  # the pre-roll ended at the start
    path = JointPath.from_dict(plan.approach_path, sw.phase_t0)
    assert tb.l1.stream is not None and sw.phase == "approach"
    l1, w = tb.l1, tb.world
    q_refs, ts, track = [], [], 0.0
    looked = False
    while sw.phase in ("approach",) or (sw.phase == "windup" and tb.t < sw.phase_t0 + 0.3):
        tb.step()
        q_refs.append(l1.q_ref.copy())
        ts.append(tb.t)
        if sw.phase == "approach":
            track = max(track, float(np.abs(l1.q_ref - path(tb.t)[0]).max()))
            if not looked and tb.t > sw.phase_t0 + 0.3 * plan.approach_time:
                base._update_arm_lookahead(tb.t)
                for k, dt in enumerate(LOOKAHEAD):
                    assert np.allclose(base.arm_look[k], path(tb.t + dt)[0] - l1.q_ref, atol=1e-9)
                looked = True
    assert looked and track < 0.02  # the arm's targets followed the checked joint path
    assert tb.l1.stream is None and sw.phase == "windup"  # released at the handoff
    dq = np.abs(np.diff(np.asarray(q_refs), axis=0))
    v_step = w.arm_spec.velocity * l1.dt_cmd * 0.9  # PositionL1's rate limit per command tick
    assert np.all(dq <= v_step + 1e-9)
    t_hand = sw.phase_t0
    near = (np.asarray(ts[1:]) > t_hand - 0.05) & (np.asarray(ts[1:]) < t_hand + 0.05)
    assert near.any() and np.all(dq[near] < 0.5 * v_step)  # no step at the handoff: nowhere near the rate limit


@pytest.mark.slow
def test_the_drill_transit_ends_at_its_angled_entry_and_lines_up_with_the_screw():
    from tactile_sim.behaviors.motion_plan import preroll

    tb, base, plan = _planned("drill", 0, lambda p: np.degrees(np.linalg.norm(p.align_rotvec)) > 5.0)
    feed = base.feed
    preroll(tb, np.asarray(plan.start.q), plan.preroll_time, base=feed, plan=plan, path=plan.preroll_path)
    assert feed.phase == "transit" and tb.l1.stream is not None
    a = feed.a
    while feed.phase == "transit":
        x_t = feed.target(tb.t)
        x_fk = feed.transit.pose(tb.t)[0]
        assert np.allclose(x_t, x_fk)
        tb.step()
    assert feed.phase == "align" and tb.l1.stream is None
    p_entry, R_entry = tb.world.ctrl_pose()
    bit_entry = -R_entry[:, 1] if tb.cfg.drill.form == "pistol" else R_entry[:, 2]
    assert np.degrees(np.arccos(np.clip(bit_entry @ a, -1, 1))) > 3.0  # it came in at an angle
    while feed.phase == "align":
        tb.step()
    assert feed.phase == "approach"
    for _ in range(int(0.2 / tb.world.dt)):  # settle at the start of the feed's approach
        tb.step()
    p, R = tb.world.ctrl_pose()
    bit = -R[:, 1] if tb.cfg.drill.form == "pistol" else R[:, 2]
    assert np.degrees(np.arccos(np.clip(bit @ a, -1, 1))) < 2.0  # lined up with the screw
    lateral = (p - feed.x0) - ((p - feed.x0) @ a) * a
    assert np.linalg.norm(lateral) < 0.004
