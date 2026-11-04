"""The joint-space planner (behaviors.joint_planner): RRT-Connect, shortcutting, smoothing and timing on synthetic
obstacles, and one plan around the table on a real Vega U + WUJI world."""

import numpy as np
import pytest

from tactile_sim.assets.fetch_hands import hand_available
from tactile_sim.behaviors import joint_planner as jp

N = 7
LO, HI = -np.pi * np.ones(N), np.pi * np.ones(N)
V_LIMIT = np.full(N, 2.0)


def box(centre, half):
    """valid(q): outside an axis-aligned box in the first two joints (a wall the straight line must go around)."""
    c, h = np.asarray(centre, float), np.asarray(half, float)

    def valid(q):
        return not bool(np.all(np.abs(np.asarray(q)[:2] - c) <= h))

    return valid


# a start beside and below the hover pose from which the straight joint move swings the hammer head into the board
FAR_START = (-0.251, -0.145, -0.078)
START = np.array([-1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
GOAL = np.array([1.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0])
WALL = box([0.0, 0.0], [0.15, 0.8])  # across the straight segment in joints 1-2


def dense_valid(path: jp.JointPath, valid, n: int = 400) -> bool:
    return all(valid(q) for q in path.samples(n))


def test_a_free_straight_edge_is_returned_as_is():
    res = jp.rrt_connect(START, GOAL, lambda q: True, LO, HI, np.random.default_rng(0))
    assert res.ok and res.iterations == 0
    assert np.allclose(res.waypoints[0], START) and np.allclose(res.waypoints[-1], GOAL)


def test_the_planner_goes_around_an_obstacle_on_the_straight_line():
    assert not jp.edge_ok(START, GOAL, WALL)
    path, res = jp.plan_path(START, GOAL, WALL, LO, HI, V_LIMIT, np.random.default_rng(1))
    assert res.ok and path is not None
    assert np.allclose(path(path.t0)[0], START, atol=1e-9) and np.allclose(path(path.t_end)[0], GOAL, atol=1e-9)
    assert dense_valid(path, WALL)


def test_invalid_ends_are_reported_without_searching():
    inside = np.array([0.0, 0.0, 0, 0, 0, 0, 0])
    assert jp.rrt_connect(START, inside, WALL, LO, HI, np.random.default_rng(0)).reason == "goal_invalid"
    assert jp.rrt_connect(inside, GOAL, WALL, LO, HI, np.random.default_rng(0)).reason == "start_invalid"


def test_an_unreachable_goal_exhausts_the_budget():
    # the goal sits in a pocket sealed by a shell around it
    def sealed(q):
        d = float(np.max(np.abs(np.asarray(q)[:2] - GOAL[:2])))
        return d < 0.05 or d > 0.3

    res = jp.rrt_connect(START, GOAL, sealed, LO, HI, np.random.default_rng(0), max_iters=300)
    assert not res.ok and res.reason == "budget" and res.iterations == 300


def test_plans_are_deterministic_for_a_seed():
    a, _ = jp.plan_path(START, GOAL, WALL, LO, HI, V_LIMIT, np.random.default_rng(5))
    b, _ = jp.plan_path(START, GOAL, WALL, LO, HI, V_LIMIT, np.random.default_rng(5))
    assert a.T == b.T and all(np.array_equal(p, q) for p, q in zip(a.waypoints, b.waypoints, strict=True))


def test_shortcutting_never_lengthens_the_path():
    res = jp.rrt_connect(START, GOAL, WALL, LO, HI, np.random.default_rng(2))
    short = jp.shortcut(res.waypoints, WALL, np.random.default_rng(3))
    assert jp.path_length(short) <= jp.path_length(res.waypoints) + 1e-12
    assert all(jp.edge_ok(p, q, WALL) for p, q in zip(short[:-1], short[1:], strict=True))


def test_smoothing_keeps_a_spline_off_an_obstacle_inside_its_corner():
    # a right-angle polyline around a block placed so a spline through the corner would cut it
    wps = [np.r_[0.0, 0.0, np.zeros(5)], np.r_[1.0, 0.0, np.zeros(5)], np.r_[1.0, 1.0, np.zeros(5)],
           np.r_[2.0, 1.0, np.zeros(5)]]
    block = box([1.12, 0.12], [0.1, 0.1])
    assert all(jp.edge_ok(p, q, block) for p, q in zip(wps[:-1], wps[1:], strict=True))
    assert not dense_valid(jp.JointPath(wps, 1.0), block, 800)  # a plain spline through the corner cuts the block
    out, shape = jp.smooth(wps, block)
    path = jp.timed(out, V_LIMIT, shape=shape)
    assert dense_valid(path, block, 800)


def test_timing_starts_and_ends_at_rest_within_the_speed_margin():
    path, _ = jp.plan_path(START, GOAL, WALL, LO, HI, V_LIMIT, np.random.default_rng(1))
    assert np.allclose(path(path.t0)[1], 0.0) and np.allclose(path(path.t_end)[1], 0.0)
    assert path.peak_speed_ratio(V_LIMIT) <= jp.SPEED_MARGIN + 1e-6
    slow = jp.timed(path.waypoints, V_LIMIT, speed_frac=0.5, shape=path.shape)
    assert slow.T == pytest.approx(2.0 * path.T, rel=1e-6)


def test_a_path_is_a_joint_stream_that_holds_its_ends_and_round_trips():
    path = jp.timed([START, GOAL], V_LIMIT).retimed(3.0)
    assert np.allclose(path(0.0)[0], START) and np.allclose(path(0.0)[1], 0.0)
    assert np.allclose(path(path.t_end + 1.0)[0], GOAL)
    back = jp.JointPath.from_dict(path.to_dict(), t0=3.0)
    for t in np.linspace(3.0, path.t_end, 7):
        assert np.allclose(back(t)[0], path(t)[0]) and np.allclose(back(t)[1], path(t)[1])


@pytest.mark.slow
def test_the_planner_finds_a_way_around_the_table_on_the_hammer_world():
    if not (hand_available("wuji2") and hand_available("vega_1u")):
        pytest.skip("WUJI Hand 2 or the Vega U not cached")
    import tactile_sim.vla  # noqa: F401 - selects the GL backend before mujoco's first import
    from tactile_sim.behaviors.feasibility import CLEARANCE, ArmProbe
    from tactile_sim.vla.demos import EPISODES, demo_config

    cfg, _ = demo_config("hammer", 100_003, fast=True)
    tb, _, _ = EPISODES["hammer"](cfg, 100_003)
    w = tb.world
    probe = ArmProbe(w)
    q_hover = w.data.qpos[w.arm_qadr].copy()
    p, R = w.ctrl_pose()
    q0, ep, er = probe.ik(p + np.asarray(FAR_START), R, q_hover)
    assert ep < 1e-3 and er < 1e-2 and probe.valid(q0)
    assert not jp.edge_ok(q0, q_hover, probe.valid)  # the straight joint move hits something
    path, res = jp.plan_path(q0, q_hover, probe.valid, probe.lo, probe.hi, w.arm_spec.velocity,
                             np.random.default_rng(0))
    assert res.ok, res.reason
    assert min(probe.clearance(q)[0] for q in path.samples(100)) >= CLEARANCE
