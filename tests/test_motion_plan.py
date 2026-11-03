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
