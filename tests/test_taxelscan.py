"""TaxelScan Rev3 skins on the WUJI Hand 2: conforming layout and the RP2350 scan readout."""

import mujoco
import numpy as np
import pytest

from tactile_sim.assets.fetch_hands import hand_available, hand_xml
from tactile_sim.config import SensorsCfg
from tactile_sim.sensors.taxelscan import TaxelScanSensor, taxel_time

requires_wuji = pytest.mark.skipif(not hand_available("wuji2"), reason="WUJI Hand 2 not cached")


@pytest.fixture(scope="module")
def layout():
    from tactile_sim.model.hands.taxel_layout import conforming_layout

    return conforming_layout(str(hand_xml("wuji2")))


@requires_wuji
def test_layout_counts(layout):
    assert len(layout) == 11
    assert sum(p.n_taxels for p in layout) == 448
    assert layout[0].name == "palm" and layout[0].grid == (8, 16) and layout[0].accel
    for p in layout[1:]:
        assert p.grid == (8, 4) and not p.accel
    assert {p.name for p in layout[1:]} == {f"{f}_{s}" for f in ("thumb", "index", "middle", "ring", "pinky")
                                           for s in ("middle", "distal")}


@requires_wuji
def test_taxels_lie_on_the_palmar_surface(layout):
    m = mujoco.MjModel.from_xml_path(str(hand_xml("wuji2")))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for p in layout:
        b = m.body(p.body).id
        R, x = d.xmat[b].reshape(3, 3), d.xpos[b]
        pw, nw = p.pos @ R.T + x, p.normal @ R.T
        # every taxel sits on its link's surface: mujoco's own ray from 3 mm outside hits the link's mesh at 3 mm
        bids = {m.body(n).id for n in p.bodies}
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in bids and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                 and m.geom_contype[g] == 0]
        for i in range(0, len(pw), 7):
            o = pw[i] + 0.003 * nw[i]
            hits = [mujoco.mj_rayMesh(m, d, g, o, -nw[i]) for g in geoms]
            dist = min(h for h in hits if h >= 0)
            assert dist == pytest.approx(0.003, abs=1e-4), (p.name, i)
        assert np.allclose(np.linalg.norm(p.normal, axis=1), 1.0)
        # normals face away from the link, toward the palm side; the four fingers' skins face the palm's +y
        assert np.mean(p.normal @ np.cross(p.u, p.v)) > 0.6
        if p.name != "palm" and not p.name.startswith("thumb"):
            assert nw.mean(axis=0)[1] > 0.6, p.name
        # taxels are spread over a pad, not bunched: pitch 1.5-6 mm
        assert 1.2e-3 < min(p.pitch) and max(p.pitch) < 6e-3, p.name


def _sensor(n=32, **over):
    s = SensorsCfg(**over)
    feed = {"f": np.zeros(n)}
    sen = TaxelScanSensor("t", n, 1.25e-4, lambda: feed["f"], np.random.default_rng(0), s)
    return sen, feed, s


def test_scan_timing_and_latency():
    sen, feed, s = _sensor()
    t = 0.0
    for _ in range(int(0.02 / 1.25e-4)):
        t += 1.25e-4
        sen.step(t)
    h = sen.history()
    assert np.allclose(np.diff(h["t_sample"]), 1e-3)
    assert np.allclose(h["t_avail"] - h["t_sample"], 32 * taxel_time(s) + s.ts_latency)


def test_scan_skew_sees_a_step_partway_through_a_frame():
    # a load step arriving mid-scan shows on the late taxels of that frame only
    sen, feed, s = _sensor(n=128, ts_bandwidth=3000.0, ts_gain_mismatch=0.0, ts_offset_lsb=0.0, ts_enob=12.0)
    t, dt = 0.0, 1.25e-4
    t_step = 0.010 + 64 * taxel_time(s)
    while t < 0.015:
        t += dt
        feed["f"] = np.full(128, 10.0 if t >= t_step else 0.0)
        sen.step(t)
    h = sen.history()
    k = int(np.argmin(np.abs(h["t_sample"] - 0.010)))
    frame = h["value"][k]
    assert frame[:40].max() < 0.5 and frame[-20:].min() > 1.0
    assert h["value"][k + 2].min() > 8.0


def test_adc_resolution_and_calibration():
    sen, feed, s = _sensor(ts_gain_mismatch=0.0, ts_offset_lsb=0.0, ts_enob=12.0)
    f = np.linspace(0, 20, 32)
    est = sen.calibrate(sen.adc(f))
    # 12-bit divider: ~1 mN per count near zero, ~30 mN at 20 N
    assert np.max(np.abs(est - f)) < 0.05
    sen2, _, _ = _sensor()
    est = sen2.calibrate(sen2.adc(np.full(32, 5.0)))
    assert 0.02 < np.std(est) / 5.0 < 0.2  # the calibration residuals show as a per-taxel spread


def test_one_board_for_the_hand_must_fit_the_frame():
    s = SensorsCfg()
    with pytest.raises(ValueError, match="longer than"):
        TaxelScanSensor("t", 448, 1.25e-4, lambda: np.zeros(448), None, s)
    s0 = SensorsCfg(ts_settle=0.0)  # 448 x 2 us = 0.9 ms: fits with no settling time
    TaxelScanSensor("t", 448, 1.25e-4, lambda: np.zeros(448), None, s0)
