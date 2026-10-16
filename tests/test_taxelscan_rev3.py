"""TaxelScan fast profile (ts_model "rev3"): boards of at most 128 taxels scanning from the SYNC edge, frames
polled one period behind in board order, mu-law codes plus patch features at the host, frame drops."""

import numpy as np
import pytest

from tactile_sim.assets.fetch_hands import hand_available
from tactile_sim.config import SensorsCfg, fast_config, hand_config
from tactile_sim.sensors.taxel_codec import CodecSpec, patch_features, to_mn
from tactile_sim.sensors.taxelscan import TaxelScanRev3Sensor, board_layout, poll_slots, taxel_time

requires_wuji = pytest.mark.skipif(not hand_available("wuji2"), reason="WUJI Hand 2 not cached")

NAMES = ["palm"] + [f"{f}_{seg}" for f in ("thumb", "index", "middle", "ring", "pinky") for seg in ("middle", "distal")]
GRIDS = [(8, 16)] + [(8, 4)] * 10
DT = 1.25e-4


def cfg(**over) -> SensorsCfg:
    base = dict(ts_model="rev3", ts_taxel_time=7.8e-6)
    base.update(over)
    return SensorsCfg(**base)


def test_default_board_map_covers_the_hand_on_four_boards():
    s = cfg()
    boards = board_layout(s, NAMES, GRIDS)
    assert len(boards) == 4 and sorted(k for b in boards for k in b) == list(range(11))
    assert [sum(GRIDS[k][0] * GRIDS[k][1] for k in b) for b in boards] == [128, 128, 128, 64]
    assert [NAMES[k] for k in boards[1]] == ["thumb_middle", "thumb_distal", "index_middle", "index_distal"]


@pytest.mark.parametrize("bad, msg", [
    ((("palm",), ("palm",)), "two boards"),
    ((("palm", "thumb_middle"),), "more than"),
    ((("palm",),), "unread"),
    ((("palm", "elbow"),), "unknown patch"),
])
def test_bad_board_maps_are_rejected(bad, msg):
    with pytest.raises(ValueError, match=msg):
        board_layout(cfg(ts_board_map=bad), NAMES, GRIDS)


def test_poll_slots_follow_the_bus():
    s = cfg()
    slots = poll_slots(s, board_layout(s, NAMES, GRIDS), GRIDS)
    assert np.all(np.diff(slots) > 0)
    frame_bytes = [150, 186, 186, 98]  # U8 frames (docs/taxelscan_fast_profile.md)
    want = s.ts_poll_lead + np.cumsum([s.ts_turnaround + b * 10 / s.ts_bus_baud for b in frame_bytes])
    assert slots == pytest.approx(want)
    assert slots[-1] < 0.6e-3  # all four boards cross the bus in ~0.6 ms of the 1 ms period
    u6 = cfg(ts_code_bits=6)
    assert poll_slots(u6, board_layout(u6, NAMES, GRIDS), GRIDS)[-1] < slots[-1]


def _sensor(n_grid=(8, 4), offset=0.0, slot=3e-4, **over):
    s = cfg(**over)
    feed = {"f": np.zeros(n_grid[0] * n_grid[1])}
    sen = TaxelScanRev3Sensor("pressure_t", n_grid, DT, lambda: feed["f"], np.random.default_rng(0), s,
                              offset=offset, slot=slot, board=0)
    return sen, feed, s


def run(sen, feed, t_end, f_of_t):
    t = 0.0
    while t < t_end - 1e-12:
        t += DT
        feed["f"] = f_of_t(t)
        sen.step(t)


def test_frames_arrive_one_period_behind_in_order():
    sen, feed, s = _sensor(slot=4e-4)
    run(sen, feed, 0.03, lambda t: np.full(32, 2.0))
    h = sen.history()
    assert np.allclose(np.diff(h["t_sample"]), 1e-3)
    lag = h["t_avail"] - h["t_sample"]
    lo, hi = s.ts_bridge_latency
    assert np.all(lag >= 1e-3 + 4e-4 + lo - DT) and np.all(lag <= 1e-3 + 4e-4 + hi + DT)
    assert np.all(np.diff(h["t_avail"]) >= 0)


def test_host_sees_codes_and_features_of_the_board_frame():
    sen, feed, s = _sensor(ts_gain_mismatch=0.0, ts_offset_lsb=0.0, ts_enob=30.0, ts_bandwidth=3000.0)
    codec = CodecSpec(bits=8, mu=s.ts_mu, f_max=s.ts_f_max, contact_mn=s.ts_contact_mn)
    load = np.zeros(32)
    load[2 * 4 + 3] = 4.0  # one loaded taxel at row 2, column 3
    run(sen, feed, 0.02, lambda t: load)
    assert np.array_equal(sen.latest().value, codec.decode(sen.codes))
    s_dN, cop_r, cop_c, area, peak, rate = sen.features
    assert s_dN == pytest.approx(40, abs=1) and (cop_r, cop_c) == (2 * 256, 3 * 256) and area == 1
    assert rate == 0  # a steady load
    # the features are the codec's own, on the board's millinewton values
    f_mn = to_mn(sen.calibrate(sen.counts))
    assert np.array_equal(patch_features(f_mn, (8, 4), int(s_dN), s.ts_contact_mn)[:5], sen.features[:5])


def test_the_rate_feature_tracks_load_changes_between_frames():
    sen, feed, s = _sensor(ts_gain_mismatch=0.0, ts_offset_lsb=0.0, ts_enob=30.0, ts_bandwidth=3000.0)
    run(sen, feed, 0.03, lambda t: np.full(32, 0.1 * t / 1e-3))  # +0.1 N per taxel per ms: +3.2 N per frame
    rates = np.array(sen.feature_history)[5:, 5]
    assert np.all(np.abs(rates - 32) <= 2)


def test_dropped_frames_never_reach_the_host():
    sen, feed, s = _sensor(ts_drop_prob=1.0)
    run(sen, feed, 0.01, lambda t: np.full(32, 1.0))
    assert sen.latest().seq == -1 and sen.frame_k >= 8  # the board kept scanning


def test_a_board_scan_must_fit_the_period():
    with pytest.raises(ValueError, match="longer than"):
        _sensor(n_grid=(8, 16), ts_taxel_time=10.2e-6)  # rev3 today: ~10.2 us per taxel, 1.3 ms for 128
    _sensor(n_grid=(8, 16))  # the fast profile's 7.8 us per taxel: 1.0 ms
    with pytest.raises(ValueError, match="longer than"):
        _sensor(n_grid=(8, 4), offset=0.99e-3)  # behind 0.99 ms of earlier patches on the same board
    assert taxel_time(cfg()) == 7.8e-6 and taxel_time(SensorsCfg()) == pytest.approx(3e-6)


@requires_wuji
def test_the_hand_reads_its_skin_through_the_fast_profile():
    from tactile_sim.sim.testbed import Testbed

    tb = Testbed(hand_config("wuji2", fast_config(), sensors={"ts_model": "rev3", "ts_taxel_time": 7.8e-6}))
    hand = tb.world.hand
    sens = [tb.sensors[n] for n in hand.pressure_names]
    assert all(isinstance(x, TaxelScanRev3Sensor) for x in sens)
    assert [x.board for x in sens] == [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3]
    assert [x.offset for x in sens[1:5]] == pytest.approx([0.0, 32 * 7.8e-6, 64 * 7.8e-6, 96 * 7.8e-6])
    tb.reset(0)
    p0, r0 = tb.world.hammer_in_hand()
    tb.run_for(0.3)
    p1, _ = tb.world.hammer_in_hand()
    assert np.linalg.norm(p1 - p0) < 0.2e-3  # the grip loop still holds on the decoded codes
    assert tb.grip.measured == pytest.approx(tb.cfg.controller.grip_hold, rel=0.2)
    palm = tb.sensors["pressure_palm"]
    assert palm.features[0] > 50  # > 5 N on the palm
    lag = np.array(palm.t_avail) - np.array(palm.t_sample)
    assert np.all(lag > 1e-3) and np.all(lag < 3e-3)
