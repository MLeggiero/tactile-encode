"""M7: episode files and the run_strikes CLI."""

import numpy as np
import pytest

from tactile_sim import run_strikes
from tactile_sim.config import fast_config
from tactile_sim.logging.schema import SCHEMA_VERSION
from tactile_sim.logging.writer import read_episode


@pytest.fixture(scope="module")
def h5file(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs") / "ep.h5"
    assert run_strikes.main(["--n", "3", "--fast", "--out", str(out), "--seed", "1"]) == 0
    return out


def test_file_layout(h5file):
    d = read_episode(h5file)
    assert d["attrs"]["schema_version"] == SCHEMA_VERSION
    assert d["attrs"]["dt_phys"] == pytest.approx(0.00025)
    for key in ("physics/t", "physics/contact_force", "physics/nail_depth", "physics/hammer_in_hand_rotvec",
                "control/t", "control/tau", "control/x_eq", "control/mode", "control/impact_flag",
                "grip/setpoint", "events", "strikes", "sensors/ft/value", "sensors/pad_acc_L/t_avail"):
        assert key in d, key
    assert len(d["strikes"]) == 3
    assert d["summary"]["n_strikes"] == 3
    assert d["physics/contact_force"].max() > 100  # the blows are in the truth log


def test_sensor_rates_match_spec(h5file):
    d = read_episode(h5file)
    cfg = fast_config()
    T = d["physics/t"][-1] - d["physics/t"][0]
    for name, rate in [("ft", cfg.sensors.ft_rate), ("pad_acc_L", cfg.sensors.accel_rate),
                       ("pressure_L", cfg.sensors.pressure_rate), ("joint_pos", cfg.sensors.joint_rate)]:
        ts = d[f"sensors/{name}/t_sample"]
        assert len(ts) == pytest.approx(rate * T, rel=0.01), name
        assert np.allclose(np.diff(ts), 1.0 / rate, atol=1e-6)
    dt_l1 = np.diff(d["control/t"])
    assert np.allclose(dt_l1, 1.0 / cfg.controller.rate, atol=1e-6)


def test_strike_rows_are_consistent(h5file):
    d = read_episode(h5file)
    s = d["strikes"]
    assert np.all(np.diff(s["depth_after"]) >= -1e-6)
    hit = s["hit"]
    assert np.all(s["t_contact_truth"][hit] > s["t_swing_start"][hit])
    assert np.all((s["flag_latency"][hit] >= 0) & (s["flag_latency"][hit] <= 0.002))
    # the per-strike depth advance matches the truth trace
    t, depth = d["physics/t"], d["physics/nail_depth"]
    assert s["depth_after"][-1] == pytest.approx(depth[np.searchsorted(t, s["t_swing_start"][-1]) :].max(), abs=1e-4)


def test_npz_fallback(tmp_path):
    from tactile_sim.episode import Episode
    from tactile_sim.logging.writer import write_episode

    cfg = fast_config()
    res = Episode(cfg, seed=0).run(1)
    p = write_episode(res, tmp_path / "ep.npz", cfg, 0)
    d = read_episode(p)
    assert p.suffix == ".npz" and len(d["strikes"]) == 1 and d["attrs"]["schema_version"] == SCHEMA_VERSION
