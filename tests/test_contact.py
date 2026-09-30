"""M2: hammer-nail contact calibration with the free hammer (no arm in the loop)."""

import numpy as np
import pytest

from tactile_sim.calibrate import free_strike
from tactile_sim.config import SimConfig, fast_config
from tactile_sim.sim.truth import pulse_stats


def test_pulse_stats_bridges_chatter():
    t = np.arange(0, 0.02, 0.00025)
    f = np.where((t > 0.002) & (t < 0.005), 500.0, 0.0)
    f[(t > 0.0033) & (t < 0.0036)] = 0.0  # one-sample chatter gap
    st = pulse_stats(t, f)
    assert st["peak"] == 500.0
    assert st["width"] == pytest.approx(0.003, abs=0.0005)
    assert st["impulse"] == pytest.approx(np.trapezoid(f, t), rel=1e-9)


@pytest.fixture(scope="module")
def strike_25():
    return free_strike(fast_config(), 2.5)


def test_nominal_strike(strike_25):
    st = strike_25
    assert 0.001 <= st["width"] <= 0.006, st["width"]  # ~3 ms class pulse
    assert 100.0 <= st["peak"] <= 2000.0
    assert 0.0005 <= st["advance"] <= 0.005
    assert st["max_penetration"] < 0.0015  # never through the 1.5 mm nail head


def test_energy_bound(strike_25):
    cfg = fast_config()
    m_hammer = cfg.hammer.head_mass + cfg.hammer.handle_mass
    ke = 0.5 * m_hammer * 2.5**2
    assert strike_25["advance"] * cfg.plant.resistance_0 <= ke


def test_no_tunnelling_at_high_speed():
    st = free_strike(fast_config(), 6.0)
    assert st["max_penetration"] < 0.0015
    assert st["advance"] < fast_config().plant.proud  # hammer did not ride the nail through its stop


def test_timestep_consistency(strike_25):
    st8 = free_strike(SimConfig(), 2.5)
    assert st8["advance"] == pytest.approx(strike_25["advance"], rel=0.2)
    assert st8["peak"] == pytest.approx(strike_25["peak"], rel=0.2)


def test_restitution_falls_with_damping():
    base = fast_config()
    at_stop = base.plant.proud - 0.001
    es = [free_strike(base.replace(plant={"vdr_zeta0": z, "vdr_zeta1": 0.0}), 2.5, depth0=at_stop)["e"]
          for z in (0.2, 0.6, 1.0)]
    assert es[0] > es[1] > es[2] > 0.0


def test_velocity_dependent_restitution_sets_damping():
    cfg = fast_config(plant={"vdr_zeta0": 0.3, "vdr_zeta1": 0.1, "vdr_zeta_max": 1.0})
    from tactile_sim.sim.world import World

    w = World(cfg)
    free_strike(cfg, 4.0, world=w)
    assert w.plant.contact_zeta == pytest.approx(0.3 + 0.1 * 4.0, abs=0.05)
    k = w.plant.k_face
    assert w.model.pair_solref[w.plant.pair_id, 0] == pytest.approx(-k)


def test_resistance_grows_with_depth():
    cfg = fast_config()
    a0 = free_strike(cfg, 2.5, depth0=0.0)["advance"]
    a16 = free_strike(cfg, 2.5, depth0=0.016)["advance"]
    assert a16 < a0


def test_too_weak_strike_does_not_move_nail():
    cfg = fast_config(plant={"resistance_0": 1500.0})
    assert free_strike(cfg, 1.5)["advance"] < 1e-4
