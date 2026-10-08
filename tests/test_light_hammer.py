"""The light hammer and the residual experiments' configuration on the Vega U + WUJI."""

import numpy as np
import pytest

from tactile_sim.assets.fetch_hands import hand_available
from tactile_sim.config import LIGHT_HAMMER_OVERRIDES, fast_config, residual_config
from tactile_sim.model.hands.wuji_grasp import grasp_config

requires_wuji = pytest.mark.skipif(not hand_available("wuji2"), reason="WUJI Hand 2 not cached")


def test_residual_config_composes_the_experiment():
    cfg = residual_config(rate=250.0)
    assert cfg.arm.robot == "vega_1u" and cfg.gripper.hand == "wuji2" and cfg.gripper.hand_control == "mit"
    assert cfg.sensors.ts_model == "rev3" and cfg.residual.enabled and cfg.residual.rate == 250.0
    assert cfg.hammer.model == "primitive" and cfg.hammer.head_mass + cfg.hammer.handle_mass <= 0.35
    assert cfg.plant.resistance_0 == 150.0  # a small nail the light hammer can drive
    assert cfg.physics.timestep == 0.000125  # the 8 kHz validator unless a base is given
    assert residual_config(base=fast_config()).physics.timestep == 0.00025


def test_randomized_head_masses_share_the_light_hammer_grasp():
    a = residual_config(hammer={"head_mass": 0.5})
    g = grasp_config(a)
    assert g.hammer.head_mass == LIGHT_HAMMER_OVERRIDES["hammer"]["head_mass"]
    assert g.hammer.handle_mass == LIGHT_HAMMER_OVERRIDES["hammer"]["handle_mass"]


@requires_wuji
def test_the_vega_hammers_with_the_light_hammer_through_the_residual():
    from tactile_sim.behaviors.base_sources import SwingBase
    from tactile_sim.episode import Episode

    ep = Episode(residual_config(base=fast_config(), forecast={"mask_prob": 0.0}), seed=0)
    w = ep.tb.world
    assert w.model.body_mass[w.hammer_body] == pytest.approx(0.33, abs=1e-6)
    ep.tb.residual.base_fn = SwingBase(ep.tb, ep.swing, rng=np.random.default_rng(0))
    res = ep.run(2)
    assert res.summary["n_strikes"] == 2 and res.summary["total_depth"] > 2.5e-3  # ~2 mm a strike
    assert not ep.tb.grip.dropped
