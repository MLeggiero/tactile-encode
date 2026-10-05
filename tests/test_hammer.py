"""YCB hammer asset: frame, landmarks and mass properties."""

import numpy as np
import pytest

from tactile_sim.assets.ycb import ensure_hammer
from tactile_sim.config import fast_config
from tactile_sim.model.tool_hammer import hammer_geometry
from tactile_sim.sim.world import World

HAVE_YCB = ensure_hammer(fast_config().hammer.grip_from_head) is not None
requires_ycb = pytest.mark.skipif(not HAVE_YCB, reason="YCB hammer not cached")


@requires_ycb
def test_ycb_hammer_frame_and_mass():
    cfg = fast_config()
    geo = hammer_geometry(cfg.hammer)
    assert geo.source == "ycb"
    fx, fy, fz = geo.face_local
    assert fx == pytest.approx(-cfg.hammer.grip_from_head, abs=0.03)  # head axis ~190 mm from the grip
    assert fy < -0.03 and abs(fz) < 0.01  # face points to -y, in the handle's plane
    assert 0.012 < geo.face_radius < 0.02  # ~30 mm striking face
    assert 0.012 < geo.grip_half_width < 0.025
    w = World(cfg)
    b = w.hammer_body
    assert w.hammer_source == "ycb"
    assert w.model.body_mass[b] == pytest.approx(0.665, abs=1e-6)
    assert w.model.body_ipos[b][0] < -0.1  # centre of mass toward the steel head


@requires_ycb
def test_ycb_grasp_is_centred_and_loads_both_pads():
    w = World(fast_config())
    w.reset()
    f = w.pad_normal_forces()
    assert np.allclose(f, w.cfg.controller.grip_hold, rtol=0.1)
    for side in (0, 1):
        assert w.pad_taxels(side).sum() == pytest.approx(f[side], rel=1e-6)
        assert np.count_nonzero(w.pad_taxels(side) > 0.5) >= 8  # load spread over the pad, not one taxel


def test_primitive_hammer_fallback():
    w = World(fast_config(hammer={"model": "primitive"}))
    assert w.hammer_source == "primitive"
    assert w.model.body_mass[w.hammer_body] == pytest.approx(0.665, abs=1e-6)
