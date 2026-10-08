"""Grasp keyframe cache: "nominal" grasp masses share one keyframe across randomized tool masses; "actual" keeps
one per mass, with the same keys as before the option existed."""

import pytest

from tactile_sim.config import fast_config, hand_config
from tactile_sim.model.hands.wuji_grasp import cache_path, grasp_config


@pytest.fixture
def hand_file(tmp_path):
    p = tmp_path / "hand.xml"
    p.write_text("<mujoco/>")
    return p


def cfg(mode: str, head_scale: float = 1.0):
    c = hand_config("wuji2", fast_config(), gripper={"grasp_mass": mode})
    return c.replace(hammer={"head_mass": c.hammer.head_mass * head_scale})


def test_nominal_mode_ignores_the_tool_mass(hand_file):
    assert cache_path(cfg("nominal", 0.8), hand_file) == cache_path(cfg("nominal", 1.2), hand_file)
    assert grasp_config(cfg("nominal", 0.8)).hammer.head_mass == pytest.approx(0.45)


def test_actual_mode_keys_on_the_mass_as_before(hand_file):
    assert cache_path(cfg("actual", 0.8), hand_file) != cache_path(cfg("actual", 1.2), hand_file)
    # at the nominal mass both modes are the same keyframe
    assert cache_path(cfg("actual", 1.0), hand_file) == cache_path(cfg("nominal", 1.3), hand_file)


def test_nominal_mode_ignores_the_skin_contact_too(hand_file):
    a = cfg("nominal").replace(gripper={"hand_friction": 0.6, "hand_solref": (0.003, 1.0)})
    assert cache_path(a, hand_file) == cache_path(cfg("nominal"), hand_file)
    b = cfg("actual").replace(gripper={"hand_friction": 0.6})
    assert cache_path(b, hand_file) != cache_path(cfg("actual"), hand_file)


def test_geometry_still_changes_the_key(hand_file):
    a = cfg("nominal")
    b = a.replace(hammer={"grip_from_head": a.hammer.grip_from_head + 0.02})
    assert cache_path(a, hand_file) != cache_path(b, hand_file)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        grasp_config(cfg("heavy"))
