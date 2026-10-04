"""Replay of recorded tool motions: retargeting onto our nail, the feasibility time warp, the pinned sources, and an
Adroit demonstration replayed on the FR3."""

import numpy as np
import pytest

from tactile_sim.config import fast_config
from tactile_sim.model.builder import nail_head_target, strike_axis, tcp_rotation
from tactile_sim.replay.retarget import map_motion, task_frame
from tactile_sim.replay.sources import ToolMotion, available, load

requires_adroit = pytest.mark.skipif(not available("adroit"), reason="Adroit demos not cached")
requires_dtb = pytest.mark.skipif(not available("dextoolbench"), reason="DexToolBench trajectories not cached")


def synthetic_motion(n_blows=3, period=0.6, speed=1.5, dt=0.01):
    """A hammer face swinging along +x onto a target at the origin, tool tilted 30 deg off square, wobbling 1 cm."""
    t = np.arange(0.0, n_blows * period + 0.3, dt)
    phase = (t % period) / period
    x = -0.12 * (1 - np.cos(2 * np.pi * phase)) / 2 * speed
    wob = 0.01 * np.sin(2 * np.pi * t / (n_blows * period))
    face = np.column_stack([x, wob, np.zeros_like(t)])
    tilt = np.radians(30)
    # our tool axes: x along the handle (here world -z), -y the face normal (along +x tilted about z)
    n = np.array([np.cos(tilt), np.sin(tilt), 0.0])
    Rm = np.column_stack([[0, 0, -1.0], -n, np.cross([0, 0, -1.0], -n)])
    R = np.repeat(Rm[None], len(t), axis=0)
    contacts = [int(round(k * period / dt)) for k in range(n_blows)]
    return ToolMotion("synthetic", t, face, R, contacts[0], np.array([1.0, 0, 0]), contacts)


def test_task_frame_is_a_rotation():
    F = task_frame(np.array([0.3, 0.9, -0.1]), np.array([1.0, 0, 0]))
    assert np.allclose(F.T @ F, np.eye(3), atol=1e-12) and np.linalg.det(F) == pytest.approx(1.0)
    assert np.allclose(F[:, 0], np.array([0.3, 0.9, -0.1]) / np.linalg.norm([0.3, 0.9, -0.1]))


def test_map_motion_puts_every_blow_on_the_nail_square():
    cfg = fast_config()
    m = synthetic_motion()
    face, R = map_motion(m, cfg, engage=0.004)
    nail, a = nail_head_target(cfg), strike_axis(cfg)
    for i in m.contacts:
        assert np.allclose(face[i], nail + 0.004 * a, atol=1e-3)  # every recorded contact aimed at our nail
    assert -R[m.contact_index, :, 1] @ a == pytest.approx(1.0, abs=1e-9)  # face square to the strike axis
    h = tcp_rotation(cfg)[:, 0]
    assert R[m.contact_index, :, 0] @ h > 0.85  # the handle keeps our grasp's side
    face_raw, R_raw = map_motion(m, cfg, engage=0.004, align_face=False, aim_each=False)
    assert -R_raw[m.contact_index, :, 1] @ a == pytest.approx(np.cos(np.radians(30)), abs=1e-6)


@pytest.fixture(scope="module")
def world():
    from tactile_sim.sim.world import World

    w = World(fast_config())
    w.reset()
    return w


def test_plan_slows_only_what_the_arm_cannot_follow(world):
    from tactile_sim.replay.retarget import plan_replay

    slow = plan_replay(synthetic_motion(speed=0.3), world, check_joints=False)
    assert slow.time_scale == pytest.approx(1.0, abs=0.02)  # feasible as recorded: untouched
    fast = plan_replay(synthetic_motion(speed=3.0, period=0.4), world, v_max=1.0, check_joints=False)
    assert fast.time_scale > 1.1
    assert fast.notes["v_peak"] <= 1.0 * 1.1
    assert np.all(np.diff(fast.t_contacts) > 0)


@requires_adroit
def test_adroit_demos_load_with_their_blows():
    m = load("adroit", demo=0)
    assert len(m.contacts) >= 5 and m.meta["dt"] == 0.01
    assert abs(m.axis[2]) < 1e-3  # the DAPG nail is driven horizontally
    k = m.contact_index
    assert np.linalg.norm(m.face[k] - np.array(m.meta["nail"])) < 0.05  # the face is at the nail when it touches
    assert m.normal[k] @ m.axis > 0.5  # face toward the board
    v = np.linalg.norm(np.diff(m.face, axis=0), axis=1) / 0.01
    assert v[k - 1] > 0.8  # a human blow arrives at ~1 m/s


@requires_dtb
def test_dextoolbench_hammer_loads():
    m = load("dextoolbench", task="hammer/claw_hammer/swing_side")
    assert len(m.t) > 30 and len(m.contacts) >= 1
    assert m.normal[m.contact_index] @ m.axis > 0.8


@requires_adroit
def test_adroit_replay_on_the_fr3():
    from tactile_sim.replay.runner import ReplayEpisode

    res = ReplayEpisode(fast_config(), motion=load("adroit", demo=22)).run()
    s = res.summary
    assert s["blows"] >= 2 and s["peak_force"] > 20.0
    assert s["tracking_rms"] < 0.01 and s["limit_violations"] == ""
    assert "pressure_L" in res.sensors or any(k.startswith("pressure") for k in res.sensors)
    assert s["time_scale"] > 1.5  # the FR3 cannot match a human's hand accelerations
