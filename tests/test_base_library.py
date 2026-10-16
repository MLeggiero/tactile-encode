"""The base library (behaviors.base_library): the WUJI drives' state carried across a library entry's start, the arm's
command-tick phase, a recorded episode's streams and outcomes, and an open-loop replay reproducing the closed-loop
strike that recorded it."""

import numpy as np
import pytest

from tactile_sim.assets.fetch_hands import hand_available
from tactile_sim.behaviors.base_library import (
    DRIVE_STATE,
    PARAM_KEYS,
    arm_tick_phase_ms,
    assemble,
    base_signature,
    drive_state,
    entry_stream,
    episode_config,
    replay_entry,
    set_drive_state,
)
from tactile_sim.config import WujiDriveCfg
from tactile_sim.control.interface import HandCommand
from tactile_sim.sim.wuji_drive import WujiMitDrive, joint_group

requires_vega = pytest.mark.skipif(not (hand_available("vega_1u") and hand_available("wuji2")),
                                   reason="Vega U / WUJI not cached")
FINGERS = ("index_finger", "middle_finger", "ring_finger", "pinky")
NAMES = ([f"r_thumb_{j}" for j in ("cmc_flex", "cmc_abd", "mcp", "ip")] +
         [f"r_{f}_{j}" for f in FINGERS for j in ("mcp_flex", "mcp_abd", "pip", "dip")])
RATED = {"mcp_flex": 2.0, "mcp_abd": 0.2, "pip_dip": 0.3, "thumb_cmc": 0.6, "thumb_mcp_ip": 0.3}
TAU = np.array([RATED[joint_group(n)] for n in NAMES])


def test_drive_state_carries_commands_in_flight_across_a_clock_restart():
    cfg = WujiDriveCfg(cmd_latency=0.002, cmd_jitter=0.0, sdk_lpf_hz=100.0, tau_i=0.0003)
    dt = 0.00025
    a = WujiMitDrive(cfg, NAMES, TAU, dt, np.random.default_rng(0))
    rng = np.random.default_rng(1)
    t = 0.0
    for k in range(37):  # mid-cycle, with commands in transit
        if k % 4 == 0:
            a.set_host_command(HandCommand.from_array(t, rng.normal(0, 0.3, (5, 20))))
        a.update(t, rng.normal(0, 0.2, 20), rng.normal(0, 1, 20))
        t += dt
    assert len(a.line) > 0
    st = drive_state(a, t)
    assert set(st) == set(DRIVE_STATE)
    b = WujiMitDrive(cfg, NAMES, TAU, dt)
    set_drive_state(b, st, t0=0.0)  # the same drive, its clock restarted at 0
    for k in range(40):
        q, qd = rng.normal(0, 0.2, 20), rng.normal(0, 1, 20)
        if k % 4 == 1:
            cmd = rng.normal(0, 0.3, (5, 20))
            a.set_host_command(HandCommand.from_array(t, cmd))
            b.set_host_command(HandCommand.from_array(k * dt, cmd))
        assert np.allclose(a.update(t, q, qd), b.update(k * dt, q, qd), atol=1e-12)
        t += dt


def test_arm_ticks_fall_on_the_absolute_10_ms_grid():
    assert [arm_tick_phase_ms(t) for t in (0.705, 1.2, 2.165000001, 3.0049999)] == [5, 0, 5, 5]


def test_episode_configuration_and_base_signature():
    cfg, draw = episode_config(3)
    assert cfg.scene.reduced and cfg.physics.timestep == 0.00025 and cfg.physics.noslip_iterations == 0
    assert cfg.hammer.head_mass == draw["hammer"]["head_mass"]
    nominal, none = episode_config(3, randomize=False)
    assert none is None and nominal.hammer.head_mass != cfg.hammer.head_mass
    sig = base_signature(cfg)
    assert sig == base_signature(nominal)  # the randomization does not change the base
    assert set(sig) >= {"grip_loop", "grip_profile", "grip_margin_step"}


@pytest.fixture(scope="module")
def recorded():
    from tactile_sim.behaviors.base_library import record_episode

    return record_episode(5, n_strikes=1)


@requires_vega
def test_a_recorded_swing_has_its_streams_state_and_outcome(recorded):
    ep = recorded
    assert len(ep["entries"]) == 1 and set(ep["params"]) == set(PARAM_KEYS)
    e = ep["entries"][0]
    rows = ep["rows"]
    assert e["len"] > 1000 and np.all(np.diff(rows["t"]) > 0)
    assert e["outcome"]["hit"] and e["outcome"]["advance"] > 0.5e-3
    assert e["outcome"]["peak2"] <= e["outcome"]["peak"]
    # the arm's stream only changes on its 100 Hz command ticks
    q = rows["arm_q"][e["k"]:e["k"] + e["len"]]
    changes = np.flatnonzero(np.any(np.diff(q, axis=0) != 0, axis=1)) + 1
    ph = arm_tick_phase_ms(e["t0"])
    assert len(changes) > 50 and np.all((changes - ph) % 10 == 0)
    lib = assemble([ep])
    st = entry_stream(lib, 0)
    assert st["arm_q"].shape == (e["len"], 7) and st["look"].shape == (e["len"], 21)
    assert np.nanmin(st["fc_t_event"][st["fc_valid"]]) > 0.0  # the predicted contact lies ahead of the windup


@requires_vega
def test_replaying_an_entry_open_loop_reproduces_the_strike(recorded):
    lib = assemble([recorded])
    r = replay_entry(lib, 0)["outcome"]
    rec = {k: float(lib[f"e_{k}"][0]) for k in ("t_contact", "advance", "peak2", "v_face")}
    assert r["hit"]
    assert abs(r["t_contact"] - rec["t_contact"]) <= 0.0015
    assert r["advance"] == pytest.approx(rec["advance"], rel=0.1)
    assert r["peak2"] == pytest.approx(rec["peak2"], rel=0.1)
    assert r["v_face"] == pytest.approx(rec["v_face"], abs=0.03)
