"""Saw and driver tasks: the plant models, and the FR3 + Franka Hand running each with its scripted behavior."""

import numpy as np
import pytest

from tactile_sim.config import fast_config, task_config
from tactile_sim.model.plants import DrillPlant, SawPlant


def saw_cfg(**over):
    return task_config("saw", fast_config(), **over)


def drill_cfg(**over):
    return task_config("drill", fast_config(), **over)


# ---------------------------------------------------------------- models (no simulation)
def test_saw_model():
    p = SawPlant(saw_cfg(saw={"knot": (0.005, 0.010, 3.0)}))
    k = p.cfg.saw.k_cut
    assert p.removal_resistance(0.002) == k and p.removal_resistance(0.007) == 3 * k  # the knot is harder to cut
    assert p.cutting_stroke(0.3) and not p.cutting_stroke(-0.3)  # a western saw cuts on the push
    assert SawPlant(saw_cfg(saw={"cut_on": "pull"})).cutting_stroke(-0.3)


def test_screw_model():
    p = DrillPlant(drill_cfg())
    dc = p.cfg.drill
    taus = [p.screw_torque(z) for z in np.linspace(0, dc.screw_len, 5)]
    assert np.all(np.diff(taus) > 0) and taus[-1] < dc.clutch_torque  # drives in under the clutch setting
    assert p.screw_torque(dc.screw_len + 0.001) > dc.clutch_torque  # seating: the clutch must slip
    assert p.cam_limit(60.0, 0.0) == pytest.approx(dc.cam_ratio * 60.0)
    assert p.cam_limit(60.0, 0.1) < p.cam_limit(60.0, 0.0)  # a tilted bit cams out sooner
    assert p.cam_limit(60.0, dc.cam_max_angle) == 0.0
    assert p.feed_per_rev(dc.thrust_f0) == 0.0 < p.feed_per_rev(dc.thrust_f0 + 50)
    assert p.motor_torque(1.0, 0.0) == dc.stall_torque and p.motor_torque(1.0, dc.free_speed) == 0.0


def test_task_config():
    assert task_config("nail").plant.kind == "nail"
    with pytest.raises(ValueError):
        task_config("chisel")
    from tactile_sim.model.builder import build_scene

    with pytest.raises(ValueError):
        build_scene(task_config("saw", fast_config(gripper={"hand": "wuji2"})))


# ---------------------------------------------------------------- tasks on the FR3 + Franka Hand
def run(cfg, t_max):
    from tactile_sim.tool_task import ToolTask

    return ToolTask(cfg, t_max=t_max).run()


@pytest.fixture(scope="module")
def saw_run():
    return run(saw_cfg(saw={"cut_target": 0.004}), 8.0)


def test_saw_cuts_within_limits(saw_run):
    s, T = saw_run.summary, saw_run.truth
    assert s["done"] and s["progress"] >= 0.004
    assert s["strokes"] >= 3
    v = T["stroke_speed"]
    assert v.max() > 0.2 and v.min() < -0.2  # the load at the grip reverses with every stroke
    assert np.max(T["saw_cut_force"]) > 10.0
    # the pads flex a few degrees with each stroke and spring back; what remains is slip
    assert np.degrees(s["max_slip_rot"]) < 10.0
    assert s["net_slip_trans"] < 0.003 and np.degrees(s["net_slip_rot"]) < 2.0
    assert s["limit_violations"] == ""


@pytest.fixture(scope="module")
def screw_run():
    return run(drill_cfg(), 6.0)


def test_driver_seats_a_screw(screw_run):
    s, T = screw_run.summary, screw_run.truth
    assert s["done"] and s["cam_outs"] == 0
    assert s["progress"] >= 0.022 - 1e-4
    assert np.any(T["clutch_slipping"] > 0)  # seating ends with the clutch ratcheting
    assert s["peak_torque"] <= 1.05 * screw_run_clutch()
    # the clutch's ratchet peaks (~2.5 Nm) beat the pads' hold about the bit for a moment: a few degrees of roll
    assert np.degrees(s["net_slip_rot"]) < 4.0 and s["limit_violations"] == ""


def screw_run_clutch() -> float:
    return drill_cfg().drill.clutch_torque


def test_light_push_cams_out():
    res = run(drill_cfg(drill={"push_force": 35.0}), 4.0)
    s = res.summary
    assert s["cam_outs"] >= 1 and not s["done"]  # the screw's torque outgrows what a 35 N push holds in the recess
    assert s["state"] == "stripped"


def test_drill_breaks_through():
    res = run(drill_cfg(drill={"mode": "hole", "board_thickness": 0.008}), 6.0)
    s, T = res.summary, res.truth
    assert s["done"] and s["state"] == "through"
    i = int(np.argmax(T["drill_depth"] >= 0.008))
    assert T["drill_axial_force"][i - 50:i].max() > 5.0  # supported until the exit
    # through: the bit loses its support (the jerk the arm must catch)
    assert T["drill_axial_force"][i + 5:].max() < 1.0
    assert s["limit_violations"] == ""
