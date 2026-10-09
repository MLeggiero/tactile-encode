"""The 12 V drill/driver (plants/drill.py on plants/drivetrain.py): the drivetrain's physics, the virtual dyno
against the nominal tool's published ratings, the hole's laws, a clutch set too high, the vibration weighting, the
held drill's vibration (its spectrum, and its rating as an upper bound), and the v2 preset reproducing the previous
model's traces."""

import json
from pathlib import Path

import numpy as np
import pytest

from tactile_sim.assets.fetch_hands import hand_available
from tactile_sim.config import (
    DRILL_V2_PRESET,
    PRIMITIVE_PISTOL_GRIPPER,
    VEGA_DRILL_OVERRIDES,
    VEGA_DRILL_V2_PRESET,
    DrillCfg,
    fast_config,
    hand_config,
    task_config,
)
from tactile_sim.model.plants import DrillPlant, drill_dyno
from tactile_sim.model.plants.drivetrain import Drivetrain, DrivetrainParams

requires_wuji = pytest.mark.skipif(not hand_available("wuji2"), reason="WUJI Hand 2 not cached")
P = DrivetrainParams.from_cfg(DrillCfg())
DT = 1e-4


def drill_cfg(**drill):
    return task_config("drill", fast_config(), drill=drill) if drill else task_config("drill", fast_config())


def vega_cfg(**drill):
    cfg = hand_config("wuji2", fast_config(), robot="vega_1u").replace(**VEGA_DRILL_OVERRIDES)
    return cfg.replace(drill=drill) if drill else cfg


def run(cfg, t_max, seed=0):
    from tactile_sim.tool_task import ToolTask

    return ToolTask(cfg, seed=seed, t_max=t_max).run()


# ---------------------------------------------------------------- the drivetrain alone
def test_motor_line_and_brake():
    dr = Drivetrain(P)
    assert dr.motor_torque(1.0, 0.0) == P.stall_torque and dr.motor_torque(1.0, P.free_speed) == 0.0
    assert dr.motor_torque(0.5, 0.0) == pytest.approx(0.5 * P.stall_torque)  # the PWM moves the line, same slope
    assert dr.motor_torque(0.0, P.free_speed) == -P.brake_torque  # released: the shorted motor brakes, capped
    assert dr.motor_torque(0.0, 0.01 * P.free_speed) == pytest.approx(-0.01 * P.stall_torque)


def test_a_free_spin_up_kicks_the_hand_with_the_motor_shafts_torque():
    dr = Drivetrain(P)
    kick = []
    for _ in range(int(0.3 / DT)):
        dr.step(1.0, 0.0, DT)
        kick.append(dr.tau_housing)
    # nothing on the chuck: the housing feels the internals' angular momentum change, J_am dw/dt, which is the
    # stall torque scaled by J_am / J_spin (~1/N), not the stall torque itself (v2's housing torque was -tau_m)
    assert max(kick) == pytest.approx(P.stall_torque * P.momentum_inertia / P.spindle_inertia, rel=0.02)
    assert max(kick) < 0.05 * P.stall_torque
    assert drill_dyno.momentum_residual(P) < 1e-9


def test_the_clutch_holds_then_ratchets_at_the_motors_pace():
    dr = Drivetrain(P)
    for _ in range(int(0.3 / DT)):
        dr.step(1.0, 0.0, DT)
    brk = P.clutch_torque * (1 + P.clutch_overshoot)
    for need in np.linspace(0.0, 0.999 * brk, 200):  # below the breakaway the clutch is rigid
        dr.step(1.0, need, DT)
        assert not dr.slipping and dr.tau_chuck == pytest.approx(need)
    dr.step(1.0, 1.01 * brk, DT)
    assert dr.slipping
    ct = drill_dyno.clutch_test(P)
    assert ct["peak_torque"] <= P.clutch_torque + 1e-9
    assert P.clutch_low * P.clutch_torque * P.clutch_duty < ct["mean_torque"] < P.clutch_torque
    assert ct["click_rate"] == pytest.approx(ct["click_rate_model"], rel=0.05)
    assert ct["slip_speed"] == pytest.approx(P.steady_speed(1.0, ct["mean_torque"]), rel=0.03)
    assert 40.0 < ct["click_rate"] < 60.0  # ~50 clicks a second for the nominal tool at a full trigger


def test_the_dyno_reproduces_the_nominal_tools_ratings():
    # the Makita DF333D: 0-450 rpm in low gear, 14 / 30 Nm soft / hard joint (ISO 5393), 170 W
    rpm = drill_dyno.no_load_speed(P) * 60 / (2 * np.pi)
    assert 440 < rpm < 452
    assert drill_dyno.soft_torque(P) == pytest.approx(14.0, abs=0.5)
    assert drill_dyno.hard_torque(P) == pytest.approx(30.0, abs=1.0)
    assert drill_dyno.fit_spindle_inertia(P, 30.0) == pytest.approx(P.spindle_inertia, rel=0.03)
    assert P.stall_torque * P.free_speed / 4 == pytest.approx(170.0, rel=0.05)  # the peak power
    assert 0.1 < drill_dyno.brake_stop_time(P) < 0.3


def test_the_wh_weighting():
    fs = 4000.0
    t = np.arange(0.0, 3.0, 1 / fs)

    def a_hv(f):
        acc = np.zeros((len(t), 3))
        acc[:, 0] = np.sin(2 * np.pi * f * t)
        return drill_dyno.wh_rms(acc, fs, skip=1.0)

    assert a_hv(10.0) == pytest.approx(1 / np.sqrt(2), rel=0.1)  # the flat band
    assert a_hv(160.0) == pytest.approx(0.1 / np.sqrt(2), rel=0.15)  # -6 dB per octave above 16 Hz
    assert a_hv(1.0) < 0.05  # below the band


# ---------------------------------------------------------------- the plant's laws (no simulation)
def test_the_holes_laws():
    p = DrillPlant(drill_cfg(mode="hole"))
    dc = p.cfg.drill
    f = p.feed_per_rev(70.0)
    assert 3e-4 < f < 7e-4  # ~0.5 mm a turn under a 70 N push
    # a 4 mm bit cuts with ~0.1 Nm, far under the clutch: the breakthrough's thrust drop is the event, not torque
    assert 0.05 < p.hole_torque(f, 0.0) < 0.2 * dc.clutch_torque
    assert p.hole_torque(f, 0.03) > p.hole_torque(f, 0.0) and p.hole_torque(2 * f, 0.0) > p.hole_torque(f, 0.0)


# ---------------------------------------------------------------- tasks on the FR3 + Franka Hand
@pytest.mark.parametrize("over, state", [({"clutch_torque": 4.0, "strip_torque": 2.5}, "spun"),
                                         ({"clutch_torque": 4.0, "screw_twist_off": 2.2}, "snapped")])
def test_a_clutch_set_too_high_ruins_the_screw(over, state):
    s = run(drill_cfg(**over), 4.0).summary
    assert s["state"] == state and not s["done"]
    assert s["peak_torque"] < max(over.get("strip_torque", 9.0), over.get("screw_twist_off", 0.0)) + 0.1


# ---------------------------------------------------------------- the held drill's vibration
@requires_wuji
@pytest.mark.slow
def test_the_held_drill_vibrates_at_the_motors_speed_within_its_rating():
    from scipy.signal import welch

    from tactile_sim.tool_task import ToolTask

    cfg = vega_cfg()
    tt = ToolTask(cfg, seed=0, t_max=6.0)
    w, plant = tt.tb.world, tt.tb.world.plant
    acc, phase, speed = [], [], []

    def probe(t):
        acc.append(w.sensor("hammer_acc").copy())
        phase.append(tt.behavior.phase)
        speed.append(plant.w)

    tt.tb.sched.register("imu_probe", 1.0 / w.dt, probe)
    assert tt.run().summary["done"]
    m = np.array(phase) == "drive"
    a, fs = np.array(acc)[m], 1.0 / w.dt
    # 12 V drill/drivers declare a_h < 2.5 m/s^2 for screwdriving (EN 62841-2-1, the hand on the grip). It bounds the
    # drill's shaking from above only: the Wh weighting falls 6 dB an octave above 16 Hz, so the motor's ~270 Hz
    # unbalance barely counts (the weighted value comes from the drive's own motion)
    assert drill_dyno.wh_rms(a, fs, skip=0.2) < 2.5
    # the motor's unbalance stands out of the tool's spectrum at the rotor's speed
    f, P = welch(a - a.mean(0), fs=fs, nperseg=1024, axis=0)
    P = P.sum(1)
    f_motor = cfg.drill.gear_ratio * float(np.median(np.array(speed)[m])) / (2 * np.pi)

    def band(lo, hi):
        return P[(f >= lo) & (f < hi)].mean()

    side = 0.5 * (band(f_motor - 90, f_motor - 40) + band(f_motor + 40, f_motor + 90))
    assert 240 < f_motor < 300 and band(f_motor - 15, f_motor + 15) > 10 * side


# ---------------------------------------------------------------- the v2 preset reproduces v2
REF = json.loads((Path(__file__).parent / "data" / "drill_v2_ref.json").read_text())


def _compare(name, cfg, t_max):
    ref = REF["runs"][name]
    res = run(cfg, t_max)
    for k in REF["keys"]:
        got = res.truth[k][::REF["every_ms"]]
        np.testing.assert_allclose(got, np.asarray(ref[k]), rtol=1e-7, atol=1e-7, err_msg=f"{name}: {k}")


@pytest.mark.parametrize("name, over, t_max", [("fr3_screw", {}, 3.0),
                                               ("fr3_hole", {"mode": "hole", "board_thickness": 0.008}, 4.0)])
def test_the_v2_preset_reproduces_v2(name, over, t_max):
    _compare(name, drill_cfg(**{**DRILL_V2_PRESET, **over}), t_max)


@requires_wuji
def test_the_v2_preset_reproduces_v2_on_the_vega():
    _compare("vega_screw", vega_cfg(**VEGA_DRILL_V2_PRESET).replace(gripper=PRIMITIVE_PISTOL_GRIPPER), 3.0)
