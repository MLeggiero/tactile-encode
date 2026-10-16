"""WUJI Hand 2 MIT-mode drive model: SDK cycle, transport delay, servo law, current clamp and loop, thermal
derating, readback sensors, and the hand running its grasp through the drives."""

import numpy as np
import pytest

from tactile_sim.assets.fetch_hands import hand_available
from tactile_sim.config import WujiDriveCfg, fast_config, hand_config
from tactile_sim.control.hand import WrapSynergy
from tactile_sim.control.interface import HandCommand
from tactile_sim.sim.wuji_drive import JOINT_GROUPS, WujiMitDrive, joint_group

requires_wuji = pytest.mark.skipif(not hand_available("wuji2"), reason="WUJI Hand 2 not cached")

FINGERS = ("index_finger", "middle_finger", "ring_finger", "pinky")
NAMES = ([f"r_thumb_{j}" for j in ("cmc_flex", "cmc_abd", "mcp", "ip")] +
         [f"r_{f}_{j}" for f in FINGERS for j in ("mcp_flex", "mcp_abd", "pip", "dip")])
RATED = {"mcp_flex": 2.0, "mcp_abd": 0.2, "pip_dip": 0.3, "thumb_cmc": 0.6, "thumb_mcp_ip": 0.3}
TAU = np.array([RATED[joint_group(n)] for n in NAMES])
DT = 0.00025


def drive(**over) -> WujiMitDrive:
    base = dict(cmd_latency=0.0, cmd_jitter=0.0, tau_i=0.0)
    base.update(over)
    return WujiMitDrive(WujiDriveCfg(**base), NAMES, TAU, DT, np.random.default_rng(0))


def cmd(t=0.0, q=0.0, qd=0.0, kp=0.0, kd=0.0, tau=0.0, n=20) -> HandCommand:
    f = np.full
    return HandCommand(t, f(n, q, float), f(n, qd, float), f(n, kp, float), f(n, kd, float), f(n, tau, float))


def run(dr: WujiMitDrive, t0: float, steps: int, q=None, qd=None) -> list[np.ndarray]:
    q = np.zeros(dr.n) if q is None else q
    qd = np.zeros(dr.n) if qd is None else qd
    return [dr.update(t0 + k * DT, q, qd) for k in range(steps)]


def test_joint_groups_cover_both_hands():
    for n in NAMES:
        assert joint_group(n) in JOINT_GROUPS
        assert joint_group(n.replace("r_", "l_", 1)) == joint_group(n)
    assert [joint_group(n) for n in NAMES[:4]] == ["thumb_cmc", "thumb_cmc", "thumb_mcp_ip", "thumb_mcp_ip"]
    with pytest.raises(ValueError):
        joint_group("r_wrist")


def test_no_torque_before_the_first_command():
    dr = drive()
    assert np.all(run(dr, 0.0, 8)[-1] == 0.0)


def test_mit_law_once_the_command_arrives():
    dr = drive()
    q = np.linspace(-0.2, 0.2, 20)
    qd = np.linspace(1.0, -1.0, 20)
    dr.set_host_command(cmd(q=0.1, qd=0.5, kp=0.3, kd=0.01, tau=0.02))
    tau = run(dr, 0.0, 2, q, qd)[-1]
    want = 0.3 * (0.1 - q) + 0.01 * (0.5 - qd) + 0.02
    assert tau == pytest.approx(want, abs=1e-12)


def test_transport_delay_and_sdk_cycle():
    dr = drive(cmd_latency=0.001)
    dr.set_host_command(cmd(tau=0.1))
    out = run(dr, 0.0, 8)  # the SDK latches at t = 0; the command reaches the drivers at 1 ms
    on = [bool(np.all(x > 0)) for x in out]
    assert on == [False] * 4 + [True] * 4
    # a command written between SDK cycles waits for the next one (1 ms later), then for the transport
    dr2 = drive(cmd_latency=0.001)
    run(dr2, 0.0, 1)  # cycle at t = 0 with no command
    dr2.set_host_command(cmd(tau=0.1))
    out2 = run(dr2, DT, 11)
    first = next(k for k, x in enumerate(out2) if np.all(x > 0))
    assert (first + 1) * DT == pytest.approx(0.002)


def test_jitter_never_reorders_commands():
    dr = drive(cmd_latency=0.001, cmd_jitter=0.0009)
    seen = []
    for k in range(400):
        t = k * DT
        dr.set_host_command(cmd(t=t, tau=float(k) * 1e-4))
        dr.update(t, np.zeros(20), np.zeros(20))
        assert np.all(np.diff([a for a, _ in dr.line]) >= 0)
        seen.append(dr.active[4, 0])
    assert np.all(np.diff(seen) >= 0)  # a driver never sees an older command after a newer one
    assert seen[-1] > 0


def test_sdk_low_pass_on_the_position_target():
    f_c = 50.0
    dr = drive(sdk_lpf_hz=f_c, sdk_lpf_fields=("q",))
    dr.set_host_command(cmd(q=0.0, kp=1.0))
    run(dr, 0.0, 4)  # the first SDK cycle starts the filter at q* = 0
    dr.set_host_command(cmd(q=0.1, kp=1.0, tau=0.05))  # stays under every joint's rating
    taus = run(dr, 4 * DT, 4 * 40)  # 40 SDK cycles
    q_star = np.array([x[0] - 0.05 for x in taus[3::4]])  # tau = kp (q* - 0) + 0.05
    a = 1.0 - np.exp(-2 * np.pi * f_c / 1000.0)
    want = 0.1 * (1.0 - (1.0 - a) ** np.arange(1, 41))
    assert q_star == pytest.approx(want, rel=1e-9)
    assert taus[0][0] == pytest.approx(0.1 * a + 0.05)  # the feedforward is not filtered


def test_current_is_clamped_at_the_effort_limit_and_lags_the_command():
    dr = drive()
    dr.set_host_command(cmd(tau=100.0))
    assert run(dr, 0.0, 2)[-1] == pytest.approx(TAU)  # effort_limit = rated current
    dr = drive(effort_limit_ratio=1.5, i_peak_ratio=1.2)
    dr.set_host_command(cmd(tau=-100.0))
    assert run(dr, 0.0, 2)[-1] == pytest.approx(-1.2 * TAU)  # never past the driver's peak
    tau_i = 0.0005
    dr = drive(tau_i=tau_i)
    dr.set_host_command(cmd(tau=0.1))
    out = run(dr, 0.0, 8)  # the command is active from step 0; the current rises with tau_i
    k = 2  # three steps of 0.25 ms = 1.5 tau_i
    assert out[k][0] == pytest.approx(0.1 * (1 - np.exp(-(k + 1) * DT / tau_i)), rel=1e-9)


def test_a_torque_constant_error_reaches_the_joint_not_the_current():
    """WujiDriveCfg.kt_scale: the SDK converts with the nominal Kt, the motors deliver kt_scale x the torque per amp;
    the current (what the host reads back) and its ceiling do not move."""
    ref, err = drive(), drive(kt_scale=1.2)
    for d in (ref, err):
        d.set_host_command(cmd(tau=0.1))
    a, b = run(ref, 0.0, 4)[-1], run(err, 0.0, 4)[-1]
    assert b == pytest.approx(1.2 * a, rel=1e-12) and err.i == pytest.approx(ref.i, rel=1e-12)
    ref, err = drive(), drive(kt_scale=0.85)
    for d in (ref, err):
        d.set_host_command(cmd(tau=100.0))  # at the current ceiling
    a, b = run(ref, 0.0, 2)[-1], run(err, 0.0, 2)[-1]
    assert a == pytest.approx(TAU) and b == pytest.approx(0.85 * TAU) and err.i == pytest.approx(ref.i)


def test_winding_heats_derates_and_faults():
    c = dict(effort_limit_ratio=2.0, i_peak_ratio=2.0, c_th=0.05, r_th=50.0, t_warn=40.0, t_max=60.0)
    dr = drive(**c)
    dr.set_host_command(cmd(tau=100.0))
    out, temps = [], []
    for k in range(4000):  # 1 s asking for twice the rated current
        out.append(dr.update(k * DT, np.zeros(20), np.zeros(20)))
        temps.append(dr.T[0])
    out = np.array(out)
    assert max(temps) >= 60.0 and np.all(dr.fault)
    assert np.all(out[-1] == 0.0) and dr.T[0] < 60.0  # off and cooling, still faulted
    # the ceiling falls from twice the rating once the winding passes t_warn, to derate_floor of it at t_max
    k_warn = int(np.argmax(np.abs(out[:, 0]) < 2.0 * TAU[0] - 1e-9))
    assert temps[k_warn] > 40.0 and temps[k_warn - 1] <= 40.0 + 0.1
    k_fault = int(np.argmax(np.abs(out[:, 0]) == 0.0))
    assert out[k_fault - 1, 0] == pytest.approx(0.5 * 2.0 * TAU[0], rel=0.01)
    dr.reset()
    assert not dr.fault.any() and np.all(dr.T == 25.0)


def test_servo_rate_holds_the_torque_between_servo_ticks():
    dr = drive(servo_rate=1000.0)
    dr.set_host_command(cmd(kp=1.0, q=0.0))
    q = np.zeros(20)
    taus = []
    for k in range(8):
        q = q - 0.01  # the joint moves every physics step
        taus.append(dr.update(k * DT, q, np.zeros(20))[0])
    assert taus[0] == taus[1] == taus[2] == taus[3] != taus[4]


def test_time_shift_rebases_the_drive_clock():
    dr = drive(cmd_latency=0.001)
    dr.reset(0.4)
    dr.set_host_command(cmd(tau=0.1))
    run(dr, 0.4, 2)  # latched by the SDK at t = 0.4 s, still in transit
    dr.shift_time(-0.4)
    out = run(dr, 0.0005, 6)
    assert next(k for k, x in enumerate(out) if np.all(x > 0)) == 2  # arrives at 1 ms on the new clock


@pytest.mark.parametrize("synergy", ["rated", "tendon"])
def test_synergy_as_mit_commands_matches_the_joint_law(synergy):
    joints = list(zip(NAMES, TAU, strict=True))
    syn = WrapSynergy.for_wuji(joints, (1, 0, 1, 1), (0.0, -0.8, 0.0, 0.0), synergy)
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = rng.uniform(0, 1)
        q, qd = rng.uniform(-0.5, 1.2, 20), rng.uniform(-3, 3, 20)
        qs, qds, kp, kd, tau_ff = syn.mit_command(s, q)
        tau = np.clip(kp * (qs - q) + kd * (qds - qd) + tau_ff, -TAU, TAU)
        assert tau == pytest.approx(syn.torque(s, q, qd), abs=1e-12)


# ---------------------------------------------------------------- the hand on the testbed


def mit_cfg(**over):
    return hand_config("wuji2", fast_config(), gripper={"hand_control": "mit"}, **over)


@pytest.fixture(scope="module")
def mit_tb():
    from tactile_sim.sim.testbed import Testbed

    return Testbed(mit_cfg())


@requires_wuji
def test_mit_drives_hold_the_wrap_and_report_through_the_sdk(mit_tb):
    tb = mit_tb
    tb.reset(0)
    hand = tb.world.hand
    assert hand.drive is not None and "hand_q" in tb.sensors and "hand_current" in tb.sensors
    m = tb.world.model
    assert m.actuator_ctrlrange[hand.act][:, 1] == pytest.approx(2.0 * hand.tau_max)  # the drives clamp instead
    p0, r0 = tb.world.hammer_in_hand()
    tb.run_for(0.5)
    p1, r1 = tb.world.hammer_in_hand()
    assert np.linalg.norm(p1 - p0) < 0.2e-3 and np.degrees(np.linalg.norm(r1 - r0)) < 0.3
    assert tb.grip.measured == pytest.approx(tb.cfg.controller.grip_hold, rel=0.2)
    # readback: encoder angle and current, published at the SDK rate and delayed by the feedback latency
    sq, si = tb.sensors["hand_q"], tb.sensors["hand_current"]
    assert sq.spec.rate_hz == 1000.0 and sq.spec.latency_s == tb.cfg.wuji.fb_latency
    assert sq.latest().value == pytest.approx(tb.world.data.qpos[hand.qadr], abs=2e-3)
    assert np.all(np.abs(si.latest().value) <= hand.drive.i_limit + 0.05)
    assert np.any(np.abs(si.latest().value) > 0.05)  # the closing joints carry current
    assert hand.drive.T.max() < tb.cfg.wuji.t_warn


@requires_wuji
def test_an_external_command_replaces_the_synergy(mit_tb):
    tb = mit_tb
    tb.reset(0)
    hand, w = tb.world.hand, tb.world
    tb.run_for(0.1)
    hand.set_hand_command(HandCommand.zeros(tb.t, 20, src="test"))  # limp hand
    tb.run_for(0.01)
    assert np.all(np.abs(hand.drive.torque) < 1e-6)
    hand.set_hand_command(None)  # back to the synergy
    tb.run_for(0.01)
    assert np.abs(hand.drive.torque).max() > 0.1
    assert w.hand.drive.n_sdk > 0
