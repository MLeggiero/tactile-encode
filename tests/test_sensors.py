"""M3: rate-limited sensor models and the scheduler."""

import numpy as np
import pytest

from tactile_sim.sensors import SensorSpec, build_sensor_suite
from tactile_sim.sensors.base import RateLimitedSensor
from tactile_sim.sensors.pressure import center_of_pressure
from tactile_sim.sim.scheduler import RateScheduler

DT = 0.000125  # 8 kHz physics


def run(sensor, signal_fn, T):
    """Drive a sensor with signal_fn(t) for T seconds; returns the list of latest() values per step."""
    state = {"t": 0.0}
    sensor.read_fn = lambda: np.atleast_1d(signal_fn(state["t"]))
    sensor.reset(0.0)
    out = []
    n = int(round(T / DT))
    for k in range(1, n + 1):
        state["t"] = k * DT
        sensor.step(state["t"])
        out.append((state["t"], sensor.latest().value.copy()))
    return out


def make(**kw):
    spec = SensorSpec(**{"name": "s", "rate_hz": 1000.0, "dim": 1, **kw})
    return RateLimitedSensor(spec, DT, lambda: np.zeros(spec.dim), np.random.default_rng(0))


@pytest.mark.parametrize("rate", [8000.0, 4000.0, 1000.0, 500.0, 200.0, 30.0])
def test_sample_count(rate):
    s = make(rate_hz=rate)
    run(s, lambda t: 0.0, 1.0)
    assert abs(len(s.values) - rate) <= 1


def test_latency_of_step_input():
    s = make(rate_hz=1000.0, latency_s=0.002, decimation="zoh")
    out = run(s, lambda t: 1.0 if t >= 0.0100 - 1e-12 else 0.0, 0.03)
    h = s.history()
    assert np.allclose(h["t_avail"] - h["t_sample"], 0.002)
    first = next(t for t, v in out if v[0] > 0.5)
    assert first == pytest.approx(0.012, abs=DT * 1.01)


def test_zero_order_hold_between_samples():
    s = make(rate_hz=500.0, decimation="zoh")
    out = run(s, lambda t: np.sin(2 * np.pi * 7 * t), 0.1)
    vals = np.array([v[0] for _, v in out])
    changes = np.count_nonzero(np.diff(vals))
    assert changes <= 0.1 * 500 + 1  # only changes when a new sample lands


def test_noise_std_and_bias():
    s = make(rate_hz=4000.0, noise_std=0.3, decimation="zoh")
    run(s, lambda t: 0.0, 1.0)
    v = s.history()["value"][:, 0]
    assert np.std(v) == pytest.approx(0.3, rel=0.15)
    b = make(rate_hz=1000.0, bias_std=1.0)
    biases = []
    for seed in range(40):
        b.rng = np.random.default_rng(seed)
        b.reset()
        biases.append(b.bias[0])
    assert 0.5 < np.std(biases) < 1.5


def test_saturation_and_quantization():
    s = make(rate_hz=1000.0, saturation=2.0, quant_step=0.5, decimation="zoh")
    run(s, lambda t: 10 * np.sin(2 * np.pi * 5 * t), 0.2)
    v = s.history()["value"][:, 0]
    assert v.max() == 2.0 and v.min() == -2.0
    assert np.allclose(np.round(v / 0.5) * 0.5, v)
    s2 = make(rate_hz=1000.0, saturation=(0.0, 5.0), decimation="zoh")
    run(s2, lambda t: -3.0, 0.01)
    assert s2.history()["value"].min() == 0.0


@pytest.mark.parametrize("fc", [100.0, 800.0])
def test_bandwidth_minus_3db(fc):
    s = make(rate_hz=8000.0, bandwidth_hz=fc, decimation="zoh")
    run(s, lambda t: np.sin(2 * np.pi * fc * t), 0.5)
    v = s.history()["value"][:, 0]
    amp = np.max(np.abs(v[len(v) // 2:]))
    assert amp == pytest.approx(1 / np.sqrt(2), rel=0.2)
    s_low = make(rate_hz=8000.0, bandwidth_hz=fc, decimation="zoh")
    run(s_low, lambda t: np.sin(2 * np.pi * fc / 10 * t), 0.5)
    assert np.max(np.abs(s_low.history()["value"][2000:, 0])) > 0.95


def test_mean_decimation_is_antialiasing():
    # a tone at the sensor's rate aliases to DC under zoh; the boxcar mean suppresses it
    f = 1000.0
    zoh = make(rate_hz=f, decimation="zoh")
    mean = make(rate_hz=f, decimation="mean")
    sig = lambda t: np.cos(2 * np.pi * f * t)  # noqa: E731
    run(zoh, sig, 0.1)
    run(mean, sig, 0.1)
    z = zoh.history()["value"][:, 0]
    assert abs(z.mean()) > 0.5 and z.std() < 1e-6  # aliased to a constant (DC) by point sampling
    assert np.mean(np.abs(mean.history()["value"])) < 0.05


def test_timestamps_monotonic_and_window():
    s = make(rate_hz=4000.0, latency_s=0.0005, dim=3)
    run(s, lambda t: np.array([t, 2 * t, 3 * t]), 0.05)
    h = s.history()
    assert np.all(np.diff(h["t_sample"]) > 0) and np.all(np.diff(h["t_avail"]) > 0)
    w = s.window(32)
    assert w.shape == (32, 3) and np.all(np.diff(w[:, 0]) > 0)


def test_bandwidth_above_nyquist_rejected():
    with pytest.raises(ValueError):
        make(bandwidth_hz=5000.0)


def test_scheduler_rates():
    sch = RateScheduler(DT)
    hits = {"l1": 0, "grip": 0, "l2": 0}
    sch.register("l1", 1000.0, lambda t: hits.__setitem__("l1", hits["l1"] + 1))
    sch.register("grip", 500.0, lambda t: hits.__setitem__("grip", hits["grip"] + 1))
    sch.register("l2", 200.0, lambda t: hits.__setitem__("l2", hits["l2"] + 1))
    for k in range(8000):
        sch.tick(k * DT)
    assert hits == {"l1": 1000, "grip": 500, "l2": 200}
    with pytest.raises(ValueError):
        sch.register("bad", 3000.0, lambda t: None)


def test_center_of_pressure():
    tax = np.zeros(64)
    tax[0] = 10.0  # row 0, col 0 corner
    cop = center_of_pressure(tax, (0.0085, 0.002, 0.0085), (8, 8))
    assert cop[0] < 0 and cop[1] < 0
    assert np.allclose(center_of_pressure(np.ones(64), (0.0085, 0.002, 0.0085), (8, 8)), 0.0)


def test_suite_on_world(settled_world):
    w = settled_world
    suite = build_sensor_suite(w, rng=np.random.default_rng(1))
    suite.reset(w.t)
    t0 = w.t
    for _ in range(int(0.2 / w.dt)):
        w.set_grip_force(w.cfg.controller.grip_hold)
        w.set_arm_torque(w.hold_torque(w.q_hover))
        w.step()
        suite.step(w.t)
    T = w.t - t0
    for name, rate in [("ft", w.cfg.sensors.ft_rate), ("pad_acc_L", w.cfg.sensors.accel_rate),
                       ("pressure_L", w.cfg.sensors.pressure_rate), ("joint_pos", w.cfg.sensors.joint_rate)]:
        assert abs(len(suite[name].values) - rate * T) <= 2 + 1e-6, name
    # static grasp: F/T z ~ weight below the sensor, grip ~ 40 N per pad, pad accel ~ 1 g
    assert abs(suite.latest("ft")[2]) == pytest.approx(15.0, abs=2.0)
    assert suite.latest("pressure_L").sum() == pytest.approx(w.cfg.controller.grip_hold, abs=5.0)
    assert np.linalg.norm(suite.latest("pad_acc_L")) == pytest.approx(9.81, abs=0.5)


def test_camera_degrades_gracefully(settled_world):
    import warnings

    from tactile_sim.sensors.camera import make_camera

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cam = make_camera(settled_world, settled_world.cfg, np.random.default_rng(0))
    assert cam is None or cam.spec.rate_hz == settled_world.cfg.sensors.camera_rate
