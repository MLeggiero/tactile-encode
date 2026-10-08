"""Synthetic load generator: schedules, the shape of each regime, the summed wrench and the forecast truth."""

import numpy as np
import pytest

from tactile_sim.config import LoadGenCfg
from tactile_sim.model.loadgen import KINDS, LoadEvent, _end, forecast_truth, sample_schedule, wrench


def schedules(n=200, t_end=10.0, **over):
    cfg = LoadGenCfg(**over)
    rng = np.random.default_rng(0)
    return cfg, [sample_schedule(cfg, rng, t_end) for _ in range(n)]


def test_schedules_mix_every_regime_without_overlap():
    cfg, eps = schedules()
    kinds = [e.kind for ev in eps for e in ev]
    frac = np.array([kinds.count(k) for k in KINDS]) / len(kinds)
    assert frac == pytest.approx(np.asarray(cfg.p_kind) / sum(cfg.p_kind), abs=0.05)
    for ev in eps:
        spans = [((e.t0 - e.duration) if (e.kind == "step" and e.sign > 0) else e.t0, _end(e)) for e in ev]
        for (a0, a1), (b0, _) in zip(spans, spans[1:], strict=False):
            assert b0 >= a1 - 1e-12
        assert all(0.2 <= s0 and s1 <= 10.0 + 2.0 for s0, s1 in spans)
    n = np.mean([len(ev) for ev in eps])
    assert 0.8 < n / 9.8 < cfg.rate  # events per second: below the Poisson rate, events take time


def test_an_impulse_is_a_half_sine_with_the_right_impulse():
    ev = LoadEvent("impulse", 1.0, 0.004, 500.0, direction=np.array([0.0, 0.0, 1.0]))
    t = np.linspace(0.99, 1.01, 20001)
    f = np.array([wrench([ev], x)[0][2] for x in t])
    assert np.trapezoid(f, t) == pytest.approx(500.0 * 2 * 0.004 / np.pi, rel=1e-3)
    assert f.max() == pytest.approx(500.0, rel=1e-3) and f[t < 1.0].max() == 0.0 and f[t > 1.004].max() == 0.0


def test_periodic_loads_oscillate_at_their_frequency():
    cfg, eps = schedules(n=40, p_kind=(0, 1, 0, 0, 0), periodic_s=(1.0, 1.0), drift=0.0)
    for ev in eps[:10]:
        e = ev[0]
        t = e.t0 + np.arange(0, 1.0, 1e-4)
        y = np.array([e.value(x) for x in t])
        spec = np.abs(np.fft.rfft(y))
        f = np.fft.rfftfreq(len(y), 1e-4)
        assert f[np.argmax(spec)] == pytest.approx(e.freq, abs=1.5)
        assert 1.0 <= e.freq <= 100.0 and np.abs(y).max() == pytest.approx(e.amp, rel=0.05)


def test_a_breakthrough_step_drops_the_push():
    ev = LoadEvent("step", 2.0, 0.5, 60.0, sign=1.0)
    assert wrench([ev], 1.6)[0][2] == 60.0 and wrench([ev], 2.0001)[0][2] == 0.0 and wrench([ev], 1.4)[0][2] == 0.0
    up = LoadEvent("step", 2.0, 0.5, 60.0, sign=-1.0)
    assert wrench([up], 1.99)[0][2] == 0.0 and wrench([up], 2.2)[0][2] == 60.0


def test_binds_and_periodic_torques_twist_about_the_long_axis():
    bind = LoadEvent("bind", 1.0, 0.05, 2.0, torque=True)
    f, tq = wrench([bind], 1.025)
    assert np.all(f == 0) and tq[0] == pytest.approx(2.0) and np.all(tq[1:] == 0)


def test_forecast_truth_follows_the_schedule():
    evs = [LoadEvent("impulse", 1.0, 0.004, 300.0), LoadEvent("step", 2.5, 0.5, 60.0, sign=1.0),
           LoadEvent("periodic", 3.0, 0.5, 20.0, freq=25.0)]
    a = forecast_truth(evs, 0.5)
    assert a.valid and a.t_event == 1.0 and a.regime == "impulse" and a.amp == 300.0
    b = forecast_truth(evs, 1.5)  # during the step's push: the breakthrough at 2.5 s is next
    assert b.t_event == 2.5 and b.regime == "step"
    c = forecast_truth(evs, 3.2)
    assert c.regime == "periodic" and c.f_dom == 25.0 and c.t_event == 3.0
    assert not forecast_truth(evs, 4.0).valid
