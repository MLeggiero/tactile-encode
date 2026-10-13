"""Synthetic load generator: schedules, the shape of each regime, the summed wrench and the forecast truth."""

import numpy as np
import pytest

from tactile_sim.config import LoadGenCfg
from tactile_sim.model.loadgen import (
    KINDS,
    LoadEvent,
    _end,
    forecast_truth,
    kind_probs,
    sample_schedule,
    wrench,
)

# the residual's training loads (tactile_rl.mdp.loadgen_command.wide_loads, without its band)
WIDE = dict(p_kind=(0.30, 0.21, 0.13, 0.13, 0.08, 0.15), cone_deg=60.0, bind_axes="xyz")


def schedules(n=200, t_end=10.0, **over):
    cfg = LoadGenCfg(**over)
    rng = np.random.default_rng(0)
    return cfg, [sample_schedule(cfg, rng, t_end) for _ in range(n)]


@pytest.mark.parametrize("over", [{}, WIDE], ids=["default", "wide"])
def test_schedules_mix_every_regime_without_overlap(over):
    cfg, eps = schedules(**over)
    kinds = [e.kind for ev in eps for e in ev]
    frac = np.array([kinds.count(k) for k in KINDS]) / len(kinds)
    assert frac == pytest.approx(kind_probs(cfg), abs=0.05)
    for ev in eps:
        spans = [((e.t0 - e.duration) if (e.kind == "step" and e.sign > 0) else e.t0, _end(e)) for e in ev]
        for (_, a1), (b0, _) in zip(spans, spans[1:], strict=False):
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
    _, eps = schedules(n=100)  # the default configuration: no torque impulses, every bind about +x
    assert not any(e.kind == "torque_impulse" or e.axis is not None for ev in eps for e in ev)


def test_wide_loads_twist_about_every_axis_and_strike_off_axis():
    _, eps = schedules(n=400, **WIDE)
    ev = [e for s in eps for e in s]
    for kind in ("bind", "torque_impulse"):
        axes = np.array([e.axis for e in ev if e.kind == kind])
        assert len(axes) > 200 and np.all(np.abs(axes).sum(1) == 1.0)  # a signed unit tool axis
        share = np.array([np.mean(axes[:, i] == s) for i in range(3) for s in (1.0, -1.0)])
        assert share == pytest.approx(np.full(6, 1 / 6), abs=0.05)
    z = np.array([e.direction[2] for e in ev if e.kind == "impulse"])
    off = np.degrees(np.arccos(np.clip(z, -1, 1)))
    assert off.max() <= 60.0 + 1e-9 and np.mean(off > 20.0) > 0.5  # most of the cap lies past the old 20 degrees
    ti = next(e for e in ev if e.kind == "torque_impulse")
    t = ti.t0 + 0.5 * ti.duration
    f, tq = wrench([ti], t)
    assert np.all(f == 0) and tq == pytest.approx(ti.amp * np.asarray(ti.axis))  # the half-sine's peak
    assert forecast_truth([ti], ti.t0 - 0.1).regime == "impulse"
    with pytest.raises(ValueError):
        sample_schedule(LoadGenCfg(bind_axes="xy", p_kind=(0, 0, 0, 1, 0)), np.random.default_rng(0), 5.0)


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
