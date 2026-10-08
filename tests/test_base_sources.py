"""Base sources for the residual: the forecast's error model, and the scripted hammer swing as a base (synergy hand
command, arm lookahead, contact-time forecast) on the Vega U + WUJI through the MIT drives and the TaxelScan fast
profile."""

import numpy as np
import pytest

from tactile_sim.assets.fetch_hands import hand_available
from tactile_sim.behaviors.forecast import ForecastModel, ForecastTruth
from tactile_sim.config import ForecastCfg, fast_config, hand_config
from tactile_sim.control.obs_spec import LOOKAHEAD, N_ARM, N_JOINTS, REGIMES

requires_wuji = pytest.mark.skipif(not hand_available("wuji2"), reason="WUJI Hand 2 not cached")


def model(**over):
    base = dict(mask_prob=0.0, dropout=0.0, bias_sigma=0.0, jitter_sigma=0.0, class_confusion=0.0, freq_err=0.0,
                amp_err=0.0, update_rate=10.0, latency=0.1)
    base.update(over)
    return ForecastModel(ForecastCfg(**base), np.random.default_rng(0))


def run(fm, truth_fn, t_end, dt=0.001):
    out = []
    for k in range(int(round(t_end / dt))):
        t = k * dt
        out.append((t, fm(t, truth_fn(t))))
    return out


def test_an_exact_forecast_arrives_after_its_latency_on_the_update_grid():
    fm = model()
    out = run(fm, lambda t: ForecastTruth(True, 1.0, "impulse", 0.0, 200.0), 0.5)
    first = next(t for t, v in out if v[0] == 1.0)
    assert first == pytest.approx(0.1)  # issued at t = 0, seen 100 ms later
    for t, v in out:
        if v[0] == 1.0:
            age = v[-1]
            assert 0.1 - 1e-9 <= age < 0.2 + 1e-9  # refreshed at 10 Hz, each seen 100 ms late
            assert v[1] == pytest.approx(1.0 - t)  # time to the event, from the forecast's own estimate
            assert v[3 + REGIMES.index("impulse")] == 1.0 and v[3:8].sum() == pytest.approx(1.0)
            assert v[10] == pytest.approx(2.0)  # 200 N in 100 N units


def test_masked_episodes_see_nothing():
    fm = model(mask_prob=1.0)
    assert all(np.all(v == 0) for _, v in run(fm, lambda t: ForecastTruth(True, 0.5), 0.5))


def test_errors_bias_jitter_and_confusion():
    fm = model(bias_sigma=0.02, jitter_sigma=0.01, class_confusion=0.3, amp_err=0.2)
    bias = fm.bias
    errs = [v[1] - (1.0 - t) for t, v in run(fm, lambda t: ForecastTruth(True, 1.0, "periodic", 25.0, 100.0), 0.9)
            if v[0] == 1.0]
    assert np.mean(errs) == pytest.approx(bias, abs=0.012)
    _, v = run(fm, lambda t: ForecastTruth(True, 1.0, "periodic", 25.0, 100.0), 0.2)[-1]
    p = v[3:8]
    assert p.sum() == pytest.approx(1.0) and p[REGIMES.index("periodic")] >= 0.7 - 1e-9
    assert p[REGIMES.index("periodic")] == p.max()


def test_no_event_ahead_clears_the_forecast():
    fm = model(latency=0.0)
    out = run(fm, lambda t: ForecastTruth(t < 0.25, 0.3), 0.5)
    assert out[100][1][0] == 1.0 and out[-1][1][0] == 0.0


def test_dropout_keeps_the_last_forecast():
    fm = model(dropout=0.5, latency=0.0)
    out = run(fm, lambda t: ForecastTruth(True, 2.0), 2.0)
    ages = np.array([v[-1] for _, v in out if v[0] == 1.0])
    assert ages.max() > 0.1  # a dropped update leaves the previous one in place, older


@requires_wuji
def test_swing_base_drives_a_vega_wuji_episode_through_the_residual():
    from tactile_sim.behaviors.base_sources import SwingBase
    from tactile_sim.episode import Episode

    cfg = hand_config("wuji2", fast_config(), robot="vega_1u", gripper={"hand_control": "mit"},
                      sensors={"ts_model": "rev3", "ts_taxel_time": 7.8e-6}, residual={"enabled": True},
                      forecast={"mask_prob": 0.0})
    ep = Episode(cfg, seed=0)
    tb = ep.tb
    base = SwingBase(tb, ep.swing, rng=np.random.default_rng(1))
    seen = []

    def recording(t):
        out = base(t)
        seen.append((t, ep.swing.phase, out[1].copy(), out[3].copy()))
        return out

    tb.residual.base_fn = recording
    res = ep.run(2)
    assert res.summary["n_strikes"] == 2 and res.summary["total_depth"] > 0.5e-3
    assert not tb.grip.dropped
    assert tb.residual.n_nonfinite == 0 and tb.residual.n_ticks > 1000
    look = np.array([s[2] for s in seen])
    arm = look.reshape(len(look), len(LOOKAHEAD), N_JOINTS + N_ARM)[:, :, N_JOINTS:]
    swinging = np.array([s[1] == "swing" for s in seen])
    assert np.abs(arm[swinging]).max() > 0.01  # the arm moves ahead of the targets during the swing
    fc = np.array([s[3] for s in seen])
    lead = fc[swinging & (fc[:, 0] == 1.0), 1]
    assert len(lead) and lead.max() > 0.05  # the contact is forecast well before it happens
