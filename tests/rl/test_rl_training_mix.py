"""The residual's training distribution of 2026-10-13 without an env: the VLA-style hand base (TorchHandTable against
tactile_sim.vla.action.HandTable, its 30 Hz grid), the wide loads' band file (loadgen_command.wide_loads) and the
calibration's band choice (loadgen_calibration.choose_band). The env side is in test_rl_loadgen (C3)."""

import json
import math

import numpy as np
import pytest

pytestmark = [pytest.mark.rl]
torch = pytest.importorskip("torch")


def _table(seed: int = 0, J: int = 20):
    from tactile_sim.vla.action import HandTable

    rng = np.random.default_rng(seed)
    return HandTable(rng.uniform(1.0, 5.0, J), rng.uniform(0.01, 0.1, J), rng.normal(0.0, 0.3, J))


def _synergy_base(rng, N: int, J: int = 20) -> torch.Tensor:
    """A synergy-like base (N, 5, J): torque on half the joints (Kp 0), a PD on the others."""
    q_t = rng.normal(0.0, 0.5, (N, J))
    kp = np.where(np.arange(J) % 2 == 0, 0.0, rng.uniform(0.5, 3.0, (N, J)))
    kd = np.full((N, J), 0.05)
    tau = rng.normal(0.0, 0.4, (N, J))
    return torch.as_tensor(np.stack([q_t, np.zeros((N, J)), kp, kd, tau], axis=1), dtype=torch.float32)


def test_vla_table_gives_the_scripted_static_torque_and_holds_its_targets_on_the_grid():
    from tactile_rl.mdp.action_map import TorchHandTable

    tab = _table()
    N, J, hz = 6, 20, 30.0
    vt = TorchHandTable(tab, N, "cpu", refresh_hz=hz)
    torch.manual_seed(0)
    vt.reset(torch.arange(N), p=1.0)
    vt.active[0] = False  # a scripted episode
    has_q = torch.ones(N, dtype=torch.bool)
    has_q[1] = False  # the host has not read the angles yet
    rng = np.random.default_rng(1)
    refreshed = {e: [] for e in range(2, N)}
    prev = None
    for k in range(240):  # 240 ms of 1 kHz policy ticks
        t = torch.full((N,), k * 1e-3, dtype=torch.float64)
        base = _synergy_base(rng, N)
        q = torch.as_tensor(rng.normal(0.0, 0.5, (N, J)), dtype=torch.float64)
        out = vt.apply(base, q, t, has_q)
        again = vt.apply(_synergy_base(rng, N), q + 1.0, t, has_q)  # a second caller at this time: the same command
        assert torch.equal(out[2:], again[2:])
        assert torch.equal(out[:2], base[:2])  # scripted, or no reading yet: the synergy's command
        assert torch.equal(out[2:, 2], vt.kp.expand(N - 2, -1)) and torch.equal(out[2:, 4], vt.tau.expand(N - 2, -1))
        assert not bool(out[2:, 1].any()) and torch.equal(out[2:, 3], vt.kd.expand(N - 2, -1))
        for e in range(2, N):
            if prev is None or not torch.equal(out[e, 0], prev[e, 0]):
                refreshed[e].append(k)
                # a refresh: under the table, q* gives the synergy base's static torque at the angles read now
                want = tab.equivalent_targets(base[e].double().numpy(), q[e].numpy())
                np.testing.assert_allclose(out[e, 0].double().numpy(), want, rtol=1e-5, atol=1e-5)
                tq_tab = tab.kp * (out[e, 0].double().numpy() - q[e].numpy()) + tab.tau
                b = base[e].double().numpy()
                np.testing.assert_allclose(tq_tab, b[2] * (b[0] - q[e].numpy()) + b[4], rtol=1e-4, atol=1e-4)
        prev = out.clone()
    for e in range(2, N):
        ks = np.array(refreshed[e])
        assert ks[0] == 0  # the first tick with a reading
        gaps = np.diff(ks)
        # held between the 30 Hz grid's points (the first one comes at the episode's phase)
        assert 1 <= gaps[0] <= 34 and np.all((gaps[1:] >= 33) & (gaps[1:] <= 34))
        ph = float(vt.phase[e])
        assert 0.0 <= ph < 1.0 / hz
        for kk in ks[1:]:  # each refresh is the first tick at or past a grid point of the episode's phase
            g = math.floor((kk * 1e-3 - ph) * hz + 1e-9)
            assert (kk - 1) * 1e-3 < ph + g / hz + 1e-9 <= kk * 1e-3 + 1e-9
    # a new episode: drawn again, refreshed at once
    vt.reset(torch.tensor([2]), p=0.0)
    assert not bool(vt.active[2]) and vt.next_t[2] == -math.inf


def test_vla_share_is_the_episodes_probability():
    from tactile_rl.mdp.action_map import TorchHandTable

    vt = TorchHandTable(_table(), 20000, "cpu")
    torch.manual_seed(3)
    vt.reset(torch.arange(20000), p=0.4)
    assert float(vt.active.float().mean()) == pytest.approx(0.4, abs=0.015)
    with pytest.raises(ValueError):
        from tactile_sim.vla.action import HandTable

        TorchHandTable(HandTable(np.zeros(20), np.ones(20), np.zeros(20)), 4, "cpu")


def test_wide_loads_take_a_calibrated_band(tmp_path, monkeypatch):
    from tactile_rl.mdp.loadgen_command import BAND_KEYS, WIDE_CONE_DEG, WIDE_P_KIND, wide_loads
    from tactile_sim.config import LoadGenCfg

    monkeypatch.delenv("TACTILE_LOADGEN_BAND", raising=False)
    cfg = wide_loads()
    assert cfg.cone_deg == WIDE_CONE_DEG == 60.0 and cfg.bind_axes == "xyz" and cfg.p_kind == WIDE_P_KIND
    assert cfg.impulse_n == LoadGenCfg().impulse_n  # no band: LoadGenCfg's ranges
    path = tmp_path / "band.json"
    band = {k: [1.0 + i, 10.0 + i] for i, k in enumerate(BAND_KEYS)}
    path.write_text(json.dumps({"band": band, "notes": {}}))
    monkeypatch.setenv("TACTILE_LOADGEN_BAND", str(path))
    cfg = wide_loads()
    assert all(getattr(cfg, k) == tuple(v) for k, v in band.items())
    assert wide_loads({"impulse_n": [5, 50]}).impulse_n == (5.0, 50.0)
    for bad in ({"impulse_n": [50, 5]}, {"impulse_ms": [1, 2]}, {"push_n": [0, 2]}):
        with pytest.raises(ValueError):
            wide_loads(bad)


def test_the_calibration_chooses_the_band_from_the_losses():
    from tactile_rl.bench.loadgen_calibration import BAND_OF, GRID, choose_band

    rows = []
    for name, (_, _, amps) in GRID.items():
        for i, a in enumerate(amps):
            lost = {"impulse": max(0.0, min(1.0, (i - 4) * 0.2)),  # 0 up to the 5th cell, 20 % per cell after
                    "push": 0.0}.get(name, 1.0 if i >= 1 else 0.0)
            rows.append({"kind": name, "amp": a, "hold": 0.5, "lost_event": lost})
    out = choose_band(rows, 0.5, target=0.4)
    lo, hi = out["band"]["impulse_n"]
    a = GRID["impulse"][2]
    # 40 % between the 6th (20 %) and 7th (40 %) cells' amplitudes: at the 7th; the low end its share, never above
    # the largest amplitude without a loss (the 5th)
    assert hi == pytest.approx(a[6]) and lo == pytest.approx(min(BAND_OF["impulse"][1] * a[6], a[4]))
    assert out["band"]["push_n"][1] == GRID["push"][2][-1] and "never" in out["notes"]["push"]
    lo_b, hi_b = out["band"]["bind_nm"]  # 0 at the grid's bottom, 100 % above: 40 % of the way up in log
    b = GRID["bind"][2]
    assert hi_b == pytest.approx(math.exp(math.log(b[0]) + 0.4 * (math.log(b[1]) - math.log(b[0]))))
    assert lo_b <= b[0]
    assert set(out["band"]) == {k for k, _ in BAND_OF.values()}
