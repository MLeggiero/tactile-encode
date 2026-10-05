"""M8: Gymnasium environment and domain randomization."""

import warnings

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from tactile_sim.config import fast_config  # noqa: E402
from tactile_sim.env.gym_env import TactileHammerEnv  # noqa: E402
from tactile_sim.sim.randomize import apply_dr, sample_dr  # noqa: E402

HOLD = np.array([0, 0, 0, 1.0, 0, 40.0], dtype=np.float32)


def test_env_checker():
    from gymnasium.utils.env_checker import check_env

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(TactileHammerEnv(), skip_render_check=True)


def test_registered():
    env = gym.make("TactileHammer-v0")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_hold_steps_are_finite_and_in_space():
    env = TactileHammerEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(50):
        obs, r, term, trunc, info = env.step(HOLD)
        assert env.observation_space.contains(obs)
        assert np.isfinite(r) and not term
    assert all(np.all(np.isfinite(v)) for v in obs.values())
    assert info["nail_depth"] < 1e-6
    assert obs["grip"][0] == pytest.approx(40.0, abs=3.0)
    assert obs["ft"].shape == (env.n_ft, 6) and np.abs(obs["ft"]).sum() > 0


def test_seeded_reset_is_reproducible():
    rng = np.random.default_rng(3)
    actions = [np.array([0, rng.uniform(-0.05, 0.05), 0, 1.0, 0, 40.0], dtype=np.float32) for _ in range(20)]
    outs = []
    for _ in range(2):
        env = TactileHammerEnv()
        env.reset(seed=7)
        for a in actions:
            obs, *_ = env.step(a)
        outs.append(obs)
    for k in outs[0]:
        assert np.allclose(outs[0][k], outs[1][k]), k


def test_step_command_strikes_the_nail():
    env = TactileHammerEnv()
    env.reset(seed=0)
    impacts = 0
    info = {}
    for k in range(200):
        t = k * env.dt_env
        # a crude L2: back off, then step the offset through the nail with full strike-axis feedforward (the
        # FR3's 1000 Nm/s torque-rate limit turns the step into a ramp, so it needs the push)
        dy = -0.10 * min(1.0, t / 0.3) if t < 0.5 else 0.25
        ff = 0.0 if t < 0.5 else 60.0
        obs, r, term, trunc, info = env.step(np.array([0, dy, 0, 1.5, ff, 70.0], dtype=np.float32))
        impacts += int(obs["impact"][0])
        if term or trunc:
            break
    assert impacts >= 1
    assert info["nail_depth"] > 0


def test_domain_randomization_ranges_and_seeding():
    cfg = fast_config(dr={"enabled": True})
    a = sample_dr(cfg, np.random.default_rng(1))
    b = sample_dr(cfg, np.random.default_rng(1))
    c = sample_dr(cfg, np.random.default_rng(2))
    assert a == b and a != c
    for s in [sample_dr(cfg, np.random.default_rng(k)) for k in range(30)]:
        r = cfg.dr
        assert r.resistance_0[0] <= s.resistance_0 <= r.resistance_0[1]
        assert r.head_mass_scale[0] <= s.head_mass_scale <= r.head_mass_scale[1]
        assert r.pad_friction[0] <= s.pad_friction <= r.pad_friction[1]
        assert r.sensor_noise_scale[0] <= s.sensor_noise_scale <= r.sensor_noise_scale[1]
    cfg2 = apply_dr(cfg, a)
    assert cfg2.plant.resistance_0 == a.resistance_0
    assert cfg2.hammer.head_mass == pytest.approx(cfg.hammer.head_mass * a.head_mass_scale)


def test_randomized_env_resets_with_new_parameters():
    env = TactileHammerEnv(cfg=fast_config(dr={"enabled": True}))
    _, i1 = env.reset(seed=1)
    m1 = env.tb.world.model.body_mass[env.tb.world.hammer_body]
    _, i2 = env.reset(seed=2)
    m2 = env.tb.world.model.body_mass[env.tb.world.hammer_body]
    assert i1["dr"] != i2["dr"] and m1 != m2
    obs, *_ = env.step(HOLD)
    assert env.observation_space.contains(obs)
