"""Milestone 5: the 12 V drill on the GPU: the drivetrain and the plant's laws batched in torch against the CPU
(tactile_sim.model.plants.drivetrain / drill), and the virtual dyno on the batched drivetrain."""

from dataclasses import replace

import numpy as np
import pytest

pytestmark = [pytest.mark.rl]
torch = pytest.importorskip("torch")


def _params():
    from tactile_sim.config import residual_config
    from tactile_sim.model.plants.drivetrain import DrivetrainParams

    p = DrivetrainParams.from_cfg(residual_config(task="drill").drill)
    return [p, replace(p, trigger_slew=8.0, clutch_torque=1.2, clutch_detents=9), replace(p, clutch_lock=True)]


@pytest.mark.parametrize("hold", [True, False])
def test_drivetrain_matches_the_cpu_step_for_step(hold):
    from tactile_rl.parity.drill_plant import _drive
    from tactile_sim.model.plants.drivetrain import Drivetrain

    params = _params()
    cpu = [Drivetrain(p) for p in params]
    gpu = _drive(params)
    rng = np.random.default_rng(3)
    dt = 2.5e-4
    trig = need = None
    for i in range(4000):
        if i % 300 == 0:
            trig = rng.choice([0.0, 0.3, 1.0], size=len(params))
            need = rng.uniform(-0.5, 3.0, size=len(params))
        turn_g = gpu.step(torch.as_tensor(trig), torch.as_tensor(need), dt, hold=hold).numpy()
        for k, c in enumerate(cpu):
            turn_c = c.step(float(trig[k]), float(need[k]), dt, hold=hold)
            assert turn_g[k] == pytest.approx(turn_c, abs=1e-12)
            assert float(gpu.w[k]) == pytest.approx(c.w, abs=1e-10)
            assert float(gpu.tau_housing[k]) == pytest.approx(c.tau_housing, abs=1e-8)
            assert int(gpu.clicks[k]) == c.clicks and bool(gpu.slipping[k]) == c.slipping


def test_batched_dyno_matches_the_cpu_dyno():
    from tactile_rl.parity.drill_plant import clutch_test, no_load_speed
    from tactile_sim.model.plants import drill_dyno as dy

    params = _params()[:2]
    nl = no_load_speed(params)
    ct = clutch_test(params)
    for k, p in enumerate(params):
        assert nl[k] == pytest.approx(dy.no_load_speed(p), rel=1e-12)
        c = dy.clutch_test(p)
        assert ct["mean_torque"][k] == pytest.approx(c["mean_torque"], rel=1e-12)
        assert ct["click_rate"][k] == pytest.approx(c["click_rate"], rel=1e-12)
    assert nl[0] * 60 / (2 * np.pi) == pytest.approx(424.0, abs=5.0)  # the 12 V drill's low gear, ~425 rpm


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["screw", "hole"])
def test_drill_laws_replay_a_cpu_episode_exactly(mode):
    """A zero-residual drill episode on the CPU (Vega U + WUJI, the pistol driver, the trigger pulled by the index
    finger): the GPU laws on its recorded inputs give the CPU plant's wrench, states and drivetrain."""
    pytest.importorskip("mujoco")
    from tactile_rl.parity.drill_plant import record_cpu, replay

    r = replay(record_cpu(mode, seed=0, fast=True, t_max=4.0))
    assert r["steps"] > 1000
    assert r["max_abs_force"] < 1e-9 and r["max_abs_torque"] < 1e-9 and r["max_abs_depth"] < 1e-12
    assert r["max_abs_w"] < 1e-9 and r["state_mismatches"] == 0 and r["click_mismatches"] == 0
