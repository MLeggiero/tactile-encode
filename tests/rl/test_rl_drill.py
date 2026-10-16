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
    assert nl[0] * 60 / (2 * np.pi) == pytest.approx(450.0, abs=5.0)  # the DF333D's low gear, 450 rpm


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


needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def test_training_parts_carry_the_drill_and_the_trigger_finger_cap():
    pytest.importorskip("mujoco")
    from tactile_sim.sim.training_parts import export_training_parts, residual_training_config

    p = export_training_parts(residual_training_config(1000.0, task="drill"))
    assert p.nail is None and p.drill is not None and p.drill.trigger_joint == "trigger_slide"
    assert len(p.skin) == 11 and len(p.synergy.trigger) == 3
    caps = dict(zip(p.skin_pairs, p.skin_caps, strict=True))
    assert any(c == pytest.approx(0.9) for c in caps.values())  # the trigger finger's pairs
    assert p.fixed_friction_pairs and set(p.fixed_friction_pairs) <= set(p.skin_pairs)  # the trigger's face


@needs_gpu
@pytest.mark.gpu
@pytest.mark.slow
def test_a_drill_episode_replays_on_the_gpu(tmp_path):
    """One zero-residual screw on the CPU (the training physics), replayed open loop on the GPU: it seats within
    50 ms of the CPU, at the same depth, with the clutch's clicks and the tool's slip close."""
    pytest.importorskip("mjlab")
    from tactile_rl.parity.drill_episodes import gpu_replay
    from tactile_sim.behaviors.drill_library import assemble_drill, record_drill_episode

    path = tmp_path / "drill.npz"
    np.savez_compressed(path, **assemble_drill([record_drill_episode(0, "screw")], {"mode": "screw"}))
    r = gpu_replay(str(path))["rows"][0]
    assert r["done_gpu"] and r["done_cpu"]
    assert abs(r["t_done_gpu"] - r["t_done_cpu"]) < 0.05
    assert r["depth_gpu"] == pytest.approx(r["depth_cpu"], abs=2e-4)
    assert abs(r["clicks_gpu"] - r["clicks_cpu"]) <= 8
    assert r["net_slip_rot_gpu"] == pytest.approx(r["net_slip_rot_cpu"], abs=0.02)
