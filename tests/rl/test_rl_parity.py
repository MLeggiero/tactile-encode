"""CPU/GPU parity, P0 and P1 (tactile_rl.parity): the mjlab model equals the testbed's reduced-scene model field by
field, the GPU settle reproduces World.reset, and one physics step from identical states agrees with CPU MuJoCo to
within float32 arithmetic (see tactile_rl/README.md for the measured numbers and what they mean)."""

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.rl]  # torch, mjlab and CUDA (.venv-rl) or skipped
torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from tactile_rl.parity.model_fields import compare_models, compare_warp  # noqa: E402
from tactile_rl.parity.one_step import format_report, run_p1  # noqa: E402
from tactile_rl.tasks.hold import hold_env_cfg  # noqa: E402
from tactile_sim.sim.training_parts import config_from_key, training_config, training_parts  # noqa: E402


@pytest.fixture(scope="module")
def env4():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = hold_env_cfg(num_envs=4)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    yield env, cfg.actions["residual"].scene_key
    env.close()


def test_p0_the_gpu_model_is_the_cpu_model(env4):
    from tactile_sim.sim.world import World

    env, key = env4
    cpu = World(config_from_key(key)).model
    assert compare_models(cpu, env.sim.mj_model) == []
    # MuJoCo Warp stores the model in float32 and raises the solver tolerance to 1e-6; nothing else may differ
    assert [d for d in compare_warp(cpu, env.sim.wp_model) if "opt.tolerance" not in d] == []
    # against the validator (noslip 5), only noslip differs: MuJoCo Warp has no noslip solver
    validator = World(training_config(noslip_iterations=5)).model
    assert compare_models(validator, env.sim.mj_model) == ["opt.noslip_iterations: cpu 5 gpu 0"]


def test_p0_the_gpu_settle_reproduces_world_reset(env4):
    import warp as wp

    from tactile_rl.mdp.events import tactile_state, weld_id

    env, key = env4
    parts = training_parts(key)
    bank = tactile_state(env).bank
    eq = wp.to_torch(env.sim.wp_model.eq_data)[0, weld_id(env, parts.weld)].cpu().numpy()
    assert np.allclose(eq, parts.weld_eq_data, atol=1e-6)
    p_cpu, r_cpu = parts.hammer_in_hand
    rel = bank.hammer_rel.cpu().numpy()
    assert np.abs(rel[:, :3] - p_cpu).max() < 1e-4  # m
    assert np.degrees(np.abs(rel[:, 3:] - r_cpu).max()) < 0.1
    q = bank.qpos.cpu().numpy()
    arm = slice(7, 14)  # after the hammer's free joint
    assert np.abs(q[:, arm] - parts.qpos_settled[arm]).max() < 1e-4
    hand = slice(14, 34)
    assert np.abs(q[:, hand] - parts.qpos_settled[hand]).max() < 0.01
    assert torch.all(bank.drive["t"] == 0) and torch.all(bank.drive["host_valid"])


@pytest.fixture(scope="module")
def p1():
    return {m: run_p1(multiccd=m) for m in (True, False)}


def test_p1_smooth_dynamics_and_contact_geometry_agree(p1):
    for r in p1.values():
        g = r["gpu"]
        print(format_report(r, "P1"))
        assert g["qfrc_bias_p99"] < 1e-5 and g["qfrc_passive_p99"] < 1e-5
        assert g["actuator_force_p99"] < 1e-4 and g["qacc_smooth_p99"] < 1e-4  # float32 ctrl and state
        assert g["dqpos_max"] < 1e-3


def test_p1_one_step_qacc_and_contact_forces(p1):
    """Where both sides find the same contacts, qacc agrees to the plan's 1e-3 at the median (kinetic-energy norm,
    relative to the unconstrained acceleration); forces to a few 1e-3. The tails, and the force medians above 1e-3,
    are float32 arithmetic in a stiff, ill-conditioned grasp: the CPU differs as much from itself when its state moves
    by ~10 float32 ulps (`floor` at 1 ulp is reported as a reference). Bounds here catch regressions; README has the
    numbers."""
    off, on = p1[False], p1[True]
    for r in (off, on):
        g, f = r["gpu"], r["floor"]
        assert g["qacc_M_same_median"] < 1e-3
        assert g["force_same_median"] < 5e-3
        assert g["qacc_M_same_p99"] < 0.2 and g["force_same_p99"] < 0.5
        assert f["same_contacts"] >= 0.95 * f["states"]  # the CPU's own contact set is stable to 1 ulp
    # contact detection: single-point convex collisions agree in most states; with multi-contact on (the testbed's
    # default) MuJoCo Warp often adds a second point to the palm/handle contact where CPU MuJoCo has one
    assert off["gpu"]["same_contacts"] >= 0.85 * off["gpu"]["states"]
    assert on["gpu"]["same_contacts"] >= 0.5 * on["gpu"]["states"]
    assert off["gpu"]["force_unmatched_frac"] < 0.05 and on["gpu"]["force_unmatched_frac"] < 0.05
