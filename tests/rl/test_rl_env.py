"""The GPU training env (tactile_rl): task registration, the torch action map and synergy against their numpy
references, a zero residual reproducing the base command exactly, resets from the settled-state bank, a held hammer,
and a short PPO run through mjlab's runner."""

import types

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.rl]  # torch, mjlab and CUDA (.venv-rl) or skipped
torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from tactile_rl.mdp.action_map import TorchActionMap, TorchSynergy  # noqa: E402
from tactile_rl.tasks.hold import RATES, decimation_for, hold_env_cfg, task_id  # noqa: E402
from tactile_sim.config import ResidualCfg  # noqa: E402
from tactile_sim.control.hand import WrapSynergy  # noqa: E402
from tactile_sim.control.obs_spec import ActionSpec, JointParams, compose, offsets, slew  # noqa: E402
from tactile_sim.control.residual_l2 import ResidualHandL2  # noqa: E402
from tactile_sim.sim.wuji_drive import joint_group  # noqa: E402

FINGERS = ("index_finger", "middle_finger", "ring_finger", "pinky")
NAMES = ([f"r_thumb_{j}" for j in ("cmc_flex", "cmc_abd", "mcp", "ip")] +
         [f"r_{f}_{j}" for f in FINGERS for j in ("mcp_flex", "mcp_abd", "pip", "dip")])
RATED = {"mcp_flex": 2.0, "mcp_abd": 0.2, "pip_dip": 0.3, "thumb_cmc": 0.6, "thumb_mcp_ip": 0.3}
TAU = np.array([RATED[joint_group(n)] for n in NAMES])
DEV = "cuda"


def test_tasks_register_at_every_rate():
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    import tactile_rl.tasks  # noqa: F401

    for rate in RATES:
        assert task_id(rate) in list_tasks()
        cfg = load_env_cfg(task_id(rate))
        assert cfg.decimation == decimation_for(rate, 0.00025) == round(4000 / rate)
        assert cfg.sim.mujoco.timestep == 0.00025 and cfg.sim.mujoco.cone == "elliptic"
        rl = load_rl_cfg(task_id(rate))
        assert rl.algorithm.gamma == pytest.approx(np.exp(-1.0 / rate / 0.5))
    with pytest.raises(ValueError):
        decimation_for(300.0, 0.00025)


def test_action_map_matches_obs_spec():
    rng = np.random.default_rng(0)
    spec = ActionSpec()
    jp = JointParams.for_joints(NAMES, TAU, (0.37, 0.43, 0.22, 0.55, 0.22), rng.uniform(1e-4, 5e-4, 20))
    syn = WrapSynergy.for_wuji(list(zip(NAMES, TAU, strict=True)), (1, 0, 1, 1), (0.0, -0.8, 0.0, 0.0), "rated")
    trig = [NAMES.index(j) for j in ResidualCfg().trigger_joints]
    tm = TorchActionMap(spec, jp, DEV, close_dir=syn.close_dir, trigger_idx=trig)
    n = 64
    a = rng.uniform(-1, 1, (n, 60))
    prev = rng.normal(0, 0.05, (n, 3, 20))
    base = np.stack([np.stack(syn.mit_command(s, q)) for s, q in zip(rng.uniform(0, 1, n),
                                                                        rng.uniform(-0.5, 1.2, (n, 20)), strict=True)])
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=DEV)  # noqa: E731
    off_t = tm.offsets(t(a))
    slewed_t = tm.slew(t(prev), off_t, 0.004)
    cmd_t = tm.compose(t(base), slewed_t).cpu().numpy()
    for i in range(n):
        off = offsets(spec, jp, a[i])
        assert off_t[i].cpu().numpy() == pytest.approx(off, abs=1e-5, rel=1e-5)
        cmd = compose(spec, jp, base[i], slew(spec, jp, prev[i], off, 0.004))
        assert cmd_t[i] == pytest.approx(cmd, abs=1e-5, rel=1e-5)
    # the drill-trigger interlock against the testbed's
    fake = types.SimpleNamespace(trig=np.array(trig), close_dir=syn.close_dir, cfg=ResidualCfg())
    for trigger in (False, True):
        guard_t = tm.trigger_guard(off_t, torch.full((n,), trigger, device=DEV)).cpu().numpy()
        for i in range(n):
            want = ResidualHandL2._trigger_guard(fake, off_t[i].cpu().numpy().astype(float), trigger)
            assert guard_t[i] == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize("synergy", ["rated", "tendon"])
def test_synergy_matches_wrap_synergy(synergy):
    syn = WrapSynergy.for_wuji(list(zip(NAMES, TAU, strict=True)), (1, 0, 1, 1), (0.0, -0.8, 0.0, 0.0), synergy)
    ts = TorchSynergy(syn, DEV)
    rng = np.random.default_rng(1)
    s, q = rng.uniform(-0.2, 1.2, 32), rng.uniform(-0.5, 1.2, (32, 20))
    out = ts.mit_command(torch.as_tensor(s, dtype=torch.float32, device=DEV),
                         torch.as_tensor(q, dtype=torch.float32, device=DEV)).cpu().numpy()
    for i in range(32):
        assert out[i] == pytest.approx(np.stack(syn.mit_command(s[i], q[i])), abs=1e-6)


def test_zero_action_composes_the_base_exactly():
    spec = ActionSpec()
    jp = JointParams.for_joints(NAMES, TAU, (0.37, 0.43, 0.22, 0.55, 0.22), np.full(20, 3e-4))
    syn = WrapSynergy.for_wuji(list(zip(NAMES, TAU, strict=True)), (1, 0, 1, 1), (0.0, -0.8, 0.0, 0.0), "rated")
    tm = TorchActionMap(spec, jp, DEV, close_dir=syn.close_dir, trigger_idx=[12, 14, 15])
    base = TorchSynergy(syn, DEV).mit_command(torch.rand(16, device=DEV), torch.rand(16, 20, device=DEV))
    off = tm.trigger_guard(tm.slew(torch.zeros(16, 3, 20, device=DEV),
                                   tm.offsets(torch.tanh(torch.zeros(16, 60, device=DEV))), 0.001),
                           torch.zeros(16, dtype=torch.bool, device=DEV))
    assert torch.equal(tm.compose(base, off), base)


@pytest.fixture(scope="module")
def env64():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = hold_env_cfg(num_envs=64, episode_s=0.5)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    yield env, cfg.actions["residual"].scene_key
    env.close()


def test_zero_residual_reproduces_the_base_command_in_the_env(env64):
    env, _ = env64
    env.reset()
    term = env.action_manager.get_term("residual")
    drive = term.drive
    zero = torch.zeros(env.num_envs, 60, device=env.device)
    for _ in range(20):
        env.step(zero)
        assert torch.equal(term.command, term.base)
        assert torch.equal(drive.host, term.command)
    # a residual moves the command within its bounds, and decays back to the base under the slew limit
    env.step(torch.full_like(zero, 3.0))
    off = term.off
    assert torch.all(off[:, 0].abs() <= term.map.rate[0] * env.step_dt + 1e-6)
    assert not torch.equal(term.command, term.base)
    for _ in range(200):
        env.step(zero)
    assert torch.equal(term.command, term.base)


def test_bank_resets_and_hold(env64):
    from tactile_rl.mdp.events import tactile_state
    from tactile_rl.mdp.kinematics import hammer_in_hand

    env, key = env64
    obs, _ = env.reset()
    assert obs["actor"].shape == (env.num_envs, 100) and torch.isfinite(obs["actor"]).all()
    st = tactile_state(env)
    assert st.bank.size == env.num_envs
    rel = hammer_in_hand(env, key)
    assert torch.allclose(rel, st.ref_rel, atol=1e-5)  # each env starts on its bank entry
    assert torch.all(env.sim.data.time == 0)
    zero = torch.zeros(env.num_envs, 60, device=env.device)
    rewards = []
    for _ in range(300):  # 0.3 s
        _, rew, term, trunc, _ = env.step(zero)
        rewards.append(rew)
    assert not term.any()
    d = hammer_in_hand(env, key) - st.ref_rel
    assert torch.linalg.norm(d[:, :3], dim=1).max() < 0.2e-3
    r = torch.stack(rewards)
    assert torch.isfinite(r).all()
    # the 0.5 s episodes time out and restart from the bank
    for _ in range(250):
        _, _, term, trunc, _ = env.step(zero)
    assert int(env.sim.data.overflow.max()) == 0
    assert (env.episode_length_buf < 250).all()


def test_short_ppo_run(tmp_path):
    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from tactile_rl.tasks.rl_cfg import ppo_runner_cfg

    env = ManagerBasedRlEnv(hold_env_cfg(250.0, num_envs=128, episode_s=0.2), device="cuda:0")
    try:
        rl = ppo_runner_cfg(250.0)
        rl.max_iterations = 2
        runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env), asdict(rl), str(tmp_path), "cuda:0")
        runner.learn(num_learning_iterations=2)
        assert runner.current_learning_iteration >= 1
    finally:
        env.close()
