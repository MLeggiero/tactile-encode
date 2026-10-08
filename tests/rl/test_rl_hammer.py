"""The GPU hammer task (Tactile-Hammer-VegaWuji-*): registration, the observation groups (the student exactly
obs_spec's layout), a zero residual passing the base through, the stale-input decay, the base library played on the
GPU hitting the nail as the CPU did, the nail plant's per-world hooks, perturbed bases, and a short PPO run."""

import math

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.rl]  # torch, mjlab and CUDA (.venv-rl) or skipped
torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from tactile_rl.mdp.observations import PRIVILEGED  # noqa: E402
from tactile_rl.tasks.hammer import hammer_env_cfg, hammer_task_id  # noqa: E402
from tactile_rl.tasks.hold import RATES, decimation_for  # noqa: E402
from tactile_sim.control.obs_spec import obs_dim, obs_slices, stack_for  # noqa: E402


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """A small base library: two episodes of two swings (CPU, ~1 min)."""
    from tactile_sim.behaviors.base_library import assemble, record_episode

    path = tmp_path_factory.mktemp("lib") / "lib.npz"
    np.savez_compressed(path, **assemble([record_episode(s, n_strikes=2) for s in (0, 1)]))
    return str(path)


def make_env(library, n, rate=1000.0, **kw):
    from mjlab.envs import ManagerBasedRlEnv

    return ManagerBasedRlEnv(hammer_env_cfg(rate, library=library, num_envs=n, **kw), device="cuda:0")


def test_hammer_tasks_register_at_every_rate():
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    import tactile_rl.tasks  # noqa: F401

    for rate in RATES:
        assert hammer_task_id(rate) in list_tasks()
        cfg = load_env_cfg(hammer_task_id(rate))
        assert cfg.decimation == decimation_for(rate, 0.00025)
        assert set(cfg.observations) == {"student", "teacher", "critic"}
        assert load_rl_cfg(hammer_task_id(rate)).obs_groups["actor"] == ("teacher",)


@pytest.mark.parametrize("rate", [1000.0, 250.0])
def test_observation_groups_and_a_zero_residual(library, rate):
    env = make_env(library, 4, rate, perturb=False, assign="sequential")
    try:
        obs, _ = env.reset()
        stack = stack_for(rate)
        assert obs["student"].shape == (4, obs_dim(stack))
        assert obs["teacher"].shape == (4, obs_dim(stack) + sum(PRIVILEGED.values())) == obs["critic"].shape
        assert torch.equal(obs["teacher"][:, :obs_dim(stack)], obs["student"])
        term = env.action_manager.get_term("residual")
        assert term.map.trig.numel() == 0  # the trigger interlock is the drill task's only (trigger_indices)
        zero = torch.zeros(4, 60, device=env.device)
        for _ in range(int(0.03 * rate)):
            obs, *_ = env.step(zero)
            assert torch.equal(term.command, term.base) and torch.equal(term.drive.host, term.command)
        o = obs["student"].cpu().numpy()
        sl = obs_slices(stack)
        assert np.all(np.isfinite(o))
        assert o[:, sl["taxel_codes"]].max() > 0  # the hand holds the hammer: loaded taxels
        assert np.abs(o[:, sl["hand_q"]]).max() > 0.1 and np.all(o[:, sl["task"]][:, 0] == 1)
        status = o[:, sl["board_status"]].reshape(4, stack, 4, 2)
        assert np.all(status[:, 0, :, 0] < 0.6)  # every board's newest frame is fresh (< 3 ms)
        assert int(term.n_stale.sum()) <= 4 * 3  # only before the first frames and readback arrive
        cur = o[:, sl["hand_current"]].reshape(4, stack, 20)
        for env_cur in cur:  # a slower policy sees every millisecond: `stack` distinct 1 kHz samples
            assert len({row.tobytes() for row in env_cur}) == stack
    finally:
        env.close()


def test_unperturbed_base_outputs_are_the_librarys(library):
    """Unperturbed, the synergy scalar and the lookahead an env sees each tick are the library's recorded values, and a
    zero-error forecast reports the swing's planned contact."""
    from dataclasses import replace

    from mjlab.envs import ManagerBasedRlEnv

    from tactile_sim.config import ForecastCfg

    cfg = hammer_env_cfg(1000.0, library=library, num_envs=4, perturb=False, assign="sequential")
    cfg.commands["base"].forecast = replace(ForecastCfg(), mask_prob=0.0, dropout=0.0, bias_sigma=0.0,
                                            jitter_sigma=0.0, class_confusion=0.0, freq_err=0.0, amp_err=0.0)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        env.reset()
        cmd = env.command_manager.get_term("base")
        lib = cmd.lib
        zero = torch.zeros(4, 60, device=env.device)
        seen_forecast = False
        for k in range(1, 1000):
            env.step(zero)
            i = torch.full((4,), k, device=env.device)
            e = cmd.entry
            assert torch.equal(cmd.s_now, lib.s[e, i])
            arm = cmd.lookahead.reshape(4, 3, 27)[:, :, 20:]
            assert torch.allclose(arm, lib.look[e, i], atol=1e-6)
            fc = cmd.forecast
            on = fc[:, 0] == 1
            if k >= 100 and bool(on.any()):
                seen_forecast = True
                # the visible update was issued on the 10 Hz grid 100 ms (the latency) or more ago, with the truth's
                # event time as of then
                t_issue = (k - 100) // 100 * 100
                t_ev = lib.fc_t_event[e, torch.full_like(i, t_issue)]
                want = (t_ev - k * 1e-3).clamp(-1.0, 1.0)
                assert torch.allclose(fc[on, 1].double(), want[on], atol=1e-5)
                assert torch.allclose(fc[on, -1].double(), torch.full_like(want[on], (k - t_issue) * 1e-3), atol=1e-5)
        assert seen_forecast
    finally:
        env.close()


def test_stale_inputs_decay_the_residual(library):
    env = make_env(library, 2, perturb=False, assign="sequential")
    try:
        env.reset()
        term = env.action_manager.get_term("residual")
        rt = env._tactile_rt
        push = torch.full((2, 60), 0.5, device=env.device)
        for _ in range(30):
            env.step(push)
        built = term.off.abs().max().item()
        assert built > 0
        rt.res.stale_taxel = -1.0  # every frame now counts as stale (ResidualHandL2's test does the same)
        try:
            for _ in range(100):
                env.step(push)
        finally:
            rt.res.stale_taxel = 0.005
        assert term.off.abs().max().item() < 0.05 * built
    finally:
        env.close()


def test_the_gpu_replays_the_library_strikes(library):
    """Unperturbed, every entry hits the nail on the GPU as on the CPU, with the same advance and blow."""
    from tactile_rl.parity.strikes import gpu_replay
    from tactile_sim.behaviors.base_library import load_library

    lib = load_library(library)
    entries = list(range(len(lib["e_t0"])))
    gpu = {r["entry"]: r for r in gpu_replay(library, entries)["rows"]}
    for e in entries:
        g = gpu[e]
        assert g["hit"] and bool(lib["e_hit"][e])
        assert abs(g["t_contact"] - lib["e_t_contact"][e]) < 0.003
        assert g["advance"] == pytest.approx(lib["e_advance"][e], rel=0.15)
        assert g["peak2"] == pytest.approx(lib["e_peak2"][e], rel=0.15)
        assert g["v_face"] == pytest.approx(lib["e_v_face"][e], abs=0.05)


def test_nail_plant_writes_its_fields_per_world(library):
    env = make_env(library, 3, perturb=False, assign="sequential")
    try:
        env.reset()
        rt = env._tactile_rt
        nail = rt.nail
        zero = torch.zeros(3, 60, device=env.device)
        seen_contact = torch.zeros(3, dtype=torch.bool, device=env.device)
        depths = []  # the nail's depth after the last two substeps: a substep's pre-step reads the previous one's

        def keep(r):
            depths.append(r.nail.depth().clone())
            del depths[:-2]

        rt.recorders.append(keep)
        for _ in range(900):  # past the blows (~0.86 s after the windup)
            env.step(zero)
            seen_contact |= nail.force > 0
        assert seen_contact.all()
        fl = env.sim.model.dof_frictionloss[:, nail.dof]
        want = (nail.r0 + nail.r_per_m * depths[0].double().clamp(min=0.0)).float()  # before the last substep
        assert torch.allclose(fl, want, rtol=1e-6)
        k = nail.k_face
        sr = env.sim.model.pair_solref[:, nail.pair]
        assert torch.allclose(sr[:, 0], (-k).float())
        zeta = (-sr[:, 1].double()) / (2.0 * torch.sqrt(k))
        assert torch.all(zeta >= nail.zeta0 - 1e-5) and torch.all(zeta <= nail.zeta_max + 1e-5)
        assert torch.isfinite(nail.contact_zeta).all()  # the damping ratio frozen at each blow's onset
        assert (rt.f_peak > 50).all() and (rt.n_blows >= 1).all()
    finally:
        env.close()


def test_a_perturbed_base(library):
    env = make_env(library, 64)
    try:
        env.reset()
        cmd = env.command_manager.get_term("base")
        c = cmd.cfg
        for x, (lo, hi) in ((cmd.warp, c.time_warp), (cmd.amp, c.amplitude), (cmd.delay, c.start_delay)):
            assert float(x.min()) >= lo and float(x.max()) <= hi and float(x.std()) > 0
        tau = torch.zeros(64, dtype=torch.float64, device=env.device)
        first = cmd.arm_target(tau)
        lib = cmd.lib
        e = cmd.entry
        want = (lib.arm_q[e, 0] + lib.arm_droop[e, 0] + cmd.kd_kp * cmd.amp[:, None] * cmd.warp[:, None] *
                lib.arm_qd[e, 0]).float()
        assert torch.allclose(first, want, atol=1e-6)  # the base holds its first sample through the delay
        zero = torch.zeros(64, 60, device=env.device)
        for _ in range(200):
            obs, *_ = env.step(zero)
        for g in obs.values():
            assert torch.isfinite(g).all()
        look = cmd.lookahead.reshape(64, 3, 27)[:, :, 20:]
        assert torch.isfinite(look).all() and float(look.abs().max()) > 0
    finally:
        env.close()


def test_short_ppo_run_on_the_hammer_task(library, tmp_path):
    from dataclasses import asdict

    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from tactile_rl.tasks.rl_cfg import ppo_runner_cfg

    env = make_env(library, 64, 250.0, episode_s=0.3)
    try:
        rl = ppo_runner_cfg(250.0, experiment="tactile_hammer", obs_groups={"actor": ("teacher",),
                                                                            "critic": ("critic",)})
        rl.max_iterations = 2
        runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env), asdict(rl), str(tmp_path), "cuda:0")
        runner.learn(num_learning_iterations=2)
        assert runner.current_learning_iteration >= 1
        assert math.isfinite(float(env.reward_buf.mean()))
    finally:
        env.close()
