"""Milestone 3: the load generator batched on the GPU (schedules, wrench and forecast truth against
tactile_sim.model.loadgen), the Tactile-LoadGen task (observation groups, a zero residual, loads on the tool, finite
substep-integrated rewards the same at every policy rate), the reset randomization, the curriculum's stages, and the
teacher (zero-initialized GRU actor, the entropy schedule, a short PPO run)."""

import math

import numpy as np
import pytest

pytestmark = [pytest.mark.rl]  # torch (.venv-rl); the env tests also need mjlab and CUDA
torch = pytest.importorskip("torch")

from tactile_rl.plants.loadgen import TorchLoadGen  # noqa: E402
from tactile_sim.config import LoadGenCfg  # noqa: E402
from tactile_sim.control.obs_spec import ACTION_DIM, REGIMES  # noqa: E402
from tactile_sim.model.loadgen import KINDS, forecast_truth, kind_probs, sample_schedule, wrench  # noqa: E402

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
# the residual's training loads (loadgen_command.wide_loads without a band)
WIDE = dict(p_kind=(0.30, 0.21, 0.13, 0.13, 0.08, 0.15), cone_deg=60.0, bind_axes="xyz")


# ---------------------------------------------------------------------------------------------- the load generator


@pytest.mark.parametrize("over", [{}, WIDE], ids=["default", "wide"])
def test_batched_schedules_match_the_cpu_sampler_statistically(over):
    cfg = LoadGenCfg(**over)
    n, t_end = 4000, 3.0
    g = TorchLoadGen(cfg, n, "cpu", slots=24, generator=torch.Generator().manual_seed(0))
    g.sample(torch.arange(n), t_end, t_start=0.2)
    assert int(g.overflow.sum()) == 0
    rng = np.random.default_rng(0)
    cpu = [sample_schedule(cfg, rng, t_end, t_start=0.2) for _ in range(n)]
    n_cpu = np.mean([len(s) for s in cpu])
    n_gpu = float(g.valid.sum(1).double().mean())
    assert abs(n_gpu - n_cpu) / n_cpu < 0.05
    kinds_cpu = np.bincount([KINDS.index(e.kind) for s in cpu for e in s], minlength=6) / sum(len(s) for s in cpu)
    kv = g.kind[g.valid]
    kinds_gpu = torch.bincount(kv, minlength=6).double().numpy() / len(kv)
    assert np.abs(kinds_gpu - kinds_cpu).max() < 0.02
    for k, name in enumerate(KINDS):  # per-kind amplitude and duration means
        if kind_probs(cfg)[k] == 0:
            assert not bool((g.valid & (g.kind == k)).any())
            continue
        a_cpu = np.mean([e.amp for s in cpu for e in s if e.kind == name])
        d_cpu = np.mean([e.duration for s in cpu for e in s if e.kind == name])
        sel = g.valid & (g.kind == k)
        assert abs(float(g.amp[sel].mean()) - a_cpu) / a_cpu < 0.05
        assert abs(float(g.dur[sel].mean()) - d_cpu) / d_cpu < 0.05
    # torques' axes: +x unless the kind takes any axis (each of the six signed axes alike)
    for k, name in enumerate(KINDS):
        sel = g.valid & g.torque & (g.kind == k)
        if not bool(sel.any()):
            continue
        ax = g.tax[sel].numpy()
        assert np.all(np.abs(ax).sum(1) == 1.0)
        if name == "torque_impulse" or (name == "bind" and cfg.bind_axes == "xyz"):
            share = np.array([np.mean(ax[:, i] == s) for i in range(3) for s in (1.0, -1.0)])
            assert share == pytest.approx(np.full(6, 1 / 6), abs=0.04)
        else:
            assert np.all(ax[:, 0] == 1.0)
    imp = g.valid & (g.kind == 0)
    off = torch.rad2deg(torch.acos(g.dir[imp][:, 2].clamp(-1, 1)))
    assert float(off.max()) <= cfg.cone_deg + 1e-6 and float(off.max()) > 0.9 * cfg.cone_deg
    # events never overlap within an episode and stay inside it
    t0, end = g.t0.clone(), g.end()
    start = torch.where((g.kind == 2) & (g.sign > 0), t0 - g.dur, t0)
    ok = g.valid[:, 1:] & g.valid[:, :-1]
    assert bool((start[:, 1:][ok] >= end[:, :-1][ok] - 1e-9).all())
    assert bool((t0[g.valid] >= 0.2).all()) and bool((t0[g.valid] < t_end + 1.0).all())


@pytest.mark.parametrize("over", [{}, WIDE], ids=["default", "wide"])
def test_wrench_and_forecast_truth_match_the_cpu_schedule(over):
    cfg = LoadGenCfg(**over)
    rng = np.random.default_rng(3)
    scheds = [sample_schedule(cfg, rng, 4.0) for _ in range(8)]
    g = TorchLoadGen(cfg, len(scheds), "cpu", slots=24)
    for e, s in enumerate(scheds):
        g.set_events(e, s)
    for t in np.linspace(0.0, 4.2, 1500):
        f, tq = g.wrench(torch.full((len(scheds),), t, dtype=torch.float64))
        tr = g.truth(torch.full((len(scheds),), t, dtype=torch.float64))
        for e, s in enumerate(scheds):
            fc, tc = wrench(s, t)
            np.testing.assert_allclose(f[e].numpy(), fc, atol=1e-9)
            np.testing.assert_allclose(tq[e].numpy(), tc, atol=1e-9)
            ft = forecast_truth(s, t)
            assert bool(tr["valid"][e]) == ft.valid
            if ft.valid:
                assert float(tr["t_event"][e]) == pytest.approx(ft.t_event)
                assert REGIMES[int(tr["regime"][e])] == ft.regime
                assert float(tr["f"][e]) == pytest.approx(ft.f_dom)
                assert float(tr["amp"][e]) == pytest.approx(ft.amp)


def test_amplitude_scale():
    g = TorchLoadGen(LoadGenCfg(), 64, "cpu", generator=torch.Generator().manual_seed(1))
    g.sample(torch.arange(64), 3.0, amp_scale=0.25)
    t = torch.linspace(0.2, 3.0, 2000, dtype=torch.float64)
    vmax = max(float(g.values(torch.full((64,), float(x), dtype=torch.float64)).abs().max()) for x in t)
    assert vmax <= 0.25 * 700.0 * 2.0 + 1e-6  # the largest amplitude (with a periodic drift of at most 2), scaled


# ---------------------------------------------------------------------------------------------- the teacher's parts


def test_zero_initialized_gru_actor_and_the_entropy_schedule():
    from tensordict import TensorDict

    from tactile_rl.algos.models import ZeroInitRNNModel
    from tactile_rl.algos.ppo import ScheduledPPO
    from tactile_rl.tasks.rl_cfg import discount, teacher_runner_cfg

    obs = TensorDict({"teacher": torch.randn(5, 888)}, batch_size=[5])
    m = ZeroInitRNNModel(obs, {"actor": ["teacher"]}, "actor", ACTION_DIM, hidden_dims=(256, 128),
                         obs_normalization=True,
                         rnn_type="gru", rnn_hidden_dim=256,
                         distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3})
    assert m.is_recurrent and isinstance(m.rnn.rnn, torch.nn.GRU) and m.rnn.rnn.hidden_size == 256
    assert torch.equal(m(obs), torch.zeros(5, ACTION_DIM))  # a = 0: the base
    # the normalizer: a feature that never moved is not blown up when it does
    from tactile_rl.algos.models import RobustNormalization

    norm = RobustNormalization(3)
    norm.update(torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 2.0]]))
    y = norm(torch.tensor([[1.0, 2.0, 2.0]]))
    assert float(y[0, 0]) == pytest.approx(5.0) and float(y[0, 1]) == pytest.approx(0.0)
    assert isinstance(m.obs_normalizer, RobustNormalization)
    cfg = teacher_runner_cfg(1000.0, "loadgen", max_iterations=100)
    assert cfg.actor.rnn_type == "gru" and cfg.critic.hidden_dims == (512, 256, 128)
    assert cfg.num_steps_per_env == 128 and cfg.obs_groups == {"actor": ("teacher",), "critic": ("teacher",)}
    gamma, lam = discount(1000.0, "loadgen")
    assert cfg.algorithm.gamma == pytest.approx(math.exp(-1e-3 / 0.3)) == gamma
    assert cfg.algorithm.lam == pytest.approx(math.exp(-1e-3 / 0.15)) == lam
    assert teacher_runner_cfg(250.0, "hammer").algorithm.gamma == pytest.approx(0.9920, abs=1e-4)
    assert teacher_runner_cfg(100.0, "hammer").num_steps_per_env == 13
    p = ScheduledPPO.__new__(ScheduledPPO)
    p.entropy_coef_start, p.entropy_coef_final, p.entropy_decay_iterations = 1e-3, 0.0, 100
    assert p.entropy_at(0) == 1e-3 and p.entropy_at(50) == pytest.approx(5e-4) and p.entropy_at(500) == 0.0


# ---------------------------------------------------------------------------------------------- the task (GPU)


def make_env(n, rate=1000.0, **kw):
    pytest.importorskip("mjlab")
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.tasks.loadgen import loadgen_env_cfg

    return ManagerBasedRlEnv(loadgen_env_cfg(rate, num_envs=n, **kw), device="cuda:0")


@needs_gpu
@pytest.mark.gpu
def test_loadgen_tasks_register_at_every_rate():
    pytest.importorskip("mjlab")
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    import tactile_rl.tasks  # noqa: F401
    from tactile_rl.tasks.hold import RATES, decimation_for
    from tactile_rl.tasks.loadgen import loadgen_task_id

    for rate in RATES:
        assert loadgen_task_id(rate) in list_tasks()
        assert load_env_cfg(loadgen_task_id(rate)).decimation == decimation_for(rate, 0.00025)
        rl = load_rl_cfg(loadgen_task_id(rate))
        assert rl.actor.rnn_type == "gru" and rl.obs_groups["actor"] == ("teacher",)


@needs_gpu
@pytest.mark.gpu
def test_loadgen_env_observations_loads_and_a_zero_residual():
    from tactile_rl.mdp.observations import PRIVILEGED_BY_TASK
    from tactile_sim.control.obs_spec import obs_dim

    env = make_env(16, start_stage=2, max_stage=2)
    try:
        obs, _ = env.reset()
        assert obs["student"].shape == (16, obs_dim(1))
        assert obs["teacher"].shape == (16, obs_dim(1) + sum(PRIVILEGED_BY_TASK["loadgen"].values()))
        assert torch.equal(obs["teacher"][:, :obs_dim(1)], obs["student"])
        cmd = env.command_manager.get_term("base")
        act = env.action_manager.get_term("residual")
        zero = torch.zeros(16, env.action_manager.total_action_dim, device=env.device)
        body = cmd.app.body
        loaded = torch.zeros(16, dtype=torch.bool, device=env.device)
        total = torch.zeros(16, device=env.device)
        for _ in range(600):
            _, rew, *_ = env.step(zero)
            total += rew
            torch.testing.assert_close(act.command, act.base)  # a zero residual passes the base through
            loaded |= env.sim.data.xfrc_applied[:, body].abs().sum(1) > 0
        assert bool(torch.isfinite(total).all())
        # the loads land on the tool exactly when the schedule has one under way
        assert bool(loaded.any())
        t = cmd.rt.t
        f, tq = cmd.gen.wrench(t - env.physics_dt)
        on = (f.abs().sum(1) + tq.abs().sum(1)) > 0
        live = env.episode_length_buf > 1  # not reset in the last step
        applied = env.sim.data.xfrc_applied[:, body].abs().sum(1) > 0
        assert bool((on == applied)[live].all())
    finally:
        env.close()


@needs_gpu
@pytest.mark.gpu
def test_rewards_are_integrated_over_substeps():
    """The same hold at 1 kHz and at 250 Hz collects the same integrated penalties (a per-step reward would differ
    four-fold)."""
    sums = {}
    for rate in (1000.0, 250.0):
        env = make_env(32, rate, start_stage=0, max_stage=0, randomize=False)
        try:
            env.reset()
            zero = torch.zeros(32, env.action_manager.total_action_dim, device=env.device)
            n = int(round(0.4 / env.step_dt))
            for _ in range(n):
                env.step(zero)
            rm = env.reward_manager
            sums[rate] = {k: float(v.mean()) for k, v in rm._episode_sums.items()}
        finally:
            env.close()
    for k in ("effort_i2t", "pose_drift"):
        a, b = sums[1000.0][k], sums[250.0][k]
        assert a < 0 and b < 0
        assert abs(a - b) / abs(a) < 0.15, (k, a, b)


@needs_gpu
@pytest.mark.gpu
def test_reset_randomization_writes_per_world_parameters():
    from tactile_rl.mdp.actions import hand_drive

    env = make_env(64, start_stage=2, max_stage=2)
    try:
        env.reset()
        dr = env._tactile_dr
        c = dr.cfg
        m = env.sim.model
        sp = dr.skin_pairs
        mu = m.pair_friction[:, sp, 0]
        assert float(mu.min()) >= c.hand_friction[0] - 1e-6 and float(mu.max()) <= c.hand_friction[1] + 1e-6
        assert float(mu.std()) > 0.05  # per world
        d0 = m.pair_solimp[:, sp, 0]
        assert float(d0.min()) >= 0.85 - 1e-6 and float(d0.max()) <= 0.95 + 1e-6
        hm = dr.v["head_mass"]
        assert float(hm.min()) >= c.head_mass[0] and float(hm.max()) <= c.head_mass[1]
        mass = m.body_mass[:, dr.hammer_bid]
        # the hammer body's mass is the head's plus the handle's (the YCB's 0.215 kg, the light hammer's 0.10)
        torch.testing.assert_close(mass.double(), hm + dr.parts.cfg.hammer.handle_mass, atol=2e-3, rtol=0)
        drive = hand_drive(env.scene["robot"])
        lat = drive.cmd_latency
        assert float(lat.min()) >= c.cmd_latency[0] - 1e-9 and float(lat.max()) <= c.cmd_latency[1] + 1e-9
        rt = env._tactile_rt
        assert float(rt.scan.f_half.min()) >= c.ts_f_half[0] and float(rt.scan.f_half.max()) <= c.ts_f_half[1]
        assert float((1 - rt.scan.alive).mean()) <= 0.03
        fc = env.command_manager.get_term("base").forecaster
        assert float(fc.latency.min()) >= 0.05 - 1e-9 and float(fc.latency.max()) <= 0.25 + 1e-9
        assert float(fc.mask_prob.mean()) == pytest.approx(0.25)
        from tactile_rl.mdp.randomization import PARAM_DIM

        assert dr.param_vector().shape == (64, PARAM_DIM) == (64, 34)
    finally:
        env.close()


@needs_gpu
@pytest.mark.gpu
def test_curriculum_stages():
    env = make_env(16, start_stage=0, max_stage=2, min_episodes=10, min_events=10)
    try:
        env.reset()
        cur = env.curriculum_manager._term_cfgs[0].func
        cmd = env.command_manager.get_term("base")
        dr = env._tactile_dr
        assert cur.stage == 0 and not cmd.enabled and dr.scale == pytest.approx(0.1) and dr.forecast_mode == "clean"
        dev = env.device
        cur.win = torch.tensor([100.0, 5.0, 0.0, 0.0], dtype=torch.float64, device=dev)  # 5 % drops: stay
        cur._evaluate()
        assert cur.stage == 0
        cur.win = torch.tensor([100.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=dev)
        cur._evaluate()
        assert cur.stage == 1 and cmd.enabled and cmd.amp_scale == pytest.approx(0.25)
        assert cmd.hold_level == (0.5, 1.0)
        assert dr.scale == pytest.approx(0.3) and dr.forecast_mode == "clean"
        cur.win = torch.tensor([100.0, 0.0, 200.0, 150.0], dtype=torch.float64, device=dev)  # 75 % within target
        cur._evaluate()
        assert cur.stage == 1
        cur.win = torch.tensor([100.0, 0.0, 200.0, 170.0], dtype=torch.float64, device=dev)
        cur._evaluate()
        assert cur.stage == 2 and cmd.amp_scale == 1.0 and dr.scale == 1.0 and cmd.hold_level == (0.3, 0.8)
        assert dr.forecast_mode == "noisy" and dr.forecast_masking
        # a stage is judged on the episodes that began in it: one under way at the change counts for neither stage
        env.step(torch.zeros(16, env.action_manager.total_action_dim, device=dev))
        cur.win.zero_()
        cur.ep_stage[:8], cur.ep_stage[8:] = 1, 2
        cur(env, torch.arange(16, device=dev))
        assert float(cur.win[0]) == 8.0 and bool((cur.ep_stage == 2).all())
    finally:
        env.close()


@needs_gpu
@pytest.mark.gpu
def test_short_teacher_ppo_run(tmp_path):
    from dataclasses import asdict

    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from tactile_rl.tasks.rl_cfg import teacher_runner_cfg

    env = make_env(64, start_stage=1, max_stage=1)
    try:
        rl = teacher_runner_cfg(1000.0, "loadgen", max_iterations=2)
        rl.num_steps_per_env = 8
        runner = MjlabOnPolicyRunner(RslRlVecEnvWrapper(env), asdict(rl), str(tmp_path), "cuda:0")
        runner.learn(num_learning_iterations=2)
        assert runner.current_learning_iteration >= 1
        assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
        assert runner.alg.entropy_coef < 1e-3  # the schedule moved
    finally:
        env.close()


# ---------------------------------------------------------------------------------------------- the C3 mix (GPU)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """A small base library: two episodes of two swings (CPU, ~1 min)."""
    from tactile_sim.behaviors.base_library import assemble, record_episode

    path = tmp_path_factory.mktemp("lib") / "lib.npz"
    np.savez_compressed(path, **assemble([record_episode(s, n_strikes=2) for s in (0, 1)]))
    return str(path)


@needs_gpu
@pytest.mark.gpu
def test_c3_mixes_hammer_and_load_generator_episodes(library):
    pytest.importorskip("mjlab")
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.mdp.observations import PRIVILEGED_BY_TASK
    from tactile_rl.tasks.residual import residual_env_cfg
    from tactile_sim.control.obs_spec import TASKS, obs_dim, obs_slices

    env = ManagerBasedRlEnv(residual_env_cfg(1000.0, library=library, num_envs=64, start_stage=3, max_stage=3,
                                             nan_policy="error"), device="cuda:0")
    try:
        obs, _ = env.reset()
        assert obs["teacher"].shape == (64, obs_dim(1) + sum(PRIVILEGED_BY_TASK["mixed"].values()))
        cmd = env.command_manager.get_term("base")
        ham = cmd.is_hammer
        assert 0.4 < float(ham.float().mean()) < 0.95  # 70 % hammer
        assert cmd.hold_level == (0.3, 1.2) and env._tactile_dr.scale == 1.0
        # the task block says which, a hammer env has no loads, its hold level is drawn
        task = obs["student"][:, obs_slices(1)["task"]]
        assert torch.equal(task.argmax(1), torch.where(ham, TASKS.index("hammer"), TASKS.index("loadgen")))
        assert not bool(cmd.lg.gen.valid[ham].any())
        lv = torch.where(ham, cmd.ham.level, cmd.lg.level)
        assert float(lv.min()) >= 0.3 - 1e-6 and float(lv.max()) <= 1.2 + 1e-6
        act = env.action_manager.get_term("residual")
        zero = torch.zeros(64, env.action_manager.total_action_dim, device=env.device)
        total = torch.zeros(64, device=env.device)
        assert bool(torch.isnan(cmd.ham.lib.fc_t_event).any())  # the library's "no event" (after each contact)
        for _ in range(1000):
            ham_now = cmd.is_hammer.clone()
            obs, rew, *_ = env.step(zero)
            total += rew
            torch.testing.assert_close(act.command, act.base)
            # only a hammer episode can end with its entry; the teacher's observation stays finite (no event: 0)
            assert not bool((env.termination_manager.get_term("base_done") & ~ham_now).any())
            assert bool(torch.isfinite(obs["teacher"]).all())
        assert bool(torch.isfinite(total).all())
    finally:
        env.close()


@needs_gpu
@pytest.mark.gpu
def test_c3_training_distribution(library):
    """The training distribution of 2026-10-13: every episode's hand/tool friction drawn from TRAIN_HAND_FRICTION (a
    hammer episode's too, over its library entry's), VLA-style hand bases in a share of the episodes (the observation
    shows the base the residual composes on; the table's gains; targets held between the 30 Hz grid's points), the
    wide loads, and the stage cap per env count."""
    pytest.importorskip("mjlab")
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.mdp.randomization import PARAM_DIM
    from tactile_rl.tasks.residual import CAP_REF_ENVS, STAGE_CAP_S, TRAIN_HAND_FRICTION, residual_env_cfg
    from tactile_sim.control.obs_spec import obs_slices

    N = 64
    env = ManagerBasedRlEnv(residual_env_cfg(1000.0, library=library, num_envs=N, start_stage=3, max_stage=3,
                                             vla_frac=0.5, nan_policy="error"), device="cuda:0")
    try:
        obs, _ = env.reset()
        cmd = env.command_manager.get_term("base")
        dr, rt, m = env._tactile_dr, env._tactile_rt, env.sim.model
        cur = env.curriculum_manager._term_cfgs[0].func
        assert cur.stage_cap_s == pytest.approx(STAGE_CAP_S * N / CAP_REF_ENVS)
        mu = m.pair_friction[:, dr.skin_pairs, 0]
        lo, hi = TRAIN_HAND_FRICTION
        assert float(mu.min()) >= lo - 1e-6 and float(mu.max()) <= hi + 1e-6
        assert bool((mu == mu[:, :1]).all())  # one friction per env, on every skin pair
        torch.testing.assert_close(mu[:, 0].double(), dr.v["hand_friction"], atol=1e-6, rtol=0)
        ham = cmd.is_hammer
        assert float(mu[ham, 0].min()) < 1.1  # below the library's range: the hammer's own draw
        torch.testing.assert_close(cmd.ham.mu[ham], dr.v["hand_friction"][ham])
        vla = rt.vla
        assert 0.2 < float(vla.active.float().mean()) < 0.8
        pv = dr.param_vector()
        assert pv.shape == (N, PARAM_DIM) and torch.equal(pv[:, -1].bool(), vla.active)
        g = cmd.lg.gen.cfg
        assert g.cone_deg == 60.0 and g.bind_axes == "xyz" and len(g.p_kind) == 6
        act = env.action_manager.get_term("residual")
        zero = torch.zeros(N, env.action_manager.total_action_dim, device=env.device)
        sl = obs_slices(1)
        refreshes = torch.zeros(N, device=env.device)
        reset = torch.zeros(N, dtype=torch.bool, device=env.device)
        prev_q, prev_next, prev_t = None, None, None
        for _ in range(100):
            obs, *_ = env.step(zero)
            base = act.base_command()  # what the residual composes on at the next step
            torch.testing.assert_close(obs["student"][:, sl["base_hand_now"]], base[:, 0])
            on = vla.active & (rt.rb_q.seq >= 0)
            assert torch.equal(base[on][:, 2], vla.kp.expand(int(on.sum()), -1))
            assert torch.equal(base[on][:, 4], vla.tau.expand(int(on.sum()), -1))
            if prev_q is not None:
                t = rt.t
                same_ep = t > prev_t  # (an env that reset restarted its clock)
                held = on & same_ep & (t < prev_next - 1e-9)
                assert torch.equal(base[held][:, 0], prev_q[held])
                refreshes += (on & same_ep & ~torch.all(base[:, 0] == prev_q, dim=1)).float()
                reset |= ~same_ep
            prev_q, prev_next, prev_t = base[:, 0].clone(), vla.next_t.clone(), rt.t.clone()
        # VLA-style for the whole 100 ms: after the first reading's refresh (at 1 ms, not counted) the grid's points
        whole = vla.active & ~reset
        assert bool(whole.any()) and float(refreshes[whole].max()) <= 3.0 and float(refreshes[whole].min()) >= 2.0
    finally:
        env.close()


@needs_gpu
@pytest.mark.gpu
def test_the_curriculum_reaches_c3(library):
    pytest.importorskip("mjlab")
    from mjlab.envs import ManagerBasedRlEnv

    from tactile_rl.tasks.residual import residual_env_cfg

    env = ManagerBasedRlEnv(residual_env_cfg(1000.0, library=library, num_envs=16, start_stage=2, max_stage=3,
                                             min_episodes=10, min_events=10), device="cuda:0")
    try:
        env.reset()
        cur = env.curriculum_manager._term_cfgs[0].func
        cmd = env.command_manager.get_term("base")
        assert cur.stage == 2 and cmd.hammer_frac == 0.0
        # 75 % within target: C2 holds until its time cap, then C3 begins anyway
        cur.win = torch.tensor([100.0, 0.0, 200.0, 150.0], dtype=torch.float64, device=env.device)
        cur._evaluate()
        assert cur.stage == 2
        cur.stage_cap_s = 0.0
        cur.win = torch.tensor([100.0, 0.0, 200.0, 150.0], dtype=torch.float64, device=env.device)
        cur._evaluate()
        assert cur.stage == 3 and cur.capped == 1 and cur.history[-1]["how"] == "capped"
        assert cmd.hammer_frac == pytest.approx(0.7) and cmd.hold_level == (0.3, 1.2)
    finally:
        env.close()
