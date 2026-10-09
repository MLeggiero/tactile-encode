"""Milestone 5: the student (tactile_rl.algos.student_model), DAgger's labels and loop (tactile_rl.algos.dagger), and
the student's ONNX export with its auxiliary output (tactile_rl.export.onnx_export)."""

import numpy as np
import pytest

pytestmark = [pytest.mark.rl]
torch = pytest.importorskip("torch")

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _trained_like(net, seed: int = 0):
    """Normalizer statistics and a non-zero action layer, as training leaves them."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        net.norm.mean.copy_(torch.randn(net.obs_dim, generator=g) * 0.2)
        var = torch.rand(net.obs_dim, generator=g) * 2.0
        var[::5] = 1e-4  # under the scale's floor
        net.norm.var.copy_(var)
        net.norm.count.fill_(1000.0)
        net.head[-1].weight.normal_(0.0, 0.2, generator=g)
        net.head[-1].bias.normal_(0.0, 0.2, generator=g)
    net.norm.frozen = True
    return net.eval()


@pytest.mark.parametrize("stack", [1, 4])
def test_student_size_zero_start_and_sequence_matches_steps(stack):
    from tactile_rl.algos.student_model import AUX_DIM, StudentNet, count_parameters

    net = StudentNet(stack)
    n = count_parameters(net)
    assert 0.35e6 < n < 0.5e6, n  # the plan's ~0.45 M
    B = 3
    x = torch.rand(5, B, net.obs_dim)
    h = net.initial_state(B)
    with torch.no_grad():
        mu, h1, aux = net.step(x[0], h)
    assert torch.equal(mu, torch.zeros_like(mu)) and aux.shape == (B, AUX_DIM)  # a fresh student is the base
    _trained_like(net)
    starts = torch.zeros(5, B, dtype=torch.bool)
    starts[3, 1] = True  # env 1's episode restarts at step 3
    h0 = torch.randn(1, B, net.hidden) * 0.1
    with torch.no_grad():
        mu_seq, aux_seq = net.sequence(x, h0, starts)
        h = h0.clone()
        for t in range(5):
            h = h * (~starts[t]).float()[None, :, None]
            mu_t, h, aux_t = net.step(x[t], h)
            torch.testing.assert_close(mu_seq[t], mu_t, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(aux_seq[t], aux_t, atol=1e-5, rtol=1e-5)


def test_future_slip_labels():
    from tactile_rl.algos.dagger import future_slip

    T, N, P = 10, 2, 11
    speed = torch.zeros(T, N, P)
    speed[:, :, 0] = torch.arange(T, dtype=torch.float32)[:, None]  # m/s
    done = torch.zeros(T, N, dtype=torch.bool)
    done[4, 1] = True  # env 1's episode ends with step 4
    out, ok = future_slip(speed, done, k=3, dt=0.001)
    # step 2: speeds at 3, 4, 5 -> 12 m/s * 1 ms = 12 mm
    assert out[2, 0, 0] == pytest.approx(12.0) and bool(ok[2, 0])
    assert not bool(ok[7:].any())  # the window leaves the rollout
    assert bool(ok[1, 1]) and not bool(ok[2, 1]) and not bool(ok[4, 1]) and bool(ok[5, 1])  # crosses the episode end


def test_student_export_parity_and_cpu_runner(tmp_path):
    pytest.importorskip("onnxruntime")
    from tactile_rl.algos.student_model import AUX_DIM, StudentNet
    from tactile_rl.export.onnx_export import check_student, export_student
    from tactile_sim.control.obs_spec import ActionSpec, spec_hash
    from tactile_sim.control.policy_io import OnnxPolicy, PolicyMeta

    net = _trained_like(StudentNet(1))
    ck = tmp_path / "student.pt"
    torch.save({"student_state_dict": net.state_dict(), "stack": 1, "hidden": net.hidden, "rate": 1000.0,
                "teacher": "teacher.pt", "iter": 7}, ck)
    out = tmp_path / "student_1k.onnx"
    info = export_student(str(ck), str(out), commit="test")
    assert info["hidden"] == 192 and info["obs_dim"] == net.obs_dim
    meta = PolicyMeta.read(out)
    assert meta.spec_hash == spec_hash(ActionSpec(), 1) and meta.hidden == 192 and meta.rate == 1000.0
    assert meta.extra["kind"] == "student" and sum(meta.extra["aux"].values()) == AUX_DIM
    rng = np.random.default_rng(0)
    obs = rng.random((40, 4, net.obs_dim)).astype(np.float32)
    dones = np.zeros((40, 4), bool)
    dones[15, 2] = True
    err = check_student(str(out), str(ck), obs, dones)
    assert err["step_max_abs_action"] < 1e-5 and err["step_max_abs_state"] < 1e-5 and err["step_max_abs_aux"] < 1e-4
    assert err["max_abs_action"] < 1e-5
    pol = OnnxPolicy(out)  # the CPU runner reads action and state, the aux output rides along
    a = pol(obs[0, 0])
    assert a.shape == (60,) and np.all(np.abs(a) <= 1.0)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    from tactile_sim.behaviors.base_library import assemble, record_episode

    path = tmp_path_factory.mktemp("lib") / "lib.npz"
    np.savez_compressed(path, **assemble([record_episode(s, n_strikes=2) for s in (0, 1)]))
    return str(path)


@needs_gpu
@pytest.mark.gpu
def test_dagger_runs_a_few_iterations(library, tmp_path):
    pytest.importorskip("mjlab")
    from tensordict import TensorDict

    from tactile_rl.algos.dagger import DaggerCfg, DaggerTrainer, load_student
    from tactile_rl.algos.models import ZeroInitRNNModel
    from tactile_rl.mdp.observations import PRIVILEGED_BY_TASK
    from tactile_sim.control.obs_spec import obs_dim

    d = obs_dim(1) + sum(PRIVILEGED_BY_TASK["mixed"].values())
    teacher = ZeroInitRNNModel(TensorDict({"teacher": torch.zeros(1, d)}, batch_size=[1]), {"actor": ["teacher"]},
                               "actor", 60, hidden_dims=(256, 128), obs_normalization=True, rnn_type="gru",
                               rnn_hidden_dim=256, distribution_cfg={"class_name": "GaussianDistribution",
                                                                     "init_std": 0.3, "std_type": "scalar"})
    with torch.no_grad():
        last = [m for m in teacher.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight.normal_(0.0, 0.05)
    tck = tmp_path / "teacher.pt"
    torch.save({"actor_state_dict": teacher.state_dict()}, tck)
    cfg = DaggerCfg(num_envs=32, iterations=3, bc_iterations=1, mix_iterations=1, rollout_s=0.032, minibatches=2,
                    epochs=1, save_every=3)
    tr = DaggerTrainer(cfg, str(tck), library, str(tmp_path / "run"), device="cuda:0")
    try:
        tr.train()
    finally:
        tr.env.close()
    assert tr.use_teacher.dtype == torch.bool
    st = load_student(str(tmp_path / "run" / "student_2.pt"))
    assert all(torch.isfinite(p).all() for p in st.parameters())
    assert float(st.norm.count) > 0
