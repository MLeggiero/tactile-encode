"""Milestone 4: the teacher's ONNX export (tactile_rl.export.onnx_export): the graph folds in the robust normalizer and
carries the GRU's state in and out, its sidecar is the CPU policy runner's, and onnxruntime agrees with the torch actor
to 1e-5 over recorded sequences with episode resets."""

import numpy as np
import pytest

pytestmark = [pytest.mark.rl]
torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
pytest.importorskip("onnxruntime")


def _actor_checkpoint(path, d: int, seed: int = 0) -> str:
    """A teacher actor as training leaves it: normalizer statistics (some scales under the floor, so the clamp acts),
    a non-zero output layer."""
    from tensordict import TensorDict

    from tactile_rl.algos.models import ZeroInitRNNModel

    torch.manual_seed(seed)
    actor = ZeroInitRNNModel(TensorDict({"teacher": torch.zeros(1, d)}, batch_size=[1]), {"actor": ["teacher"]},
                             "actor", 60, hidden_dims=(256, 128), obs_normalization=True, rnn_type="gru",
                             rnn_hidden_dim=256, distribution_cfg={"class_name": "GaussianDistribution",
                                                                   "init_std": 0.3, "std_type": "scalar"})
    norm = actor.obs_normalizer
    with torch.no_grad():
        norm._mean.copy_(torch.randn_like(norm._mean))
        std = torch.rand_like(norm._std) * 2.0
        std[:, ::7] = 0.02  # under the 0.1 floor
        norm._std.copy_(std)
        norm._var.copy_(std**2)
        last = [m for m in actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight.normal_(0.0, 0.2)
        last.bias.normal_(0.0, 0.2)
    torch.save({"actor_state_dict": actor.state_dict(), "iter": 0}, path)
    return str(path)


def test_teacher_export_matches_torch_and_runs_on_the_cpu_runner(tmp_path):
    from tactile_rl.export.onnx_export import OPSET, check_against_torch, export_teacher
    from tactile_rl.mdp.observations import PRIVILEGED_BY_TASK
    from tactile_sim.control.obs_spec import ActionSpec, obs_dim, spec_hash
    from tactile_sim.control.policy_io import OnnxPolicy, PolicyMeta

    d = obs_dim(1) + sum(PRIVILEGED_BY_TASK["mixed"].values())
    ckpt = _actor_checkpoint(tmp_path / "model.pt", d)
    out = tmp_path / "teacher.onnx"
    info = export_teacher(ckpt, str(out), "mixed", 1000.0, commit="test")
    assert info["obs_dim"] == d and info["hidden"] == 256
    meta = PolicyMeta.read(out)
    assert meta.spec_hash == spec_hash(ActionSpec(), 1) and meta.rate == 1000.0 and meta.stack == 1
    assert meta.hidden == 256 and meta.extra["obs_dim"] == d and meta.extra["opset"] == OPSET
    assert meta.extra["normalizer"] == {"std_floor": 0.1, "clip": 5.0} and meta.extra["commit"] == "test"
    assert meta.extra["codec"]["version"] == "taxel_codec_v1"
    # recorded-like inputs: wide enough that the normalizer's clip and floor act; episodes end along the way
    rng = np.random.default_rng(1)
    T, B = 64, 8
    obs = (rng.standard_normal((T, B, d)) * 3.0).astype(np.float32)
    dones = np.zeros((T, B), bool)
    dones[20, 1] = dones[40, 5] = dones[41, 0] = True
    err = check_against_torch(str(out), ckpt, obs, dones)
    assert err["max_abs_action"] < 1e-5 and err["max_abs_state"] < 1e-5
    # the CPU runner (tactile_sim.control.policy_io) loads it and carries the state
    pol = OnnxPolicy(out)
    a1 = pol(obs[0, 0])
    a2 = pol(obs[1, 0])
    assert a1.shape == (60,) and np.all(np.abs(a1) <= 1.0) and not np.allclose(a1, a2)
