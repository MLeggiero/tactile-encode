"""ONNX export of a teacher's actor (the plan's "Export", for the teacher; the students come with DAgger).

The graph is the CPU's policy convention (tactile_sim.control.policy_io): inputs `obs` [B, D] (the teacher's
observation: obs_spec's student layout followed by the privileged blocks) and `h_in` [B, H] (the GRU's state), outputs
`action` [B, 60] (squashed: tanh of the policy's mean, as the env applies it) and `h_out` [B, H]. The robust
normalizer (scale floor, clip) is folded in; opset 17; the batch axis is dynamic.

The sidecar JSON (PolicyMeta) records obs_spec's hash for the student part and its stack, the rate, the GRU's size and
the provenance: the observation group and its blocks' widths, the normalizer's floor and clip, the codec version, the
commit, the checkpoint. `check_against_torch` replays recorded observation sequences through onnxruntime and the
torch actor side by side (state carried step to step, reset where an episode ended) and reports the largest
differences; `record_inputs` records them from the teacher playing Tactile-Residual's C3 (the plan: 1e4 inputs). The
check's result goes into the sidecar ("parity").

The plan's signature has a third output, `aux` (the student's auxiliary heads); a teacher has none, so its graph has
two outputs.

    python -m tactile_rl.export.onnx_export CHECKPOINT OUT.onnx --task mixed --rate 1000 \
        [--record 64x160 --library LIB | --check RECORD.npz]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn

OPSET = 17


class TeacherGraph(nn.Module):
    """obs, h_in -> action, h_out for an rsl_rl RNNModel actor (GRU, one layer) with RobustNormalization."""

    def __init__(self, actor):
        super().__init__()
        norm = actor.obs_normalizer
        self.register_buffer("mean", norm._mean.detach().clone().float())
        self.register_buffer("std", torch.clamp(norm._std.detach().clone().float(), min=float(norm.std_floor)))
        self.clip = float(norm.clip)
        gru = actor.rnn.rnn
        if not isinstance(gru, nn.GRU) or gru.num_layers != 1:
            raise ValueError("the export takes a one-layer GRU actor")
        self.gru = nn.GRU(gru.input_size, gru.hidden_size, num_layers=1)
        self.gru.load_state_dict(gru.state_dict())
        self.mlp = actor.mlp
        self.hidden = gru.hidden_size

    def forward(self, obs: torch.Tensor, h_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
        y, h = self.gru(x.unsqueeze(0), h_in.unsqueeze(0))
        a = torch.tanh(self.mlp(y.squeeze(0)))
        return a, h.squeeze(0)


def load_actor(checkpoint: str, obs_dim: int | None = None, device: str = "cpu"):
    """The teacher's actor from a training checkpoint (rsl_rl's actor_state_dict), sizes read from it."""
    from tensordict import TensorDict

    from tactile_rl.algos.models import ZeroInitRNNModel

    sd = torch.load(checkpoint, map_location=device, weights_only=False)["actor_state_dict"]
    d = int(sd["obs_normalizer._mean"].shape[-1]) if obs_dim is None else int(obs_dim)
    hidden = int(sd["rnn.rnn.weight_hh_l0"].shape[1])
    lin = sorted((k for k in sd if k.startswith("mlp.") and k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
    dims = [int(sd[k].shape[0]) for k in lin]
    std_type = "log" if "distribution.log_std_param" in sd else "scalar"
    obs = TensorDict({"teacher": torch.zeros(1, d)}, batch_size=[1])
    actor = ZeroInitRNNModel(obs, {"actor": ["teacher"]}, "actor", dims[-1], hidden_dims=tuple(dims[:-1]),
                             obs_normalization=True, rnn_type="gru", rnn_hidden_dim=hidden,
                             distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                                               "std_type": std_type})
    actor.load_state_dict(sd)
    return actor.eval()


def _commit() -> str:
    """The checkout's HEAD; outside a checkout (a synced copy) the COMMIT file at the repo root, if any."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:  # noqa: BLE001 - outside a checkout
        f = Path(__file__).resolve().parents[2] / "COMMIT"
        return f.read_text().strip() if f.exists() else "unknown"


def export_teacher(checkpoint: str, out: str, task: str, rate: float, commit: str | None = None) -> dict:
    from tactile_rl.mdp.observations import PRIVILEGED_BY_TASK
    from tactile_sim.control.obs_spec import ActionSpec, obs_dim, spec_hash, stack_for
    from tactile_sim.control.policy_io import PolicyMeta
    from tactile_sim.sensors.taxel_codec import CodecSpec

    actor = load_actor(checkpoint)
    graph = TeacherGraph(actor).eval()
    stack = stack_for(rate)
    d_student = obs_dim(stack)
    widths = PRIVILEGED_BY_TASK[task]
    D = int(graph.mean.shape[-1])
    if D != d_student + sum(widths.values()):
        raise ValueError(f"the actor reads {D} values, the {task} teacher at {rate:g} Hz {d_student} + "
                         f"{sum(widths.values())}")
    out = str(out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    obs = torch.zeros(1, D)
    h = torch.zeros(1, graph.hidden)
    torch.onnx.export(graph, (obs, h), out, input_names=["obs", "h_in"], output_names=["action", "h_out"],
                      dynamic_axes={"obs": {0: "batch"}, "h_in": {0: "batch"}, "action": {0: "batch"},
                                    "h_out": {0: "batch"}}, opset_version=OPSET, dynamo=False)
    codec = CodecSpec()
    meta = PolicyMeta(spec_hash=spec_hash(ActionSpec(), stack), rate=float(rate), stack=stack, hidden=graph.hidden,
                      name=Path(out).stem,
                      extra={"kind": "teacher", "task": task, "obs_group": "teacher", "obs_dim": D,
                             "student_dim": d_student, "privileged": widths,
                             "normalizer": {"std_floor": float(actor.obs_normalizer.std_floor),
                                            "clip": float(actor.obs_normalizer.clip)},
                             "codec": {"bits": codec.bits, "mu": codec.mu, "f_max": codec.f_max,
                                       "version": "taxel_codec_v1"},
                             "action": "tanh(mean)", "opset": OPSET, "commit": commit or _commit(),
                             "checkpoint": str(checkpoint)})
    meta.write(out)
    return {"onnx": out, "obs_dim": D, "hidden": graph.hidden}


def check_against_torch(onnx_path: str, checkpoint: str, obs_seq: np.ndarray, dones: np.ndarray | None = None
                        ) -> dict[str, float]:
    """obs_seq (T, B, D) recorded observations (dones (T, B): a new episode begins after that step); the actor in torch
    (rsl_rl's inference path: normalizer, GRU with its stored state, MLP, tanh) and the ONNX graph step by step, state
    carried and reset alike. Returns the largest absolute differences of the action and of the state, each runtime
    carrying its own state (free running: float32 differences accumulate through the recurrence), and per step (the
    graph fed torch's state: the graph's own error)."""
    import onnxruntime as ort
    from tensordict import TensorDict

    actor = load_actor(checkpoint)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    T, B, _ = obs_seq.shape
    H = actor.rnn.rnn.hidden_size
    h = np.zeros((B, H), dtype=np.float32)
    actor.reset()
    err = {"max_abs_action": 0.0, "max_abs_state": 0.0, "step_max_abs_action": 0.0, "step_max_abs_state": 0.0}

    def upd(k, x, y):
        err[k] = max(err[k], float(np.abs(x - y).max()))

    with torch.inference_mode():
        for t in range(T):
            hs = actor.get_hidden_state()
            h_prev = np.zeros((B, H), dtype=np.float32) if hs is None else hs[0].numpy().copy()
            o = torch.as_tensor(obs_seq[t], dtype=torch.float32)
            a_t = torch.tanh(actor(TensorDict({"teacher": o}, batch_size=[B]))).numpy()
            h_t = actor.get_hidden_state()[0].numpy()
            x = obs_seq[t].astype(np.float32)
            a_o, h = sess.run(["action", "h_out"], {"obs": x, "h_in": h})
            a_1, h_1 = sess.run(["action", "h_out"], {"obs": x, "h_in": h_prev})
            upd("max_abs_action", a_o, a_t)
            upd("max_abs_state", h, h_t)
            upd("step_max_abs_action", a_1, a_t)
            upd("step_max_abs_state", h_1, h_t)
            if dones is not None and dones[t].any():
                actor.reset(torch.as_tensor(dones[t]))
                h[dones[t].astype(bool)] = 0.0
    return {"steps": T, "batch": B, "inputs": T * B, **err}


def record_inputs(checkpoint: str, rate: float, num_envs: int = 64, steps: int = 160, library: str | None = None,
                  stage: int = 3, seed: int = 0, device: str = "cuda:0") -> dict[str, np.ndarray]:
    """The teacher's observations as it plays (its mean action) in Tactile-Residual at `stage` (C3: hammer and load
    generator episodes, the full randomization): obs (steps, num_envs, D) and dones (steps, num_envs)."""
    from mjlab.envs import ManagerBasedRlEnv
    from tensordict import TensorDict

    from tactile_rl.tasks.common import TEACHER_GROUPS
    from tactile_rl.tasks.residual import residual_env_cfg

    torch.manual_seed(seed)
    actor = load_actor(checkpoint, device=device).to(device)
    env = ManagerBasedRlEnv(residual_env_cfg(rate, library=library, num_envs=num_envs, start_stage=stage,
                                             max_stage=stage, obs_groups=TEACHER_GROUPS), device=device)
    obs_rec, done_rec = [], []
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            actor.reset()
            for _ in range(steps):
                o = obs["teacher"]
                obs_rec.append(o.cpu().numpy().astype(np.float32))
                a = actor(TensorDict({"teacher": o}, batch_size=[num_envs]))
                obs, _, term, trunc, _ = env.step(a)
                d = term | trunc
                done_rec.append(d.cpu().numpy())
                actor.reset(d)
    finally:
        env.close()
    return {"obs": np.stack(obs_rec), "dones": np.stack(done_rec)}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - export script
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--task", default="mixed")
    ap.add_argument("--rate", type=float, default=1000.0)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--check", default=None, help="an .npz of recorded observations (obs (T, B, D), dones (T, B))")
    ap.add_argument("--record", default=None, help="ENVSxSTEPS: record the teacher in Tactile-Residual (C3) first, "
                                                    "save OUT's .npz, check on it")
    ap.add_argument("--library", default=None)
    a = ap.parse_args(argv)
    info = export_teacher(a.checkpoint, a.out, a.task, a.rate, a.commit)
    print(json.dumps(info), flush=True)
    rec_path = a.check
    if a.record:
        n, t = (int(x) for x in a.record.lower().split("x"))
        rec = record_inputs(a.checkpoint, a.rate, n, t, a.library)
        rec_path = str(Path(a.out).with_suffix(".inputs.npz"))
        np.savez_compressed(rec_path, **rec)
    if rec_path:
        rec = np.load(rec_path)
        res = check_against_torch(a.out, a.checkpoint, rec["obs"], rec["dones"] if "dones" in rec else None)
        res["inputs_file"] = Path(rec_path).name
        print(json.dumps(res), flush=True)
        side = Path(a.out).with_suffix(".json")
        meta = json.loads(side.read_text())
        meta["parity"] = res
        side.write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
