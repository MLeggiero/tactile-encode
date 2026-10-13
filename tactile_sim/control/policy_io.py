"""Exported residual policies: an ONNX graph plus a sidecar JSON (<name>.onnx, <name>.json).

The graph maps (obs [1, D], h_in [1, H]) to (action [1, 40], h_out [1, H]): the action already squashed into
[-1, 1], the recurrent state carried by the caller between ticks (a state that turns non-finite restarts from zero:
the caller, ResidualHandL2, counts it). A feedforward policy has no h_in/h_out. The
sidecar records the interface it was trained on (obs_spec.spec_hash with its stack depth), its rate, hidden size
and provenance; a policy only runs where its hash and rate match (ResidualHandL2 checks).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PolicyMeta:
    spec_hash: str
    rate: float
    stack: int
    hidden: int = 0
    name: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def read(cls, onnx_path: str | Path) -> PolicyMeta:
        d = json.loads(Path(onnx_path).with_suffix(".json").read_text())
        known = {k: d.pop(k) for k in ("spec_hash", "rate", "stack", "hidden", "name") if k in d}
        return cls(**known, extra=d)

    def write(self, onnx_path: str | Path) -> None:
        d = {"spec_hash": self.spec_hash, "rate": self.rate, "stack": self.stack, "hidden": self.hidden,
             "name": self.name, **self.extra}
        Path(onnx_path).with_suffix(".json").write_text(json.dumps(d, indent=1))


class OnnxPolicy:
    """Runs an exported policy on the CPU, one thread, carrying its recurrent state."""

    def __init__(self, path: str | Path, threads: int = 1):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.meta = PolicyMeta.read(path)
        names = {i.name for i in self.sess.get_inputs()}
        self.recurrent = "h_in" in names
        if self.recurrent != (self.meta.hidden > 0):
            raise ValueError(f"{path}: graph and sidecar disagree on the recurrent state")
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, self.meta.hidden), dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        feed = {"obs": np.asarray(obs, dtype=np.float32)[None]}
        if self.recurrent:
            feed["h_in"] = self.h
            a, self.h = self.sess.run(["action", "h_out"], feed)
        else:
            (a,) = self.sess.run(["action"], feed)
        return np.asarray(a[0], dtype=float)
