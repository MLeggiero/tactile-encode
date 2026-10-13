"""Writes tests/data/tiny_policy{,_v3}.{onnx,json} and tiny_policy{,_v3}_io.npz: a small random GRU residual policy
exported the way trained students are (obs, h_in) -> (action, h_out), and its torch outputs on a fixed input sequence,
for observation layout 2 (tiny_policy) and 3 (tiny_policy_v3). Run in the training env (torch + onnx):
PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python tests/data/make_tiny_policy.py [2 3]"""

import sys
from pathlib import Path

import numpy as np
import torch

from tactile_sim.control.obs_spec import ACTION_DIM, ActionSpec, obs_dim, spec_hash
from tactile_sim.control.policy_io import PolicyMeta

OUT = Path(__file__).parent
H, A = 8, ACTION_DIM


class Tiny(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.enc = torch.nn.Linear(d, 16)
        self.gru = torch.nn.GRUCell(16, H)
        self.head = torch.nn.Linear(H, A)

    def forward(self, obs, h_in):
        h = self.gru(torch.tanh(self.enc(obs)), h_in)
        return torch.tanh(self.head(h)), h


def write(version: int) -> None:
    stem = "tiny_policy" if version == 2 else f"tiny_policy_v{version}"
    d = obs_dim(1, version)
    torch.manual_seed(0)
    net = Tiny(d).eval()
    torch.onnx.export(net, (torch.zeros(1, d), torch.zeros(1, H)), str(OUT / f"{stem}.onnx"),
                      input_names=["obs", "h_in"], output_names=["action", "h_out"], opset_version=17, dynamo=False)
    PolicyMeta(spec_hash(ActionSpec(), 1, version), 1000.0, 1, H, "tiny",
               extra={"obs_version": version}).write(OUT / f"{stem}.onnx")
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(5, d)).astype(np.float32)
    h = torch.zeros(1, H)
    acts = []
    with torch.no_grad():
        for o in obs:
            a, h = net(torch.from_numpy(o)[None], h)
            acts.append(a[0].numpy())
    np.savez(OUT / f"{stem}_io.npz", obs=obs, action=np.array(acts))
    print("wrote", OUT / f"{stem}.onnx")


if __name__ == "__main__":
    for v in (int(x) for x in (sys.argv[1:] or ["3"])):
        write(v)
