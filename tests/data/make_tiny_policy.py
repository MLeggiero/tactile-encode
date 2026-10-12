"""Writes tests/data/tiny_policy.{onnx,json,npz}: a small random GRU residual policy exported the way trained
students are (obs, h_in) -> (action, h_out), and its torch outputs on a fixed input sequence. Run in the training
env (torch + onnx): PYTHONPATH=. ~/tactile-encode/.venv-rl/bin/python tests/data/make_tiny_policy.py"""

from pathlib import Path

import numpy as np
import torch

from tactile_sim.control.obs_spec import ACTION_DIM, ActionSpec, obs_dim, spec_hash
from tactile_sim.control.policy_io import PolicyMeta

OUT = Path(__file__).parent
D, H, A = obs_dim(1), 8, ACTION_DIM


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Linear(D, 16)
        self.gru = torch.nn.GRUCell(16, H)
        self.head = torch.nn.Linear(H, A)

    def forward(self, obs, h_in):
        h = self.gru(torch.tanh(self.enc(obs)), h_in)
        return torch.tanh(self.head(h)), h


torch.manual_seed(0)
net = Tiny().eval()
torch.onnx.export(net, (torch.zeros(1, D), torch.zeros(1, H)), str(OUT / "tiny_policy.onnx"),
                  input_names=["obs", "h_in"], output_names=["action", "h_out"], opset_version=17, dynamo=False)
PolicyMeta(spec_hash(ActionSpec(), 1), 1000.0, 1, H, "tiny").write(OUT / "tiny_policy.onnx")
rng = np.random.default_rng(0)
obs = rng.normal(size=(5, D)).astype(np.float32)
h = torch.zeros(1, H)
acts = []
with torch.no_grad():
    for o in obs:
        a, h = net(torch.from_numpy(o)[None], h)
        acts.append(a[0].numpy())
np.savez(OUT / "tiny_policy_io.npz", obs=obs, action=np.array(acts))
print("wrote", OUT / "tiny_policy.onnx")
