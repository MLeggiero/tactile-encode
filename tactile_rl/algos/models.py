"""The teacher's actor: rsl_rl's RNNModel (normalized observations -> GRU -> MLP head) with its output layer
zero-initialized, so the policy's mean is 0 at the start of training and a = 0 reproduces the base exactly."""

from __future__ import annotations

import torch
from rsl_rl.models.rnn_model import RNNModel


def zero_last_linear(module: torch.nn.Module) -> torch.nn.Linear:
    last = [m for m in module.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.weight.zero_()
        if last.bias is not None:
            last.bias.zero_()
    return last


class ZeroInitRNNModel(RNNModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        zero_last_linear(self.mlp)
