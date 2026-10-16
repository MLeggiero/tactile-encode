"""The teacher's networks on top of rsl_rl's models.

- `ZeroInitRNNModel`: rsl_rl's RNNModel (normalized observations -> GRU -> MLP head) with its output layer
  zero-initialized, so the policy's mean is 0 at the start of training and a = 0 reproduces the base exactly.
- `RobustNormalization`: the observation normalizer both networks use. rsl_rl's EmpiricalNormalization divides by
  std + 0.01, so a feature that never moved (the load generator's wrench through the hold stage C0, an untouched
  patch's taxels) is scaled up 100-fold when it first moves, and its running variance (a count-weighted average over
  millions of samples) takes hundreds of iterations to catch up: the policy's outputs then swing between rollout and
  update and the adaptive learning rate sits at its 1e-5 floor. Here the scale is at least `std_floor` (the inputs
  are pre-scaled to O(1)) and the normalized values are clipped to +-`clip`.
"""

from __future__ import annotations

import torch
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.models.rnn_model import RNNModel
from rsl_rl.modules import EmpiricalNormalization


class RobustNormalization(EmpiricalNormalization):
    def __init__(self, shape, eps: float = 1e-2, until: int | None = None, std_floor: float = 0.1,
                 clip: float = 5.0):
        super().__init__(shape, eps, until)
        self.std_floor = float(std_floor)
        self.clip = float(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (x - self._mean) / torch.clamp(self._std, min=self.std_floor)
        return torch.clamp(y, -self.clip, self.clip)


def zero_last_linear(module: torch.nn.Module) -> torch.nn.Linear:
    last = [m for m in module.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.weight.zero_()
        if last.bias is not None:
            last.bias.zero_()
    return last


def _robust(model: MLPModel) -> None:
    if model.obs_normalization:
        model.obs_normalizer = RobustNormalization(model.obs_dim)


class ZeroInitRNNModel(RNNModel):
    """The teacher's actor (see module doc)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _robust(self)
        zero_last_linear(self.mlp)


class RobustMLPModel(MLPModel):
    """The critic: rsl_rl's MLPModel with the robust normalizer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _robust(self)
