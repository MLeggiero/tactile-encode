"""rsl_rl's PPO with the plan's entropy schedule: the entropy coefficient goes linearly from `entropy_coef` to
`entropy_coef_final` over `entropy_decay_iterations` updates, then stays."""

from __future__ import annotations

from rsl_rl.algorithms.ppo import PPO


class ScheduledPPO(PPO):
    def __init__(self, *args, entropy_coef_final: float = 0.0, entropy_decay_iterations: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_coef_start = float(self.entropy_coef)
        self.entropy_coef_final = float(entropy_coef_final)
        self.entropy_decay_iterations = max(1, int(entropy_decay_iterations))
        self.n_updates = 0

    def entropy_at(self, n: int) -> float:
        f = min(1.0, n / self.entropy_decay_iterations)
        return self.entropy_coef_start + f * (self.entropy_coef_final - self.entropy_coef_start)

    def update(self) -> dict[str, float]:
        self.entropy_coef = self.entropy_at(self.n_updates)
        out = super().update()
        self.n_updates += 1
        out = dict(out)
        out["entropy_coef"] = self.entropy_coef
        return out
