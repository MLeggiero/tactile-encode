"""PPO settings per policy rate, from the plan's "PPO per rate" table: horizons fixed in seconds (tau_h 0.5 s for the
hammer, 1.0 s for the drill, 0.3 s for the load generator; tau_lambda 0.15 s), rollouts of 0.128 s, initial action
noise proportional to sqrt(rate) (0.30 at 1 kHz).

`teacher_runner_cfg` is the plan's teacher: privileged PPO with an asymmetric critic. The actor reads the `teacher`
group (normalized) through a GRU of 256 units and a (256, 128) MLP head whose output layer is zero-initialized
(tactile_rl.algos.models.ZeroInitRNNModel); the critic is an MLP (512, 256, 128) on the same privileged observation
(the plan: the critic sees what the teacher sees), a separate network without memory. Both normalize with
tactile_rl.algos.models.RobustNormalization (a floor on the scale, normalized values clipped to +-5). 4096 envs,
learning rate 3e-4 adaptive with a desired KL of 0.005 (the plan's; under the load generator it holds the rate at
1e-5-1e-4, and 0.02 lets it settle near 3e-4: `desired_kl`), clip 0.2, 5 epochs over 8 mini-batches (512 envs'
128-step sequences each: the GRU's backpropagation through time fits the 16 GB card), the entropy coefficient 1e-3
decaying linearly to 0 (tactile_rl.algos.ppo.ScheduledPPO). `ppo_runner_cfg` is milestone 1's MLP setting for the
hold tasks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

TAU_H = {"hammer": 0.5, "drill": 1.0, "loadgen": 0.3, "hold": 0.5}  # s, discount horizons
TAU_LAMBDA = 0.15  # s
ROLLOUT_S = 0.128
SIGMA_1KHZ = 0.30  # initial action noise before tanh at 1 kHz (scales with sqrt(rate))


@dataclass
class ScheduledPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """RslRlPpoAlgorithmCfg with the entropy schedule (ScheduledPPO)."""

    entropy_coef_final: float = 0.0
    entropy_decay_iterations: int = 1000
    class_name: str = "tactile_rl.algos.ppo:ScheduledPPO"


def discount(rate: float, task: str = "hammer") -> tuple[float, float]:
    """(gamma, lambda) for a policy at `rate`: e^(-dt / tau_h), e^(-dt / tau_lambda)."""
    dt = 1.0 / rate
    return math.exp(-dt / TAU_H[task]), math.exp(-dt / TAU_LAMBDA)


def ppo_runner_cfg(rate: float, experiment: str = "tactile_hold",
                   obs_groups: dict[str, tuple[str, ...]] | None = None, task: str = "hold") -> RslRlOnPolicyRunnerCfg:
    gamma, lam = discount(rate, task)
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(256, 128), activation="elu", obs_normalization=True,
                            distribution_cfg={"class_name": "GaussianDistribution",
                                              "init_std": SIGMA_1KHZ * math.sqrt(rate / 1000.0), "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(learning_rate=3e-4, schedule="adaptive", desired_kl=0.005, clip_param=0.2,
                                       num_learning_epochs=5, num_mini_batches=4, entropy_coef=1e-3,
                                       gamma=gamma, lam=lam),
        num_steps_per_env=max(1, round(ROLLOUT_S * rate)),
        max_iterations=1000,
        save_interval=100,
        experiment_name=f"{experiment}_{rate:g}hz",
        logger="tensorboard",
        upload_model=False,
        obs_groups=obs_groups or {"actor": ("actor",), "critic": ("critic",)},
    )


def teacher_runner_cfg(rate: float, task: str, max_iterations: int = 2000, entropy_decay_iterations: int | None = None,
                       desired_kl: float = 0.005) -> RslRlOnPolicyRunnerCfg:
    """The plan's teacher for `task` ("hammer", "drill", "loadgen") at `rate` (see module doc)."""
    gamma, lam = discount(rate, task)
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(256, 128), activation="elu", obs_normalization=True,
                            rnn_type="gru", rnn_hidden_dim=256, rnn_num_layers=1,
                            class_name="tactile_rl.algos.models:ZeroInitRNNModel",
                            distribution_cfg={"class_name": "GaussianDistribution",
                                              "init_std": SIGMA_1KHZ * math.sqrt(rate / 1000.0), "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                             class_name="tactile_rl.algos.models:RobustMLPModel"),
        algorithm=ScheduledPpoAlgorithmCfg(learning_rate=3e-4, schedule="adaptive", desired_kl=desired_kl,
                                           clip_param=0.2, num_learning_epochs=5, num_mini_batches=8, entropy_coef=1e-3,
                                           entropy_coef_final=0.0,
                                           entropy_decay_iterations=entropy_decay_iterations or max_iterations,
                                           gamma=gamma, lam=lam),
        num_steps_per_env=max(1, round(ROLLOUT_S * rate)),
        max_iterations=max_iterations,
        save_interval=50,
        experiment_name=f"tactile_{task}_teacher_{rate:g}hz",
        logger="tensorboard",
        upload_model=False,
        obs_groups={"actor": ("teacher",), "critic": ("teacher",)},
    )
