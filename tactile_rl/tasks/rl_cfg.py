"""PPO settings per policy rate, from the plan's "PPO per rate" table: horizons fixed in seconds (tau_h 0.5 s for the
hammer, tau_lambda 0.15 s), rollouts of 0.128 s, initial action noise proportional to sqrt(rate)."""

from __future__ import annotations

import math

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

TAU_H = 0.5  # s, discount horizon (hammer)
TAU_LAMBDA = 0.15  # s
ROLLOUT_S = 0.128


def ppo_runner_cfg(rate: float, experiment: str = "tactile_hold",
                   obs_groups: dict[str, tuple[str, ...]] | None = None) -> RslRlOnPolicyRunnerCfg:
    dt = 1.0 / rate
    # Milestone 1 uses MLPs: the plan's teacher (GRU 256 + MLP head, zero-initialized output layer) needs the full
    # observation of milestone 2
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(256, 128), activation="elu", obs_normalization=True,
                            distribution_cfg={"class_name": "GaussianDistribution",
                                              "init_std": 0.30 * math.sqrt(rate / 1000.0), "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(learning_rate=3e-4, schedule="adaptive", desired_kl=0.005, clip_param=0.2,
                                       num_learning_epochs=5, num_mini_batches=4, entropy_coef=1e-3,
                                       gamma=math.exp(-dt / TAU_H), lam=math.exp(-dt / TAU_LAMBDA)),
        num_steps_per_env=max(1, round(ROLLOUT_S * rate)),
        max_iterations=1000,
        save_interval=100,
        experiment_name=f"{experiment}_{rate:g}hz",
        logger="tensorboard",
        upload_model=False,
        obs_groups=obs_groups or {"actor": ("actor",), "critic": ("critic",)},
    )
