"""pi0.5 (openpi) on a Vega U + WUJI dataset (tactile_sim.vla.lerobot_export): policy transforms and training configs.

Standalone (it imports openpi, not tactile_sim). In an openpi checkout:

    cp tactile_sim/vla/finetune/openpi_vega_wuji.py <openpi>/src/openpi/policies/vega_wuji_policy.py
    # in <openpi>/src/openpi/training/config.py, at the end of the _CONFIGS list:
    #     *__import__("openpi.policies.vega_wuji_policy", fromlist=["_"]).train_configs(),
    ln -s <dataset> ~/.cache/huggingface/lerobot/local/vega_wuji_tools      # LeRobot finds local repos here
    uv run scripts/compute_norm_stats.py --config-name pi05_vega_wuji
    uv run python -c "from openpi.policies.vega_wuji_policy import widen_norm_stats as w; \
        w('assets/pi05_vega_wuji/local/vega_wuji_tools/norm_stats.json')"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_vega_wuji --exp-name=<run> --overwrite
    uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_vega_wuji --policy.dir=<checkpoint>

The dataset's state and action are 27-D (right arm 7, WUJI Hand 2 20), inside pi0.5's 32-D action space, so the
base checkpoint's projections load as they are (no re-initialized layers). Arm actions train as deltas from the
chunk's first state (DeltaActions on the first 7), hand targets stay absolute, as in the GR00T config. The head
camera is base_0_rgb; both wrist slots are masked. Full fine-tuning needs more than 70 GB (the 96 GB GPU); the
LoRA variant (pi05_vega_wuji_lora) needs about 23 GB.

Scripted demos hold some hand targets still (the abduction joints' keyframe), so their 1st-99th percentile range
is zero, and openpi's quantile normalization divides by it without clipping: widen_norm_stats gives every range at
least min_width. On the desktop torchcodec needs FFmpeg 4-7 libraries: LD_LIBRARY_PATH=~/scratch/envs/ffmpeg7/lib.

Served policy inputs (tactile_sim.vla.client.OpenPiPolicy sends these): {"observation/image": uint8 (H, W, 3),
"observation/state": float32 (27,), "prompt": str}; output {"actions": (H, 27)}.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from openpi import transforms
from openpi.models import model as _model

STATE_DIM = ACTION_DIM = 27
N_ARM = 7


@dataclasses.dataclass(frozen=True)
class VegaWujiInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        image = np.asarray(data["observation/image"])
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).astype(np.uint8)
        if image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)
        out = {
            "state": np.asarray(data["observation/state"], np.float32),
            "image": {"base_0_rgb": image, "left_wrist_0_rgb": np.zeros_like(image),
                      "right_wrist_0_rgb": np.zeros_like(image)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.False_, "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], np.float32)
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class VegaWujiOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}


def train_configs(repo_id: str = "local/vega_wuji_tools") -> list:
    from openpi.models import pi0_config
    from openpi.training import config as C
    from openpi.training import optimizer as _optimizer
    from openpi.training import weight_loaders

    @dataclasses.dataclass(frozen=True)
    class LeRobotVegaWujiDataConfig(C.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            repack = transforms.Group(inputs=[transforms.RepackTransform({
                "observation/image": "observation.images.ego_view",
                "observation/state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            })])
            mask = transforms.make_bool_mask(N_ARM, -(ACTION_DIM - N_ARM))
            data_transforms = transforms.Group(inputs=[VegaWujiInputs(model_type=model_config.model_type)],
                                               outputs=[VegaWujiOutputs()]).push(
                inputs=[transforms.DeltaActions(mask)], outputs=[transforms.AbsoluteActions(mask)])
            return dataclasses.replace(self.create_base_config(assets_dirs, model_config),
                                       repack_transforms=repack, data_transforms=data_transforms,
                                       model_transforms=C.ModelTransformFactory()(model_config),
                                       action_sequence_keys=("action",))

    data = LeRobotVegaWujiDataConfig(repo_id=repo_id, base_config=C.DataConfig(prompt_from_task=True))
    base = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")
    full = C.TrainConfig(
        name="pi05_vega_wuji",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=16, discrete_state_input=False),
        data=data, batch_size=32, num_train_steps=30_000,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000,
                                                   decay_lr=5e-6),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0), ema_decay=0.999, weight_loader=base)
    lora_model = pi0_config.Pi0Config(pi05=True, action_horizon=16, discrete_state_input=False,
                                      paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    lora = dataclasses.replace(full, name="pi05_vega_wuji_lora", model=lora_model, ema_decay=None,
                               freeze_filter=lora_model.get_freeze_filter())
    # the pipeline end to end on a tiny random model (fits any GPU): data, transforms, norm stats, training steps
    debug = dataclasses.replace(full, name="pi05_vega_wuji_debug", batch_size=2, num_train_steps=10,
                                model=pi0_config.Pi0Config(pi05=True, action_horizon=16, discrete_state_input=False,
                                                           paligemma_variant="dummy", action_expert_variant="dummy"),
                                weight_loader=weight_loaders.NoOpWeightLoader(), ema_decay=None, overwrite=True,
                                exp_name="debug", wandb_enabled=False,
                                data=dataclasses.replace(data, assets=C.AssetsConfig(
                                    assets_dir="./assets/pi05_vega_wuji", asset_id=repo_id)))
    return [full, lora, debug]


def widen_norm_stats(path: str, min_width: float = 0.02) -> None:
    """Give every state/action dimension a 1st-99th percentile range of at least min_width (rad), centred where it
    was, and a standard deviation of at least a quarter of that, in an openpi norm_stats.json."""
    import json

    with open(path) as f:
        stats = json.load(f)
    for v in stats["norm_stats"].values():
        lo, hi = np.asarray(v["q01"], float), np.asarray(v["q99"], float)
        mid, half = 0.5 * (lo + hi), 0.5 * np.maximum(hi - lo, min_width)
        v["q01"], v["q99"] = (mid - half).tolist(), (mid + half).tolist()
        v["std"] = np.maximum(np.asarray(v["std"], float), 0.25 * min_width).tolist()
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
