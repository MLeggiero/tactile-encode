"""Fine-tune GR00T N1.7 on a Vega U + WUJI dataset (tactile_sim.vla.lerobot_export), with memory profiles.

Runs inside Isaac-GR00T's environment (it imports gr00t, not tactile_sim). It is GR00T's own
gr00t/experiment/launch_finetune.py with the dataset, embodiment and modality config filled in and a profile:

- "16gb" (an RTX 5070 Ti): the diffusion transformer frozen, only the new embodiment's state/action encoders and
  decoder trained (about 0.33 B parameters); frozen weights in bf16, gradient checkpointing, micro-batches of 16
  with gradient accumulation. A first model, to test the pipeline and the closed loop; it adapts less than
- "full" (the user's 96 GB GPU, overnight): GR00T's default post-training recipe, projector and diffusion
  transformer trained (1.6 B parameters, about 35 GB at batch 32).

Before training, GR00T needs its relative-action statistics:

    python gr00t/data/stats.py --dataset-path <ds> --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path tactile_sim/vla/finetune/gr00t_vega_wuji_config.py

torchcodec (GR00T's video decoder) needs FFmpeg 4-7 shared libraries; on the desktop they come from a separate
conda env: LD_LIBRARY_PATH=~/scratch/envs/ffmpeg7/lib.

    cd ~/Isaac-GR00T && LD_LIBRARY_PATH=~/scratch/envs/ffmpeg7/lib python \
        <repo>/tactile_sim/vla/finetune/gr00t_finetune.py --dataset-path <ds> --output-dir <out> --profile 16gb
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILES = {
    "16gb": {"tune_diffusion_model": False, "load_bf16": True, "gradient_checkpointing": True, "micro_batch": 16,
             "grad_accum": 4, "max_steps": 20_000},
    "full": {"tune_diffusion_model": True, "load_bf16": False, "gradient_checkpointing": False, "micro_batch": 32,
             "grad_accum": 1, "max_steps": 20_000},
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="16gb")
    ap.add_argument("--base-model-path", default="nvidia/GR00T-N1.7-3B")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--save-steps", type=int, default=2000)
    ap.add_argument("--state-dropout", type=float, default=0.05,
                    help="lower than GR00T's 0.2: a hammer swing depends on where the arm is")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tune-visual", action="store_true", help="also tune the vision encoder (more memory)")
    a = ap.parse_args(argv)
    p = dict(PROFILES[a.profile])
    for k in ("max_steps", "micro_batch", "grad_accum"):
        if getattr(a, k) is not None:
            p[k] = getattr(a, k)

    os.environ.setdefault("LOGURU_LEVEL", "INFO")
    sys.path.append(str(HERE))
    importlib.import_module("gr00t_vega_wuji_config")  # registers the NEW_EMBODIMENT modality config
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.experiment import run

    tag = EmbodimentTag.resolve("NEW_EMBODIMENT").value
    config = get_default_config().load_dict({"data": {"download_cache": False, "datasets": [
        {"dataset_paths": [a.dataset_path], "mix_ratio": 1.0, "embodiment_tag": tag}]}})
    config.load_config_path = None
    m = config.model
    m.tune_llm = False
    m.tune_visual = a.tune_visual
    m.tune_projector = True
    m.tune_diffusion_model = p["tune_diffusion_model"]
    m.state_dropout_prob = a.state_dropout
    m.random_rotation_angle = None
    m.color_jitter_params = None
    m.extra_augmentation_config = None
    m.load_bf16 = p["load_bf16"]
    m.reproject_vision = False
    m.model_name = "nvidia/Cosmos-Reason2-2B"
    m.backbone_trainable_params_fp32 = True
    m.use_relative_action = True
    t = config.training
    t.start_from_checkpoint = a.base_model_path
    t.optim = "adamw_torch"
    t.global_batch_size = p["micro_batch"]  # GR00T's "global" batch is the per-step batch on one GPU
    t.gradient_accumulation_steps = p["grad_accum"]
    t.gradient_checkpointing = p["gradient_checkpointing"]
    t.dataloader_num_workers = a.workers
    t.learning_rate = a.learning_rate
    t.output_dir = a.output_dir
    t.save_steps = a.save_steps
    t.save_total_limit = 3
    t.num_gpus = 1
    t.use_wandb = False
    t.max_steps = p["max_steps"]
    t.warmup_ratio = 0.05
    t.save_only_model = False
    t.experiment_name = Path(a.output_dir).name
    print(f"GR00T N1.7 fine-tune, profile {a.profile}: {p}", flush=True)
    run(config)


if __name__ == "__main__":
    main()
