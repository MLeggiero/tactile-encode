"""Fine-tune GR00T N1.7 on a Vega U + WUJI dataset (tactile_sim.vla.lerobot_export), with memory profiles.

Runs inside Isaac-GR00T's environment (it imports gr00t, not tactile_sim). It is GR00T's own
gr00t/experiment/launch_finetune.py with the dataset, embodiment and modality config filled in and a profile:

- "16gb" (an RTX 5070 Ti with nothing else on it): the diffusion transformer and the VL self-attention frozen,
  only the new embodiment's state/action encoders and decoder trained (about 0.33 B parameters); frozen weights in
  bf16, gradient checkpointing, micro-batches of 4 with 16-step gradient accumulation. It does not fit next to
  other GPU work (RL training, the demos' renderers). A first model, to test the pipeline and the closed loop; it
  adapts less than
- "full" (the user's 96 GB GPU, overnight): GR00T's default post-training recipe, projector and diffusion
  transformer trained (1.6 B parameters, about 35 GB at batch 32).

Before training, GR00T needs its relative-action statistics:

    python gr00t/data/stats.py --dataset-path <ds> --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path tactile_sim/vla/finetune/gr00t_vega_wuji_config.py

On the desktop two libraries GR00T's environment expects come from separate conda envs, leaving the user's
`gr00t` env as it is: torchcodec (the video decoder) needs FFmpeg 4-7 shared libraries
(LD_LIBRARY_PATH=~/scratch/envs/ffmpeg7/lib), and the installed deepspeed, which transformers imports even on one
GPU, needs an nvcc to read the CUDA version (CUDA_HOME=~/scratch/envs/cuda128, conda-forge cuda-nvcc 12.8).

    cd ~/Isaac-GR00T && CUDA_HOME=~/scratch/envs/cuda128 LD_LIBRARY_PATH=~/scratch/envs/ffmpeg7/lib python \
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
    "16gb": {"tune_diffusion_model": False, "tune_vlln": False, "load_bf16": True, "frozen_bf16": True,
             "gradient_checkpointing": True, "micro_batch": 4, "grad_accum": 16, "max_steps": 20_000},
    "full": {"tune_diffusion_model": True, "tune_vlln": True, "load_bf16": False, "frozen_bf16": False,
             "gradient_checkpointing": False, "micro_batch": 32, "grad_accum": 1, "max_steps": 20_000},
}


def store_frozen_action_head_in_bf16() -> None:
    """Keep the action head's frozen modules (the diffusion transformer, the VL self-attention) in bf16 instead of
    GR00T's fp32: about 2.6 GB less on the GPU. Training runs under bf16 autocast, so their compute is unchanged."""
    import torch
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    original = Gr00tN1d7Pipeline.return_model

    def return_model(self):
        model = original(self)
        head = model.action_head
        for name in ("model", "vl_self_attention"):
            mod = getattr(head, name, None)
            if mod is not None and not any(p.requires_grad for p in mod.parameters()):
                mod.to(torch.bfloat16)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"frozen action-head modules in bf16; {n_train / 1e6:.0f} M trainable parameters", flush=True)
        return model

    Gr00tN1d7Pipeline.return_model = return_model


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="16gb")
    ap.add_argument("--base-model-path", default="nvidia/GR00T-N1.7-3B")
    ap.add_argument("--backbone", default="nvidia/Cosmos-Reason2-2B",
                    help="the VLM: its hub id, or a local copy under a path ending in nvidia/Cosmos-Reason2-2B (GR00T "
                         "tells backbones apart by that name). Offline, use a local copy: given a hub id, "
                         "transformers asks the hub about the tokenizer even when it is cached")
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
    m.tune_vlln = p["tune_vlln"]
    m.state_dropout_prob = a.state_dropout
    m.random_rotation_angle = None
    m.color_jitter_params = None
    m.extra_augmentation_config = None
    m.load_bf16 = p["load_bf16"]
    m.reproject_vision = False
    m.model_name = a.backbone
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
    t.experiment_name = None  # checkpoints straight under output_dir
    # a requeued (preempted) job picks up from its last checkpoint; newer GR00T makes resuming a config field
    resume = any(Path(a.output_dir).glob("checkpoint-*"))
    if hasattr(t, "resume_from_checkpoint"):
        t.resume_from_checkpoint = resume
    if p["frozen_bf16"]:
        store_frozen_action_head_in_bf16()
    print(f"GR00T N1.7 fine-tune, profile {a.profile}: {p}; resuming: {resume}", flush=True)
    run(config)


if __name__ == "__main__":
    main()
