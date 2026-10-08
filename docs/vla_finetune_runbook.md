# VLA fine-tune runbook (the 96 GB GPU)

The bake-off between GR00T N1.7 and π0.5 (`docs/vla_survey.md`) on the sim demos of the Vega U + WUJI Hand 2
(`docs/vla_in_sim.md`). Both full recipes need more than the desktop's 16 GB; this is what to run on the 96 GB
GPU, overnight, and what to bring back.

## 1. Get the dataset and the repo

Copy from the desktop (`desktop` below is its ssh host alias): the LeRobot dataset, and the repo at `feat/residual-l2` (only
`tactile_sim/vla/finetune/` is needed on the GPU machine).

```bash
rsync -a desktop:tactile-encode/runs/lerobot/vega_wuji_tools/ ~/data/vega_wuji_tools/
```

```bash
rsync -a desktop:tactile-encode/tactile_sim/vla/finetune/ ~/vega_wuji_finetune/
```

(`vega_wuji_tools` holds both tasks, "hammer the nail into the board" and "drive the screw into the board"; the
hammer-only `vega_wuji_hammer` works the same way.)

## 2. GR00T N1.7 (full recipe: projector + diffusion transformer, batch 32, 20k steps)

Set up Isaac-GR00T per its README (`uv` or its conda instructions; Python 3.10, torch 2.7 cu128). Accept the
`nvidia/Cosmos-Reason2-2B` license on Hugging Face with the account you log in with. If torchcodec cannot find
FFmpeg 4-7 libraries, or deepspeed asks for `CUDA_HOME`, see the desktop workarounds in `docs/vla_in_sim.md`.

```bash
cd Isaac-GR00T && python gr00t/data/stats.py --dataset-path ~/data/vega_wuji_tools --embodiment-tag NEW_EMBODIMENT --modality-config-path ~/vega_wuji_finetune/gr00t_vega_wuji_config.py
```

```bash
cd Isaac-GR00T && python ~/vega_wuji_finetune/gr00t_finetune.py --dataset-path ~/data/vega_wuji_tools --output-dir ~/runs/gr00t_vega_wuji --profile full
```

About 35 GB of GPU memory. Checkpoints every 2000 steps; the last one is what to bring back.

## 3. π0.5 (full fine-tune, batch 32, 30k steps)

```bash
git clone https://github.com/Physical-Intelligence/openpi.git && cd openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Then follow the header of `~/vega_wuji_finetune/openpi_vega_wuji.py`: copy it to
`src/openpi/policies/vega_wuji_policy.py`, add its line to `_CONFIGS` in `src/openpi/training/config.py`, and link
the dataset where LeRobot looks for local datasets (use the repo id `local/vega_wuji_tools`, or pass
`repo_id="local/vega_wuji_tools"` to `train_configs`):

```bash
mkdir -p ~/.cache/huggingface/lerobot/local && ln -sfn ~/data/vega_wuji_tools ~/.cache/huggingface/lerobot/local/vega_wuji_tools
```

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_vega_wuji
```

```bash
uv run python -c "from openpi.policies.vega_wuji_policy import widen_norm_stats as w; w('assets/pi05_vega_wuji/local/vega_wuji_tools/norm_stats.json')"
```

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_vega_wuji --exp-name=vega_wuji --overwrite
```

More than 70 GB. On an RTX PRO 6000 (Blackwell) use one GPU (JAX multi-GPU collectives hang there). The tiny
`pi05_vega_wuji_debug` config runs the same pipeline in a few minutes if anything needs checking first.

## 4. Bring back, and the closed-loop runs (desktop)

Copy the final checkpoints to the desktop (`~/runs/vla/`). Serve one model at a time on the 5070 Ti (GR00T:
`gr00t/eval/run_gr00t_server.py --model-path <ckpt> --embodiment-tag NEW_EMBODIMENT --modality-config-path
.../gr00t_vega_wuji_config.py --port 5555`; π0.5: `scripts/serve_policy.py policy:checkpoint
--policy.config=pi05_vega_wuji --policy.dir=<ckpt>`), then run E2 from the repo:

```bash
python -m tactile_sim.experiments.vla_experiment run --task hammer --server groot://localhost:5555 --seeds 100 --workers 6 --out runs/e2/groot_hammer
```

The same with `--task drill`, and with `--server openpi://localhost:8000` for π0.5. The bake-off compares the two
on `done`, slip and drops; pick GR00T if it beats π0.5 by about ten points or more.
