#!/usr/bin/env bash
# pi0.5 full fine-tune on the Vega U + WUJI sim demos on GT's Skynet: 4x A40 with FSDP (the full fine-tune needs
# more than one 48 GB GPU), overcap with requeue; openpi resumes from its last checkpoint. The base checkpoint
# downloads from openpi's public bucket into ~/scratch/DexVLA/tactile/openpi-cache on the first run.
#
#   sbatch ~/scratch/DexVLA/tactile/finetune/skynet_pi05.sbatch
#   sbatch --export=ALL,CONFIG=pi05_vega_wuji_lora --gres=gpu:a40:1 --cpus-per-task=6 .../skynet_pi05.sbatch   # LoRA
#   sbatch --export=ALL,EXP=smoke,STEPS=20 --time=01:00:00 .../skynet_pi05.sbatch                              # smoke
# DATASET picks the dataset under ../data (default vega_wuji_tools); its norm stats are recomputed when it changes.
#SBATCH --job-name=tactile-pi05
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:4
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/nethome/%u/scratch/DexVLA/tactile/logs/%x-%j.log
set -euo pipefail
work="$HOME/scratch/DexVLA/tactile"
meta="$HOME/scratch/DexVLA/meta"
config="${CONFIG:-pi05_vega_wuji}"
export OPENPI_DATA_HOME="$work/openpi-cache" WANDB_MODE=disabled
export LD_LIBRARY_PATH="$meta/.pixi/envs/data/lib:${LD_LIBRARY_PATH:-}"  # FFmpeg for torchcodec
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
src="$(readlink -f "$work/data/${DATASET:-vega_wuji_tools}")"
# LeRobot opens and seeks a video for every sample. From the shared /coc storage (with GR00T runs reading alongside)
# that took ~0.7 s a sample, 22 s a step; the dataset (a few hundred MB) goes on the node's own disk instead
export HF_LEROBOT_HOME="${TMPDIR:-/tmp}/tactile-lerobot-$SLURM_JOB_ID"
trap 'rm -rf "$HF_LEROBOT_HOME"' EXIT
mkdir -p "$HF_LEROBOT_HOME/local"
cp -r "$src" "$HF_LEROBOT_HOME/local/vega_wuji_tools"
cd "$work/openpi"
nvidia-smi > "$work/logs/gpus-$SLURM_JOB_ID.txt"
stats="assets/$config/local/vega_wuji_tools/norm_stats.json"
if [[ ! -f "$stats" || "$(cat "$stats.src" 2>/dev/null)" != "$src" ]]; then
    mkdir -p "$(dirname "$stats")"  # from the parquet files (openpi's own pass decodes every video frame: a day)
    .venv/bin/python "$work/finetune/openpi_norm_stats.py" "$src" "$stats"
    echo "$src" > "$stats.src"
fi
exp="${EXP:-vega_wuji}"
if [[ -d "checkpoints/$config/$exp" ]]; then flag=--resume; else flag=--overwrite; fi
.venv/bin/python scripts/train.py "$config" --exp-name="$exp" "$flag" --fsdp-devices="${SLURM_GPUS_ON_NODE:-4}" \
    --num-workers="${WORKERS:-12}" ${STEPS:+--num-train-steps="$STEPS" --save-interval="$STEPS"}
