#!/usr/bin/env bash
# GR00T N1.7 fine-tune on the Vega U + WUJI sim demos, on GT's Skynet (Slurm), reusing the DexVLA project's GR00T
# runtime and base weights (~/scratch/DexVLA/meta: .runtime/gr00t17 = torch 2.9 cu128 + flash-attn, the pixi
# "data" env for FFmpeg, data/models/GR00T-N1.7-3B, the Cosmos-Reason2-2B cache) read-only. This project's files
# live under ~/scratch/DexVLA/tactile. Overcap is preemptible: the job requeues and resumes from its last
# checkpoint (gr00t_finetune.py).
#
#   sbatch --export=ALL,PROFILE=full,MAX_STEPS=20000 ~/scratch/DexVLA/tactile/finetune/skynet_gr00t.sbatch
#   sbatch --export=ALL,PROFILE=full,MAX_STEPS=20,RUN=smoke --time=00:30:00 .../skynet_gr00t.sbatch   # smoke
#SBATCH --job-name=tactile-gr00t
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/nethome/%u/scratch/DexVLA/tactile/logs/%x-%j.log
set -euo pipefail
meta="$HOME/scratch/DexVLA/meta"
work="$HOME/scratch/DexVLA/tactile"
run="$work/runs/gr00t_${RUN:-${PROFILE:-full}}"
mkdir -p "$run"
nvidia-smi > "$run/gpus-$SLURM_JOB_ID.txt"
export PATH="$meta/.runtime/gr00t17/bin:$meta/.pixi/envs/data/bin:$PATH"
export LD_LIBRARY_PATH="$meta/.pixi/envs/data/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_CACHE="$meta/data/hf-cache/hub" HF_HUB_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DS_BUILD_OPS=0 TRITON_CACHE_DIR="$work/.runtime/triton-cache"
cd "$meta/repos/Isaac-GR00T"
ds="$work/data/${DATASET:-vega_wuji_tools}"
# GR00T's own statistics (relative actions included), by this GR00T version, once per dataset
[[ -f "$ds/meta/relative_stats.json" && -f "$ds/meta/.stats_by_cluster_gr00t" ]] || {
    python gr00t/data/stats.py --dataset-path "$ds" --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "$work/finetune/gr00t_vega_wuji_config.py" && touch "$ds/meta/.stats_by_cluster_gr00t"; }
python "$work/finetune/gr00t_finetune.py" --dataset-path "$ds" \
    --output-dir "$run" --profile "${PROFILE:-full}" --max-steps "${MAX_STEPS:-20000}" \
    --base-model-path "$meta/data/models/GR00T-N1.7-3B" --save-steps "${SAVE_STEPS:-1000}" --workers 4
