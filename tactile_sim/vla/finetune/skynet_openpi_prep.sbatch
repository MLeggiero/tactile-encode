#!/usr/bin/env bash
# pi0.5's inputs on GT's Skynet, as a CPU job before the GPU one (skynet_pi05.sbatch): the base checkpoint from
# openpi's public bucket into ~/scratch/DexVLA/tactile/openpi-cache, and the dataset's norm stats (widened, keyed to
# the dataset as the GPU job keys them), so four A40s don't wait on a download or a pass over the videos.
#
#   sbatch --export=ALL,DATASET=vega_wuji_tools_v2 ~/scratch/DexVLA/tactile/finetune/skynet_openpi_prep.sbatch
#SBATCH --job-name=tactile-openpi-prep
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/nethome/%u/scratch/DexVLA/tactile/logs/%x-%j.log
set -euo pipefail
work="$HOME/scratch/DexVLA/tactile"
meta="$HOME/scratch/DexVLA/meta"
config="${CONFIG:-pi05_vega_wuji}"
export HF_LEROBOT_HOME="$work/lerobot" OPENPI_DATA_HOME="$work/openpi-cache" JAX_PLATFORMS=cpu
export LD_LIBRARY_PATH="$meta/.pixi/envs/data/lib:${LD_LIBRARY_PATH:-}"  # FFmpeg for torchcodec
mkdir -p "$HF_LEROBOT_HOME/local"
ln -sfn "$work/data/${DATASET:-vega_wuji_tools}" "$HF_LEROBOT_HOME/local/vega_wuji_tools"
cd "$work/openpi"
.venv/bin/python -c "
from openpi.shared import download
print('base checkpoint:', download.maybe_download('gs://openpi-assets/checkpoints/pi05_base/params'))"
stats="assets/$config/local/vega_wuji_tools/norm_stats.json"
src="$(readlink -f "$HF_LEROBOT_HOME/local/vega_wuji_tools")"
if [[ ! -f "$stats" || "$(cat "$stats.src" 2>/dev/null)" != "$src" ]]; then
    .venv/bin/python scripts/compute_norm_stats.py --config-name "$config"
    .venv/bin/python -c "from openpi.policies.vega_wuji_policy import widen_norm_stats as w; w('$stats')"
    echo "$src" > "$stats.src"
fi
echo "norm stats for $src: $stats"
