#!/usr/bin/env bash
# Build openpi (pi0.5) for this project on GT's Skynet, as a CPU job (Skynet keeps installs off the login nodes):
# openpi cloned under ~/scratch/DexVLA/tactile/openpi, its uv environment there, uv itself from the DexVLA
# project's pixi env. Adds the Vega U + WUJI policy module and its config entry (openpi_vega_wuji.py).
#
#   sbatch ~/scratch/DexVLA/tactile/finetune/skynet_openpi_env.sbatch
#SBATCH --job-name=tactile-openpi-env
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nethome/%u/scratch/DexVLA/tactile/logs/%x-%j.log
set -euo pipefail
work="$HOME/scratch/DexVLA/tactile"
uv="$HOME/scratch/DexVLA/meta/.pixi/envs/sft/bin/uv"
export UV_CACHE_DIR="$work/.runtime/uv-cache" UV_PYTHON_INSTALL_DIR="$work/.runtime/uv-python" GIT_LFS_SKIP_SMUDGE=1
cd "$work"
[[ -d openpi ]] || git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git
cd openpi
"$uv" sync
"$uv" pip install -e .
cp "$work/finetune/openpi_vega_wuji.py" src/openpi/policies/vega_wuji_policy.py
grep -q vega_wuji_policy src/openpi/training/config.py || python3 - <<'PY'
from pathlib import Path
p = Path("src/openpi/training/config.py")
s = p.read_text()
i = s.index("\n]\n", s.index("_CONFIGS = ["))
s = s[:i] + '\n    *__import__("openpi.policies.vega_wuji_policy", fromlist=["_"]).train_configs(),' + s[i:]
p.write_text(s)
PY
.venv/bin/python -c "from openpi.training import config as C; print([C.get_config(n).name for n in ('pi05_vega_wuji', 'pi05_vega_wuji_lora', 'pi05_vega_wuji_debug')])"
