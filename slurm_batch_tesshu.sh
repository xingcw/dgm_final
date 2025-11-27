#!/bin/bash
#SBATCH --job-name=pnp_eval
#SBATCH --partition=standby
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/pnp_eval_%j.out
#SBATCH --error=logs/pnp_eval_%j.err

set -euo pipefail

###############################
# Clean environment (modules, Isaac, etc.)
###############################
module purge 2>/dev/null || true
unset PYTHONPATH
unset PYTHONHOME
unset LD_PRELOAD 2>/dev/null || true

###############################
# Paths
###############################
CONDA_ROOT="$HOME/miniconda3"
PROJECT_ROOT="$HOME/dgm_final"
ENV_NAME="p2p_sd"   # <<< use the new env on /shared_data0

###############################
# Caches to /shared_data0
###############################
BASE_SHARED="/shared_data0/$USER"

export HF_HOME="$BASE_SHARED/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$BASE_SHARED/torch_cache"
export XDG_CACHE_HOME="$BASE_SHARED/.cache"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME"

###############################
# Conda env
###############################
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

###############################
# Run the eval script
###############################
cd "${PROJECT_ROOT}"

mkdir -p logs
chmod +x PnPInversion/eval_script.sh

bash PnPInversion/eval_script.sh
