#!/usr/bin/env bash
set -euo pipefail

###############################
# Config (edit if you need)
###############################
SCRATCH_BASE="/scratch/tmp/${USER}"
CONDA_ROOT="${HOME}/miniconda3"
ENV_NAME="p2p"
ENV_PYTHON="3.9"
REPO_DIR="/home/ftesshu/dgm_final/PnPInversion"
REQ_FILE="environment/p2p_requirements.txt"

echo "=== Step 0: Install Miniconda (if needed) ==="
if [ ! -d "${CONDA_ROOT}" ]; then
  if [ ! -f "miniconda.sh" ]; then
    echo "ERROR: miniconda.sh not found in current directory."
    exit 1
  fi
  bash miniconda.sh -b -p "${CONDA_ROOT}"
else
  echo "Miniconda already present at ${CONDA_ROOT}"
fi

echo "=== Step 1: Enable conda in this script ==="
# Use profile script directly instead of relying on conda init
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

echo "=== Step 2: Put pkgs/envs on scratch ==="
mkdir -p "${SCRATCH_BASE}/conda_pkgs" "${SCRATCH_BASE}/conda_envs"

# --add may fail if already present, so ignore errors with '|| true'
conda config --add pkgs_dirs "${SCRATCH_BASE}/conda_pkgs" || true
conda config --add envs_dirs "${SCRATCH_BASE}/conda_envs" || true

echo "Conda pkgs/envs dirs:"
conda config --show pkgs_dirs envs_dirs

echo "=== Step 3: Install mamba in base ==="
conda activate base
conda install -n base -c conda-forge mamba -y

echo "=== Step 4: Create ${ENV_NAME} env (Python ${ENV_PYTHON}) ==="
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  mamba create -n "${ENV_NAME}" python="${ENV_PYTHON}" -y
else
  echo "Environment ${ENV_NAME} already exists, skipping creation."
fi

conda activate "${ENV_NAME}"

echo "=== Step 5: Clean any existing torch installs ==="
mamba remove pytorch torchvision torchaudio cudatoolkit -y || true
pip uninstall -y torch torchvision torchaudio || true

echo "=== Step 6: Install PyTorch 1.12.1 (cu113) via pip ==="
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113

echo "=== Step 7: Install repo requirements ==="
if [ ! -d "${REPO_DIR}" ]; then
  echo "ERROR: repo directory ${REPO_DIR} not found."
  exit 1
fi

cd "${REPO_DIR}"

if [ ! -f "${REQ_FILE}" ]; then
  echo "ERROR: requirements file ${REQ_FILE} not found in ${REPO_DIR}."
  exit 1
fi

pip install -r "${REQ_FILE}"

echo "=== Step 8: Sanity-check torch ==="
python - << 'EOF'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch file:", torch.__file__)
EOF

echo
echo "=== DONE ==="
echo "To use the env in a new shell, run:"
echo "  source ${CONDA_ROOT}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
