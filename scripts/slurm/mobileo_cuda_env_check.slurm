#!/bin/bash
#SBATCH --job-name=mobileo-cuda-check
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/Mobile-O-runs/logs/slurm-%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/dbartaula/Mobile-O-runs/logs/slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="/iopsstor/scratch/cscs/dbartaula/Mobile-O-Experiments"
RUN_ROOT="${RUN_ROOT:-/iopsstor/scratch/cscs/dbartaula/Mobile-O-runs}"
LOG_ROOT="$RUN_ROOT/logs"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT"

export PYTHONNOUSERSITE=1

# Conda activation
eval "$(conda shell.bash hook)"
conda activate mobileo2

echo "==== Host / Job Info ===="
hostname
nvidia-smi || true

echo "==== CUDA Modules ===="
module avail cuda 2>&1 || true

echo "==== Try Generic CUDA Module Load ===="
module load cuda || true

if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
  fi
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "==== CUDA Toolkit Checks ===="
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
which nvcc || true
nvcc --version || true
python -c "from torch.utils.cpp_extension import CUDA_HOME; print('torch CUDA_HOME=', CUDA_HOME)"

echo "==== Python Package Checks ===="
which python
python -V
python -c "import sys,diffusers,transformers,torch; print('python=',sys.executable); print('torch=',torch.__version__); print('diffusers=',diffusers.__version__); print('transformers=',transformers.__version__)"
python -c "from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel; print('UNet2DConditionModel import OK')"

echo "==== Done ===="


