#!/bin/bash
#SBATCH --job-name=mobileo-stage12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/Mobile-O-runs/logs/slurm-%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/dbartaula/Mobile-O-runs/logs/slurm-%x-%j.err

set -euo pipefail

# Usage:
#   sbatch scripts/slurm/train_stage1_stage2_4gpu.slurm pretrain 1.5B
#   sbatch scripts/slurm/train_stage1_stage2_4gpu.slurm sft 1.5B
#   sbatch scripts/slurm/train_stage1_stage2_4gpu.slurm pretrain 0.5B
#   sbatch scripts/slurm/train_stage1_stage2_4gpu.slurm sft 0.5B

STAGE="${1:-pretrain}"   # pretrain|sft
SIZE="${2:-1.5B}"        # 1.5B|0.5B

REPO_ROOT="/iopsstor/scratch/cscs/dbartaula/Mobile-O-Experiments"
cd "$REPO_ROOT"
RUN_ROOT="${RUN_ROOT:-/iopsstor/scratch/cscs/dbartaula/Mobile-O-runs}"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/checkpoints"

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=INFO
export WANDB_PROJECT="${WANDB_PROJECT:-Mobile-O}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-stage12-4gpu}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONNOUSERSITE=1

# Conda activation

eval "$(conda shell.bash hook)"
conda activate mobileo2
# Optional CUDA toolkit diagnostics (no hard fail)
if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  echo "CUDA_HOME=$CUDA_HOME"
  nvcc --version
else
  echo "nvcc not found; continuing without CUDA toolkit (no DeepSpeed)."
fi
python -c "from torch.utils.cpp_extension import CUDA_HOME; print('torch CUDA_HOME=', CUDA_HOME)"

# Runtime diagnostics
which python
python -V
python -c "import sys; print(sys.executable)"
PYTHONNOUSERSITE=1 python -c "import sys,diffusers,transformers; print('python=',sys.executable); print('diffusers=',diffusers.__version__); print('transformers=',transformers.__version__)"
PYTHONNOUSERSITE=1 python -c "from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel; print('UNet2DConditionModel import OK')"

if [[ "$SIZE" == "1.5B" ]]; then
  if [[ "$STAGE" == "pretrain" ]]; then
    srun bash "$REPO_ROOT/scripts/Mobile-O-1.5B/pre_train.sh"
  elif [[ "$STAGE" == "sft" ]]; then
    srun bash "$REPO_ROOT/scripts/Mobile-O-1.5B/sft.sh"
  else
    echo "Unknown stage: $STAGE"
    exit 1
  fi
elif [[ "$SIZE" == "0.5B" ]]; then
  if [[ "$STAGE" == "pretrain" ]]; then
    srun bash "$REPO_ROOT/scripts/Mobile-O-0.5B/pre_train.sh"
  elif [[ "$STAGE" == "sft" ]]; then
    srun bash "$REPO_ROOT/scripts/Mobile-O-0.5B/sft.sh"
  else
    echo "Unknown stage: $STAGE"
    exit 1
  fi
else
  echo "Unknown size: $SIZE"
  exit 1
fi





















