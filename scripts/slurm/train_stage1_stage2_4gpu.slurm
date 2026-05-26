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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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









