#!/bin/bash
#SBATCH --job-name=mobileo-05b-sft-xf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/Mobile-O-runs/logs/slurm-%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/dbartaula/Mobile-O-runs/logs/slurm-%x-%j.err

set -euo pipefail
REPO_ROOT="/iopsstor/scratch/cscs/dbartaula/Mobile-O-Experiments"
cd "$REPO_ROOT"
RUN_ROOT="${RUN_ROOT:-/iopsstor/scratch/cscs/dbartaula/Mobile-O-runs}"
LOG_ROOT="$RUN_ROOT/logs"
CKPT_ROOT="$RUN_ROOT/checkpoints"
mkdir -p "$LOG_ROOT" "$CKPT_ROOT"

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONNOUSERSITE=1

# Activate conda environment explicitly
source /iopsstor/scratch/cscs/dbartaula/miniforge3/etc/profile.d/conda.sh
conda activate mobileo2

# Runtime diagnostics
which python
python -V
python -c "import sys; print(sys.executable)"
PYTHONNOUSERSITE=1 python -c "import sys,diffusers,transformers; print('python=',sys.executable); print('diffusers=',diffusers.__version__); print('transformers=',transformers.__version__)"
PYTHONNOUSERSITE=1 python -c "from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel; print('UNet2DConditionModel import OK')"
export WANDB_API_KEY="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"
export WANDB_PROJECT="Mobile-O"
export WANDB_ENTITY="078bct-anandi-tribhuvan-university-institute-of-engineering"

RUN_NAME=Mobile-O-0.5B-SFT-xformers

PYTHONNOUSERSITE=1 srun torchrun --nnodes=1 --nproc_per_node=4 --log-dir "$LOG_ROOT/torchrun-${SLURM_JOB_ID}" --redirects 3 --tee 3 "$REPO_ROOT/mobileo/train/train_xformers.py" \
    --deepspeed ./deepspeed_scripts/zero1.json \
    --diffusion_name_or_path Efficient-Large-Model/Sana_600M_512px_diffusers \
    --vlm_num_layers 4 \
    --model_name_or_path $CKPT_ROOT/Mobile-O-0.5B-Pretrain \
    --version qwen \
    --data_type mix \
    --image_folder data/Mobile-O-SFT \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_backbone True \
    --fp16 True \
    --bf16 False \
    --is_train True \
    --output_dir $CKPT_ROOT/$RUN_NAME \
    --aspect_ratio_size 512 512 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine_with_min_lr \
    --adam_beta2 0.95 \
    --model_max_length 512 \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --seed 42 \
    --run_name $RUN_NAME 2>&1 | tee $LOG_ROOT/$RUN_NAME.txt














