#!/bin/bash
#SBATCH --job-name=mobileo-05b-posttrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs checkpoints

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

# W&B
export WANDB_API_KEY="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"
export WANDB_PROJECT="Mobile-O"
export WANDB_ENTITY="078bct-anandi-tribhuvan-university-institute-of-engineering"

RUN_NAME=Mobile-O-0.5B-Post-Train

srun torchrun --nnodes=1 --nproc_per_node=4 mobileo/train/train_mem.py \
    --deepspeed ./deepspeed_scripts/zero1.json \
    --is_train True \
    --diffusion_name_or_path Efficient-Large-Model/Sana_600M_512px_diffusers \
    --model_name_or_path checkpoints/Mobile-O-0.5B-SFT \
    --version qwen_v2 \
    --data_type mix \
    --image_folder data/Mobile-O-Post-Train \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_backbone True \
    --fp16 True \
    --bf16 False \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --output_dir checkpoints/$RUN_NAME \
    --num_train_epochs 7 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine_with_min_lr \
    --max_grad_norm 1.0 \
    --adam_beta2 0.95 \
    --model_max_length 512 \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME 2>&1 | tee logs/$RUN_NAME.txt
