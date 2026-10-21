#!/bin/bash
# Student Phase 1: projector / mm_adapter pretrain (frozen student LLM).
#
# Teacher stack (separate projector — do not reuse here):
#   LLM:     Qwen/Qwen2.5-7B-Instruct  (pretrain_teacher.sh, finetune.sh)
#   Projector: checkpoints/llavaqwen2.5-7b-pretrain/mm_projector.bin
#   LoRA teacher (Phase 3 REF_MLLM): checkpoints/llavaqwen2.5-7b-finetune-lora
#
# This script trains a separate projector for the smaller student LM on the same
# LCS-558K caption data and CLIP ViT-L/336 vision tower as the teacher.
#
# Usage:
#   cd MoD/LLava_MoD
#   export PYTHONPATH="${PWD}:${PYTHONPATH}"
#   bash shells/train/qwen/pretrain_student.sh
#
# Artifact:
#   checkpoints/llavaqwen2.5-3b-student-pretrain/mm_projector.bin
#   → POLICY_MLP_ADAPTOR in dense2dense_distillation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../conda_env.sh"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# WANDB (optional): set WANDB_API_KEY to enable online logging
# WANDB_API_KEY=''
if [ -n "${WANDB_API_KEY}" ]; then
    export WANDB_API_KEY
    export WANDB_MODE=online
    wandb login "${WANDB_API_KEY}"
    wandb online
else
    export WANDB_MODE=disabled
fi

# Deepspeed config
DEEPSPEED_CONFIG='./llavamod/config/dpconfig/zero2.json'

# Hugging Face connectivity
# Set OFFLINE=1 only if models/datasets are already cached locally.
OFFLINE="${OFFLINE:-0}"
if [ "${OFFLINE}" = "1" ]; then
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
else
    export HF_DATASETS_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
fi

# Dataset (same as pretrain_teacher.sh — LLaVA LCS-558K)
JSON_FILE='./llavamod/data/PreTrain/blip_laion_cc_sbu_558k.json'
IMAGE_FOLDER='./llavamod/data/PreTrain/Images'

# Teacher reference (for documentation only; not loaded in this stage)
TEACHER_LLM='Qwen/Qwen2.5-7B-Instruct'
TEACHER_PRETRAIN='./checkpoints/llavaqwen2.5-7b-pretrain/mm_projector.bin'

# Student LLM + vision (match teacher vision encoder / projector type)
LLM='Qwen/Qwen2.5-3B-Instruct'
VISION_ENCODER='openai/clip-vit-large-patch14-336'

# Output dir (student projector — separate from llavaqwen2.5-7b-pretrain)
OUTPUT_DIR='./checkpoints/llavaqwen2.5-3b-student-pretrain'

# Multi-GPU: use all 8 GPUs by default. Override with NUM_GPUS=4 or CUDA_VISIBLE_DEVICES=0,1,2,3.
NUM_GPUS="${NUM_GPUS:-8}"

deepspeed --num_gpus "${NUM_GPUS}" llavamod/train/train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${LLM} \
    --version qwen \
    --data_path ${JSON_FILE} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ${VISION_ENCODER} \
    --image_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --attn_implementation sdpa \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 2 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to ${REPORT_TO:-none}
