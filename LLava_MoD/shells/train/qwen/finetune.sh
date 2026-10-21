#!/bin/bash
# Teacher Phase 2: full-parameter SFT (all LLM + mm_projector weights).
# Compare with LoRA SFT: shells/train/qwen/finetune_lora.sh
#
# Prerequisites:
#   checkpoints/llavaqwen2.5-7b-pretrain/mm_projector.bin  (pretrain_teacher.sh)
#
# Output:
#   checkpoints/llavaqwen2.5-7b-finetune/
#
# Phase 3 (full teacher): in dense2dense_distillation.sh set
#   REF_MLLM='./checkpoints/llavaqwen2.5-7b-finetune'
#   REF_MLLM_BASE=''   # no LoRA merge
#
# Usage:
#   cd MoD/LLava_MoD
#   unset CUDA_VISIBLE_DEVICES
#   bash shells/train/qwen/finetune.sh
#
# If OOM on 8x40GB:
#   DEEPSPEED_CONFIG=./llavamod/config/dpconfig/zero3_offload.json PER_DEVICE_BATCH=1 bash shells/train/qwen/finetune.sh

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

# Deepspeed config (ZeRO-3 shards model weights across GPUs; required for full 7B finetune on 8x A100 40GB)
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./llavamod/config/dpconfig/zero3.json}"

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

# Dataset
JSON_FILE='/home/ubuntu/llava-test/LLaVA-Instruct-150K/teacher_training_data.json'
IMAGE_FOLDER='/home/ubuntu/llava-test/LLaVA-Instruct-150K'

# Teacher (Qwen2.5-7B — same generation as student)
LLM='Qwen/Qwen2.5-7B-Instruct'
VISION_ENCODER='openai/clip-vit-large-patch14-336'
MLP_ADAPTOR='./checkpoints/llavaqwen2.5-7b-pretrain/mm_projector.bin'

# Output dir
OUTPUT_DIR='./checkpoints/llavaqwen2.5-7b-finetune'

# Multi-GPU: use all 8x A100 by default. Override with NUM_GPUS=4 or CUDA_VISIBLE_DEVICES=0,1,2,3.
NUM_GPUS="${NUM_GPUS:-8}"

deepspeed --num_gpus "${NUM_GPUS}" llavamod/train/train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${LLM} \
    --version qwen \
    --data_path ${JSON_FILE} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ${VISION_ENCODER} \
    --image_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ${MLP_ADAPTOR} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH:-2} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACCUM:-8} \
    --attn_implementation sdpa \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate ${LR:-2e-5} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length ${MAX_LEN:-2048} \
    --gradient_checkpointing True \
    --dataloader_num_workers ${DATALOADER_WORKERS:-8} \
    --lazy_preprocess True \
    --report_to ${REPORT_TO:-none}
