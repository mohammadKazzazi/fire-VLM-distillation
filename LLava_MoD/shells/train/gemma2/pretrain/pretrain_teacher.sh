#!/bin/bash
# Teacher Phase 1: projector pretrain (frozen Gemma2-9B LLM).
#
# Prerequisite:
#   huggingface-cli login  # accept Google Gemma license
#
# Usage:
#   cd MoD/LLava_MoD
#   bash shells/train/gemma2/pretrain/pretrain_teacher.sh
#
# Artifact:
#   checkpoints/llavagemma2-9b-pretrain/mm_projector.bin

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_gemma2.env.sh"
# shellcheck disable=SC1091
source "${REPO_ROOT}/shells/conda_env.sh"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY
    export WANDB_MODE=online
    wandb login "${WANDB_API_KEY}"
    wandb online
else
    export WANDB_MODE=disabled
fi

DEEPSPEED_CONFIG='./llavamod/config/dpconfig/zero2.json'
OFFLINE="${OFFLINE:-0}"
if [ "${OFFLINE}" = "1" ]; then
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
else
    export HF_DATASETS_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
fi

JSON_FILE='./llavamod/data/PreTrain/blip_laion_cc_sbu_558k.json'
IMAGE_FOLDER='./llavamod/data/PreTrain/Images'
LLM="${TEACHER_LLM}"
VISION_ENCODER='openai/clip-vit-large-patch14-336'
OUTPUT_DIR='./checkpoints/llavagemma2-9b-pretrain'
NUM_GPUS="${NUM_GPUS:-8}"

deepspeed --num_gpus "${NUM_GPUS}" llavamod/train/train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${LLM} \
    --version ${CONV_VERSION} \
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
    --attn_implementation ${ATTN_IMPLEMENTATION} \
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
