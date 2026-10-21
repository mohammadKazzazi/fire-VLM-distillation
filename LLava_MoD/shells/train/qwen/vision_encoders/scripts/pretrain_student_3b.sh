#!/bin/bash
# Student Phase 1b — Qwen2.5-3B projector pretrain for a vision-encoder track.
#
# Usage:
#   VE_SLUG=mobileclip_s2 bash scripts/pretrain_student_3b.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_ve.env.sh"
# shellcheck disable=SC1091
source "${REPO_ROOT}/shells/conda_env.sh"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [ -n "${WANDB_API_KEY:-}" ]; then export WANDB_API_KEY; export WANDB_MODE=online; wandb login "${WANDB_API_KEY}"; wandb online; else export WANDB_MODE=disabled; fi

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./llavamod/config/dpconfig/zero2.json}"
JSON_FILE='./llavamod/data/PreTrain/blip_laion_cc_sbu_558k.json'
IMAGE_FOLDER='./llavamod/data/PreTrain/Images'
OUTPUT_DIR="${STUDENT_PRETRAIN_DIR}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29513}"

[[ -f "${JSON_FILE}" ]] || { echo "Missing pretrain JSON: ${REPO_ROOT}/${JSON_FILE}"; exit 1; }
[[ -d "${IMAGE_FOLDER}" ]] || { echo "Missing pretrain Images: ${REPO_ROOT}/${IMAGE_FOLDER}"; exit 1; }

echo "=== Student 3B projector pretrain | VE=${VE_SLUG} | ${VISION_ENCODER} ==="
echo "OUTPUT_DIR=${OUTPUT_DIR}"

deepspeed --num_gpus "${NUM_GPUS}" --master_port "${MASTER_PORT}" llavamod/train/train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${STUDENT_LLM} \
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
    --per_device_train_batch_size ${PER_DEVICE_BATCH:-8} \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps ${GRAD_ACCUM:-2} \
    --attn_implementation sdpa \
    --eval_strategy "no" \
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
