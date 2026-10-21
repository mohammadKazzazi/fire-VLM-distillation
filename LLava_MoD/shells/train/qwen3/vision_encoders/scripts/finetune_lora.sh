#!/bin/bash
# Teacher Phase 2 — LoRA SFT (fire teacher) for a vision-encoder track.
#
# Usage:
#   VE_SLUG=mobileclip_s2 bash scripts/finetune_lora.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_ve.env.sh"
# shellcheck disable=SC1091
source "${REPO_ROOT}/shells/conda_env.sh"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [ -n "${WANDB_API_KEY:-}" ]; then export WANDB_API_KEY; export WANDB_MODE=online; wandb login "${WANDB_API_KEY}"; wandb online; REPORT_TO=wandb; else export WANDB_MODE=disabled; REPORT_TO=none; fi

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./llavamod/config/dpconfig/zero2.json}"
JSON_FILE="${JSON_FILE:-${WORKSPACE_ROOT}/LLaVA-Instruct-150K/teacher_training_data.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${WORKSPACE_ROOT}/LLaVA-Instruct-150K}"
OUTPUT_DIR="${TEACHER_LORA_DIR}"
NUM_GPUS="${NUM_GPUS:-8}"

[[ -f "${TEACHER_PRETRAIN}" ]] || { echo "Missing teacher projector: ${TEACHER_PRETRAIN}"; exit 1; }
[[ -f "${JSON_FILE}" ]] || { echo "Missing teacher JSON: ${JSON_FILE} (WORKSPACE_ROOT=${WORKSPACE_ROOT})"; exit 1; }
[[ -d "${IMAGE_FOLDER}" ]] || { echo "Missing IMAGE_FOLDER: ${IMAGE_FOLDER}"; exit 1; }
for _link in real_fire real_non_fire stable_diff_v15; do
  [[ -e "${IMAGE_FOLDER}/${_link}" ]] || {
    echo "Missing ${IMAGE_FOLDER}/${_link} — recreate relative symlinks to detectioumfire (see HANDOFF_MOBILECLIP_GOOGLE_DRIVE.md)"
    exit 1
  }
done

echo "=== Teacher LoRA SFT | VE=${VE_SLUG} | ${VISION_ENCODER} ==="
echo "OUTPUT_DIR=${OUTPUT_DIR}"

deepspeed --num_gpus "${NUM_GPUS}" llavamod/train/train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${TEACHER_LLM} \
    --version ${CONV_VERSION} \
    --data_path ${JSON_FILE} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ${VISION_ENCODER} \
    --image_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ${TEACHER_PRETRAIN} \
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
    --gradient_accumulation_steps ${GRAD_ACCUM:-4} \
    --attn_implementation sdpa \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate ${LR:-2e-4} \
    --mm_projector_lr ${MM_PROJECTOR_LR:-2e-5} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length ${MAX_LEN:-1536} \
    --gradient_checkpointing True \
    --dataloader_num_workers ${DATALOADER_WORKERS:-4} \
    --lazy_preprocess True \
    --report_to ${REPORT_TO} \
    --lora_enable True \
    --lora_r ${LORA_R:-128} \
    --lora_alpha ${LORA_ALPHA:-256} \
    --lora_dropout 0.05
