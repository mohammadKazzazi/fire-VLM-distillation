#!/bin/bash
# Student Phase 1 (1B): projector pretrain (frozen Llama-3.2-1B LLM).
#
# Usage:
#   cd MoD/LLava_MoD
#   bash "shells/train/llama3/student 1B/pretrain_student_1b.sh"   # detached by default
#
# Monitor:
#   tail -f logs/pretrain_llama3_student_1b_latest.log
#
# Artifact:
#   checkpoints/llavallama3.2-1b-student-pretrain/mm_projector.bin

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_1b.env.sh"

DETACH="${DETACH:-1}"
if [[ "${DETACH}" == "1" && -z "${PRETRAIN_DETACHED:-}" ]]; then
  LOG_DIR="${REPO_ROOT}/logs"
  mkdir -p "${LOG_DIR}"
  STAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/pretrain_llama3_student_1b_${STAMP}.log"
  ln -sfn "$(basename "${LOG_FILE}")" "${LOG_DIR}/pretrain_llama3_student_1b_latest.log"
  PID_FILE="${LOG_DIR}/pretrain_llama3_student_1b.pid"
  echo "Detaching — safe to close terminal/Cursor."
  echo "  log:  ${LOG_FILE}"
  echo "  tail: tail -f ${LOG_DIR}/pretrain_llama3_student_1b_latest.log"
  PRETRAIN_DETACHED=1 DETACH=0 nohup env PRETRAIN_DETACHED=1 DETACH=0 NUM_GPUS="${NUM_GPUS:-8}" \
    MASTER_PORT="${MASTER_PORT:-29501}" \
    bash "${BASH_SOURCE[0]}" >>"${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
  echo "  pid:  $(cat "${PID_FILE}") (also in ${PID_FILE})"
  exit 0
fi

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
LLM="${STUDENT_LLM}"
VISION_ENCODER='openai/clip-vit-large-patch14-336'
OUTPUT_DIR='./checkpoints/llavallama3.2-1b-student-pretrain'
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

deepspeed --num_gpus "${NUM_GPUS}" --master_port "${MASTER_PORT}" llavamod/train/train.py \
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
