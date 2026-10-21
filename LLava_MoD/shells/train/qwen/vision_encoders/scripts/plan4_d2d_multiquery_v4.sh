#!/bin/bash
# Plan 4 v4 D2D kd_lm — Qwen2.5-3B student for a vision-encoder track.
#
# Usage:
#   VE_SLUG=mobileclip_s2 SEED=42 bash scripts/plan4_d2d_multiquery_v4.sh
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
JSON_FILE="${JSON_FILE:-${WORKSPACE_ROOT}/LLaVA-Instruct-150K/dense2dense_fire_nonbbox_v4.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${WORKSPACE_ROOT}/LLaVA-Instruct-150K}"
OUTPUT_DIR="${OUTPUT_DIR:-${STUDENT_D2D_DIR}}"
LOSS_TYPE="${LOSS_TYPE:-kd_lm}"
REF_MLLM="${TEACHER_LORA_DIR}"

[[ -f "${JSON_FILE}" ]] || { echo "Missing JSON: ${JSON_FILE}"; exit 1; }
[[ -f "${TEACHER_LORA}" ]] || { echo "Missing teacher LoRA: ${TEACHER_LORA}"; exit 1; }
[[ -f "${STUDENT_PROJECTOR}" ]] || { echo "Missing student projector: ${STUDENT_PROJECTOR}"; exit 1; }
[[ "${LOSS_TYPE}" == "kd_lm" ]] || { echo "Plan 4 expects LOSS_TYPE=kd_lm (got ${LOSS_TYPE})"; exit 1; }
[[ -d "${IMAGE_FOLDER}/real_fire" && -d "${IMAGE_FOLDER}/real_non_fire" ]] \
  || { echo "Missing image symlinks under ${IMAGE_FOLDER}"; exit 1; }

echo "=== Plan 4 v4 D2D (Qwen2.5-3B) | VE=${VE_SLUG} | SEED=${SEED} ==="
echo "Teacher: ${REF_MLLM}"
echo "Student projector: ${STUDENT_PROJECTOR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "VISION_ENCODER=${VISION_ENCODER}"

deepspeed llavamod/train/align_train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --ref_model_name_or_path ${REF_MLLM} \
    --ref_model_base ${TEACHER_LLM} \
    --policy_model_name_or_path ${STUDENT_LLM} --policy_pretrain_mm_mlp_adapter ${STUDENT_PROJECTOR} \
    --policy_model_type dense --ref_model_type dense --loss_type ${LOSS_TYPE} \
    --moe_loss_enable False --moe_enable False --moe_finetune False \
    --num_experts 4 --top_k_experts 2 --capacity_factor 1.5 \
    --moe_mode sparse --use_residual False --router_aux_loss_coef 0.01 \
    --train_modules mlp.gate_proj mlp.up_proj mlp.down_proj wg \
    --distill_all_tokens False \
    --version ${CONV_VERSION} \
    --data_path ${JSON_FILE} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower ${VISION_ENCODER} \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS:-1} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --attn_implementation sdpa \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate ${LR:-2e-5} \
    --mm_projector_lr 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to ${REPORT_TO} \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05
