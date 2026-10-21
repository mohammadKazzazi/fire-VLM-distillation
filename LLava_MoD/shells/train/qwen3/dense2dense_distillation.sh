#!/bin/bash
# D2D mimic KD — Qwen3-8B LoRA teacher → Qwen3-4B LoRA student.
# Default LOSS_TYPE is kd_lm (required for deployable Yes/No prefixes).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_qwen3.env.sh"
# shellcheck disable=SC1091
source "${REPO_ROOT}/shells/conda_env.sh"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

if [ -n "${WANDB_API_KEY}" ]; then
    export WANDB_API_KEY
    export WANDB_MODE=online
    wandb login "${WANDB_API_KEY}"
    wandb online
    REPORT_TO=wandb
else
    export WANDB_MODE=disabled
    REPORT_TO=none
fi

# torch 2.4: zero2_offload JIT-compiles cpu_adam against the wrong libtorch ABI.
# Use zero2 (no CPU offload); override with DEEPSPEED_CONFIG if needed.
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./llavamod/config/dpconfig/zero2.json}"

JSON_FILE="${JSON_FILE:-/home/ubuntu/llava-test/LLaVA-Instruct-150K/dense2dense_fire_nonbbox_v4.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/ubuntu/llava-test/LLaVA-Instruct-150K}"

REF_MLLM='./checkpoints/llavaqwen3-8b-finetune-lora'
REF_MLLM_BASE="${TEACHER_LLM}"
POLICY_LLM="${STUDENT_LLM}"
POLICY_MLP_ADAPTOR="${STUDENT_PROJECTOR}"
VISION_ENCODER='openai/clip-vit-large-patch14-336'

POLICY_MODEL_TYPE='dense'
REF_MODEL_TYPE='dense'
LOSS_TYPE="${LOSS_TYPE:-kd_lm}"
DISTILL_ALL_TOKENS=False

MOE_LOSS_ENABLE=False
MOE_ENABLE=False
MOE_FINETUNE=False
MOE_MODE="sparse"
NUM_EXPERTS=4
TOP_K_EXPERTS=2
USE_RESIDUAL=False
ROUTER_AUX_LOSS_COEF=0.01
CAPACITY_FACTOR=1.5

SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen3-4b-seed42}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"

OFFLINE="${OFFLINE:-0}"
if [ "${OFFLINE}" = "1" ]; then
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
else
    export HF_DATASETS_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
fi

deepspeed llavamod/train/align_train.py \
    --deepspeed ${DEEPSPEED_CONFIG} \
    --ref_model_name_or_path ${REF_MLLM} \
    --ref_model_base ${REF_MLLM_BASE} \
    --policy_model_name_or_path ${POLICY_LLM} --policy_pretrain_mm_mlp_adapter ${POLICY_MLP_ADAPTOR} \
    --policy_model_type ${POLICY_MODEL_TYPE} --ref_model_type ${REF_MODEL_TYPE} --loss_type ${LOSS_TYPE} \
    --moe_loss_enable ${MOE_LOSS_ENABLE} --moe_enable ${MOE_ENABLE} --moe_finetune ${MOE_FINETUNE} \
    --num_experts ${NUM_EXPERTS} --top_k_experts ${TOP_K_EXPERTS} --capacity_factor ${CAPACITY_FACTOR} \
    --moe_mode ${MOE_MODE} --use_residual ${USE_RESIDUAL} --router_aux_loss_coef ${ROUTER_AUX_LOSS_COEF} \
    --train_modules mlp.gate_proj mlp.up_proj mlp.down_proj wg \
    --distill_all_tokens ${DISTILL_ALL_TOKENS} \
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
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
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
    --report_to ${REPORT_TO:-none} \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05
