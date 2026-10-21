# Vision encoder registry for Qwen3 VE ablation tracks.
# Source from vision_encoders/scripts/* — do not execute directly.
#
# Usage:
#   VE_SLUG=siglip2_b384 bash scripts/pretrain_teacher.sh

VE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${VE_DIR}/common_qwen3.env.sh"

VE_SLUG="${VE_SLUG:-clip_vitl336}"

case "${VE_SLUG}" in
  clip_vitl336)
    VISION_ENCODER='openai/clip-vit-large-patch14-336'
    VE_SHORT='clip-vitl336'
    VE_HIDDEN=1024
    VE_PATCHES=576
    VE_RES=336
    ;;
  siglip2_b384)
    VISION_ENCODER='google/siglip2-base-patch16-384'
    VE_SHORT='siglip2-b384'
    VE_HIDDEN=768
    VE_PATCHES=576
    VE_RES=384
    VE_VISION_PARAMS_M=93
    ;;
  mobileclip_s2)
    VISION_ENCODER='mobileclip:MobileCLIP-S2:datacompdr'
    VE_SHORT='mobileclip-s2'
    VE_HIDDEN=1280
    VE_PATCHES=64
    VE_RES=256
    ;;
  *)
    echo "Unknown VE_SLUG=${VE_SLUG}. Use: clip_vitl336 | siglip2_b384 | mobileclip_s2" >&2
    exit 1
    ;;
esac

export VISION_ENCODER VE_SHORT VE_HIDDEN VE_PATCHES VE_RES VE_SLUG

# Teacher artifacts (per VE)
# clip_vitl336 reuses the legacy unsuffixed Qwen3-8B CLIP-L teacher (already trained).
if [[ "${VE_SLUG}" == "clip_vitl336" ]]; then
  TEACHER_PRETRAIN="${REPO_ROOT}/checkpoints/llavaqwen3-8b-pretrain/mm_projector.bin"
  TEACHER_PRETRAIN_DIR="${REPO_ROOT}/checkpoints/llavaqwen3-8b-pretrain"
  TEACHER_LORA="${REPO_ROOT}/checkpoints/llavaqwen3-8b-finetune-lora/adapter_config.json"
  TEACHER_LORA_DIR="${REPO_ROOT}/checkpoints/llavaqwen3-8b-finetune-lora"
  TEACHER_REUSE_LEGACY=1
else
  TEACHER_PRETRAIN="${REPO_ROOT}/checkpoints/llavaqwen3-8b-pretrain-${VE_SHORT}/mm_projector.bin"
  TEACHER_PRETRAIN_DIR="${REPO_ROOT}/checkpoints/llavaqwen3-8b-pretrain-${VE_SHORT}"
  TEACHER_LORA="${REPO_ROOT}/checkpoints/llavaqwen3-8b-finetune-lora-${VE_SHORT}/adapter_config.json"
  TEACHER_LORA_DIR="${REPO_ROOT}/checkpoints/llavaqwen3-8b-finetune-lora-${VE_SHORT}"
  TEACHER_REUSE_LEGACY=0
fi

# Student 0.6B artifacts (per VE)
STUDENT_LLM='Qwen/Qwen3-0.6B'
STUDENT_PROJECTOR="${REPO_ROOT}/checkpoints/llavaqwen3-0.6b-student-pretrain-${VE_SHORT}/mm_projector.bin"
STUDENT_PRETRAIN_DIR="${REPO_ROOT}/checkpoints/llavaqwen3-0.6b-student-pretrain-${VE_SHORT}"

# Default checkpoint naming: include VE + seed
SEED="${SEED:-42}"
STUDENT_D2D_DIR="${REPO_ROOT}/checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen3-0.6b-${VE_SHORT}-seed${SEED}"
EVAL_ANSWERS="${EVAL_RUNS}/students/qwen3-0.6b/${VE_SHORT}/plan4-v4-kd-lm-seed${SEED}/val_answers.jsonl"

export TEACHER_PRETRAIN TEACHER_PRETRAIN_DIR TEACHER_LORA TEACHER_LORA_DIR TEACHER_REUSE_LEGACY
export STUDENT_LLM STUDENT_PROJECTOR STUDENT_PRETRAIN_DIR STUDENT_D2D_DIR EVAL_ANSWERS
