# Vision encoder registry for Qwen2.5-3B VE tracks.
# Source from vision_encoders/scripts/* — do not execute directly.
#
# Usage:
#   VE_SLUG=mobileclip_s2 bash scripts/pretrain_teacher.sh

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_qwen.env.sh"

VE_SLUG="${VE_SLUG:-mobileclip_s2}"

case "${VE_SLUG}" in
  mobileclip_s2)
    VISION_ENCODER='mobileclip:MobileCLIP-S2:datacompdr'
    VE_SHORT='mobileclip-s2'
    VE_HIDDEN=1280
    VE_PATCHES=64
    VE_RES=256
    ;;
  *)
    echo "Unknown VE_SLUG=${VE_SLUG}. Use: mobileclip_s2" >&2
    echo "CLIP ViT-L Qwen2.5-3B already exists as the default CLIP-family track" >&2
    echo "(shells/train/qwen/plan4_d2d_multiquery_v4.sh). Do not reuse those checkpoints here." >&2
    exit 1
    ;;
esac

export VISION_ENCODER VE_SHORT VE_HIDDEN VE_PATCHES VE_RES VE_SLUG

# Teacher artifacts (per VE) — always train from base; never reuse CLIP teacher.
TEACHER_PRETRAIN="${REPO_ROOT}/checkpoints/llavaqwen2.5-7b-pretrain-${VE_SHORT}/mm_projector.bin"
TEACHER_PRETRAIN_DIR="${REPO_ROOT}/checkpoints/llavaqwen2.5-7b-pretrain-${VE_SHORT}"
TEACHER_LORA="${REPO_ROOT}/checkpoints/llavaqwen2.5-7b-finetune-lora-${VE_SHORT}/adapter_config.json"
TEACHER_LORA_DIR="${REPO_ROOT}/checkpoints/llavaqwen2.5-7b-finetune-lora-${VE_SHORT}"

# Student 3B artifacts (per VE)
STUDENT_PROJECTOR="${REPO_ROOT}/checkpoints/llavaqwen2.5-3b-student-pretrain-${VE_SHORT}/mm_projector.bin"
STUDENT_PRETRAIN_DIR="${REPO_ROOT}/checkpoints/llavaqwen2.5-3b-student-pretrain-${VE_SHORT}"

SEED="${SEED:-42}"
STUDENT_D2D_DIR="${REPO_ROOT}/checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen2.5-3b-${VE_SHORT}-seed${SEED}"
EVAL_ANSWERS="${EVAL_RUNS}/students/qwen2.5-3b/${VE_SHORT}/plan4-v4-kd-lm-seed${SEED}/val_answers.jsonl"

export TEACHER_PRETRAIN TEACHER_PRETRAIN_DIR TEACHER_LORA TEACHER_LORA_DIR
export STUDENT_PROJECTOR STUDENT_PRETRAIN_DIR STUDENT_D2D_DIR EVAL_ANSWERS SEED
