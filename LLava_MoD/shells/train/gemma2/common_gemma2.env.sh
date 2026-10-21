# Shared paths for Gemma2-9B teacher → Gemma2-2B student pipeline.
# Source from other scripts — do not execute directly.
GEMMA2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${GEMMA2_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../../.." && pwd)"

TEACHER_LLM='google/gemma-2-9b-it'
STUDENT_LLM='google/gemma-2-2b-it'

TEACHER_PRETRAIN="${REPO_ROOT}/checkpoints/llavagemma2-9b-pretrain/mm_projector.bin"
TEACHER_LORA="${REPO_ROOT}/checkpoints/llavagemma2-9b-finetune-lora/adapter_config.json"
STUDENT_PROJECTOR="${REPO_ROOT}/checkpoints/llavagemma2-2b-student-pretrain/mm_projector.bin"

D2D_SCRIPT="${GEMMA2_DIR}/dense2dense_distillation.sh"
EVAL_ROOT="${REPO_ROOT}/../detectioumfire/eval"
EVAL_LAUNCHER="${EVAL_ROOT}/launchers/gemma2"
EVAL_RUNS="${EVAL_ROOT}/runs"
EVAL_DIR="${EVAL_LAUNCHER}"
CONV_VERSION='gemma_2'
# Gemma2 training in this repo requires eager attention (see train.py / align_train.py).
ATTN_IMPLEMENTATION='eager'
