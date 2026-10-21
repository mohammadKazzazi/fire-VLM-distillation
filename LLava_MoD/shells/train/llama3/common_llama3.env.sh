# Shared paths for Llama 3.1 teacher → Llama 3.2 student pipeline (default: 3B).
# Edge 1B student: shells/train/llama3/student 1B/common_1b.env.sh
# Source from other scripts — do not execute directly.
LLAMA3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LLAMA3_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../../.." && pwd)"

TEACHER_LLM='meta-llama/Llama-3.1-8B-Instruct'
STUDENT_LLM='meta-llama/Llama-3.2-3B-Instruct'

TEACHER_PRETRAIN="${REPO_ROOT}/checkpoints/llavallama3.1-8b-pretrain/mm_projector.bin"
TEACHER_LORA="${REPO_ROOT}/checkpoints/llavallama3.1-8b-finetune-lora/adapter_config.json"
STUDENT_PROJECTOR="${REPO_ROOT}/checkpoints/llavallama3.2-3b-student-pretrain/mm_projector.bin"

D2D_SCRIPT="${LLAMA3_DIR}/dense2dense_distillation.sh"
EVAL_ROOT="${REPO_ROOT}/../detectioumfire/eval"
EVAL_LAUNCHER="${EVAL_ROOT}/launchers/llama3"
EVAL_RUNS="${EVAL_ROOT}/runs"
EVAL_DIR="${EVAL_LAUNCHER}"
CONV_VERSION='llama_3'
