# Shared paths for Qwen2.5-0.5B student pipeline. Source from other scripts — do not execute directly.
STUDENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${STUDENT_DIR}/../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../../.." && pwd)"

POLICY_LLM='Qwen/Qwen2.5-0.5B-Instruct'
STUDENT_PROJECTOR="${REPO_ROOT}/checkpoints/llavaqwen2.5-0.5b-student-pretrain/mm_projector.bin"
TEACHER_LORA="${REPO_ROOT}/checkpoints/llavaqwen2.5-7b-finetune-lora/adapter_config.json"
D2D_SCRIPT="${STUDENT_DIR}/dense2dense_distillation_0.5b.sh"
EVAL_ROOT="${REPO_ROOT}/../detectioumfire/eval"
EVAL_LAUNCHER="${EVAL_ROOT}/launchers/qwen2.5"
EVAL_RUNS="${EVAL_ROOT}/runs"
EVAL_DIR="${EVAL_LAUNCHER}"
