# Shared paths for Qwen3-8B teacher → Qwen3-4B student pipeline.
# Source from other scripts — do not execute directly.
#
# Paths are resolved from this file's location (no hardcoded /home/...):
#   REPO_ROOT      = .../LLava_MoD
#   WORKSPACE_ROOT = parent that contains LLaVA_MoD/ and LLaVA-Instruct-150K/
# Override with: export WORKSPACE_ROOT=/your/root   (optional)
QWEN3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${QWEN3_DIR}/../../.." && pwd)"
if [[ -z "${WORKSPACE_ROOT:-}" ]]; then
  WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../../.." && pwd)"
fi
export REPO_ROOT WORKSPACE_ROOT

TEACHER_LLM='Qwen/Qwen3-8B'
STUDENT_LLM='Qwen/Qwen3-4B'

TEACHER_PRETRAIN="${REPO_ROOT}/checkpoints/llavaqwen3-8b-pretrain/mm_projector.bin"
TEACHER_LORA="${REPO_ROOT}/checkpoints/llavaqwen3-8b-finetune-lora/adapter_config.json"
STUDENT_PROJECTOR="${REPO_ROOT}/checkpoints/llavaqwen3-4b-student-pretrain/mm_projector.bin"

D2D_SCRIPT="${QWEN3_DIR}/dense2dense_distillation.sh"
EVAL_ROOT="${REPO_ROOT}/../detectioumfire/eval"
EVAL_LAUNCHER="${EVAL_ROOT}/launchers/qwen3"
EVAL_RUNS="${EVAL_ROOT}/runs"
EVAL_DIR="${EVAL_LAUNCHER}"
CONV_VERSION='qwen3'
