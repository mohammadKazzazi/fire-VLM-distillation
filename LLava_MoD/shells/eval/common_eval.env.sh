# Shared paths for LLava_MoD Detectium val wrappers. Source from plan*_val.sh — do not execute directly.
COMMON_EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${COMMON_EVAL_DIR}/../.." && pwd)"
EVAL_ROOT="${REPO_ROOT}/../detectioumfire/eval"
EVAL_LAUNCHER="${EVAL_ROOT}/launchers/qwen2.5"
EVAL_RUNS="${EVAL_ROOT}/runs"
