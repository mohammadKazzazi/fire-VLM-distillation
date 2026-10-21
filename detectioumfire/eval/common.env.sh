# Shared Detectium eval paths. Source from launcher scripts — do not execute directly.
COMMON_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="${EVAL_ROOT:-${COMMON_ENV_DIR}}"
DETECTIUM_DIR="$(cd "${EVAL_ROOT}/.." && pwd)"
EVAL_LIB="${EVAL_ROOT}/lib"
EVAL_RUNS="${EVAL_ROOT}/runs"
EVAL_COMPARISONS="${EVAL_ROOT}/comparisons"
