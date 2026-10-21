# Shared paths for Qwen2.5-7B teacher → Qwen2.5-3B student VE tracks.
# Source from vision_encoders scripts — do not execute directly.
#
# Paths are resolved from this file's location (no hardcoded /home/...):
#   REPO_ROOT      = .../LLava_MoD
#   WORKSPACE_ROOT = parent that contains LLaVA_MoD/ and LLaVA-Instruct-150K/
# Override with: export WORKSPACE_ROOT=/your/root   (optional)
VE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${VE_DIR}/../../../.." && pwd)"
if [[ -z "${WORKSPACE_ROOT:-}" ]]; then
  WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../../.." && pwd)"
fi
export REPO_ROOT WORKSPACE_ROOT VE_DIR

TEACHER_LLM='Qwen/Qwen2.5-7B-Instruct'
STUDENT_LLM='Qwen/Qwen2.5-3B-Instruct'
CONV_VERSION='qwen'
CONV_MODE='qwen'

EVAL_ROOT="${REPO_ROOT}/../detectioumfire/eval"
EVAL_LAUNCHER="${EVAL_ROOT}/launchers/qwen2.5"
EVAL_RUNS="${EVAL_ROOT}/runs"

export TEACHER_LLM STUDENT_LLM CONV_VERSION CONV_MODE
export EVAL_ROOT EVAL_LAUNCHER EVAL_RUNS
