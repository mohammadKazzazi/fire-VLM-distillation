# Shared paths for Qwen3-0.6B edge student (same 8B teacher as 4B pipeline).
# Source from other scripts — do not execute directly.
STUDENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${STUDENT_DIR}/../common_qwen3.env.sh"

STUDENT_LLM='Qwen/Qwen3-0.6B'
POLICY_LLM="${STUDENT_LLM}"
STUDENT_PROJECTOR="${REPO_ROOT}/checkpoints/llavaqwen3-0.6b-student-pretrain/mm_projector.bin"
D2D_SCRIPT="${STUDENT_DIR}/dense2dense_distillation_0.6b.sh"
