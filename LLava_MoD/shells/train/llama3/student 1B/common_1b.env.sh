# Shared paths for Llama-3.2-1B edge student (same 3.1 teacher as 3B pipeline).
# Source from other scripts — do not execute directly.
STUDENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${STUDENT_DIR}/../common_llama3.env.sh"

STUDENT_LLM='meta-llama/Llama-3.2-1B-Instruct'
POLICY_LLM="${STUDENT_LLM}"
STUDENT_PROJECTOR="${REPO_ROOT}/checkpoints/llavallama3.2-1b-student-pretrain/mm_projector.bin"
D2D_SCRIPT="${STUDENT_DIR}/dense2dense_distillation_1b.sh"
