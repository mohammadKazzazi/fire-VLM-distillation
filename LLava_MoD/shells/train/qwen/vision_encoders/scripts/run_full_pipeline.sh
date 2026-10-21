#!/usr/bin/env bash
# Full Qwen2.5-3B MobileCLIP track: teacher → student projector → Plan 4 D2D seeds.
#
# Usage (from MoD/LLava_MoD):
#   VE_SLUG=mobileclip_s2 bash shells/train/qwen/vision_encoders/scripts/run_full_pipeline.sh
#
# Env:
#   SEEDS="42 123 456"   default
#   RUN_VAL=0            set 1 to eval after each D2D (needs eval fix pack on Triton)
#   SKIP_EXISTING=1      skip finished stages
#   NUM_GPUS=4
#   DEEPSPEED_CONFIG=./llavamod/config/dpconfig/zero2.json
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VE_SLUG="${VE_SLUG:-mobileclip_s2}"
SEEDS="${SEEDS:-42 123 456}"
RUN_VAL="${RUN_VAL:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
export VE_SLUG SEEDS RUN_VAL SKIP_EXISTING

echo "============================================================"
echo "Qwen2.5-3B VE full pipeline"
echo "VE_SLUG=${VE_SLUG}"
echo "SEEDS=${SEEDS}"
echo "RUN_VAL=${RUN_VAL}"
echo "============================================================"

bash "${SCRIPT_DIR}/verify_layout.sh"
bash "${SCRIPT_DIR}/run_teacher_pipeline.sh"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_ve.env.sh"
if [[ -f "${STUDENT_PROJECTOR}" && "${SKIP_EXISTING}" == "1" ]]; then
  echo "Student projector exists: ${STUDENT_PROJECTOR} — skipping pretrain"
else
  bash "${SCRIPT_DIR}/pretrain_student_3b.sh"
fi

bash "${SCRIPT_DIR}/run_multi_seed.sh"

echo "Full pipeline complete for VE=${VE_SLUG}"
