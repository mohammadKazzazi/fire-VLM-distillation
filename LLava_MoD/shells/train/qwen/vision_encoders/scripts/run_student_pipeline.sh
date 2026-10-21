#!/usr/bin/env bash
# Run Qwen2.5-3B student pipeline for one vision encoder (requires teacher done).
#
# Usage:
#   VE_SLUG=mobileclip_s2 SEED=42 bash scripts/run_student_pipeline.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VE_SLUG="${VE_SLUG:?Set VE_SLUG=mobileclip_s2}"
SEED="${SEED:-42}"
export VE_SLUG SEED

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_ve.env.sh"

if [[ ! -f "${STUDENT_PROJECTOR}" ]]; then
  bash "${SCRIPT_DIR}/pretrain_student_3b.sh"
else
  echo "Student projector exists: ${STUDENT_PROJECTOR} — skipping pretrain"
fi

bash "${SCRIPT_DIR}/plan4_d2d_multiquery_v4.sh"
if [[ "${RUN_VAL:-0}" == "1" ]]; then
  NUM_GPUS="${NUM_GPUS:-8}" bash "${SCRIPT_DIR}/plan4_d2d_multiquery_v4_val.sh"
fi

echo "Student pipeline complete for VE=${VE_SLUG} SEED=${SEED}"
