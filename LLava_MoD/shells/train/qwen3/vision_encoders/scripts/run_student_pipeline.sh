#!/usr/bin/env bash
# Run full Qwen3-0.6B student pipeline for one vision encoder (requires teacher done).
#
# Usage:
#   VE_SLUG=mobileclip_s2 SEED=42 bash scripts/run_student_pipeline.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VE_SLUG="${VE_SLUG:?Set VE_SLUG}"
SEED="${SEED:-42}"
export VE_SLUG SEED

bash "${SCRIPT_DIR}/pretrain_student_0.6b.sh"
bash "${SCRIPT_DIR}/plan4_d2d_multiquery_v4.sh"
NUM_GPUS="${NUM_GPUS:-8}" bash "${SCRIPT_DIR}/plan4_d2d_multiquery_v4_val.sh"

echo "Student pipeline complete for VE=${VE_SLUG} SEED=${SEED}"
