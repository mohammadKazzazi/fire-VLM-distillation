#!/usr/bin/env bash
# Run Plan 4 v4 + val for Qwen3-0.6B across multiple seeds (after teacher+projector exist).
#
# Usage:
#   VE_SLUG=mobileclip_s2 SEEDS="42 123 456" bash scripts/run_multi_seed.sh
#
# Prefer the all-models driver when sweeping several backbones:
#   bash shells/train/run_all_models_multiseed.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VE_SLUG="${VE_SLUG:?Set VE_SLUG}"
SEEDS="${SEEDS:-42 123 456}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_VAL="${RUN_VAL:-1}"

has_checkpoint() {
  local out="$1"
  [[ -f "${out}/adapter_config.json" ]] || compgen -G "${out}/checkpoint-*" >/dev/null 2>&1
}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_ve.env.sh"

for SEED in ${SEEDS}; do
  export VE_SLUG SEED
  # Refresh STUDENT_D2D_DIR for this seed
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/../common_ve.env.sh"
  echo "========== VE=${VE_SLUG} SEED=${SEED} → ${STUDENT_D2D_DIR} =========="
  if [[ "${SKIP_EXISTING}" == "1" ]] && has_checkpoint "${STUDENT_D2D_DIR}"; then
    echo "[SKIP] exists: ${STUDENT_D2D_DIR}"
    continue
  fi
  bash "${SCRIPT_DIR}/plan4_d2d_multiquery_v4.sh"
  if [[ "${RUN_VAL}" == "1" ]]; then
    NUM_GPUS="${NUM_GPUS:-8}" bash "${SCRIPT_DIR}/plan4_d2d_multiquery_v4_val.sh"
  fi
done

echo "Multi-seed complete for VE=${VE_SLUG}: ${SEEDS}"
