#!/usr/bin/env bash
# Run Plan 4 v4 + optional val for Qwen2.5-3B across multiple seeds
# (after teacher LoRA + student projector exist).
#
# Usage:
#   VE_SLUG=mobileclip_s2 SEEDS="42 123 456" bash scripts/run_multi_seed.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VE_SLUG="${VE_SLUG:?Set VE_SLUG=mobileclip_s2}"
SEEDS="${SEEDS:-42 123 456}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_VAL="${RUN_VAL:-0}"

has_checkpoint() {
  local out="$1"
  [[ -f "${out}/adapter_config.json" ]] || compgen -G "${out}/checkpoint-*" >/dev/null 2>&1
}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_ve.env.sh"

[[ -f "${TEACHER_LORA}" ]] || { echo "Missing teacher LoRA: ${TEACHER_LORA}"; exit 1; }
[[ -f "${STUDENT_PROJECTOR}" ]] || { echo "Missing student projector: ${STUDENT_PROJECTOR}"; exit 1; }

for SEED in ${SEEDS}; do
  export VE_SLUG SEED
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
