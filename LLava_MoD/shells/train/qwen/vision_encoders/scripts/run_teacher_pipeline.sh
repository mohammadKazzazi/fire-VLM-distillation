#!/usr/bin/env bash
# Run Qwen2.5-7B teacher pipeline for one vision encoder (from base).
#
# Usage:
#   VE_SLUG=mobileclip_s2 bash scripts/run_teacher_pipeline.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VE_SLUG="${VE_SLUG:?Set VE_SLUG=mobileclip_s2}"
export VE_SLUG

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_ve.env.sh"

if [[ -f "${TEACHER_LORA}" && "${SKIP_EXISTING:-1}" == "1" ]]; then
  echo "Teacher LoRA already exists: ${TEACHER_LORA_DIR} — skipping"
  exit 0
fi

bash "${SCRIPT_DIR}/pretrain_teacher.sh"
bash "${SCRIPT_DIR}/finetune_lora.sh"

echo "Teacher pipeline complete for VE=${VE_SLUG}"
echo "  projector: ${TEACHER_PRETRAIN}"
echo "  lora:      ${TEACHER_LORA_DIR}"
