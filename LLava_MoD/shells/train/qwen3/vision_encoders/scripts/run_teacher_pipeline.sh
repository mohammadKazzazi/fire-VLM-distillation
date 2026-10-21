#!/usr/bin/env bash
# Run full teacher pipeline for one vision encoder.
#
# clip_vitl336 reuses legacy llavaqwen3-8b-pretrain + llavaqwen3-8b-finetune-lora (no retrain).
#
# Usage:
#   VE_SLUG=siglip2_b384 bash scripts/run_teacher_pipeline.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VE_SLUG="${VE_SLUG:?Set VE_SLUG=clip_vitl336|siglip2_b384|mobileclip_s2}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_ve.env.sh"

if [[ "${TEACHER_REUSE_LEGACY:-0}" == "1" ]]; then
  [[ -f "${TEACHER_PRETRAIN}" ]] || { echo "Missing legacy teacher projector: ${TEACHER_PRETRAIN}"; exit 1; }
  [[ -f "${TEACHER_LORA}" ]] || { echo "Missing legacy teacher LoRA: ${TEACHER_LORA}"; exit 1; }
  echo "Teacher reuse (CLIP-L): ${TEACHER_LORA_DIR} — skipping pretrain + LoRA SFT"
  exit 0
fi

export VE_SLUG
bash "${SCRIPT_DIR}/pretrain_teacher.sh"
bash "${SCRIPT_DIR}/finetune_lora.sh"

echo "Teacher pipeline complete for VE=${VE_SLUG}"
