#!/usr/bin/env bash
# Run all three vision-encoder tracks sequentially (teacher then student for each).
#
# Usage:
#   cd MoD/LLava_MoD
#   bash shells/train/qwen3/vision_encoders/run_all_ve_ablation.sh
#
# Optional:
#   PHASE=teacher|student|all  (default: all)
#   SEED=42
#   SKIP_TEACHER=1  (if teachers already trained)
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="${ROOT}/scripts"
PHASE="${PHASE:-all}"
SEED="${SEED:-42}"
export SEED

for VE in clip_vitl336 siglip2_b384 mobileclip_s2; do
  export VE_SLUG="${VE}"
  echo ""
  echo "############################################################"
  echo "# Vision encoder track: ${VE_SLUG}"
  echo "############################################################"
  if [[ "${PHASE}" == "teacher" || "${PHASE}" == "all" ]] && [[ "${SKIP_TEACHER:-0}" != "1" ]]; then
    bash "${SCRIPTS}/run_teacher_pipeline.sh"
  fi
  if [[ "${PHASE}" == "student" || "${PHASE}" == "all" ]]; then
    bash "${SCRIPTS}/run_student_pipeline.sh"
  fi
done

echo ""
echo "All requested VE tracks finished. See Important Notes/VISION_ENCODER_ROADMAP.md"
