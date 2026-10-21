#!/usr/bin/env bash
# Verify layout for Qwen2.5-3B MobileCLIP teacher + student + D2D.
#
# Usage:
#   bash shells/train/qwen/vision_encoders/scripts/verify_layout.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VE_SLUG="${VE_SLUG:-mobileclip_s2}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_ve.env.sh"

echo "=== Qwen2.5-3B MobileCLIP layout check ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
echo "VE_SLUG=${VE_SLUG}"
echo "VISION_ENCODER=${VISION_ENCODER}"
echo "TEACHER_LLM=${TEACHER_LLM}"
echo "STUDENT_LLM=${STUDENT_LLM}"
echo

fail=0
check() {
  local label="$1" path="$2"
  if [[ -e "${path}" ]]; then
    echo "OK   ${label}: ${path}"
  else
    echo "MISS ${label}: ${path}"
    fail=1
  fi
}

check "repo train.py" "${REPO_ROOT}/llavamod/train/train.py"
check "repo align_train.py" "${REPO_ROOT}/llavamod/train/align_train.py"
check "mobileclip encoder" "${REPO_ROOT}/llavamod/model/multimodal_encoder/mobileclip_encoder.py"
check "this VE scripts" "${REPO_ROOT}/shells/train/qwen/vision_encoders/scripts/run_full_pipeline.sh"
check "pretrain JSON" "${REPO_ROOT}/llavamod/data/PreTrain/blip_laion_cc_sbu_558k.json"
check "pretrain Images/" "${REPO_ROOT}/llavamod/data/PreTrain/Images"
check "teacher JSON" "${WORKSPACE_ROOT}/LLaVA-Instruct-150K/teacher_training_data.json"
check "D2D JSON" "${WORKSPACE_ROOT}/LLaVA-Instruct-150K/dense2dense_fire_nonbbox_v4.json"
check "Instruct root" "${WORKSPACE_ROOT}/LLaVA-Instruct-150K"
check "Detectium real_fire" "${WORKSPACE_ROOT}/LLaVA-Instruct-150K/real_fire"
check "Detectium real_non_fire" "${WORKSPACE_ROOT}/LLaVA-Instruct-150K/real_non_fire"
check "Detectium synthetic" "${WORKSPACE_ROOT}/LLaVA-Instruct-150K/stable_diff_v15"
check "eval launcher qwen2.5" "${EVAL_LAUNCHER}/eval_detectium_val_parallel.sh"
check "zero2.json" "${REPO_ROOT}/llavamod/config/dpconfig/zero2.json"
check "reqs vision" "${REPO_ROOT}/requirements-vision-encoders.txt"

for name in real_fire real_non_fire stable_diff_v15; do
  link="${WORKSPACE_ROOT}/LLaVA-Instruct-150K/${name}"
  if [[ -L "${link}" ]]; then
    tgt="$(readlink "${link}")"
    if [[ "${tgt}" = /* ]]; then
      echo "BAD  symlink ${name} is absolute: ${tgt}"
      fail=1
    else
      echo "OK   symlink ${name} is relative -> ${tgt}"
    fi
  elif [[ -e "${link}" ]]; then
    echo "OK   ${name} exists (not a symlink)"
  else
    echo "MISS ${name}"
    fail=1
  fi
done

python3 - <<'PY' || fail=1
import importlib.util, sys
ok = True
if importlib.util.find_spec("open_clip") is None:
    print("MISS python package open_clip (pip install -r requirements-vision-encoders.txt)")
    ok = False
else:
    print("OK   python package open_clip")
sys.exit(0 if ok else 1)
PY

echo
echo "Expected checkpoints (created by training, not required yet):"
echo "  ${TEACHER_PRETRAIN}"
echo "  ${TEACHER_LORA_DIR}/"
echo "  ${STUDENT_PROJECTOR}"
echo "  ${REPO_ROOT}/checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen2.5-3b-${VE_SHORT}-seed{42,123,456}/"
echo

if [[ "${fail}" -ne 0 ]]; then
  echo "FAILED — fix layout before training."
  exit 1
fi
echo "PASSED — layout OK for Qwen2.5-3B MobileCLIP."
