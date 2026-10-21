#!/usr/bin/env bash
# Verify portable layout for MobileCLIP teacher + student Phase 1.
# Run from anywhere after unpacking the handoff tree.
#
# Usage:
#   bash shells/train/qwen3/vision_encoders/scripts/verify_portable_layout.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_ve.env.sh"

echo "=== Portable layout check ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
echo "VE_SLUG default would be: ${VE_SLUG}"
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
check "mobileclip encoder" "${REPO_ROOT}/llavamod/model/multimodal_encoder/mobileclip_encoder.py"
check "VE scripts" "${REPO_ROOT}/shells/train/qwen3/vision_encoders/scripts/run_teacher_pipeline.sh"
check "pretrain JSON" "${REPO_ROOT}/llavamod/data/PreTrain/blip_laion_cc_sbu_558k.json"
check "pretrain Images/" "${REPO_ROOT}/llavamod/data/PreTrain/Images"
check "teacher JSON" "${WORKSPACE_ROOT}/LLaVA-Instruct-150K/teacher_training_data.json"
check "Instruct root" "${WORKSPACE_ROOT}/LLaVA-Instruct-150K"
check "Detectium real_fire (via relative symlink)" "${WORKSPACE_ROOT}/LLaVA-Instruct-150K/real_fire"
check "Detectium real_non_fire" "${WORKSPACE_ROOT}/LLaVA-Instruct-150K/real_non_fire"
check "Detectium synthetic" "${WORKSPACE_ROOT}/LLaVA-Instruct-150K/stable_diff_v15"
check "coco train2017" "${WORKSPACE_ROOT}/LLaVA-Instruct-150K/coco/train2017"
check "reqs vision" "${REPO_ROOT}/requirements-vision-encoders.txt"

# Symlinks must be relative (no /home/... absolute targets)
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
  else
    echo "MISS symlink ${name}"
    fail=1
  fi
done

# Spot-check a few teacher images resolve
python3 - <<'PY' || fail=1
import json, os, sys
from pathlib import Path
ws = Path(os.environ["WORKSPACE_ROOT"])
root = ws / "LLaVA-Instruct-150K"
with open(root / "teacher_training_data.json") as f:
    data = json.load(f)
ok = bad = 0
samples = []
for r in data:
    img = r.get("image")
    if not img:
        continue
    p = root / img
    if p.exists():
        ok += 1
        if len(samples) < 3:
            samples.append(str(img))
    else:
        bad += 1
        if bad <= 5:
            print(f"MISS image: {img}")
print(f"Teacher image resolve: ok={ok} bad={bad} (sampled {samples})")
if bad:
    sys.exit(1)
PY

echo
if [[ "${fail}" -ne 0 ]]; then
  echo "FAILED — fix layout before training."
  echo "Expected tree:"
  echo "  <WORKSPACE_ROOT>/"
  echo "    LLaVA-Instruct-150K/   (JSON + coco + relative symlinks)"
  echo "    LLaVA_MoD/MoD/LLava_MoD/"
  echo "    LLaVA_MoD/MoD/detectioumfire/  (real + synthetic images)"
  exit 1
fi
echo "PASSED — portable layout OK. Paths are relative to unpacked tree."
