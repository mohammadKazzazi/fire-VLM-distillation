#!/usr/bin/env bash
# Build Google-Drive-friendly zip packs for MobileCLIP teacher + student Phase 1.
# Preserves relative layout so scripts need no /home/... paths.
#
# Usage (on THIS machine):
#   bash shells/train/qwen3/vision_encoders/scripts/pack_mobileclip_handoff.sh
#   # outputs under: $WORKSPACE_ROOT/handoff_mobileclip/
#
# Then upload the zip files to Google Drive and download them on the other SSH server.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_ve.env.sh"

OUT="${HANDOFF_OUT:-${WORKSPACE_ROOT}/handoff_mobileclip}"
STAGE="${OUT}/stage"
mkdir -p "${STAGE}" "${OUT}"

echo "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUT=${OUT}"

# Ensure relative symlinks before packing
INSTRUCT="${WORKSPACE_ROOT}/LLaVA-Instruct-150K"
DET_REAL="${WORKSPACE_ROOT}/LLaVA_MoD/MoD/detectioumfire/real_images/real_images"
DET_SYN="${WORKSPACE_ROOT}/LLaVA_MoD/MoD/detectioumfire/synthetic_images/synthetic_images"
(
  cd "${INSTRUCT}"
  rm -f real_fire real_non_fire stable_diff_v15
  ln -s ../LLaVA_MoD/MoD/detectioumfire/real_images/real_images/real_fire real_fire
  ln -s ../LLaVA_MoD/MoD/detectioumfire/real_images/real_images/real_non_fire real_non_fire
  ln -s ../LLaVA_MoD/MoD/detectioumfire/synthetic_images/synthetic_images/stable_diff_v15 stable_diff_v15
)

# ---------- 1) CODE (small) ----------
echo "=== Pack 1/4: code ==="
CODE_LIST="${OUT}/code_filelist.txt"
cat > "${CODE_LIST}" <<EOF
LLaVA_MoD/MoD/LLava_MoD/llavamod
LLaVA_MoD/MoD/LLava_MoD/shells
LLaVA_MoD/MoD/LLava_MoD/requirements.txt
LLaVA_MoD/MoD/LLava_MoD/requirements-vision-encoders.txt
LLaVA_MoD/MoD/LLava_MoD/requirements-backbones-llama3-qwen3.txt
LLaVA_MoD/MoD/Important Notes
README_QWEN3_VE_A100.md
EOF
# Include pyproject/setup if present
for f in setup.py setup.cfg pyproject.toml README.md; do
  [[ -f "${REPO_ROOT}/${f}" ]] && echo "LLaVA_MoD/MoD/LLava_MoD/${f}" >> "${CODE_LIST}"
done

(
  cd "${WORKSPACE_ROOT}"
  zip -r -q "${OUT}/01_code.zip" \
    LLaVA_MoD/MoD/LLava_MoD/llavamod \
    LLaVA_MoD/MoD/LLava_MoD/shells \
    LLaVA_MoD/MoD/LLava_MoD/requirements.txt \
    LLaVA_MoD/MoD/LLava_MoD/requirements-vision-encoders.txt \
    LLaVA_MoD/MoD/LLava_MoD/requirements-backbones-llama3-qwen3.txt \
    "LLaVA_MoD/MoD/Important Notes" \
    README_QWEN3_VE_A100.md \
    -x '*/__pycache__/*' '*.pyc' '*/checkpoints/*' '*/logs/*' '*/cache_dir/*' \
       '*/.git/*' 'LLaVA_MoD/MoD/LLava_MoD/llavamod/data/PreTrain/Images/*'
)

# ---------- 2) PRETRAIN JSON + Images (~27G) — split zips ----------
echo "=== Pack 2/4: PreTrain data (large) ==="
(
  cd "${WORKSPACE_ROOT}"
  zip -r -q "${OUT}/02_pretrain_json.zip" \
    LLaVA_MoD/MoD/LLava_MoD/llavamod/data/PreTrain/blip_laion_cc_sbu_558k.json \
    LLaVA_MoD/MoD/LLava_MoD/llavamod/data/PreTrain/blip_laion_cc_sbu_558k_meta.json
)
# Split image archive for Drive (~2GB parts via zip -s)
(
  cd "${WORKSPACE_ROOT}"
  zip -r -s 2000m -q "${OUT}/02_pretrain_images.zip" \
    LLaVA_MoD/MoD/LLava_MoD/llavamod/data/PreTrain/Images \
    -x '*/.cache/*'
)

# ---------- 3) Teacher JSON + slim COCO (~3G referenced only) ----------
echo "=== Pack 3/4: slim COCO + teacher JSON ==="
SLIM_COCO="${STAGE}/LLaVA-Instruct-150K/coco/train2017"
mkdir -p "${SLIM_COCO}"
python3 - <<PY
import json, shutil
from pathlib import Path
ws = Path("${WORKSPACE_ROOT}")
src_root = ws / "LLaVA-Instruct-150K"
dst_root = Path("${STAGE}") / "LLaVA-Instruct-150K"
dst_coco = dst_root / "coco" / "train2017"
dst_coco.mkdir(parents=True, exist_ok=True)
with open(src_root / "teacher_training_data.json") as f:
    data = json.load(f)
n = 0
for r in data:
    img = r.get("image") or ""
    if not img.startswith("coco/train2017/"):
        continue
    src = src_root / img
    dst = dst_root / img
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        n += 1
print(f"Copied {n} referenced COCO images")
shutil.copy2(src_root / "teacher_training_data.json", dst_root / "teacher_training_data.json")
# relative symlinks inside stage (same relative targets as real tree)
for name, tgt in [
    ("real_fire", "../LLaVA_MoD/MoD/detectioumfire/real_images/real_images/real_fire"),
    ("real_non_fire", "../LLaVA_MoD/MoD/detectioumfire/real_images/real_images/real_non_fire"),
    ("stable_diff_v15", "../LLaVA_MoD/MoD/detectioumfire/synthetic_images/synthetic_images/stable_diff_v15"),
]:
    link = dst_root / name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(tgt)
PY
(
  cd "${STAGE}"
  zip -r -s 2000m -q "${OUT}/03_teacher_json_slim_coco.zip" LLaVA-Instruct-150K
)

# ---------- 4) Detectium images needed by teacher ----------
echo "=== Pack 4/4: Detectium real + synthetic ==="
(
  cd "${WORKSPACE_ROOT}"
  zip -r -s 2000m -q "${OUT}/04_detectium_images.zip" \
    LLaVA_MoD/MoD/detectioumfire/real_images/real_images/real_fire \
    LLaVA_MoD/MoD/detectioumfire/real_images/real_images/real_non_fire \
    LLaVA_MoD/MoD/detectioumfire/synthetic_images/synthetic_images/stable_diff_v15 \
    -x '*.cache*'
)

# Handoff README inside OUT
cp "${WORKSPACE_ROOT}/LLaVA_MoD/MoD/Important Notes/HANDOFF_MOBILECLIP_GOOGLE_DRIVE.md" \
  "${OUT}/README_UNPACK.md" 2>/dev/null || true

echo
echo "=== Done. Upload these from ${OUT}: ==="
ls -lh "${OUT}"/*.z* 2>/dev/null || ls -lh "${OUT}"
echo
echo "On the remote SSH server, unpack preserving relative paths (see HANDOFF doc)."
