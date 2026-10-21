#!/usr/bin/env bash
# Build ADDON packs for multi-seed D2D that merge cleanly with packs 01–04.
#
# Does NOT re-pack PreTrain images / Detectium / code already sent.
# Unpack with:  unzip -n 05_....zip -d "$WS"   # -n = never overwrite
#
# Usage:
#   bash shells/train/qwen3/vision_encoders/scripts/pack_multiseed_addon.sh
#
# Optional (large): also pack slim teacher LoRAs + student projectors:
#   INCLUDE_CHECKPOINTS=1 bash .../pack_multiseed_addon.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_ve.env.sh"

OUT="${HANDOFF_OUT:-${WORKSPACE_ROOT}/handoff_multiseed_addon}"
mkdir -p "${OUT}"
INCLUDE_CHECKPOINTS="${INCLUDE_CHECKPOINTS:-0}"

echo "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "OUT=${OUT}"

# ---------- 05: D2D JSON + eval + multi-seed scripts (small) ----------
echo "=== Pack 05: D2D JSON + eval + multiseed scripts ==="
(
  cd "${WORKSPACE_ROOT}"
  zip -r -q "${OUT}/05_d2d_multiseed_addon.zip" \
    LLaVA-Instruct-150K/dense2dense_fire_nonbbox_v4.json \
    LLaVA_MoD/MoD/detectioumfire/detectium_llavamod_real_full_val.json \
    LLaVA_MoD/MoD/detectioumfire/export_val_questions_jsonl.py \
    LLaVA_MoD/MoD/detectioumfire/val_questions.jsonl \
    LLaVA_MoD/MoD/detectioumfire/eval \
    LLaVA_MoD/MoD/LLava_MoD/shells/train/run_all_models_multiseed.sh \
    LLaVA_MoD/MoD/LLava_MoD/shells/train/run_all_plan4_remaining_seeds.sh \
    LLaVA_MoD/MoD/LLava_MoD/shells/train/qwen3/vision_encoders \
    "LLaVA_MoD/MoD/Important Notes/HANDOFF_MULTISEED_ADDON.md" \
    -x '*/__pycache__/*' '*.pyc' '*/runs/*' '*/.cache/*'
)

if [[ "${INCLUDE_CHECKPOINTS}" == "1" ]]; then
  echo "=== Pack 06: slim teacher LoRAs (no checkpoint-* dirs) ==="
  STAGE="${OUT}/stage_ckpts"
  rm -rf "${STAGE}"
  mkdir -p "${STAGE}/LLaVA_MoD/MoD/LLava_MoD/checkpoints"

  copy_slim_lora() {
    local src_name="$1"
    local src="${REPO_ROOT}/checkpoints/${src_name}"
    local dst="${STAGE}/LLaVA_MoD/MoD/LLava_MoD/checkpoints/${src_name}"
    [[ -d "${src}" ]] || { echo "WARN missing ${src_name}"; return; }
    mkdir -p "${dst}"
    for f in adapter_config.json adapter_model.bin non_lora_trainables.bin config.json README.md trainer_state.json; do
      [[ -f "${src}/${f}" ]] && cp -a "${src}/${f}" "${dst}/"
    done
    echo "  slim ${src_name}"
  }

  copy_slim_lora llavaqwen2.5-7b-finetune-lora
  copy_slim_lora llavagemma2-9b-finetune-lora
  copy_slim_lora llavallama3.1-8b-finetune-lora
  copy_slim_lora llavaqwen3-8b-finetune-lora

  echo "=== Pack 07: student projectors ==="
  for d in \
    llavaqwen2.5-0.5b-student-pretrain \
    llavaqwen2.5-1.5b-student-pretrain \
    llavaqwen2.5-3b-student-pretrain \
    llavagemma2-2b-student-pretrain \
    llavallama3.2-1b-student-pretrain \
    llavallama3.2-3b-student-pretrain \
    llavaqwen3-4b-student-pretrain
  do
    src="${REPO_ROOT}/checkpoints/${d}"
    if [[ -f "${src}/mm_projector.bin" ]]; then
      mkdir -p "${STAGE}/LLaVA_MoD/MoD/LLava_MoD/checkpoints/${d}"
      cp -a "${src}/mm_projector.bin" "${STAGE}/LLaVA_MoD/MoD/LLava_MoD/checkpoints/${d}/"
      # copy small sidecars if present
      for f in config.json README.md; do
        [[ -f "${src}/${f}" ]] && cp -a "${src}/${f}" "${STAGE}/LLaVA_MoD/MoD/LLava_MoD/checkpoints/${d}/"
      done
      echo "  projector ${d}"
    fi
  done

  (
    cd "${STAGE}"
    zip -r -s 2000m -q "${OUT}/06_teacher_loras_slim.zip" LLaVA_MoD/MoD/LLava_MoD/checkpoints \
      -i 'LLaVA_MoD/MoD/LLava_MoD/checkpoints/llava*-finetune-lora/*'
    zip -r -q "${OUT}/07_student_projectors.zip" LLaVA_MoD/MoD/LLava_MoD/checkpoints \
      -i 'LLaVA_MoD/MoD/LLava_MoD/checkpoints/*student-pretrain*/*'
  )
fi

cp "${WORKSPACE_ROOT}/LLaVA_MoD/MoD/Important Notes/HANDOFF_MULTISEED_ADDON.md" \
  "${OUT}/README_MERGE.md" 2>/dev/null || true

echo
echo "=== Done. Upload from ${OUT}: ==="
ls -lh "${OUT}"/* 2>/dev/null | grep -v stage || ls -lh "${OUT}"
echo
echo "On remote: unzip -n 05_d2d_multiseed_addon.zip -d \"\$WS\""
echo "(-n = never overwrite files from packs 01–04)"
