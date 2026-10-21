#!/usr/bin/env bash
# Run Plan 4 v4 kd_lm multi-seed for ALL Track-A students (+ optional VE 0.6B tracks).
#
# Default seeds: 123 456  (seed 42 already done for CLIP-family models)
# VE tracks (clip_vitl336 / mobileclip_s2) are included when their projectors exist.
#
# Usage (from MoD/LLava_MoD):
#   bash shells/train/run_all_models_multiseed.sh
#   SEEDS="42 123 456" bash shells/train/run_all_models_multiseed.sh
#   SEEDS="123 456" RUN_VAL=1 NUM_GPUS=4 bash shells/train/run_all_models_multiseed.sh
#   DRY_RUN=1 bash shells/train/run_all_models_multiseed.sh
#
# Detached:
#   nohup bash shells/train/run_all_models_multiseed.sh >> logs/all_models_multiseed.log 2>&1 &
#
# Env:
#   SEEDS="123 456"       — space-separated seeds (default)
#   RUN_VAL=1             — run val after each D2D (default 0)
#   SKIP_EXISTING=1       — skip if adapter_config.json exists (default 1)
#   INCLUDE_VE=1|0|auto   — Qwen3-0.6B CLIP/MobileCLIP tracks (default auto)
#   INCLUDE_CLIP_FAMILY=1 — Qwen2.5 / Gemma2 / Llama3 / Qwen3-4B (default 1)
#   DRY_RUN=1             — print plan only
#   NUM_GPUS=4            — passed through to training/val shells
#   ONLY="Llama3-1B,Qwen3-4B,VE-mobileclip_s2"  — optional filter (comma-separated names)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

SEEDS=(${SEEDS:-123 456})
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_VAL="${RUN_VAL:-0}"
INCLUDE_CLIP_FAMILY="${INCLUDE_CLIP_FAMILY:-1}"
INCLUDE_VE="${INCLUDE_VE:-auto}"
ONLY="${ONLY:-}"

# name | kind=clip_family|ve | plan4 script | val script or "-" | ckpt base (no -seed) | prereq file | extra env
CLIP_FAMILY_MODELS=(
  "Qwen2.5-3B|clip_family|shells/train/qwen/plan4_d2d_multiquery_v4.sh|shells/eval/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-kd-lm|checkpoints/llavaqwen2.5-3b-student-pretrain/mm_projector.bin|"
  "Qwen2.5-0.5B|clip_family|shells/train/qwen/student 0.5B/plan4_d2d_multiquery_v4.sh|shells/train/qwen/student 0.5B/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-kd-lm-0.5b|checkpoints/llavaqwen2.5-0.5b-student-pretrain/mm_projector.bin|"
  "Qwen2.5-1.5B|clip_family|shells/train/qwen/student 1.5B/plan4_d2d_multiquery_v4.sh|shells/train/qwen/student 1.5B/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-kd-lm-1.5b|checkpoints/llavaqwen2.5-1.5b-student-pretrain/mm_projector.bin|"
  "Gemma2-2B|clip_family|shells/train/gemma2/plan4_d2d_multiquery_v4.sh|shells/train/gemma2/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-kd-lm-gemma2-2b|checkpoints/llavagemma2-2b-student-pretrain/mm_projector.bin|"
  "Llama3-3B|clip_family|shells/train/llama3/plan4_d2d_multiquery_v4.sh|shells/train/llama3/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-kd-lm-llama3-3b|checkpoints/llavallama3.2-3b-student-pretrain/mm_projector.bin|"
  "Llama3-1B|clip_family|shells/train/llama3/student 1B/plan4_d2d_multiquery_v4.sh|shells/train/llama3/student 1B/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-kd-lm-llama3-1b|checkpoints/llavallama3.2-1b-student-pretrain/mm_projector.bin|"
  "Qwen3-4B|clip_family|shells/train/qwen3/plan4_d2d_multiquery_v4.sh|shells/train/qwen3/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-kd-lm-qwen3-4b|checkpoints/llavaqwen3-4b-student-pretrain/mm_projector.bin|"
)

VE_MODELS=(
  "VE-clip_vitl336|ve|shells/train/qwen3/vision_encoders/scripts/plan4_d2d_multiquery_v4.sh|shells/train/qwen3/vision_encoders/scripts/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-kd-lm-qwen3-0.6b-clip-vitl336|checkpoints/llavaqwen3-0.6b-student-pretrain-clip-vitl336/mm_projector.bin|VE_SLUG=clip_vitl336"
  "VE-mobileclip_s2|ve|shells/train/qwen3/vision_encoders/scripts/plan4_d2d_multiquery_v4.sh|shells/train/qwen3/vision_encoders/scripts/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-kd-lm-qwen3-0.6b-mobileclip-s2|checkpoints/llavaqwen3-0.6b-student-pretrain-mobileclip-s2/mm_projector.bin|VE_SLUG=mobileclip_s2"
)

MODELS=()
if [[ "${INCLUDE_CLIP_FAMILY}" == "1" ]]; then
  MODELS+=("${CLIP_FAMILY_MODELS[@]}")
fi

ve_projector_ready() {
  local path="$1"
  [[ -f "${REPO_ROOT}/${path}" ]]
}

if [[ "${INCLUDE_VE}" == "auto" ]]; then
  INCLUDE_VE=0
  for entry in "${VE_MODELS[@]}"; do
    IFS='|' read -r _n _k _p _v _c prereq _e <<< "${entry}"
    if ve_projector_ready "${prereq}"; then
      INCLUDE_VE=1
      break
    fi
  done
fi
if [[ "${INCLUDE_VE}" == "1" ]]; then
  MODELS+=("${VE_MODELS[@]}")
fi

want_name() {
  local name="$1"
  [[ -z "${ONLY}" ]] && return 0
  IFS=',' read -ra parts <<< "${ONLY}"
  for p in "${parts[@]}"; do
    [[ "${p}" == "${name}" ]] && return 0
  done
  return 1
}

has_checkpoint() {
  local out="$1"
  [[ -f "${out}/adapter_config.json" ]] || compgen -G "${out}/checkpoint-*" >/dev/null 2>&1
}

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source shells/conda_env.sh
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES

# Critical D2D JSON (addon pack)
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${REPO_ROOT}/../../.." && pwd)}"
D2D_JSON="${WORKSPACE_ROOT}/LLaVA-Instruct-150K/dense2dense_fire_nonbbox_v4.json"
if [[ ! -f "${D2D_JSON}" ]]; then
  echo "ERROR: missing D2D JSON: ${D2D_JSON}" >&2
  echo "Unpack addon pack 05 (see HANDOFF_MULTISEED_ADDON.md) into the same WORKSPACE_ROOT." >&2
  exit 1
fi

TOTAL=0
SKIPPED=0
PLANNED=0
FAILED=0

echo "============================================================"
echo "All-models Plan 4 v4 multi-seed"
echo "Started:          $(stamp)"
echo "Repo:             ${REPO_ROOT}"
echo "Workspace:        ${WORKSPACE_ROOT}"
echo "Seeds:            ${SEEDS[*]}"
echo "RUN_VAL:          ${RUN_VAL}"
echo "INCLUDE_CLIP:     ${INCLUDE_CLIP_FAMILY}"
echo "INCLUDE_VE:       ${INCLUDE_VE}"
echo "ONLY:             ${ONLY:-"(all)"}"
echo "SKIP_EXISTING:    ${SKIP_EXISTING}"
echo "NUM_GPUS:         ${NUM_GPUS:-"(script default)"}"
echo "============================================================"
echo

for SEED in "${SEEDS[@]}"; do
  echo "########## SEED=${SEED} ##########"
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r NAME KIND PLAN4_SCRIPT VAL_SCRIPT CKPT_BASE PREREQ EXTRA_ENV <<< "${entry}"
    want_name "${NAME}" || continue

    PLAN4_PATH="${REPO_ROOT}/${PLAN4_SCRIPT}"
    PREREQ_PATH="${REPO_ROOT}/${PREREQ}"
    OUTPUT_DIR="${REPO_ROOT}/checkpoints/${CKPT_BASE}-seed${SEED}"
    TOTAL=$((TOTAL + 1))

    if [[ ! -f "${PLAN4_PATH}" ]]; then
      echo "[SKIP] ${NAME} seed=${SEED}: missing script ${PLAN4_SCRIPT}"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
    if [[ ! -f "${PREREQ_PATH}" ]]; then
      echo "[SKIP] ${NAME} seed=${SEED}: missing projector ${PREREQ}"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
    if [[ "${SKIP_EXISTING}" == "1" ]] && has_checkpoint "${OUTPUT_DIR}"; then
      echo "[SKIP] ${NAME} seed=${SEED}: exists → ${OUTPUT_DIR}"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    PLANNED=$((PLANNED + 1))
    RUN_LOG="${LOG_DIR}/multiseed_${CKPT_BASE}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

    echo
    echo "------------------------------------------------------------"
    echo "[RUN]  ${NAME} (${KIND})  SEED=${SEED}"
    echo "       out: ${OUTPUT_DIR}"
    echo "       log: ${RUN_LOG}"
    echo "------------------------------------------------------------"

    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    # shellcheck disable=SC2086
    export SEED="${SEED}"
    export OUTPUT_DIR="${OUTPUT_DIR}"
    if [[ -n "${EXTRA_ENV}" ]]; then
      # e.g. VE_SLUG=mobileclip_s2
      export ${EXTRA_ENV}
    fi

    set +e
    if [[ "${KIND}" == "ve" ]]; then
      VE_SLUG="${VE_SLUG}" SEED="${SEED}" \
        bash "${PLAN4_PATH}" 2>&1 | tee "${RUN_LOG}"
      rc=${PIPESTATUS[0]}
    else
      SEED="${SEED}" OUTPUT_DIR="${OUTPUT_DIR}" DETACH=0 \
        bash "${PLAN4_PATH}" 2>&1 | tee "${RUN_LOG}"
      rc=${PIPESTATUS[0]}
    fi
    set -e

    if [[ "${rc}" -ne 0 ]] || ! has_checkpoint "${OUTPUT_DIR}"; then
      echo "[FAIL] ${NAME} seed=${SEED} (rc=${rc})" >&2
      FAILED=$((FAILED + 1))
      exit 1
    fi
    echo "[OK]   ${NAME} seed=${SEED}"

    if [[ "${RUN_VAL}" == "1" && "${VAL_SCRIPT}" != "-" ]]; then
      VAL_PATH="${REPO_ROOT}/${VAL_SCRIPT}"
      if [[ -f "${VAL_PATH}" ]]; then
        echo "[VAL]  ${NAME} seed=${SEED}"
        if [[ "${KIND}" == "ve" ]]; then
          NUM_GPUS="${NUM_GPUS:-4}" VE_SLUG="${VE_SLUG}" SEED="${SEED}" \
            bash "${VAL_PATH}" 2>&1 | tee -a "${RUN_LOG}"
        else
          MODEL_PATH="${OUTPUT_DIR}" NUM_GPUS="${NUM_GPUS:-4}" \
            bash "${VAL_PATH}" 2>&1 | tee -a "${RUN_LOG}"
        fi
      else
        echo "[WARN] val script missing: ${VAL_SCRIPT}"
      fi
    fi
  done
  echo
done

echo "============================================================"
echo "Finished: $(stamp)"
echo "Slots seen: ${TOTAL}  Planned: ${PLANNED}  Skipped: ${SKIPPED}  Failed: ${FAILED}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 — nothing trained."
fi
echo "============================================================"
