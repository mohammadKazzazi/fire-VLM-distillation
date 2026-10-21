#!/usr/bin/env bash
# Plan 4 v4 D2D ablation: LOSS_TYPE=only_kd @ seed 42 for all CLIP-family students.
#
# Bypasses plan4_d2d_multiquery_v4.sh wrappers (those reject only_kd) and calls each
# family's dense2dense_distillation*.sh directly.
#
# Skip rule: if an only_kd checkpoint already exists (e.g. historical Gemma2), skip it.
# If missing, train it too.
#
# Usage (from MoD/LLava_MoD on the GPU training server):
#   bash shells/train/run_all_models_only_kd_seed42.sh
#   DRY_RUN=1 bash shells/train/run_all_models_only_kd_seed42.sh
#   ONLY="Qwen2.5-0.5B,Llama3-1B,Gemma2-2B" bash shells/train/run_all_models_only_kd_seed42.sh
#   RUN_VAL=1 NUM_GPUS=4 bash shells/train/run_all_models_only_kd_seed42.sh
#
# Detached:
#   nohup bash shells/train/run_all_models_only_kd_seed42.sh >> logs/all_models_only_kd_seed42.log 2>&1 &
#
# Env:
#   SEED=42                 — default (override only if intentional)
#   SKIP_EXISTING=1         — skip when adapter_config.json exists (default 1)
#   RUN_VAL=0               — optional Detectium val after each D2D
#   INCLUDE_VE=0|1|auto     — also Qwen3-0.6B CLIP/MobileCLIP tracks (default 0)
#   ONLY="Gemma2-2B,..."    — optional name filter
#   DRY_RUN=1               — print plan only
#   NUM_GPUS=4              — passed to val scripts when RUN_VAL=1
#
# Note: only_kd typically does NOT learn Yes./No. prefixes. Prefer content-based
# scoring (yesno_content) for val, not prefix accuracy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

SEED="${SEED:-42}"
LOSS_TYPE="${LOSS_TYPE:-only_kd}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_VAL="${RUN_VAL:-0}"
INCLUDE_VE="${INCLUDE_VE:-0}"
ONLY="${ONLY:-}"

if [[ "${LOSS_TYPE}" != "only_kd" ]]; then
  echo "ERROR: this runner is for only_kd (got LOSS_TYPE=${LOSS_TYPE})" >&2
  exit 1
fi

# name | d2d script | val script or "-" | ckpt base (no -seed) | projector prereq | extra env
CLIP_FAMILY_MODELS=(
  "Qwen2.5-3B|shells/train/qwen/dense2dense_distillation.sh|shells/eval/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-only-kd|checkpoints/llavaqwen2.5-3b-student-pretrain/mm_projector.bin|"
  "Qwen2.5-0.5B|shells/train/qwen/student 0.5B/dense2dense_distillation_0.5b.sh|shells/train/qwen/student 0.5B/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-only-kd-0.5b|checkpoints/llavaqwen2.5-0.5b-student-pretrain/mm_projector.bin|"
  "Qwen2.5-1.5B|shells/train/qwen/student 1.5B/dense2dense_distillation_1.5b.sh|shells/train/qwen/student 1.5B/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-only-kd-1.5b|checkpoints/llavaqwen2.5-1.5b-student-pretrain/mm_projector.bin|"
  "Gemma2-2B|shells/train/gemma2/dense2dense_distillation.sh|shells/train/gemma2/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-only-kd-gemma2-2b|checkpoints/llavagemma2-2b-student-pretrain/mm_projector.bin|"
  "Llama3-3B|shells/train/llama3/dense2dense_distillation.sh|shells/train/llama3/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-only-kd-llama3-3b|checkpoints/llavallama3.2-3b-student-pretrain/mm_projector.bin|"
  "Llama3-1B|shells/train/llama3/student 1B/dense2dense_distillation_1b.sh|shells/train/llama3/student 1B/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-only-kd-llama3-1b|checkpoints/llavallama3.2-1b-student-pretrain/mm_projector.bin|"
  "Qwen3-4B|shells/train/qwen3/dense2dense_distillation.sh|shells/train/qwen3/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-only-kd-qwen3-4b|checkpoints/llavaqwen3-4b-student-pretrain/mm_projector.bin|"
)

VE_MODELS=(
  "VE-clip_vitl336|shells/train/qwen3/vision_encoders/scripts/plan4_d2d_multiquery_v4.sh|shells/train/qwen3/vision_encoders/scripts/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-only-kd-qwen3-0.6b-clip-vitl336|checkpoints/llavaqwen3-0.6b-student-pretrain-clip-vitl336/mm_projector.bin|VE_SLUG=clip_vitl336"
  "VE-mobileclip_s2|shells/train/qwen3/vision_encoders/scripts/plan4_d2d_multiquery_v4.sh|shells/train/qwen3/vision_encoders/scripts/plan4_d2d_multiquery_v4_val.sh|plan4-d2d-multiquery-v4-only-kd-qwen3-0.6b-mobileclip-s2|checkpoints/llavaqwen3-0.6b-student-pretrain-mobileclip-s2/mm_projector.bin|VE_SLUG=mobileclip_s2"
)

MODELS=("${CLIP_FAMILY_MODELS[@]}")

ve_projector_ready() {
  local path="$1"
  [[ -f "${REPO_ROOT}/${path}" ]]
}

if [[ "${INCLUDE_VE}" == "auto" ]]; then
  INCLUDE_VE=0
  for entry in "${VE_MODELS[@]}"; do
    IFS='|' read -r _n _s _v _c prereq _e <<< "${entry}"
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

# Also treat legacy Gemma2 naming as "already have only_kd" if present.
legacy_only_kd_dirs() {
  local name="$1"
  case "${name}" in
    Gemma2-2B)
      printf '%s\n' \
        "${REPO_ROOT}/checkpoints/plan4-d2d-multiquery-v4-only-kd-gemma2-2b-seed${SEED}" \
        "${REPO_ROOT}/checkpoints/plan4-d2d-multiquery-v4-only_kd-gemma2-2b-seed${SEED}" \
        "${REPO_ROOT}/checkpoints/plan4-d2d-only-kd-gemma2-2b-seed${SEED}" \
        "${REPO_ROOT}/checkpoints/plan4-d2d-only_kd-gemma2-2b-seed${SEED}"
      ;;
  esac
}

already_have_only_kd() {
  local name="$1"
  local primary="$2"
  if has_checkpoint "${primary}"; then
    return 0
  fi
  local d
  while IFS= read -r d; do
    [[ -z "${d}" ]] && continue
    if has_checkpoint "${d}"; then
      echo "[SKIP] ${name}: legacy only_kd found → ${d}"
      return 0
    fi
  done < <(legacy_only_kd_dirs "${name}")
  return 1
}

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source shells/conda_env.sh
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${REPO_ROOT}/../../.." && pwd)}"
D2D_JSON="${JSON_FILE:-${WORKSPACE_ROOT}/LLaVA-Instruct-150K/dense2dense_fire_nonbbox_v4.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${WORKSPACE_ROOT}/LLaVA-Instruct-150K}"
if [[ ! -f "${D2D_JSON}" ]]; then
  echo "ERROR: missing D2D JSON: ${D2D_JSON}" >&2
  exit 1
fi
export JSON_FILE="${D2D_JSON}"
export IMAGE_FOLDER

TOTAL=0
SKIPPED=0
PLANNED=0
FAILED=0

echo "============================================================"
echo "All-models Plan 4 v4 only_kd @ seed ${SEED}"
echo "Started:          $(stamp)"
echo "Repo:             ${REPO_ROOT}"
echo "Workspace:        ${WORKSPACE_ROOT}"
echo "LOSS_TYPE:        ${LOSS_TYPE}"
echo "JSON:             ${JSON_FILE}"
echo "RUN_VAL:          ${RUN_VAL}"
echo "INCLUDE_VE:       ${INCLUDE_VE}"
echo "ONLY:             ${ONLY:-"(all CLIP families)"}"
echo "SKIP_EXISTING:    ${SKIP_EXISTING}"
echo "NUM_GPUS:         ${NUM_GPUS:-"(script default)"}"
echo "============================================================"
echo

for entry in "${MODELS[@]}"; do
  IFS='|' read -r NAME D2D_SCRIPT VAL_SCRIPT CKPT_BASE PREREQ EXTRA_ENV <<< "${entry}"
  want_name "${NAME}" || continue

  D2D_PATH="${REPO_ROOT}/${D2D_SCRIPT}"
  PREREQ_PATH="${REPO_ROOT}/${PREREQ}"
  OUTPUT_DIR="${REPO_ROOT}/checkpoints/${CKPT_BASE}-seed${SEED}"
  TOTAL=$((TOTAL + 1))

  if [[ ! -f "${D2D_PATH}" ]]; then
    echo "[SKIP] ${NAME}: missing script ${D2D_SCRIPT}"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi
  if [[ ! -f "${PREREQ_PATH}" ]]; then
    echo "[SKIP] ${NAME}: missing projector ${PREREQ}"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi
  if [[ "${SKIP_EXISTING}" == "1" ]] && already_have_only_kd "${NAME}" "${OUTPUT_DIR}"; then
    if has_checkpoint "${OUTPUT_DIR}"; then
      echo "[SKIP] ${NAME}: exists → ${OUTPUT_DIR}"
    fi
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  PLANNED=$((PLANNED + 1))
  RUN_LOG="${LOG_DIR}/only_kd_${CKPT_BASE}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

  echo
  echo "------------------------------------------------------------"
  echo "[RUN]  ${NAME}  SEED=${SEED}  LOSS_TYPE=${LOSS_TYPE}"
  echo "       out: ${OUTPUT_DIR}"
  echo "       log: ${RUN_LOG}"
  echo "------------------------------------------------------------"

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  mkdir -p "${OUTPUT_DIR}"
  export SEED LOSS_TYPE OUTPUT_DIR JSON_FILE IMAGE_FOLDER
  if [[ -n "${EXTRA_ENV}" ]]; then
    # shellcheck disable=SC2086
    export ${EXTRA_ENV}
  fi

  set +e
  if [[ "${NAME}" == VE-* ]]; then
    # VE plan4 script accepts LOSS_TYPE / but hardcodes kd_lm OUTPUT_DIR via common_ve —
    # force OUTPUT_DIR and LOSS_TYPE into the env; VE script uses STUDENT_D2D_DIR unless overridden.
    VE_SLUG="${VE_SLUG}" SEED="${SEED}" LOSS_TYPE="${LOSS_TYPE}" OUTPUT_DIR="${OUTPUT_DIR}" \
      bash "${D2D_PATH}" 2>&1 | tee "${RUN_LOG}"
    rc=${PIPESTATUS[0]}
  else
    SEED="${SEED}" LOSS_TYPE="${LOSS_TYPE}" OUTPUT_DIR="${OUTPUT_DIR}" \
    JSON_FILE="${JSON_FILE}" IMAGE_FOLDER="${IMAGE_FOLDER}" \
      bash "${D2D_PATH}" 2>&1 | tee "${RUN_LOG}"
    rc=${PIPESTATUS[0]}
  fi
  set -e

  if [[ "${rc}" -ne 0 ]] || ! has_checkpoint "${OUTPUT_DIR}"; then
    echo "[FAIL] ${NAME} (rc=${rc})" >&2
    FAILED=$((FAILED + 1))
    exit 1
  fi
  echo "[OK]   ${NAME}"

  if [[ "${RUN_VAL}" == "1" && "${VAL_SCRIPT}" != "-" ]]; then
    VAL_PATH="${REPO_ROOT}/${VAL_SCRIPT}"
    if [[ -f "${VAL_PATH}" ]]; then
      echo "[VAL]  ${NAME}  (score with yesno_content — only_kd often lacks Yes./No. prefixes)"
      if [[ "${NAME}" == VE-* ]]; then
        NUM_GPUS="${NUM_GPUS:-4}" VE_SLUG="${VE_SLUG}" SEED="${SEED}" MODEL_PATH="${OUTPUT_DIR}" \
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
echo "============================================================"
echo "Finished: $(stamp)"
echo "Slots seen: ${TOTAL}  Planned: ${PLANNED}  Skipped: ${SKIPPED}  Failed: ${FAILED}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 — nothing trained."
fi
echo "============================================================"
