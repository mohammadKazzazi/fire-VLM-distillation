#!/usr/bin/env bash
# Plan 4 v4 — Llama-3.2-1B edge student, val-aligned multitask D2D with kd_lm (46,026 rows).
#
# Usage:
#   cd MoD/LLava_MoD
#   bash "shells/train/llama3/student 1B/plan4_d2d_multiquery_v4.sh"
#
# Prerequisite:
#   bash "shells/train/llama3/student 1B/pretrain_student_1b.sh"
#
# Monitor:
#   tail -f logs/plan4_d2d_multiquery_v4_kd_lm_llama3_1b_latest.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_1b.env.sh"

JSON_FILE="${JSON_FILE:-${WORKSPACE_ROOT}/LLaVA-Instruct-150K/dense2dense_fire_nonbbox_v4.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${WORKSPACE_ROOT}/LLaVA-Instruct-150K}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/plan4-d2d-multiquery-v4-kd-lm-llama3-1b-seed42}"
LOSS_TYPE="${LOSS_TYPE:-kd_lm}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "${JSON_FILE}" ]] || die "Missing Plan 4 v4 JSON: ${JSON_FILE}"
[[ "${LOSS_TYPE}" == "kd_lm" ]] || die "Plan 4 expects LOSS_TYPE=kd_lm (got ${LOSS_TYPE})"
[[ -f "${TEACHER_LORA}" ]] || die "Missing teacher LoRA: ${TEACHER_LORA}"
[[ -f "${STUDENT_PROJECTOR}" ]] || die "Missing 1B student projector: ${STUDENT_PROJECTOR} (run pretrain_student_1b.sh first)"

if compgen -G "${OUTPUT_DIR}/checkpoint-*" >/dev/null 2>&1; then
  echo "NOTE: ${OUTPUT_DIR} has checkpoint-* — align_train.py will resume from latest."
else
  mkdir -p "${OUTPUT_DIR}"
  echo "Fresh Plan 4 v4 (1B, kd_lm) run → ${OUTPUT_DIR}"
fi

ROWS="$(python3 -c "import json; print(len(json.load(open('${JSON_FILE}'))))")"
STEPS=$(( (ROWS + 63) / 64 ))
echo "=== Plan 4 D2D (Llama3-1B) — Multi-query v4 + kd_lm ==="
echo "JSON:       ${JSON_FILE} (${ROWS} rows)"
echo "LOSS_TYPE:  ${LOSS_TYPE}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "Est. steps: ~${STEPS}/epoch"
echo

DETACH="${DETACH:-1}"
if [[ "${DETACH}" == "1" && -z "${PLAN4_KD_LM_DETACHED:-}" ]]; then
  LOG_DIR="${REPO_ROOT}/logs"
  mkdir -p "${LOG_DIR}"
  STAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/plan4_d2d_multiquery_v4_kd_lm_llama3_1b_${STAMP}.log"
  ln -sfn "$(basename "${LOG_FILE}")" "${LOG_DIR}/plan4_d2d_multiquery_v4_kd_lm_llama3_1b_latest.log"
  PID_FILE="${LOG_DIR}/plan4_d2d_multiquery_v4_kd_lm_llama3_1b.pid"
  echo "Detaching — safe to close terminal/Cursor."
  echo "  log:  ${LOG_FILE}"
  echo "  tail: tail -f ${LOG_DIR}/plan4_d2d_multiquery_v4_kd_lm_llama3_1b_latest.log"
  PLAN4_KD_LM_DETACHED=1 DETACH=0 nohup env PLAN4_KD_LM_DETACHED=1 DETACH=0 \
    JSON_FILE="${JSON_FILE}" IMAGE_FOLDER="${IMAGE_FOLDER}" OUTPUT_DIR="${OUTPUT_DIR}" LOSS_TYPE="${LOSS_TYPE}" \
    bash "${BASH_SOURCE[0]}" >>"${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
  echo "  pid:  $(cat "${PID_FILE}") (also in ${PID_FILE})"
  exit 0
fi

cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source shells/conda_env.sh
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

JSON_FILE="${JSON_FILE}" \
IMAGE_FOLDER="${IMAGE_FOLDER}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
LOSS_TYPE="${LOSS_TYPE}" \
  bash "${D2D_SCRIPT}"
