#!/usr/bin/env bash
# Plan 4 v4 — Qwen3-4B student, val-aligned multitask D2D with kd_lm (46,026 rows).
#
# Usage:
#   cd MoD/LLava_MoD
#   bash shells/train/qwen3/plan4_d2d_multiquery_v4.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_qwen3.env.sh"

JSON_FILE="${JSON_FILE:-${WORKSPACE_ROOT}/LLaVA-Instruct-150K/dense2dense_fire_nonbbox_v4.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${WORKSPACE_ROOT}/LLaVA-Instruct-150K}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen3-4b-seed42}"
LOSS_TYPE="${LOSS_TYPE:-kd_lm}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "${JSON_FILE}" ]] || die "Missing Plan 4 v4 JSON: ${JSON_FILE}"
[[ "${LOSS_TYPE}" == "kd_lm" ]] || die "Plan 4 expects LOSS_TYPE=kd_lm (got ${LOSS_TYPE})"
[[ -f "${TEACHER_LORA}" ]] || die "Missing teacher LoRA: ${TEACHER_LORA}"
[[ -f "${STUDENT_PROJECTOR}" ]] || die "Missing student projector: ${STUDENT_PROJECTOR}"

cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source shells/conda_env.sh
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

JSON_FILE="${JSON_FILE}" \
IMAGE_FOLDER="${IMAGE_FOLDER}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
LOSS_TYPE="${LOSS_TYPE}" \
  bash "${D2D_SCRIPT}"
