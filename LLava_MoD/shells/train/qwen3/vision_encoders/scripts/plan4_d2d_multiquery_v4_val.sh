#!/usr/bin/env bash
# Eval Plan 4 v4 (Qwen3-0.6B) for a vision-encoder track.
#
# Usage:
#   VE_SLUG=clip_vitl336 SEED=42 bash scripts/plan4_d2d_multiquery_v4_val.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_ve.env.sh"

MODEL_PATH="${MODEL_PATH:-${STUDENT_D2D_DIR}}"
MODEL_BASE="${MODEL_BASE:-${STUDENT_LLM}}"
ANSWERS_OUT="${ANSWERS_OUT:-${EVAL_ANSWERS}}"
NUM_GPUS="${NUM_GPUS:-8}"

[[ -f "${MODEL_PATH}/adapter_config.json" || -f "${MODEL_PATH}/config.json" ]] \
  || { echo "ERROR: checkpoint not found: ${MODEL_PATH}"; exit 1; }

echo "=== Val eval | VE=${VE_SLUG} | SEED=${SEED} ==="
echo "MODEL_PATH=${MODEL_PATH}"
echo "ANSWERS_OUT=${ANSWERS_OUT}"

cd "${EVAL_LAUNCHER}"
MODEL_PATH="${MODEL_PATH}" MODEL_BASE="${MODEL_BASE}" \
ANSWERS_OUT="${ANSWERS_OUT}" NUM_GPUS="${NUM_GPUS}" CONV_MODE="${CONV_MODE:-qwen3}" \
  ./eval_detectium_val_parallel.sh
