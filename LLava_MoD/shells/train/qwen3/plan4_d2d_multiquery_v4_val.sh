#!/usr/bin/env bash
# Eval Plan 4 v4 (Qwen3-4B) kd_lm checkpoint on Detectium val.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_qwen3.env.sh"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen3-4b-seed42}"
MODEL_BASE="${MODEL_BASE:-${STUDENT_LLM}}"
ANSWERS_OUT="${ANSWERS_OUT:-${EVAL_RUNS}/students/qwen3-4b/plan4-v4-kd-lm-seed42/val_answers.jsonl}"
NUM_GPUS="${NUM_GPUS:-8}"

[[ -f "${MODEL_PATH}/adapter_config.json" || -f "${MODEL_PATH}/config.json" ]] \
  || { echo "ERROR: checkpoint not found: ${MODEL_PATH}"; exit 1; }

cd "${EVAL_LAUNCHER}"
MODEL_PATH="${MODEL_PATH}" MODEL_BASE="${MODEL_BASE}" \
ANSWERS_OUT="${ANSWERS_OUT}" NUM_GPUS="${NUM_GPUS}" CONV_MODE="${CONV_MODE:-qwen3}" \
  ./eval_detectium_val_parallel.sh
