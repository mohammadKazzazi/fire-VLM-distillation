#!/usr/bin/env bash
# Eval Plan 4 v4 kd_lm checkpoint on Detectium val (8 GPUs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_eval.env.sh"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/plan4-d2d-multiquery-v4-kd-lm-seed42}"
MODEL_BASE="${MODEL_BASE:-Qwen/Qwen2.5-3B-Instruct}"
ANSWERS_OUT="${ANSWERS_OUT:-${EVAL_RUNS}/students/qwen2.5-3b/plan4-v4-kd-lm-seed42/val_answers.jsonl}"
NUM_GPUS="${NUM_GPUS:-8}"

[[ -f "${MODEL_PATH}/adapter_config.json" || -f "${MODEL_PATH}/config.json" ]] \
  || { echo "ERROR: checkpoint not found: ${MODEL_PATH}"; exit 1; }

cd "${EVAL_LAUNCHER}"
MODEL_PATH="${MODEL_PATH}" MODEL_BASE="${MODEL_BASE}" \
ANSWERS_OUT="${ANSWERS_OUT}" NUM_GPUS="${NUM_GPUS}" \
  ./eval_detectium_val_parallel.sh
