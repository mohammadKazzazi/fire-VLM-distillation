#!/usr/bin/env bash
# Compare Qwen2.5-7B full finetune vs LoRA teacher on Detectium val (1452 examples).
#
# Usage:
#   cd detectioumfire/eval/launchers/qwen2.5
#   NUM_GPUS=8 ./compare_full_vs_lora.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../common.env.sh"
REPO_ROOT="$(cd "${DETECTIUM_DIR}/../LLava_MoD" && pwd)"

FULL_PATH="${FULL_PATH:-${REPO_ROOT}/checkpoints/llavaqwen2.5-7b-finetune}"
LORA_PATH="${LORA_PATH:-${REPO_ROOT}/checkpoints/llavaqwen2.5-7b-finetune-lora}"
MODEL_BASE="${MODEL_BASE:-Qwen/Qwen2.5-7B-Instruct}"
VAL_JSON="${DETECTIUM_DIR}/detectium_llavamod_real_full_val.json"

FULL_OUT="${EVAL_RUNS}/teacher/qwen2.5-7b-full/val_answers.jsonl"
LORA_OUT="${EVAL_RUNS}/teacher/qwen2.5-7b-lora/val_answers.jsonl"
COMPARE_JSON="${EVAL_COMPARISONS}/compare_full_vs_lora.json"

mkdir -p "$(dirname "${FULL_OUT}")" "$(dirname "${LORA_OUT}")" "${EVAL_COMPARISONS}"

if [[ ! -f "${FULL_PATH}/config.json" ]]; then
  echo "Missing full finetune checkpoint: ${FULL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${LORA_PATH}/adapter_config.json" && ! -f "${LORA_PATH}/config.json" ]]; then
  echo "Missing LoRA checkpoint: ${LORA_PATH}" >&2
  exit 1
fi

echo "=== Full finetune (Qwen2.5-7B) ==="
if [[ "${SKIP_FULL_INFER:-0}" == "1" && -s "${FULL_OUT}" ]]; then
  echo "  SKIP_FULL_INFER=1, reusing ${FULL_OUT}"
else
  MODEL_PATH="${FULL_PATH}" MODEL_BASE="" ANSWERS_OUT="${FULL_OUT}" RUN_SCORE=0 \
    "${SCRIPT_DIR}/eval_detectium_val_parallel.sh"
fi

echo "=== LoRA finetune (Qwen2.5-7B) ==="
MODEL_PATH="${LORA_PATH}" MODEL_BASE="${MODEL_BASE}" ANSWERS_OUT="${LORA_OUT}" RUN_SCORE=0 \
  "${SCRIPT_DIR}/eval_detectium_val_parallel.sh"

python3 "${EVAL_LIB}/score_detectium_val.py" \
  --val-json "${VAL_JSON}" \
  --answers-base "${FULL_OUT}" \
  --answers-finetuned "${LORA_OUT}" \
  --json-out "${COMPARE_JSON}"

echo "Done."
