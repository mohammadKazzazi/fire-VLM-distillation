#!/usr/bin/env bash
# Compare Gemma2-9B projector-pretrain vs LoRA SFT teacher on Detectium val (1452 rows).
#
# Usage:
#   cd detectioumfire/eval/launchers/gemma2
#   NUM_GPUS=8 ./compare_pretrain_vs_lora.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../common.env.sh"
REPO_ROOT="$(cd "${DETECTIUM_DIR}/../LLava_MoD" && pwd)"

PRETRAIN_PATH="${PRETRAIN_PATH:-${REPO_ROOT}/checkpoints/llavagemma2-9b-pretrain}"
LORA_PATH="${LORA_PATH:-${REPO_ROOT}/checkpoints/llavagemma2-9b-finetune-lora}"
MODEL_BASE="${MODEL_BASE:-google/gemma-2-9b-it}"
VAL_JSON="${DETECTIUM_DIR}/detectium_llavamod_real_full_val.json"

PRETRAIN_OUT="${EVAL_RUNS}/teacher/gemma2-9b-pretrain/val_answers.jsonl"
LORA_OUT="${EVAL_RUNS}/teacher/gemma2-9b-lora/val_answers.jsonl"
COMPARE_JSON="${EVAL_COMPARISONS}/compare_gemma2_pretrain_vs_lora.json"

mkdir -p "$(dirname "${PRETRAIN_OUT}")" "$(dirname "${LORA_OUT}")" "${EVAL_COMPARISONS}"

if [[ ! -f "${PRETRAIN_PATH}/config.json" || ! -f "${PRETRAIN_PATH}/mm_projector.bin" ]]; then
  echo "Missing pretrain checkpoint: ${PRETRAIN_PATH}" >&2
  exit 1
fi
if [[ ! -f "${LORA_PATH}/adapter_config.json" ]]; then
  echo "Missing LoRA checkpoint: ${LORA_PATH}" >&2
  exit 1
fi

echo "=== Projector pretrain (Gemma2-9B, phase 1) ==="
if [[ "${SKIP_PRETRAIN_INFER:-0}" == "1" && -s "${PRETRAIN_OUT}" ]]; then
  echo "  SKIP_PRETRAIN_INFER=1, reusing ${PRETRAIN_OUT}"
else
  MODEL_PATH="${PRETRAIN_PATH}" MODEL_BASE="${MODEL_BASE}" ANSWERS_OUT="${PRETRAIN_OUT}" RUN_SCORE=0 \
    "${SCRIPT_DIR}/eval_detectium_val_parallel.sh"
fi

echo "=== LoRA SFT (Gemma2-9B, phase 2) ==="
if [[ "${SKIP_LORA_INFER:-0}" == "1" && -s "${LORA_OUT}" ]]; then
  echo "  SKIP_LORA_INFER=1, reusing ${LORA_OUT}"
else
  MODEL_PATH="${LORA_PATH}" MODEL_BASE="${MODEL_BASE}" ANSWERS_OUT="${LORA_OUT}" RUN_SCORE=0 \
    "${SCRIPT_DIR}/eval_detectium_val_parallel.sh"
fi

python3 "${EVAL_LIB}/score_detectium_val.py" \
  --val-json "${VAL_JSON}" \
  --answers-base "${PRETRAIN_OUT}" \
  --answers-finetuned "${LORA_OUT}" \
  --json-out "${COMPARE_JSON}"

python3 "${EVAL_LIB}/score_fire_reasoning_table4.py" \
  --val-json "${VAL_JSON}" \
  --model "Gemma2-9B pretrain (phase 1)" "${PRETRAIN_OUT}" \
  --model "Gemma2-9B LoRA (phase 2)" "${LORA_OUT}" \
  --json-out "${EVAL_COMPARISONS}/table4_gemma2_teacher.json"

echo "Done."
