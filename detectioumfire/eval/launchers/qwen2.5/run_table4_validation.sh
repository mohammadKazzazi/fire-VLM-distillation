#!/usr/bin/env bash
# Table 4-style fire reasoning scores for Qwen2.5-7B teachers (full + LoRA).
#
# Usage:
#   cd detectioumfire/eval/launchers/qwen2.5
#   ./run_table4_validation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../common.env.sh"

FULL_OUT="${EVAL_RUNS}/teacher/qwen2.5-7b-full/val_answers.jsonl"
LORA_OUT="${EVAL_RUNS}/teacher/qwen2.5-7b-lora/val_answers.jsonl"

for f in "${FULL_OUT}" "${LORA_OUT}"; do
  if [[ ! -s "${f}" ]]; then
    echo "Missing ${f} — run: NUM_GPUS=8 ./compare_full_vs_lora.sh" >&2
    exit 1
  fi
done

python3 "${EVAL_LIB}/score_fire_reasoning_table4.py" \
  --answers-full "${FULL_OUT}" \
  --answers-lora "${LORA_OUT}" \
  --label-full "Qwen2.5-7B full SFT (llavaqwen2.5-7b-finetune)" \
  --label-lora "Qwen2.5-7B LoRA SFT (llavaqwen2.5-7b-finetune-lora)" \
  --json-out "${EVAL_COMPARISONS}/table4_teacher_scores.json"
