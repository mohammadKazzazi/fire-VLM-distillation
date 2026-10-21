#!/usr/bin/env bash
# Table 4-style fire reasoning scores — Plan 4 v4 kd_lm + only_kd students + teachers.
#
# only_kd fire/non-fire uses full-text content scoring (not Yes/No prefix).
#
# Usage:
#   cd detectioumfire/eval/launchers/qwen2.5
#   ./run_table4_student_validation.sh
#
# Outputs:
#   comparisons/table4_all_plans.json
#   comparisons/table4_all_plans_with_yesno.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../common.env.sh"

VAL_JSON="${DETECTIUM_DIR}/detectium_llavamod_real_full_val.json"
RUNS="${EVAL_RUNS}"
MODEL_ARGS=()
MODELS_JSON="$(mktemp)"
trap 'rm -f "${MODELS_JSON}"' EXIT

python3 - <<'PY' >"${MODELS_JSON}"
import json
print(json.dumps({}))
PY

add_model() {
  local label="$1" file="$2"
  if [[ ! -s "${file}" ]]; then
    echo "Skip (missing): ${file}" >&2
    return 0
  fi
  MODEL_ARGS+=(--model "${label}" "${file}")
  python3 - <<PY
import json
from pathlib import Path
path = Path("${MODELS_JSON}")
data = json.loads(path.read_text())
data["""${label}"""] = """${file}"""
path.write_text(json.dumps(data, indent=2) + "\n")
PY
}

# Plan 4 v4 kd_lm students (CLIP-family, all seeds on disk)
for seed in 42 123 456; do
  add_model "Plan 4 v4 kd_lm (Qwen2.5-0.5B) seed${seed}" "${RUNS}/students/qwen2.5-0.5b/plan4-v4-kd-lm-seed${seed}/val_answers.jsonl"
  add_model "Plan 4 v4 kd_lm (Qwen2.5-1.5B) seed${seed}" "${RUNS}/students/qwen2.5-1.5b/plan4-v4-kd-lm-seed${seed}/val_answers.jsonl"
  add_model "Plan 4 v4 kd_lm (Qwen2.5-3B) seed${seed}" "${RUNS}/students/qwen2.5-3b/plan4-v4-kd-lm-seed${seed}/val_answers.jsonl"
  add_model "Plan 4 v4 kd_lm (Gemma2-2B) seed${seed}" "${RUNS}/students/gemma2-2b/plan4-v4-kd-lm-seed${seed}/val_answers.jsonl"
  add_model "Plan 4 v4 kd_lm (Llama3-3B) seed${seed}" "${RUNS}/students/llama3-3b/plan4-v4-kd-lm-seed${seed}/val_answers.jsonl"
  add_model "Plan 4 v4 kd_lm (Llama3-1B) seed${seed}" "${RUNS}/students/llama3-1b/plan4-v4-kd-lm-seed${seed}/val_answers.jsonl"
  add_model "Plan 4 v4 kd_lm (Qwen3-4B) seed${seed}" "${RUNS}/students/qwen3-4b/plan4-v4-kd-lm-seed${seed}/val_answers.jsonl"
done

# only_kd ablation (seed42) — fire/non-fire from answer body, not Yes/No prefix
add_model "Plan 4 v4 only_kd (Qwen2.5-0.5B) seed42" "${RUNS}/students/qwen2.5-0.5b/plan4-v4-only-kd-seed42/val_answers.jsonl"
add_model "Plan 4 v4 only_kd (Qwen2.5-1.5B) seed42" "${RUNS}/students/qwen2.5-1.5b/plan4-v4-only-kd-seed42/val_answers.jsonl"
add_model "Plan 4 v4 only_kd (Qwen2.5-3B) seed42" "${RUNS}/students/qwen2.5-3b/plan4-v4-only-kd-seed42/val_answers.jsonl"
add_model "Plan 4 v4 only_kd (Gemma2-2B) seed42" "${RUNS}/students/gemma2-2b/plan4-v4-only-kd-seed42/val_answers.jsonl"
add_model "Plan 4 v4 only_kd (Llama3-3B) seed42" "${RUNS}/students/llama3-3b/plan4-v4-only-kd-seed42/val_answers.jsonl"
add_model "Plan 4 v4 only_kd (Llama3-1B) seed42" "${RUNS}/students/llama3-1b/plan4-v4-only-kd-seed42/val_answers.jsonl"
add_model "Plan 4 v4 only_kd (Qwen3-4B) seed42" "${RUNS}/students/qwen3-4b/plan4-v4-only-kd-seed42/val_answers.jsonl"

# Qwen3-0.6B MobileCLIP (already on this box)
for seed in 42 123 456; do
  add_model "Plan 4 v4 kd_lm (Qwen3-0.6B MobileCLIP) seed${seed}" "${RUNS}/students/qwen3-0.6b/mobileclip-s2/plan4-v4-kd-lm-seed${seed}/val_answers.jsonl"
done
# Qwen3-0.6B CLIP ViT-L/336 (seed answers appear as they land)
for seed in 42 123 456; do
  add_model "Plan 4 v4 kd_lm (Qwen3-0.6B CLIP-L) seed${seed}" "${RUNS}/students/qwen3-0.6b/clip-vitl336/plan4-v4-kd-lm-seed${seed}/val_answers.jsonl"
done
add_model "Plan 4 v4 kd_lm (Qwen3-0.6B)" "${RUNS}/students/qwen3-0.6b/plan4-v4-kd-lm-seed42/val_answers.jsonl"

# Teachers
add_model "Teacher LoRA 7B (Qwen2.5)" "${RUNS}/teacher/qwen2.5-7b-lora/val_answers.jsonl"
add_model "Teacher LoRA 9B (Gemma2)" "${RUNS}/teacher/gemma2-9b-lora/val_answers.jsonl"
add_model "Teacher LoRA 8B (Llama3)" "${RUNS}/teacher/llama3-8b-lora/val_answers.jsonl"
add_model "Teacher LoRA 8B (Qwen3)" "${RUNS}/teacher/qwen3-8b-lora/val_answers.jsonl"

if [[ ${#MODEL_ARGS[@]} -eq 0 ]]; then
  echo "No answer files found under ${RUNS}" >&2
  exit 1
fi

python3 "${EVAL_LIB}/score_fire_reasoning_table4.py" \
  --val-json "${VAL_JSON}" \
  "${MODEL_ARGS[@]}" \
  --json-out "${EVAL_COMPARISONS}/table4_all_plans.json"

python3 "${EVAL_LIB}/merge_table4_yesno.py" \
  --table4-json "${EVAL_COMPARISONS}/table4_all_plans.json" \
  --val-json "${VAL_JSON}" \
  --models-json "${MODELS_JSON}" \
  --json-out "${EVAL_COMPARISONS}/table4_all_plans_with_yesno.json"
