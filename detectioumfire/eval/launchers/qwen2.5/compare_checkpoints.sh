#!/usr/bin/env bash
# Run val inference + score for two Qwen2.5-7B checkpoints side by side.
#
# Usage:
#   cd detectioumfire/eval/launchers/qwen2.5
#   ./compare_checkpoints.sh BASE_CKPT FINETUNED_CKPT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../common.env.sh"
REPO_ROOT="$(cd "${DETECTIUM_DIR}/../LLava_MoD" && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 BASE_MODEL_PATH FINETUNED_MODEL_PATH" >&2
  exit 1
fi

BASE_PATH="$1"
FT_PATH="$2"
SCRATCH="${EVAL_RUNS}/_scratch/compare_checkpoints"
BASE_OUT="${SCRATCH}/val_answers_base.jsonl"
FT_OUT="${SCRATCH}/val_answers_finetuned.jsonl"
VAL_JSON="${DETECTIUM_DIR}/detectium_llavamod_real_full_val.json"

mkdir -p "${SCRATCH}"

if [[ "${BASE_PATH}" != /* ]]; then
  BASE_PATH="${REPO_ROOT}/${BASE_PATH}"
fi
if [[ "${FT_PATH}" != /* ]]; then
  FT_PATH="${REPO_ROOT}/${FT_PATH}"
fi

echo "=== Base checkpoint (Qwen2.5-7B) ==="
MODEL_PATH="${BASE_PATH}" ANSWERS_OUT="${BASE_OUT}" RUN_SCORE=0 \
  "${SCRIPT_DIR}/eval_detectium_val_parallel.sh"

echo "=== Fine-tuned checkpoint (Qwen2.5-7B) ==="
MODEL_PATH="${FT_PATH}" ANSWERS_OUT="${FT_OUT}" RUN_SCORE=0 \
  "${SCRIPT_DIR}/eval_detectium_val_parallel.sh"

python3 "${EVAL_LIB}/score_detectium_val.py" \
  --val-json "${VAL_JSON}" \
  --answers-base "${BASE_OUT}" \
  --answers-finetuned "${FT_OUT}" \
  --json-out "${SCRATCH}/compare_metrics.json"

echo "Done."
