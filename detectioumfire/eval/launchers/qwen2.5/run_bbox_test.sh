#!/usr/bin/env bash
# Fire bbox smoke test (Qwen2.5-7B teacher) on 5 synthetic images.
#
# Usage:
#   cd detectioumfire/eval/launchers/qwen2.5
#   ./run_bbox_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../common.env.sh"
REPO_ROOT="$(cd "${DETECTIUM_DIR}/../LLava_MoD" && pwd)"
WORKSPACE_ROOT="$(cd "${DETECTIUM_DIR}/../../.." && pwd)"

# shellcheck disable=SC1091
source "${REPO_ROOT}/shells/conda_env.sh"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/llavaqwen2.5-7b-finetune}"
MODEL_BASE="${MODEL_BASE:-}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${WORKSPACE_ROOT}/LLaVA-Instruct-150K}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_RUNS}/_scratch/bbox_vis}"

args=(
  python "${EVAL_LIB}/test_fire_bbox.py"
  --model-path "${MODEL_PATH}"
  --image-folder "${IMAGE_FOLDER}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${MODEL_BASE}" ]]; then
  args+=(--model-base "${MODEL_BASE}")
fi

"${args[@]}"
