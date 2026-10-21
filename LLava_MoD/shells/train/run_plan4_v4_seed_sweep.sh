#!/usr/bin/env bash
# Run Plan 4 v4 D2D for seeds 42, 123, 456 (same train JSON, different --seed).
#
# Teacher LoRA + student projector are FIXED — only D2D is re-run per seed.
# Train/val split is FIXED — this is NOT a 3-fold data split experiment.
#
# Usage (example — Qwen3-0.6B):
#   cd MoD/LLava_MoD
#   BASE_OUTPUT_DIR=./checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen3-0.6b \
#   PLAN4_SCRIPT="shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4.sh" \
#     bash shells/train/run_plan4_v4_seed_sweep.sh
#
# BASE_OUTPUT_DIR must NOT include -seed{N}; the script appends it.
# Skip seed 42 if already trained: SEEDS="123 456" bash ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SEEDS=(${SEEDS:-42 123 456})

PLAN4_SCRIPT="${PLAN4_SCRIPT:?Set PLAN4_SCRIPT to a plan4_d2d_multiquery_v4.sh path}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:?Set BASE_OUTPUT_DIR without -seed suffix (e.g. ./checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen3-0.6b)}"

[[ -f "${REPO_ROOT}/${PLAN4_SCRIPT}" ]] || { echo "Missing ${REPO_ROOT}/${PLAN4_SCRIPT}" >&2; exit 1; }

cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source shells/conda_env.sh
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=== Plan 4 v4 random-seed sweep (NOT train/val splits) ==="
echo "Script:           ${PLAN4_SCRIPT}"
echo "Base OUTPUT_DIR:  ${BASE_OUTPUT_DIR}"
echo "Seeds:            ${SEEDS[*]}"
echo

for SEED in "${SEEDS[@]}"; do
  OUT="${BASE_OUTPUT_DIR}-seed${SEED}"
  echo "--- SEED=${SEED} → ${OUT} ---"
  SEED="${SEED}" OUTPUT_DIR="${OUT}" DETACH="${DETACH:-1}" \
    bash "${PLAN4_SCRIPT}"
  echo
done

echo "Done. Eval each checkpoint to runs/.../plan4-v4-kd-lm-seed{N}/"
