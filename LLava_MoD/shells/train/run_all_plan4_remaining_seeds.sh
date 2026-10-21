#!/usr/bin/env bash
# Train ALL Plan 4 v4 kd_lm students for remaining random seeds (default: 123, 456).
#
# Seed 42 checkpoints already exist (*-seed42/). This script runs D2D sequentially:
#   for each seed → for each backbone → train (blocking) → next
#
# Teacher LoRA + student projectors are fixed. Same train JSON for every run.
#
# Usage:
#   cd MoD/LLava_MoD
#   bash shells/train/run_all_plan4_remaining_seeds.sh
#
# Detached (safe to close terminal):
#   nohup bash shells/train/run_all_plan4_remaining_seeds.sh >> logs/plan4_all_seeds_sweep.log 2>&1 &
#   tail -f logs/plan4_all_seeds_sweep.log
#
# Env:
#   SEEDS="123 456"     — seeds to train (default; seed 42 skipped)
#   SKIP_EXISTING=1     — skip if OUTPUT_DIR already has adapter_config.json (default)
#   INCLUDE_QWEN3_0_6B=1 — include 0.6B if pretrain exists (default: auto)
#   DRY_RUN=1           — print plan only, do not train

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

SEEDS=(${SEEDS:-123 456})
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

# name | plan4 script (relative to REPO_ROOT) | checkpoint base (no -seed suffix) | prerequisite file
MODELS=(
  "Qwen2.5-3B|shells/train/qwen/plan4_d2d_multiquery_v4.sh|plan4-d2d-multiquery-v4-kd-lm|checkpoints/llavaqwen2.5-3b-student-pretrain/mm_projector.bin"
  "Qwen2.5-0.5B|shells/train/qwen/student 0.5B/plan4_d2d_multiquery_v4.sh|plan4-d2d-multiquery-v4-kd-lm-0.5b|checkpoints/llavaqwen2.5-0.5b-student-pretrain/mm_projector.bin"
  "Qwen2.5-1.5B|shells/train/qwen/student 1.5B/plan4_d2d_multiquery_v4.sh|plan4-d2d-multiquery-v4-kd-lm-1.5b|checkpoints/llavaqwen2.5-1.5b-student-pretrain/mm_projector.bin"
  "Gemma2-2B|shells/train/gemma2/plan4_d2d_multiquery_v4.sh|plan4-d2d-multiquery-v4-kd-lm-gemma2-2b|checkpoints/llavagemma2-2b-student-pretrain/mm_projector.bin"
  "Llama3-3B|shells/train/llama3/plan4_d2d_multiquery_v4.sh|plan4-d2d-multiquery-v4-kd-lm-llama3-3b|checkpoints/llavallama3.2-3b-student-pretrain/mm_projector.bin"
  "Llama3-1B|shells/train/llama3/student 1B/plan4_d2d_multiquery_v4.sh|plan4-d2d-multiquery-v4-kd-lm-llama3-1b|checkpoints/llavallama3.2-1b-student-pretrain/mm_projector.bin"
  "Qwen3-4B|shells/train/qwen3/plan4_d2d_multiquery_v4.sh|plan4-d2d-multiquery-v4-kd-lm-qwen3-4b|checkpoints/llavaqwen3-4b-student-pretrain/mm_projector.bin"
)

QWEN3_0_6B_PRETRAIN="${REPO_ROOT}/checkpoints/llavaqwen3-0.6b-student-pretrain/mm_projector.bin"
INCLUDE_QWEN3_0_6B="${INCLUDE_QWEN3_0_6B:-auto}"
if [[ "${INCLUDE_QWEN3_0_6B}" == "auto" ]]; then
  if [[ -f "${QWEN3_0_6B_PRETRAIN}" ]]; then
    INCLUDE_QWEN3_0_6B=1
  else
    INCLUDE_QWEN3_0_6B=0
  fi
fi
if [[ "${INCLUDE_QWEN3_0_6B}" == "1" ]]; then
  MODELS+=("Qwen3-0.6B|shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4.sh|plan4-d2d-multiquery-v4-kd-lm-qwen3-0.6b|checkpoints/llavaqwen3-0.6b-student-pretrain/mm_projector.bin")
fi

has_checkpoint() {
  local out="$1"
  [[ -f "${out}/adapter_config.json" ]] || compgen -G "${out}/checkpoint-*" >/dev/null 2>&1
}

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source shells/conda_env.sh
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES
unset WANDB_API_KEY

TOTAL=0
SKIPPED=0
PLANNED=0

echo "============================================================"
echo "Plan 4 v4 — all-model seed sweep"
echo "Started: $(stamp)"
echo "Repo:    ${REPO_ROOT}"
echo "Seeds:   ${SEEDS[*]}"
echo "Models:  ${#MODELS[@]} (+ Qwen3-0.6B if pretrain ready: ${INCLUDE_QWEN3_0_6B})"
echo "Mode:    sequential (DETACH=0), SKIP_EXISTING=${SKIP_EXISTING}"
echo "============================================================"
echo

for SEED in "${SEEDS[@]}"; do
  echo "########## SEED=${SEED} ##########"
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r NAME PLAN4_SCRIPT CKPT_BASE PREREQ <<< "${entry}"
    PLAN4_PATH="${REPO_ROOT}/${PLAN4_SCRIPT}"
    PREREQ_PATH="${REPO_ROOT}/${PREREQ}"
    OUTPUT_DIR="${REPO_ROOT}/checkpoints/${CKPT_BASE}-seed${SEED}"
    TOTAL=$((TOTAL + 1))

    if [[ ! -f "${PLAN4_PATH}" ]]; then
      echo "[SKIP] ${NAME} seed=${SEED}: missing script ${PLAN4_SCRIPT}"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
    if [[ ! -f "${PREREQ_PATH}" ]]; then
      echo "[SKIP] ${NAME} seed=${SEED}: missing prerequisite ${PREREQ}"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
    if [[ "${SKIP_EXISTING}" == "1" ]] && has_checkpoint "${OUTPUT_DIR}"; then
      echo "[SKIP] ${NAME} seed=${SEED}: checkpoint exists → ${OUTPUT_DIR}"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    PLANNED=$((PLANNED + 1))
    RUN_LOG="${LOG_DIR}/plan4_${CKPT_BASE}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

    echo
    echo "------------------------------------------------------------"
    echo "[RUN]  ${NAME}  SEED=${SEED}"
    echo "       script: ${PLAN4_SCRIPT}"
    echo "       out:    ${OUTPUT_DIR}"
    echo "       log:    ${RUN_LOG}"
    echo "------------------------------------------------------------"

    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    export SEED="${SEED}"
    SEED="${SEED}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    DETACH=0 \
      bash "${PLAN4_PATH}" 2>&1 | tee "${RUN_LOG}"

    if has_checkpoint "${OUTPUT_DIR}"; then
      echo "[OK]   ${NAME} seed=${SEED} finished → ${OUTPUT_DIR}"
    else
      echo "[FAIL] ${NAME} seed=${SEED}: no checkpoint in ${OUTPUT_DIR}" >&2
      exit 1
    fi
  done
  echo
done

echo "============================================================"
echo "Finished: $(stamp)"
echo "Total slots: ${TOTAL}  Planned runs: ${PLANNED}  Skipped: ${SKIPPED}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 — no training started."
else
  echo "Next: eval each checkpoint, e.g."
  echo "  MODEL_PATH=checkpoints/plan4-d2d-multiquery-v4-kd-lm-seed123 \\"
  echo "  ANSWERS_OUT=detectioumfire/eval/runs/students/qwen2.5-3b/plan4-v4-kd-lm-seed123/val_answers.jsonl \\"
  echo "  NUM_GPUS=8 bash shells/eval/plan4_d2d_multiquery_v4_val.sh"
fi
echo "============================================================"
