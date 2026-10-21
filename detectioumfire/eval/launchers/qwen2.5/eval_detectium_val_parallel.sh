#!/usr/bin/env bash
# Run Detectium val VQA on LLaVA-MoD checkpoints — one worker per GPU.
#
# Usage:
#   cd detectioumfire/eval/launchers/<backbone>   # qwen2.5, gemma2, llama3, qwen3
#   ./eval_detectium_val_parallel.sh
#
# Optional env:
#   MODEL_PATH=...       backbone-specific default (see case block below)
#   MODEL_BASE=...       set for LoRA checkpoints (PEFT base path)
#   NUM_GPUS=8
#   CONV_MODE=...        backbone-specific default
#   RUN_SCORE=1
#   SKIP_INFER=1
#   ANSWERS_OUT=...      default: eval/runs/_scratch/val_answers.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_FAMILY="$(basename "${SCRIPT_DIR}")"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../common.env.sh"
REPO_ROOT="$(cd "${DETECTIUM_DIR}/../LLava_MoD" && pwd)"
WORKSPACE_ROOT="$(cd "${DETECTIUM_DIR}/../../.." && pwd)"

case "${LAUNCHER_FAMILY}" in
  gemma2)
    CONV_MODE_DEFAULT="gemma_2"
    DEFAULT_MODEL_BASE_DEFAULT="google/gemma-2-9b-it"
    MODEL_PATH_DEFAULT="${REPO_ROOT}/checkpoints/llavagemma2-9b-finetune-lora"
    ;;
  llama3)
    CONV_MODE_DEFAULT="llama_3"
    DEFAULT_MODEL_BASE_DEFAULT="meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH_DEFAULT="${REPO_ROOT}/checkpoints/llavallama3.1-8b-finetune-lora"
    ;;
  qwen3)
    CONV_MODE_DEFAULT="qwen3"
    DEFAULT_MODEL_BASE_DEFAULT="Qwen/Qwen3-8B"
    MODEL_PATH_DEFAULT="${REPO_ROOT}/checkpoints/llavaqwen3-8b-finetune-lora"
    ;;
  *)
    CONV_MODE_DEFAULT="qwen"
    DEFAULT_MODEL_BASE_DEFAULT="Qwen/Qwen2.5-7B-Instruct"
    MODEL_PATH_DEFAULT="${REPO_ROOT}/checkpoints/llavaqwen2.5-7b-finetune"
    ;;
esac

# shellcheck disable=SC1091
source "${REPO_ROOT}/shells/conda_env.sh"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ "${LAUNCHER_FAMILY}" == "qwen3" ]]; then
  # Student checkpoints use smaller Qwen3 bases (cached under ~/.cache/huggingface).
  # Teacher LoRA uses Qwen3-8B (repo cache_dir). Avoid re-downloading into a full cache_dir.
  _qwen3_hf_base="${MODEL_BASE:-Qwen/Qwen3-8B}"
  if [[ "${_qwen3_hf_base}" == *"8B"* ]]; then
    export HF_HOME="${REPO_ROOT}/cache_dir"
    export HUGGINGFACE_HUB_CACHE="${REPO_ROOT}/cache_dir"
    export TRANSFORMERS_CACHE="${REPO_ROOT}/cache_dir"
  else
    export HF_HOME="${HOME}/.cache/huggingface"
    export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
    export TRANSFORMERS_CACHE="${HF_HOME}/hub"
  fi
fi

NUM_GPUS="${NUM_GPUS:-}"
if [[ -z "${NUM_GPUS}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  fi
  NUM_GPUS="${NUM_GPUS:-1}"
fi

CONV_MODE="${CONV_MODE:-${CONV_MODE_DEFAULT}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
RUN_SCORE="${RUN_SCORE:-1}"
SKIP_INFER="${SKIP_INFER:-0}"
STREAM_TO_TERMINAL="${STREAM_TO_TERMINAL:-0}"
DEFAULT_MODEL_BASE="${DEFAULT_MODEL_BASE:-${DEFAULT_MODEL_BASE_DEFAULT}}"

VAL_JSON="${VAL_JSON:-${DETECTIUM_DIR}/detectium_llavamod_real_full_val.json}"
QUESTION_FILE="${QUESTION_FILE:-${DETECTIUM_DIR}/val_questions.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${WORKSPACE_ROOT}/LLaVA-Instruct-150K}"
MODEL_PATH="${MODEL_PATH:-${MODEL_PATH_DEFAULT}}"
MODEL_BASE="${MODEL_BASE:-}"

EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${EVAL_RUNS}/_logs}"
ANSWERS_OUT="${ANSWERS_OUT:-${EVAL_RUNS}/_scratch/val_answers.jsonl}"
if [[ "${ANSWERS_OUT}" != /* ]]; then
  ANSWERS_OUT="${SCRIPT_DIR}/${ANSWERS_OUT#./}"
fi
mkdir -p "${EVAL_LOG_ROOT}" "$(dirname "${ANSWERS_OUT}")"

if [[ ! -f "${QUESTION_FILE}" ]]; then
  echo "Missing ${QUESTION_FILE} — building from val JSON..."
  python3 "${DETECTIUM_DIR}/export_val_questions_jsonl.py" \
    --input "${VAL_JSON}" \
    --output "${QUESTION_FILE}"
fi

if [[ ! -d "${IMAGE_FOLDER}/real_fire" && ! -d "${IMAGE_FOLDER}/real_non_fire" ]]; then
  echo "Missing Detectium image symlinks under ${IMAGE_FOLDER}."
  echo "Run: bash ${WORKSPACE_ROOT}/LLaVA-Instruct-150K/link_detectium_images.sh"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" && ! -f "${MODEL_PATH}/adapter_config.json" ]]; then
  echo "Model not found (missing config.json or adapter_config.json): ${MODEL_PATH}"
  exit 1
fi

if [[ -z "${MODEL_BASE}" && "${MODEL_PATH}" == *lora* ]]; then
  MODEL_BASE="${DEFAULT_MODEL_BASE}"
  echo "Auto-set MODEL_BASE=${MODEL_BASE} for LoRA adapter"
fi

if [[ -z "${MODEL_BASE}" && -f "${MODEL_PATH}/adapter_config.json" && -f "${MODEL_PATH}/config.json" ]]; then
  _student_base="$(LAUNCHER_FAMILY="${LAUNCHER_FAMILY}" python3 -c "
import json, os
family = os.environ.get('LAUNCHER_FAMILY', 'qwen2.5')
cfg = json.load(open('${MODEL_PATH}/config.json'))
name = (cfg.get('_name_or_path') or '').lower()
hs = cfg.get('hidden_size', 0)
if family == 'gemma2':
    if '2b' in name or hs == 2304:
        print('google/gemma-2-2b-it')
    elif '9b' in name or hs == 3584:
        print('google/gemma-2-9b-it')
elif family == 'llama3':
    if '3.2' in name or ('3b' in name and '3.1' not in name) or hs == 3072:
        print('meta-llama/Llama-3.2-3B-Instruct')
    elif '3.1' in name or '8b' in name or hs == 4096:
        print('meta-llama/Llama-3.1-8B-Instruct')
elif family == 'qwen3':
    if '4b' in name:
        print('Qwen/Qwen3-4B')
    elif '8b' in name:
        print('Qwen/Qwen3-8B')
else:
    if '0.5b' in name or hs == 896:
        print('Qwen/Qwen2.5-0.5B-Instruct')
    elif '1.5b' in name or hs == 1536:
        print('Qwen/Qwen2.5-1.5B-Instruct')
    elif '3b' in name or hs == 2048:
        print('Qwen/Qwen2.5-3B-Instruct')
    elif '7b' in name or hs == 3584:
        print('Qwen/Qwen2.5-7B-Instruct')
" 2>/dev/null || true)"
  if [[ -n "${_student_base}" ]]; then
    MODEL_BASE="${_student_base}"
    echo "Auto-set MODEL_BASE=${MODEL_BASE} for student D2D checkpoint"
  fi
fi

run_parallel_vqa() {
  local model_path="$1"
  local model_base="$2"
  local answers_out="$3"
  local tmpdir logdir
  tmpdir="$(mktemp -d)"
  logdir="$(mktemp -d "${EVAL_LOG_ROOT}/run_XXXXXX")"
  local i gpu pids=()

  echo "=== Parallel VQA → ${answers_out} (${NUM_GPUS} GPUs, conv=${CONV_MODE}) ==="
  echo "model_path:   ${model_path}"
  if [[ -n "${model_base}" ]]; then
    echo "model_base:   ${model_base}"
  fi
  if [[ "${STREAM_TO_TERMINAL}" != "1" ]]; then
    echo "Log directory: ${logdir}"
    echo "  Example: tail -f ${logdir}/gpu0.log"
    echo
  fi

  for ((i = 0; i < NUM_GPUS; i++)); do
    gpu="$i"
    local -a vqa_args=(
      python "${EVAL_LIB}/detectium_vqa_worker.py"
      --model-path "${model_path}"
      --image-folder "${IMAGE_FOLDER}"
      --question-file "${QUESTION_FILE}"
      --answers-file "${tmpdir}/part_${i}.jsonl"
      --conv-mode "${CONV_MODE}"
      --temperature 0
      --max_new_tokens "${MAX_NEW_TOKENS}"
      --num-chunks "${NUM_GPUS}"
      --chunk-idx "${i}"
    )
    if [[ -n "${model_base}" ]]; then
      vqa_args+=(--model-base "${model_base}")
    fi

    if [[ "${STREAM_TO_TERMINAL}" == "1" ]]; then
      env CUDA_VISIBLE_DEVICES="${gpu}" "${vqa_args[@]}" &
    else
      env CUDA_VISIBLE_DEVICES="${gpu}" "${vqa_args[@]}" >"${logdir}/gpu${gpu}.log" 2>&1 &
    fi
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      echo "One GPU worker failed (pid ${pid}). Check logs in ${logdir}"
      rm -rf "${tmpdir}"
      exit 1
    fi
  done

  cat "${tmpdir}"/part_*.jsonl > "${answers_out}"
  rm -rf "${tmpdir}"
  echo "Wrote ${answers_out} ($(wc -l < "${answers_out}") lines)"
  echo
}

if [[ "${SKIP_INFER}" == "1" ]]; then
  echo "SKIP_INFER=1: reusing ${ANSWERS_OUT}"
  if [[ ! -s "${ANSWERS_OUT}" ]]; then
    echo "ERROR: ${ANSWERS_OUT} missing or empty."
    exit 1
  fi
else
  run_parallel_vqa "${MODEL_PATH}" "${MODEL_BASE}" "${ANSWERS_OUT}"
fi

if [[ "${RUN_SCORE}" == "1" ]]; then
  if [[ ! -f "${VAL_JSON}" ]]; then
    echo "Skipping score: missing ${VAL_JSON}"
    exit 0
  fi
  python3 "${EVAL_LIB}/score_detectium_val.py" \
    --val-json "${VAL_JSON}" \
    --answers "${ANSWERS_OUT}"
fi

echo "Done."
