#!/usr/bin/env bash
# Shared paths for Deploy scripts.
set -euo pipefail

DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DEPLOY_ROOT

CONFIG_FILE="${CONFIG_FILE:-${DEPLOY_ROOT}/config/students.env}"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  if [[ -f "${DEPLOY_ROOT}/config/students.env.example" ]]; then
    cp "${DEPLOY_ROOT}/config/students.env.example" "${CONFIG_FILE}"
    echo "Created ${CONFIG_FILE} from example — review before production deploy."
  else
    echo "Missing ${CONFIG_FILE}" >&2
    exit 1
  fi
fi

# shellcheck disable=SC1090
source "${CONFIG_FILE}"

LLAVA_MOD_ROOT="${LLAVA_MOD_ROOT:-${DEPLOY_ROOT}/../LLaVA_MoD/MoD/LLava_MoD}"
export LLAVA_MOD_ROOT

CONDA_ENV_SH="${LLAVA_MOD_ROOT}/shells/conda_env.sh"
if [[ -f "${CONDA_ENV_SH}" ]]; then
  # Optional on remote H1/H4 boxes (use a venv instead). Do not abort if missing.
  # shellcheck disable=SC1090
  source "${CONDA_ENV_SH}" || echo "WARNING: conda_env.sh failed — using current python: $(command -v python3)" >&2
fi

export PYTHONPATH="${LLAVA_MOD_ROOT}:${DEPLOY_ROOT}:${PYTHONPATH:-}"

resolve_preset() {
  case "${STUDENT_PRESET}" in
    qwen0.5b)
      LORA_PATH="${QWEN0_5B_LORA}"
      MODEL_BASE="${QWEN0_5B_BASE}"
      CONV_MODE="${QWEN0_5B_CONV}"
      BUNDLE_PATH="${QWEN0_5B_BUNDLE}"
      ;;
    llama3-1b)
      LORA_PATH="${LLAMA3_1B_LORA}"
      MODEL_BASE="${LLAMA3_1B_BASE}"
      CONV_MODE="${LLAMA3_1B_CONV}"
      BUNDLE_PATH="${LLAMA3_1B_BUNDLE}"
      ;;
    qwen3-0.6b)
      LORA_PATH="${QWEN3_0_6B_LORA}"
      MODEL_BASE="${QWEN3_0_6B_BASE}"
      CONV_MODE="${QWEN3_0_6B_CONV}"
      BUNDLE_PATH="${QWEN3_0_6B_BUNDLE}"
      ;;
    *)
      echo "Unknown STUDENT_PRESET=${STUDENT_PRESET}" >&2
      exit 1
      ;;
  esac
  export LORA_PATH MODEL_BASE CONV_MODE BUNDLE_PATH
  if [[ "${USE_MERGED_BUNDLE:-0}" == "1" && -d "${BUNDLE_PATH}" ]]; then
    export MODEL_PATH="${BUNDLE_PATH}"
    export MODEL_BASE=""
  else
    export MODEL_PATH="${LORA_PATH}"
  fi
  export MODEL_PATH
}

apply_offline_flags() {
  if [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
  fi
  if [[ "${TRANSFORMERS_OFFLINE:-0}" == "1" ]]; then
    export TRANSFORMERS_OFFLINE=1
  fi
}
