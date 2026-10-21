#!/usr/bin/env bash
# Run fire detection with llama.cpp mtmd (no PyTorch).
# Defaults: Q8_0 LLM + F16 mmproj + conv_phi jinja (accuracy-first).
# Speed/Pi: MODEL=...Q4_K_M.gguf MMPROJ=...Q8_0.gguf
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

GGUF_DIR="${DEPLOY_ROOT}/bundles/gguf"
BIN="${GGUF_DIR}/bin-arm64/llama-mtmd-cli"
if [[ ! -x "${BIN}" ]]; then
  BIN="${LLAMA_CPP:-/home/ubuntu/llavamod/llama.cpp}/build/bin/llama-mtmd-cli"
fi
export LD_LIBRARY_PATH="$(dirname "${BIN}"):${LD_LIBRARY_PATH:-}"

IMAGE=""
N=8
MODEL="${MODEL:-${GGUF_DIR}/fire-qwen2.5-0.5b-Q8_0.gguf}"
MMPROJ="${MMPROJ:-${GGUF_DIR}/mmproj-fire-clip-vit-l-f16.gguf}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    -n) N="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --mmproj) MMPROJ="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done
[[ -n "${IMAGE}" ]] || { echo "Usage: $0 --image path.jpg [--model …] [--mmproj …]" >&2; exit 1; }

PROMPT="$(cat "${GGUF_DIR}/fire_prompt.txt" 2>/dev/null || cat "${DEPLOY_ROOT}/config/fire_prompt.txt")"
TEMPLATE_FILE="${GGUF_DIR}/llava_qwen_phi.jinja"

exec "${BIN}" \
  -m "${MODEL}" \
  --mmproj "${MMPROJ}" \
  --image "${IMAGE}" \
  -p $'\n'"${PROMPT} Answer Yes or No only." \
  -n "${N}" -t "$(nproc)" --temp 0.0 \
  --jinja --chat-template "$(cat "${TEMPLATE_FILE}")" \
  -c 4096 --no-mmproj-offload
