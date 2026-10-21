#!/usr/bin/env bash
# Run one image through Qwen2.5-0.5B fire student on Raspberry Pi (aarch64).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMG="${1:?Usage: $0 /path/to/image.jpg}"
THREADS="${THREADS:-4}"
CTX="${CTX:-4096}"
N_TOKENS="${N_TOKENS:-64}"
PROMPT="$(cat "${ROOT}/fire_prompt.txt")"

MODEL="${ROOT}/models/fire-qwen2.5-0.5b-seed42-Q8_0.gguf"
MMPROJ="${ROOT}/models/mmproj-fire-qwen2.5-0.5b-seed42-clip-vit-l-f16.gguf"

# Prefer a local build if present; else bundled bin-arm64
if [[ -x "${ROOT}/llama.cpp/build/bin/llama-mtmd-cli" ]]; then
  BIN="${ROOT}/llama.cpp/build/bin"
elif [[ -x "${ROOT}/bin-arm64/llama-mtmd-cli" ]]; then
  BIN="${ROOT}/bin-arm64"
else
  echo "No llama-mtmd-cli found. Build llama.cpp first (see README.md)." >&2
  exit 1
fi

export LD_LIBRARY_PATH="${BIN}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${THREADS}"

exec "${BIN}/llama-mtmd-cli" \
  -m "${MODEL}" \
  --mmproj "${MMPROJ}" \
  --image "${IMG}" \
  -p "${PROMPT}" \
  -n "${N_TOKENS}" \
  -t "${THREADS}" \
  -c "${CTX}" \
  --temp 0.0 \
  --no-mmproj-offload
