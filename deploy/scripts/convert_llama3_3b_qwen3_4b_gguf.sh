#!/usr/bin/env bash
# Convert lean HF exports → GGUF on the ARM CPU host.
# Usage (on 18.185.97.91 after unpack/configure):
#   bash scripts/convert_llama3_3b_qwen3_4b_gguf.sh
#   bash scripts/convert_llama3_3b_qwen3_4b_gguf.sh llama3-3b
#   bash scripts/convert_llama3_3b_qwen3_4b_gguf.sh qwen3-4b
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

LLAMA_CPP="${LLAMA_CPP:-/home/ubuntu/llavamod/llama.cpp}"
EXPORT_ROOT="${DEPLOY_ROOT}/bundles/llamacpp_export"
GGUF_DIR="${DEPLOY_ROOT}/bundles/gguf"
CLIP_CACHE="${DEPLOY_ROOT}/cache_dir"
ONLY="${1:-both}"

mkdir -p "${GGUF_DIR}"

if [[ ! -x "${LLAMA_CPP}/build/bin/llama-quantize" ]]; then
  echo "Building llama.cpp quantize/server ..."
  cmake -S "${LLAMA_CPP}" -B "${LLAMA_CPP}/build" \
    -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
  cmake --build "${LLAMA_CPP}/build" -j"$(nproc)" --target llama-quantize llama-mtmd-cli llama-server
fi

convert_one() {
  local preset="$1"
  local stem="$2"
  local jinja="$3"
  local export_dir="${EXPORT_ROOT}/${preset}"
  local llm="${export_dir}/llm"
  local proj="${export_dir}/llava.projector"
  local f16="${GGUF_DIR}/fire-${stem}-f16.gguf"
  local q8="${GGUF_DIR}/fire-${stem}-Q8_0.gguf"
  local q4="${GGUF_DIR}/fire-${stem}-Q4_K_M.gguf"
  local mmproj_f16="${GGUF_DIR}/mmproj-fire-${stem}-clip-vit-l-f16.gguf"

  [[ -d "${llm}" ]] || { echo "Missing ${llm}"; exit 1; }
  [[ -f "${proj}" ]] || { echo "Missing ${proj}"; exit 1; }
  [[ -f "${GGUF_DIR}/${jinja}" ]] || { echo "Missing ${GGUF_DIR}/${jinja}"; exit 1; }

  echo "=== ${preset}: HF → GGUF F16 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  python3 "${LLAMA_CPP}/convert_hf_to_gguf.py" \
    "${llm}" --outfile "${f16}" --outtype f16

  echo "=== ${preset}: bake chat template ${jinja} ==="
  local tmp="${f16}.tmp"
  PYTHONPATH="${LLAMA_CPP}/gguf-py:${PYTHONPATH:-}" \
    python3 "${LLAMA_CPP}/gguf-py/gguf/scripts/gguf_new_metadata.py" \
      --force --chat-template-file "${GGUF_DIR}/${jinja}" \
      "${f16}" "${tmp}"
  mv -f "${tmp}" "${f16}"

  echo "=== ${preset}: quantize Q8_0 + Q4_K_M ==="
  "${LLAMA_CPP}/build/bin/llama-quantize" "${f16}" "${q8}" Q8_0
  "${LLAMA_CPP}/build/bin/llama-quantize" "${f16}" "${q4}" Q4_K_M

  echo "=== ${preset}: mmproj F16 (mm_vision_select_layer=-2) ==="
  local clip_snap
  clip_snap="$(find "${CLIP_CACHE}/models--openai--clip-vit-large-patch14-336/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1 || true)"
  if [[ -z "${clip_snap}" ]]; then
    echo "CLIP snapshot missing under ${CLIP_CACHE}" >&2
    exit 1
  fi
  python3 - <<PY
import json
from pathlib import Path
cfg = Path("${clip_snap}") / "config.json"
d = json.loads(cfg.read_text())
d["mm_vision_select_layer"] = -2
vc = d.get("vision_config") if isinstance(d.get("vision_config"), dict) else None
if isinstance(vc, dict):
    vc["image_aspect_ratio"] = "pad"
else:
    d["image_aspect_ratio"] = "pad"
cfg.write_text(json.dumps(d, indent=2) + "\n")
print("patched CLIP config")
PY

  local mm_tmp
  mm_tmp="$(mktemp -d)"
  python3 "${LLAMA_CPP}/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py" \
    -m "${clip_snap}" \
    --llava-projector "${proj}" \
    --projector-type mlp \
    --output-dir "${mm_tmp}" \
    || python3 "${LLAMA_CPP}/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py" \
         -m "${clip_snap}" \
         --llava-projector "${proj}" \
         --projector-type mlp \
         --clip-model-is-vision \
         --output-dir "${mm_tmp}"
  cp -f "$(ls "${mm_tmp}"/mmproj-*.gguf | head -1)" "${mmproj_f16}"
  rm -rf "${mm_tmp}"

  if [[ -f "${SCRIPT_DIR}/quantize_mmproj_q8.py" ]]; then
    python3 "${SCRIPT_DIR}/quantize_mmproj_q8.py" \
      "${mmproj_f16}" \
      "${GGUF_DIR}/mmproj-fire-${stem}-clip-vit-l-Q8_0.gguf" || true
  fi

  ls -lh "${f16}" "${q8}" "${q4}" "${mmproj_f16}"
  echo "=== ${preset}: done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}

case "${ONLY}" in
  both)
    convert_one llama3-3b llama3.2-3b llava_llama3.jinja
    convert_one qwen3-4b qwen3-4b llava_qwen3.jinja
    ;;
  llama3-3b)
    convert_one llama3-3b llama3.2-3b llava_llama3.jinja
    ;;
  qwen3-4b)
    convert_one qwen3-4b qwen3-4b llava_qwen3.jinja
    ;;
  *)
    echo "Usage: $0 [both|llama3-3b|qwen3-4b]" >&2
    exit 1
    ;;
esac

echo "All requested GGUF conversions finished."
