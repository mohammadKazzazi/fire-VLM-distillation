#!/usr/bin/env bash
# Export fire student → GGUF and smoke-test on ARM64.
#
# Artifacts:
#   fire-qwen2.5-0.5b-{f16,Q8_0,Q4_K_M}.gguf   (phi chat_template baked in)
#   mmproj-fire-clip-vit-l-{f16,Q8_0}.gguf
#
# Usage:
#   bash scripts/build_llamacpp_gguf.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

LLAMA_CPP="${LLAMA_CPP:-/home/ubuntu/llavamod/llama.cpp}"
EXPORT_DIR="${DEPLOY_ROOT}/bundles/llamacpp_export"
GGUF_DIR="${DEPLOY_ROOT}/bundles/gguf"
CLIP_CACHE="${DEPLOY_ROOT}/cache_dir"
CLIP_ID="${CLIP_MODEL:-openai/clip-vit-large-patch14-336}"
PHI_JINJA="${GGUF_DIR}/llava_qwen_phi.jinja"

mkdir -p "${GGUF_DIR}"

if [[ ! -x "${LLAMA_CPP}/build/bin/llama-quantize" ]]; then
  echo "Building llama.cpp ..."
  cmake -S "${LLAMA_CPP}" -B "${LLAMA_CPP}/build" \
    -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
  cmake --build "${LLAMA_CPP}/build" -j"$(nproc)" --target llama-quantize llama-mtmd-cli llama-server
fi

echo "=== 1) Merge LoRA + export HF LLM + projector (phi template baked) ==="
python3 "${SCRIPT_DIR}/export_for_llamacpp.py" --out "${EXPORT_DIR}"

echo "=== 2) Convert LLM → GGUF F16 ==="
pip install -q gguf protobuf sentencepiece
python3 "${LLAMA_CPP}/convert_hf_to_gguf.py" \
  "${EXPORT_DIR}/llm" \
  --outfile "${GGUF_DIR}/fire-qwen2.5-0.5b-f16.gguf" \
  --outtype f16

# Ensure phi template is in the GGUF even if convert picked tokenizer defaults
if [[ -f "${GGUF_DIR}/fire-qwen2.5-0.5b-f16.gguf" ]]; then
  echo "=== 2b) Bake conv_phi chat_template into F16 GGUF ==="
  TMP="${GGUF_DIR}/fire-qwen2.5-0.5b-f16.gguf.phi.tmp"
  PYTHONPATH="${LLAMA_CPP}/gguf-py:${PYTHONPATH:-}" \
    python3 "${LLAMA_CPP}/gguf-py/gguf/scripts/gguf_new_metadata.py" \
      --force \
      --chat-template-file "${PHI_JINJA}" \
      "${GGUF_DIR}/fire-qwen2.5-0.5b-f16.gguf" "${TMP}"
  mv -f "${TMP}" "${GGUF_DIR}/fire-qwen2.5-0.5b-f16.gguf"
fi

echo "=== 3) Quantize LLM → Q8_0 (accuracy) + Q4_K_M (size) ==="
"${LLAMA_CPP}/build/bin/llama-quantize" \
  "${GGUF_DIR}/fire-qwen2.5-0.5b-f16.gguf" \
  "${GGUF_DIR}/fire-qwen2.5-0.5b-Q8_0.gguf" \
  Q8_0
"${LLAMA_CPP}/build/bin/llama-quantize" \
  "${GGUF_DIR}/fire-qwen2.5-0.5b-f16.gguf" \
  "${GGUF_DIR}/fire-qwen2.5-0.5b-Q4_K_M.gguf" \
  Q4_K_M

echo "=== 4) Convert CLIP + projector → mmproj F16 ==="
CLIP_SNAP="$(find "${CLIP_CACHE}/models--openai--clip-vit-large-patch14-336/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1 || true)"
if [[ -z "${CLIP_SNAP}" ]]; then
  echo "CLIP snapshot not found under ${CLIP_CACHE}; downloading to ${EXPORT_DIR}/clip_hf"
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("${CLIP_ID}", local_dir="${EXPORT_DIR}/clip_hf")
print("downloaded")
PY
  CLIP_DIR="${EXPORT_DIR}/clip_hf"
else
  CLIP_DIR="${CLIP_SNAP}"
fi

# Match LLaVA-MoD training: pad aspect ratio + vision select layer -2.
# Without mm_vision_select_layer, convert_image_encoder_to_gguf.py defaults to an
# off-by-one CLIP layer vs training (hurts Llama3/Gemma students badly).
python3 - <<PY || true
import json
from pathlib import Path
cfg = Path("${CLIP_DIR}") / "config.json"
if cfg.is_file():
    d = json.loads(cfg.read_text())
    d["mm_vision_select_layer"] = -2
    vc = d.get("vision_config") if isinstance(d.get("vision_config"), dict) else None
    if isinstance(vc, dict):
        vc["image_aspect_ratio"] = "pad"
    else:
        d["image_aspect_ratio"] = "pad"
    cfg.write_text(json.dumps(d, indent=2) + "\n")
    print("set mm_vision_select_layer=-2 and image_aspect_ratio=pad on CLIP config")
PY

python3 "${LLAMA_CPP}/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py" \
  -m "${CLIP_DIR}" \
  --llava-projector "${EXPORT_DIR}/llava.projector" \
  --projector-type mlp \
  --output-dir "${GGUF_DIR}" \
  || {
    echo "Retry with --clip-model-is-vision ..."
    python3 "${LLAMA_CPP}/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py" \
      -m "${CLIP_DIR}" \
      --llava-projector "${EXPORT_DIR}/llava.projector" \
      --projector-type mlp \
      --clip-model-is-vision \
      --output-dir "${GGUF_DIR}"
  }

MMPROJ="$(ls -1 "${GGUF_DIR}"/mmproj-*.gguf 2>/dev/null | grep -v Q8 | head -1 || true)"
if [[ -n "${MMPROJ}" && "${MMPROJ}" != "${GGUF_DIR}/mmproj-fire-clip-vit-l-f16.gguf" ]]; then
  cp -f "${MMPROJ}" "${GGUF_DIR}/mmproj-fire-clip-vit-l-f16.gguf"
fi

echo "=== 4b) Quantize mmproj → Q8_0 (speed/RAM; validate accuracy) ==="
python3 "${SCRIPT_DIR}/quantize_mmproj_q8.py" \
  "${GGUF_DIR}/mmproj-fire-clip-vit-l-f16.gguf" \
  "${GGUF_DIR}/mmproj-fire-clip-vit-l-Q8_0.gguf"

echo "=== 5) Smoke test (phi jinja + Q8 LLM + F16 mmproj) ==="
IMG="$(head -1 "${DEPLOY_ROOT}/config/benchmark_images.txt")"
PROMPT="$(cat "${DEPLOY_ROOT}/config/fire_prompt.txt")"
export LD_LIBRARY_PATH="${LLAMA_CPP}/build/bin:${LD_LIBRARY_PATH:-}"

"${LLAMA_CPP}/build/bin/llama-mtmd-cli" \
  -m "${GGUF_DIR}/fire-qwen2.5-0.5b-Q8_0.gguf" \
  --mmproj "${GGUF_DIR}/mmproj-fire-clip-vit-l-f16.gguf" \
  --image "${IMG}" \
  -p $'\n'"${PROMPT} Answer Yes or No only." \
  -n 8 -t "$(nproc)" --temp 0.0 \
  --jinja --chat-template "$(cat "${PHI_JINJA}")" \
  -c 4096 --no-mmproj-offload \
  2>&1 | tee "${GGUF_DIR}/smoke_mtmd.log"

echo ""
echo "Artifacts in ${GGUF_DIR}:"
ls -lh "${GGUF_DIR}"/*.gguf "${PHI_JINJA}" 2>/dev/null || true
echo "Done."
