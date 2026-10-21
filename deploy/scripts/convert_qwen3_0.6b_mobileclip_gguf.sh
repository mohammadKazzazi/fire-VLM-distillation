#!/usr/bin/env bash
# Convert Qwen3-0.6B+MobileCLIP student LLM → GGUF Q8.
# NOTE: MobileCLIP/FastViT has no llama.cpp mmproj converter — LLM-only GGUF.
set -euo pipefail

DEPLOY_ROOT=/home/ubuntu/llavamod/llava-deploy-h1-h4-20260716/workspace/Deploy
EXPORT_SCRIPT=/home/ubuntu/llavamod/llama3-3b-qwen3-4b-students/workspace/Deploy/scripts/export_student_lean.py
CONFIG="${DEPLOY_ROOT}/config/students_qwen3_0.6b_mobileclip.env"
LLAMA_CPP=/home/ubuntu/llavamod/llama.cpp
EXPORT_DIR="${DEPLOY_ROOT}/bundles/llamacpp_export/qwen3-0.6b-mobileclip"
GGUF_DIR="${DEPLOY_ROOT}/bundles/gguf"
VENV=/home/ubuntu/llavamod/llava-deploy-h1-h4-20260716/workspace/.venv
LOG=/home/ubuntu/llavamod/pipeline_logs/convert_qwen3_0.6b_mobileclip.log

mkdir -p "${EXPORT_DIR}" "${GGUF_DIR}" "$(dirname "${LOG}")"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

{
  echo "=== lean export $(date -Is) ==="
  python3 "${EXPORT_SCRIPT}" \
    --config "${CONFIG}" \
    --preset qwen3-0.6b \
    --out "${EXPORT_DIR}"

  # Bake jinja from main Deploy (export already wrote from 3b pack path)
  if [[ -f "${GGUF_DIR}/llava_qwen3.jinja" ]]; then
    cp -f "${GGUF_DIR}/llava_qwen3.jinja" "${EXPORT_DIR}/llm/chat_template.jinja"
  fi

  echo "=== HF → Q8_0 direct $(date -Is) ==="
  python3 "${LLAMA_CPP}/convert_hf_to_gguf.py" \
    "${EXPORT_DIR}/llm" \
    --outfile "${GGUF_DIR}/fire-qwen3-0.6b-mobileclip-Q8_0.gguf" \
    --outtype q8_0

  if [[ -f "${GGUF_DIR}/llava_qwen3.jinja" ]]; then
    echo "=== bake chat_template $(date -Is) ==="
    TMP="${GGUF_DIR}/fire-qwen3-0.6b-mobileclip-Q8_0.gguf.tmp"
    PYTHONPATH="${LLAMA_CPP}/gguf-py:${PYTHONPATH:-}" \
      python3 "${LLAMA_CPP}/gguf-py/gguf/scripts/gguf_new_metadata.py" \
        --force \
        --chat-template-file "${GGUF_DIR}/llava_qwen3.jinja" \
        "${GGUF_DIR}/fire-qwen3-0.6b-mobileclip-Q8_0.gguf" "${TMP}"
    mv -f "${TMP}" "${GGUF_DIR}/fire-qwen3-0.6b-mobileclip-Q8_0.gguf"
  fi

  ls -lh "${GGUF_DIR}/fire-qwen3-0.6b-mobileclip-Q8_0.gguf"
  cat > "${GGUF_DIR}/MOBILECLIP_MMPROJ_BLOCKED.md" <<'EOF'
# MobileCLIP mmproj blocked

This student uses `mm_image_tower=mobileclip:MobileCLIP-S2:datacompdr` (open_clip FastViT).

llama.cpp `tools/mtmd` / `convert_image_encoder_to_gguf.py` only support CLIP-style
vision towers (and a few relatives). There is **no MobileCLIP/FastViT** path, so a
multimodal GGUF deploy is not possible without implementing FastViT in mtmd.

Artifacts:
- `fire-qwen3-0.6b-mobileclip-Q8_0.gguf` — merged LLM only (not runnable multimodal alone)
- Projector weights are in `bundles/llamacpp_export/qwen3-0.6b-mobileclip/llava.projector`
  (1280→1024→1024) but unused without a MobileCLIP mmproj

Accuracy / H4 latency for this student: use **PyTorch CPU** with open-clip-torch
(see `results/gguf_val/val_answers_fire_qwen3_0.6b_mobileclip_pytorch.jsonl`).
EOF

  echo "=== convert done $(date -Is) ==="
} 2>&1 | tee -a "${LOG}"
