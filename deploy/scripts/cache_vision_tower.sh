#!/usr/bin/env bash
# One-time download of CLIP vision tower for offline edge deploy.
#
# Usage:
#   bash scripts/cache_vision_tower.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-large-patch14-336}"
CACHE_DIR="${DEPLOY_ROOT}/bundles/hf_cache"
mkdir -p "${CACHE_DIR}"

export HF_HOME="${CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${CACHE_DIR}/hub"

python3 - <<PY
from huggingface_hub import snapshot_download
from transformers import CLIPVisionModel, CLIPImageProcessor

model_id = "${CLIP_MODEL}"
print(f"Downloading {model_id} -> ${CACHE_DIR}")

# Download processor/config files directly from hub cache.
CLIPImageProcessor.from_pretrained(model_id)

# Newer Transformers versions block torch<2.6 for .bin loads (CVE-2025-32434).
# Force safetensors so cache works without upgrading the training env.
try:
    CLIPVisionModel.from_pretrained(model_id, use_safetensors=True)
except Exception as exc:
    print(f"Safetensors preload failed ({exc}); downloading model snapshot only.")
    snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
        local_dir_use_symlinks=False,
    )

print("Done.")
PY

echo ""
echo "Vision tower cached under ${CACHE_DIR}"
echo "For offline deploy, set in config/students.env:"
echo "  export HF_HOME=${CACHE_DIR}"
echo "  export HUGGINGFACE_HUB_CACHE=${CACHE_DIR}/hub"
echo "  export HF_HUB_OFFLINE=1"
echo "  export TRANSFORMERS_OFFLINE=1"
