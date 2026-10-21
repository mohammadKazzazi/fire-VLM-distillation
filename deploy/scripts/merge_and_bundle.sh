#!/usr/bin/env bash
# Merge LoRA student into a single offline bundle + manifest.
#
# Usage:
#   bash scripts/merge_and_bundle.sh
#   STUDENT_PRESET=llama3-1b bash scripts/merge_and_bundle.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
resolve_preset
apply_offline_flags

if [[ ! -d "${LORA_PATH}" ]]; then
  echo "LoRA checkpoint not found: ${LORA_PATH}" >&2
  exit 1
fi

mkdir -p "$(dirname "${BUNDLE_PATH}")"
mkdir -p "${DEPLOY_ROOT}/bundles"
mkdir -p "${DEPLOY_ROOT}/results"

echo "Merging ${STUDENT_PRESET} ..."
echo "  LoRA:   ${LORA_PATH}"
echo "  Base:   ${MODEL_BASE}"
echo "  Output: ${BUNDLE_PATH}"

python3 "${LLAVA_MOD_ROOT}/scripts/merge_lora_weights.py" \
  --model-path "${LORA_PATH}" \
  --model-base "${MODEL_BASE}" \
  --save-model-path "${BUNDLE_PATH}"

# Copy adapter-side metadata if merge script omitted non-LoRA files
if [[ -f "${LORA_PATH}/non_lora_trainables.bin" && ! -f "${BUNDLE_PATH}/non_lora_trainables.bin" ]]; then
  cp "${LORA_PATH}/non_lora_trainables.bin" "${BUNDLE_PATH}/"
fi

export DEPLOY_ROOT BUNDLE_PATH LORA_PATH STUDENT_PRESET MODEL_BASE CONV_MODE CLIP_MODEL

python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

deploy = Path(os.environ["DEPLOY_ROOT"])
bundle = Path(os.environ["BUNDLE_PATH"])
lora = Path(os.environ["LORA_PATH"])

# Fix merged config: LoRA adapter config sometimes sets vocab_size == pad_token_id.
cfg_path = bundle / "config.json"
if cfg_path.is_file():
    cfg = json.loads(cfg_path.read_text())
    pad = cfg.get("pad_token_id")
    vocab = cfg.get("vocab_size")
    if pad is not None and vocab is not None and vocab <= pad:
        cfg["vocab_size"] = max(pad + 1, 151936)
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"Patched {cfg_path}: vocab_size -> {cfg['vocab_size']}")

def dir_size_gb(p: Path) -> float:
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return round(total / (1024**3), 4)

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "student_preset": os.environ["STUDENT_PRESET"],
    "source_lora": str(lora),
    "model_base": os.environ["MODEL_BASE"],
    "conv_mode": os.environ["CONV_MODE"],
    "bundle_path": str(bundle),
    "bundle_size_gb": dir_size_gb(bundle),
    "lora_size_gb": dir_size_gb(lora),
    "vision_tower": os.environ.get("CLIP_MODEL", "openai/clip-vit-large-patch14-336"),
    "offline_ready": True,
}
out = bundle / "manifest.json"
out.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Wrote {out} ({manifest['bundle_size_gb']} GB)")
PY

echo ""
echo "Next steps:"
echo "  1) Cache CLIP + base weights locally (one-time): bash scripts/cache_vision_tower.sh"
echo "  2) Test: bash scripts/run_offline_detect.sh --image /path/to/frame.jpg"
echo "  3) Benchmark: bash scripts/benchmark_edge_deploy.sh --image /path/to/frame.jpg"
