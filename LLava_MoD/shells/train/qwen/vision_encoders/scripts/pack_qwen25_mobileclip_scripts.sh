#!/usr/bin/env bash
# Pack Qwen2.5-3B MobileCLIP scripts + handoff guide (small zip, no weights).
#
# Usage (from MoD/LLava_MoD):
#   bash shells/train/qwen/vision_encoders/scripts/pack_qwen25_mobileclip_scripts.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common_qwen.env.sh"

OUT="${HANDOFF_OUT:-${WORKSPACE_ROOT}/handoff_qwen25_3b_mobileclip}"
mkdir -p "${OUT}"
ZIP="${OUT}/qwen25_3b_mobileclip_scripts.zip"

(
  cd "${WORKSPACE_ROOT}"
  zip -r -q "${ZIP}" \
    LLaVA_MoD/MoD/LLava_MoD/shells/train/qwen/vision_encoders \
    "LLaVA_MoD/MoD/Important Notes/HANDOFF_QWEN25_3B_MOBILECLIP.md" \
    -x '*/__pycache__/*' '*.pyc'
)

echo "Wrote ${ZIP}"
ls -lh "${ZIP}"
echo
echo "On Triton:"
echo "  unzip -n qwen25_3b_mobileclip_scripts.zip -d \"\$WS\""
