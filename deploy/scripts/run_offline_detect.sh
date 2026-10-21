#!/usr/bin/env bash
# Run single-image offline fire detection.
#
# Usage:
#   bash scripts/run_offline_detect.sh --image /path/to/frame.jpg
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash scripts/run_offline_detect.sh --image frame.jpg
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
resolve_preset
apply_offline_flags

python3 "${DEPLOY_ROOT}/fire_detect/daemon.py" \
  --config "${CONFIG_FILE}" \
  "$@"
