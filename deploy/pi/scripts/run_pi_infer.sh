#!/usr/bin/env bash
# Run ONNX fire inference (Pi / ARM / any host without Torch).
set -euo pipefail
PI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PI_ROOT}:${PYTHONPATH:-}"
python3 "${PI_ROOT}/fire_classifier/daemon.py" "$@"
