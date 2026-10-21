#!/usr/bin/env bash
# Benchmark ONNX classifier (no PyTorch).
set -euo pipefail
PI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PI_ROOT}:${PYTHONPATH:-}"
mkdir -p "${PI_ROOT}/results"
python3 "${PI_ROOT}/fire_classifier/bench_onnx.py" "$@"
