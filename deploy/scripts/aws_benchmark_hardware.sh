#!/usr/bin/env bash
# Run deployment benchmark for a paper hardware profile (H1–H4).
#
# Usage:
#   bash scripts/aws_benchmark_hardware.sh H3
#   bash scripts/aws_benchmark_hardware.sh H3 --image /path/to/frame.jpg
#   bash scripts/aws_benchmark_hardware.sh H4 --runs 20   # quick smoke test
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

PROFILE="${1:-}"
shift || true

if [[ -z "${PROFILE}" ]]; then
  cat >&2 <<'EOF'
Usage: bash scripts/aws_benchmark_hardware.sh <H1|H2|H3|H4> [bench args...]

Hardware profiles (provider-neutral paper labels):
  H1  ARM64 + NVIDIA T4G GPU   → launch on g5g.xlarge
  H2  Constrained x86 GPU      → launch on g6.xlarge + GPU_MEMORY_FRACTION=0.5
  H3  Full x86 GPU (L4 class)  → launch on g6.xlarge
  H4  ARM64 CPU only           → launch on c7g.2xlarge, DEVICE=cpu

Quick smoke test: append --runs 5 --warmup 3 --restarts 1
Full protocol:    default 100 runs, 20 warmup, 3 restarts
EOF
  exit 1
fi

export STUDENT_PRESET="${STUDENT_PRESET:-qwen0.5b}"
export HARDWARE_ID="${PROFILE}"

WARMUP="${BENCHMARK_WARMUP:-20}"
RUNS="${BENCHMARK_RUNS:-100}"
RESTARTS="${BENCHMARK_RESTARTS:-3}"
CONCURRENCY="${BENCHMARK_CONCURRENCY:-1}"

unset GPU_MEMORY_FRACTION || true

case "${PROFILE}" in
  H1)
    export DEVICE=cuda
    EXPECTED_INSTANCE="g5g.xlarge"
    MAPS_TO="ARM64 host + NVIDIA T4G (edge-oriented ARM–GPU architecture)"
    ;;
  H2)
    export DEVICE=cuda
    export GPU_MEMORY_FRACTION="${GPU_MEMORY_FRACTION:-0.5}"
    EXPECTED_INSTANCE="g6.xlarge"
    MAPS_TO="Constrained accelerator (~half L4 memory budget)"
    ;;
  H3)
    export DEVICE=cuda
    EXPECTED_INSTANCE="g6.xlarge"
    MAPS_TO="Full modern accelerator upper bound (L4 class)"
    ;;
  H4)
    export DEVICE=cpu
    export LOAD_4BIT=0
    export LOAD_8BIT=0
    # Same protocol as H1–H3; CPU adaptations only (threads + KV cache + optional INT8).
    export CPU_OPT="${CPU_OPT:-1}"
    # Dynamic INT8 did not beat fp32+oneDNN on Graviton smoke; keep opt-in.
    export CPU_INT8="${CPU_INT8:-0}"
    export CPU_COMPILE="${CPU_COMPILE:-0}"
    export CPU_THREADS="${CPU_THREADS:-$(nproc)}"
    export CPU_USE_CACHE="${CPU_USE_CACHE:-0}"
    EXPECTED_INSTANCE="c7g.2xlarge"
    MAPS_TO="ARM64 CPU-only feasibility baseline (CPU_OPT adaptations)"
    ;;
  *)
    echo "Unknown profile ${PROFILE}. Use H1, H2, H3, or H4." >&2
    exit 1
    ;;
esac

resolve_preset
apply_offline_flags

INSTANCE="$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo unknown)"
ARCH="$(uname -m)"

echo "=== Deployment benchmark: ${PROFILE} ==="
echo "Expected instance: ${EXPECTED_INSTANCE} (actual: ${INSTANCE})"
echo "Architecture:      ${ARCH}"
echo "Preset:            ${STUDENT_PRESET}"
echo "Maps to:           ${MAPS_TO}"
echo "Protocol defaults: warmup=${WARMUP} runs=${RUNS} restarts=${RESTARTS} concurrency=${CONCURRENCY}"
echo "  (pass --warmup/--runs/--restarts/--mode to override)"
if [[ -n "${GPU_MEMORY_FRACTION:-}" ]]; then
  echo "GPU memory cap:    ${GPU_MEMORY_FRACTION}"
fi
if [[ "${PROFILE}" == "H4" ]]; then
  echo "CPU adaptations:   CPU_OPT=${CPU_OPT:-0} CPU_INT8=${CPU_INT8:-0} CPU_COMPILE=${CPU_COMPILE:-0} threads=${CPU_THREADS:-auto}"
fi
echo ""

if [[ "${PROFILE}" == "H1" && "${ARCH}" != "aarch64" ]]; then
  echo "WARNING: H1 should run on ARM64 (g5g.xlarge). Current arch: ${ARCH}" >&2
fi
if [[ "${PROFILE}" == "H4" && "${ARCH}" != "aarch64" ]]; then
  echo "WARNING: H4 should run on ARM64 (c7g.2xlarge). Current arch: ${ARCH}" >&2
fi
if [[ "${PROFILE}" =~ ^H[123]$ && ! -x "$(command -v nvidia-smi || true)" ]]; then
  echo "WARNING: nvidia-smi not found — use a GPU instance." >&2
fi

mkdir -p "${DEPLOY_ROOT}/results"

# One-time cache while online
if [[ "${HF_HUB_OFFLINE:-0}" != "1" ]]; then
  bash "${SCRIPT_DIR}/cache_vision_tower.sh" || true
fi

BENCH_ARGS=(
  --hardware-id "${PROFILE}"
  --warmup "${WARMUP}"
  --runs "${RUNS}"
  --restarts "${RESTARTS}"
  --concurrency "${CONCURRENCY}"
  --output "${DEPLOY_ROOT}/results/benchmark_${STUDENT_PRESET}_${PROFILE}.json"
)

if [[ -n "${GPU_MEMORY_FRACTION:-}" ]]; then
  BENCH_ARGS+=(--gpu-memory-fraction "${GPU_MEMORY_FRACTION}")
fi

python3 "${DEPLOY_ROOT}/fire_detect/bench.py" \
  --config "${CONFIG_FILE}" \
  --device "${DEVICE}" \
  "${BENCH_ARGS[@]}" \
  "$@"

OUT="${DEPLOY_ROOT}/results/benchmark_${STUDENT_PRESET}_${PROFILE}.json"
if [[ -f "${OUT}" ]]; then
  python3 - <<PY
import json
import platform
import subprocess
from pathlib import Path

out = Path("${OUT}")
data = json.loads(out.read_text())
data["hardware_profile"] = {
    "id": "${PROFILE}",
    "expected_instance_type": "${EXPECTED_INSTANCE}",
    "actual_instance_type": "${INSTANCE}",
    "maps_to": """${MAPS_TO}""",
    "virtualized": True,
}
data["software"] = {
    "python": platform.python_version(),
    "pytorch": None,
    "transformers": None,
}
try:
    import torch
    data["software"]["pytorch"] = torch.__version__
except Exception:
    pass
try:
    import transformers
    data["software"]["transformers"] = transformers.__version__
except Exception:
    pass
try:
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip().split("\n")[0]
    data["gpu"]["smi"] = smi
except Exception:
    pass
out.write_text(json.dumps(data, indent=2) + "\n")
print(f"Finalized {out}")
PY
fi

echo ""
echo "Done. Results: ${OUT}"
echo "Run concurrency=4:"
echo "  BENCHMARK_CONCURRENCY=4 bash scripts/aws_benchmark_hardware.sh ${PROFILE} --output ${DEPLOY_ROOT}/results/benchmark_${STUDENT_PRESET}_${PROFILE}_c4.json"
