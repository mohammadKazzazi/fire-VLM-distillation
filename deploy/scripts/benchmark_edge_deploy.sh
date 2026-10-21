#!/usr/bin/env bash
# Latency / RAM benchmark for paper deployment table.
#
# Usage:
#   bash scripts/benchmark_edge_deploy.sh --image /path/to/frame.jpg
#   DEVICE=cpu bash scripts/benchmark_edge_deploy.sh --image frame.jpg --runs 10
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
resolve_preset
apply_offline_flags

IMAGE=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ -z "${IMAGE}" ]]; then
  # Default: first val image if Detectium symlinks exist
  CANDIDATE_ROOT="${DEPLOY_ROOT}/../LLaVA-Instruct-150K"
  IMAGE="$(find "${CANDIDATE_ROOT}/real_fire" "${CANDIDATE_ROOT}/real_non_fire" -type f \( -name '*.jpg' -o -name '*.png' \) 2>/dev/null | head -1 || true)"
  if [[ -z "${IMAGE}" ]]; then
    echo "Provide --image /path/to/frame.jpg" >&2
    exit 1
  fi
  echo "Using sample image: ${IMAGE}"
fi

mkdir -p "${DEPLOY_ROOT}/results"

python3 "${DEPLOY_ROOT}/fire_detect/bench.py" \
  --config "${CONFIG_FILE}" \
  --image "${IMAGE}" \
  --device "${DEVICE:-cuda}" \
  --warmup "${BENCHMARK_WARMUP:-20}" \
  --runs "${BENCHMARK_RUNS:-100}" \
  --restarts "${BENCHMARK_RESTARTS:-3}" \
  --concurrency "${BENCHMARK_CONCURRENCY:-1}" \
  --output "${DEPLOY_ROOT}/results/benchmark_${STUDENT_PRESET}_${DEVICE:-cuda}.json" \
  "${EXTRA[@]}"
