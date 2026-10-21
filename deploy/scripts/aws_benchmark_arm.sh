#!/usr/bin/env bash
# Legacy alias — ARM CPU feasibility benchmark (H4).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/aws_benchmark_hardware.sh" H4 "$@"
