#!/usr/bin/env bash
# Legacy alias — full GPU benchmark (H3).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/aws_benchmark_hardware.sh" H3 "$@"
