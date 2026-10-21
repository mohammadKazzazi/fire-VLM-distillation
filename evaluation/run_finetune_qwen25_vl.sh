#!/usr/bin/env bash
# Back-compat wrapper — see models/qwen25_vl_3b/run_finetune.sh
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models/qwen25_vl_3b/run_finetune.sh" "$@"
