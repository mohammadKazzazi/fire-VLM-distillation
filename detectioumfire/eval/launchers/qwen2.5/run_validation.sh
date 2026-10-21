#!/usr/bin/env bash
# End-to-end Detectium validation for a Qwen2.5-7B teacher checkpoint.
#
# Usage:
#   cd "detectioumfire/eval/launchers/qwen2.5"
#   ./run_validation.sh
#   ./run_validation.sh /path/to/checkpoint
#   NUM_GPUS=8 MODEL_BASE=Qwen/Qwen2.5-7B-Instruct ./run_validation.sh ./checkpoints/llavaqwen2.5-7b-finetune-lora

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ge 1 ]]; then
  export MODEL_PATH="$1"
fi

exec "${SCRIPT_DIR}/eval_detectium_val_parallel.sh"
