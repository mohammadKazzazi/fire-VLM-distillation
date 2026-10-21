#!/usr/bin/env bash
# Tokenizer gate for mimic KD — teacher and student must share identical token IDs.
# Run after huggingface-cli login and model download.
#
# Usage:
#   cd MoD/LLava_MoD
#   bash shells/train/llama3/verify_tokenizer.sh
#   bash shells/train/qwen3/verify_tokenizer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_llama3.env.sh"

# shellcheck disable=SC1091
source "${REPO_ROOT}/shells/conda_env.sh"
cd "${REPO_ROOT}"

python3 "${SCRIPT_DIR}/../verify_tokenizer_gate.py" \
  --teacher "${TEACHER_LLM}" \
  --student "${STUDENT_LLM}" \
  --prompts \
    "Yes." "No." \
    "Carefully analyze the image for signs of fire or smoke." \
    "Yes. A moderate fire is burning through dry grass, posing a medium risk." \
    "No. This is a non-fire scene. No visible flames or clear smoke."
