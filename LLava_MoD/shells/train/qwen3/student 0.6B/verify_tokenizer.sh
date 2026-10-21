#!/usr/bin/env bash
# Tokenizer gate — Qwen3-8B teacher ↔ Qwen3-0.6B student.
#
# Usage:
#   cd MoD/LLava_MoD
#   bash "shells/train/qwen3/student 0.6B/verify_tokenizer.sh"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_0.6b.env.sh"
# shellcheck disable=SC1091
source "${REPO_ROOT}/shells/conda_env.sh"
cd "${REPO_ROOT}"

python3 "${SCRIPT_DIR}/../../verify_tokenizer_gate.py" \
  --teacher "${TEACHER_LLM}" \
  --student "${STUDENT_LLM}" \
  --prompts \
    "Yes." "No." \
    "Carefully analyze the image for signs of fire or smoke." \
    "Yes. A moderate fire is burning through dry grass, posing a medium risk." \
    "No. This is a non-fire scene. No visible flames or clear smoke."
