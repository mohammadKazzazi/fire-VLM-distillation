#!/bin/bash
# Student Phase 1 (1.5B): projector pretrain — legacy entrypoint.
#
# Usage:
#   NUM_GPUS=8 bash shells/train/qwen/pretrain/pretrain_student_1.5b.sh
#
# Prefer: shells/train/qwen/student 1.5B/pretrain_student_1.5b.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
exec bash "${REPO_ROOT}/shells/train/qwen/student 1.5B/pretrain_student_1.5b.sh" "$@"
