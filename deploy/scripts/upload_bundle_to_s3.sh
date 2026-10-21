#!/usr/bin/env bash
# Upload merged bundle + manifest to S3 (optional — for transferring to edge device).
#
# Usage:
#   S3_URI=s3://my-bucket/fire-detect/qwen0.5b bash scripts/upload_bundle_to_s3.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
resolve_preset

S3_URI="${S3_URI:-}"
if [[ -z "${S3_URI}" ]]; then
  echo "Set S3_URI=s3://bucket/prefix" >&2
  exit 1
fi

if [[ ! -d "${BUNDLE_PATH}" ]]; then
  bash "${SCRIPT_DIR}/merge_and_bundle.sh"
fi

aws s3 sync "${BUNDLE_PATH}/" "${S3_URI}/" --exclude "*.git/*"
echo "Uploaded ${BUNDLE_PATH} -> ${S3_URI}/"

if [[ -d "${DEPLOY_ROOT}/bundles/hf_cache" ]]; then
  echo "Optional: aws s3 sync ${DEPLOY_ROOT}/bundles/hf_cache ${S3_URI}-hf-cache/"
fi
