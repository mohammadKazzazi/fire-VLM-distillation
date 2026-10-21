#!/usr/bin/env bash
# Shared paths for Evaluation baseline runs (sourced by model run scripts).

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "${EVAL_DIR}/.." && pwd)"
VENV="${EVAL_DIR}/.venv"
SCRIPTS_DIR="${EVAL_DIR}/scripts"

DETECTIUM_DIR="${ROOT}/LLaVA_MoD/MoD/detectioumfire"
EVAL_ROOT="${DETECTIUM_DIR}/eval"
VC_ROOT="${EVAL_ROOT}"
VAL_JSON="${VAL_JSON:-${DETECTIUM_DIR}/detectium_llavamod_real_full_val.json}"
QUESTION_FILE="${QUESTION_FILE:-${DETECTIUM_DIR}/val_questions.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${ROOT}/LLaVA-Instruct-150K}"
TRAIN_JSON="${TRAIN_JSON:-${DETECTIUM_DIR}/detectium_llavamod_real_full_train.json}"

ensure_venv() {
  if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "Creating Evaluation venv at ${VENV}..."
    python3 -m venv "${VENV}"
    "${VENV}/bin/pip" install --upgrade pip wheel
    "${VENV}/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    "${VENV}/bin/pip" install -r "${EVAL_DIR}/requirements.txt"
  fi
}

ensure_val_questions() {
  if [[ ! -f "${QUESTION_FILE}" ]]; then
    echo "Building ${QUESTION_FILE} from val JSON..."
    python3 "${DETECTIUM_DIR}/export_val_questions_jsonl.py" \
      --input "${VAL_JSON}" \
      --output "${QUESTION_FILE}"
  fi
}

ensure_detectium_images() {
  if [[ ! -d "${IMAGE_FOLDER}/real_fire" && ! -d "${IMAGE_FOLDER}/real_non_fire" ]]; then
    echo "Missing Detectium symlinks under ${IMAGE_FOLDER}."
    echo "Run: bash ${ROOT}/LLaVA-Instruct-150K/link_detectium_images.sh"
    exit 1
  fi
}
