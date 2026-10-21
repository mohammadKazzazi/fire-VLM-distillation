# Shared env activation for LLaVA-MoD training scripts.
# Portable: no hardcoded /home/... paths.
#
# Resolution order:
#   1. LLAVAMOD_CONDA_ENV  — absolute path to a conda env (must have bin/python)
#   2. Bundled MoD/miniconda3/envs/training (relative to this file)
#   3. LLAVAMOD_SKIP_CONDA=1 — keep current PATH / already-activated env
#   4. Else: clear error with setup hints
#
# On a new server without the bundled miniconda:
#   export LLAVAMOD_SKIP_CONDA=1
#   # after activating your own venv / conda env with torch + deps

_CONDA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_BUNDLED_CONDA_ENV="$(cd "${_CONDA_SCRIPT_DIR}/../.." && pwd)/miniconda3/envs/training"

if [[ -n "${LLAVAMOD_CONDA_ENV:-}" ]]; then
  CONDA_ENV="${LLAVAMOD_CONDA_ENV}"
  if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
    echo "LLAVAMOD_CONDA_ENV set but no python at ${CONDA_ENV}/bin/python" >&2
    exit 1
  fi
  export PATH="${CONDA_ENV}/bin:${PATH}"
elif [[ -x "${_BUNDLED_CONDA_ENV}/bin/python" ]]; then
  CONDA_ENV="${_BUNDLED_CONDA_ENV}"
  export PATH="${CONDA_ENV}/bin:${PATH}"
elif [[ "${LLAVAMOD_SKIP_CONDA:-0}" == "1" ]]; then
  if ! command -v python >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
    echo "LLAVAMOD_SKIP_CONDA=1 but no python on PATH." >&2
    exit 1
  fi
  echo "conda_env.sh: using current PATH python (LLAVAMOD_SKIP_CONDA=1)"
else
  echo "Conda env not found at ${_BUNDLED_CONDA_ENV}." >&2
  echo "On a new server, either:" >&2
  echo "  export LLAVAMOD_CONDA_ENV=/path/to/your/conda/env" >&2
  echo "  # or activate your env then: export LLAVAMOD_SKIP_CONDA=1" >&2
  echo "Then install: pip install -r requirements.txt" >&2
  echo "              pip install -r requirements-backbones-llama3-qwen3.txt" >&2
  echo "              pip install -r requirements-vision-encoders.txt" >&2
  exit 1
fi
