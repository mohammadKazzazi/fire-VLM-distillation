#!/usr/bin/env bash
# Build a transfer package for H1 / H4 deployment benchmarks on a new server.
#
# Usage (on the CURRENT g6.xlarge):
#   bash scripts/pack_for_remote_benchmark.sh
#   bash scripts/pack_for_remote_benchmark.sh --with-hf-cache   # larger, fully offline
#
# Output:
#   Deploy/transfer/llava-deploy-h1-h4-YYYYMMDD.tar.gz
#   Deploy/transfer/llava-deploy-h1-h4-YYYYMMDD/   (staging dir)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
resolve_preset

WITH_HF_CACHE=0
PACK_OUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-hf-cache) WITH_HF_CACHE=1; shift ;;
    --out-dir) PACK_OUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

STAMP="$(date +%Y%m%d)"
OUT_ROOT="${PACK_OUT_DIR:-${DEPLOY_ROOT}/transfer}"
STAGE="${OUT_ROOT}/llava-deploy-h1-h4-${STAMP}"
TAR="${OUT_ROOT}/llava-deploy-h1-h4-${STAMP}.tar.gz"

rm -rf "${STAGE}"
mkdir -p "${STAGE}/workspace/Deploy" \
         "${STAGE}/workspace/LLaVA_MoD/MoD/LLava_MoD/checkpoints" \
         "${STAGE}/workspace/LLaVA_MoD/MoD/LLava_MoD/llavamod" \
         "${STAGE}/workspace/LLaVA_MoD/MoD/LLava_MoD/shells" \
         "${STAGE}/workspace/Deploy/samples/bench_images" \
         "${STAGE}/workspace/Deploy/results"

LLAVA_SRC="${LLAVA_MOD_ROOT}"
CKPT_NAME="plan4-d2d-multiquery-v4-kd-lm-0.5b-seed42"

echo "=== Packaging H1/H4 transfer bundle ==="
echo "Stage: ${STAGE}"
echo ""

# 1) Deploy package (code + config; skip large caches/results)
echo "[1/6] Deploy code..."
rsync -a \
  --exclude 'bundles/' \
  --exclude 'cache_dir' \
  --exclude 'transfer' \
  --exclude 'results/*.json' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "${DEPLOY_ROOT}/" "${STAGE}/workspace/Deploy/"
mkdir -p "${STAGE}/workspace/Deploy/bundles" "${STAGE}/workspace/Deploy/results"

# Keep H3/H2 results for reference if present
for f in benchmark_qwen0.5b_H3.json benchmark_qwen0.5b_H2.json; do
  if [[ -f "${DEPLOY_ROOT}/results/${f}" ]]; then
    cp "${DEPLOY_ROOT}/results/${f}" "${STAGE}/workspace/Deploy/results/"
  fi
done

# 2) LLaVA-MoD inference code (no training data)
echo "[2/6] llavamod inference code..."
rsync -a \
  --exclude 'data' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "${LLAVA_SRC}/llavamod/" "${STAGE}/workspace/LLaVA_MoD/MoD/LLava_MoD/llavamod/"

# Stub conda helper so common.sh does not require the x86 training env on ARM remotes.
mkdir -p "${STAGE}/workspace/LLaVA_MoD/MoD/LLava_MoD/shells"
cat > "${STAGE}/workspace/LLaVA_MoD/MoD/LLava_MoD/shells/conda_env.sh" <<'EOF'
# Stub for remote H1/H4: activate your venv before running Deploy scripts.
# Example: source /path/to/workspace/.venv/bin/activate
return 0 2>/dev/null || true
EOF
if [[ -f "${LLAVA_SRC}/requirements.txt" ]]; then
  cp "${LLAVA_SRC}/requirements.txt" "${STAGE}/workspace/LLaVA_MoD/MoD/LLava_MoD/"
fi
# Lightweight remote requirements (preferred over full training pin set)
cat > "${STAGE}/workspace/Deploy/requirements-remote.txt" <<'EOF'
transformers>=4.45,<5
peft>=0.10
accelerate
sentencepiece
protobuf
einops
pillow
timm
shortuuid
huggingface_hub
EOF

# Top-level files sometimes imported
for f in pyproject.toml setup.py README.md; do
  [[ -f "${LLAVA_SRC}/${f}" ]] && cp "${LLAVA_SRC}/${f}" "${STAGE}/workspace/LLaVA_MoD/MoD/LLava_MoD/" || true
done

# 3) Student LoRA checkpoint only (Qwen2.5-0.5B)
echo "[3/6] Student checkpoint ${CKPT_NAME}..."
rsync -a "${LLAVA_SRC}/checkpoints/${CKPT_NAME}/" \
  "${STAGE}/workspace/LLaVA_MoD/MoD/LLava_MoD/checkpoints/${CKPT_NAME}/"

# 4) Small fixed image set for reproducible benchmarks
echo "[4/6] Sample benchmark images..."
NONFIRE_ROOT="${DEPLOY_ROOT}/../LLaVA-Instruct-150K/real_non_fire"
FIRE_ROOT="${DEPLOY_ROOT}/../LLaVA_MoD/MoD/detectioumfire/real_images/real_images/real_fire"
# Fire images may be top-level or under images/
if [[ -d "${FIRE_ROOT}/images" ]]; then
  FIRE_SEARCH="${FIRE_ROOT}/images"
else
  FIRE_SEARCH="${FIRE_ROOT}"
fi
BENCH_LIST="${STAGE}/workspace/Deploy/config/benchmark_images.txt"
: > "${BENCH_LIST}"

if [[ -d "${NONFIRE_ROOT}" ]]; then
  mapfile -t NONFIRE < <(find "${NONFIRE_ROOT}" -maxdepth 1 -name '*.jpg' | sort || true)
  NONFIRE=("${NONFIRE[@]:0:10}")
  for src in "${NONFIRE[@]}"; do
    [[ -z "${src}" ]] && continue
    base="$(basename "${src}")"
    cp "${src}" "${STAGE}/workspace/Deploy/samples/bench_images/nonfire_${base}"
    echo "samples/bench_images/nonfire_${base}" >> "${BENCH_LIST}"
  done
fi
if [[ -d "${FIRE_SEARCH}" ]]; then
  mapfile -t FIRE < <(find "${FIRE_SEARCH}" -maxdepth 1 -name '*.jpg' | sort || true)
  FIRE=("${FIRE[@]:0:10}")
  for src in "${FIRE[@]}"; do
    [[ -z "${src}" ]] && continue
    base="$(basename "${src}")"
    cp "${src}" "${STAGE}/workspace/Deploy/samples/bench_images/fire_${base}"
    echo "samples/bench_images/fire_${base}" >> "${BENCH_LIST}"
  done
fi
echo "Packed $(wc -l < "${BENCH_LIST}") benchmark images (fire + non-fire)."

# Relative paths in image list — rewrite to absolute-friendly relative from Deploy/
# Keep as relative to Deploy root (bench.py resolves relative to CWD if absolute not used)
# Convert to paths relative to Deploy for portability:
sed -i 's|^|PLACEHOLDER/|' "${BENCH_LIST}" 2>/dev/null || true
# Use Deploy-relative paths that the remote README will rewrite
python3 - <<PY
from pathlib import Path
p = Path("${BENCH_LIST}")
lines = []
for ln in p.read_text().splitlines():
    ln = ln.strip().replace("PLACEHOLDER/", "")
    if ln:
        lines.append(ln)
p.write_text("\n".join(lines) + ("\n" if lines else ""))
print(f"Wrote {len(lines)} images to benchmark_images.txt")
if len(lines) < 5:
    raise SystemExit("Too few benchmark images packed — check Detectium paths.")
PY

# Ensure demo.jpg exists
if [[ -L "${DEPLOY_ROOT}/samples/demo.jpg" || -f "${DEPLOY_ROOT}/samples/demo.jpg" ]]; then
  if [[ ! -f "${STAGE}/workspace/Deploy/samples/demo.jpg" ]]; then
    # Prefer first bench image as demo
    first="$(head -1 "${BENCH_LIST}" || true)"
    if [[ -n "${first}" && -f "${STAGE}/workspace/Deploy/${first}" ]]; then
      cp "${STAGE}/workspace/Deploy/${first}" "${STAGE}/workspace/Deploy/samples/demo.jpg"
    fi
  fi
fi

# 5) Portable students.env
echo "[5/6] Portable config..."
cat > "${STAGE}/workspace/Deploy/config/students.env" <<EOF
# Portable deploy config for H1/H4 remote servers.
# Paths below assume workspace layout:
#   <WORKSPACE>/Deploy
#   <WORKSPACE>/LLaVA_MoD/MoD/LLava_MoD

STUDENT_PRESET=qwen0.5b

# Fill WORKSPACE on the remote host (pack script leaves a placeholder).
# Or edit absolute paths after unpack.

LLAVA_MOD_ROOT=__WORKSPACE__/LLaVA_MoD/MoD/LLava_MoD
DEPLOY_ROOT=__WORKSPACE__/Deploy

QWEN0_5B_LORA=\${LLAVA_MOD_ROOT}/checkpoints/${CKPT_NAME}
QWEN0_5B_BASE=Qwen/Qwen2.5-0.5B-Instruct
QWEN0_5B_CONV=qwen
QWEN0_5B_BUNDLE=\${DEPLOY_ROOT}/bundles/qwen2-0.5b-fire-merged

CLIP_MODEL=openai/clip-vit-large-patch14-336

MAX_NEW_TOKENS=64
TEMPERATURE=0.0
DEVICE=cuda
LOAD_4BIT=0
LOAD_8BIT=0

HF_HUB_OFFLINE=0
TRANSFORMERS_OFFLINE=0
USE_MERGED_BUNDLE=0
EOF

# Python daemon cannot expand ${VAR}; write absolute-ready template for remote rewrite
cat > "${STAGE}/workspace/Deploy/config/students.env.remote" <<EOF
# Generated for remote H1/H4. Run configure_remote.sh after unpack.
STUDENT_PRESET=qwen0.5b
LLAVA_MOD_ROOT=__WORKSPACE__/LLaVA_MoD/MoD/LLava_MoD
DEPLOY_ROOT=__WORKSPACE__/Deploy
QWEN0_5B_LORA=__WORKSPACE__/LLaVA_MoD/MoD/LLava_MoD/checkpoints/${CKPT_NAME}
QWEN0_5B_BASE=Qwen/Qwen2.5-0.5B-Instruct
QWEN0_5B_CONV=qwen
QWEN0_5B_BUNDLE=__WORKSPACE__/Deploy/bundles/qwen2-0.5b-fire-merged
CLIP_MODEL=openai/clip-vit-large-patch14-336
MAX_NEW_TOKENS=64
TEMPERATURE=0.0
DEVICE=cuda
LOAD_4BIT=0
LOAD_8BIT=0
HF_HUB_OFFLINE=0
TRANSFORMERS_OFFLINE=0
USE_MERGED_BUNDLE=0
EOF

# 6) Optional HF caches (Qwen base + CLIP)
if [[ "${WITH_HF_CACHE}" == "1" ]]; then
  echo "[6/6] Packing HF caches (Qwen2.5-0.5B + CLIP)..."
  mkdir -p "${STAGE}/workspace/hf_cache/hub"
  QWEN_CACHE="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
  CLIP_CACHE="${DEPLOY_ROOT}/bundles/hf_cache/hub/models--openai--clip-vit-large-patch14-336"
  [[ -d "${QWEN_CACHE}" ]] && rsync -a "${QWEN_CACHE}" "${STAGE}/workspace/hf_cache/hub/"
  [[ -d "${CLIP_CACHE}" ]] && rsync -a "${CLIP_CACHE}" "${STAGE}/workspace/hf_cache/hub/"
  # also copy CLIP from home cache if smaller/better
  CLIP_HOME="${HOME}/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336"
  if [[ -d "${CLIP_HOME}" ]] && [[ ! -d "${STAGE}/workspace/hf_cache/hub/models--openai--clip-vit-large-patch14-336" ]]; then
    rsync -a "${CLIP_HOME}" "${STAGE}/workspace/hf_cache/hub/"
  fi
else
  echo "[6/6] Skipping HF cache (use --with-hf-cache for fully offline transfer)"
  echo "       On the remote server you will download Qwen + CLIP once while online."
fi

# Copy the remote README into the stage root
cp "${DEPLOY_ROOT}/docs/REMOTE_H1_H4_BENCHMARK.md" \
  "${STAGE}/REMOTE_H1_H4_BENCHMARK.md"

# Helper configure script
cat > "${STAGE}/configure_remote.sh" <<'EOF'
#!/usr/bin/env bash
# Run once after unpacking on the remote server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${ROOT}/workspace"
if [[ ! -d "${WS}/Deploy" ]]; then
  echo "Expected ${WS}/Deploy — unpack the tarball first." >&2
  exit 1
fi

# Rewrite absolute paths
sed "s|__WORKSPACE__|${WS}|g" \
  "${WS}/Deploy/config/students.env.remote" > "${WS}/Deploy/config/students.env"

# Rewrite image list to absolute paths
LIST="${WS}/Deploy/config/benchmark_images.txt"
if [[ -f "${LIST}" ]]; then
  tmp="$(mktemp)"
  while IFS= read -r rel; do
    [[ -z "${rel}" ]] && continue
    echo "${WS}/Deploy/${rel}"
  done < "${LIST}" > "${tmp}"
  mv "${tmp}" "${LIST}"
fi

# Point HF cache if bundled
if [[ -d "${WS}/hf_cache/hub" ]]; then
  mkdir -p "${HOME}/.cache/huggingface"
  # Prefer merging into default HF location
  mkdir -p "${HOME}/.cache/huggingface/hub"
  rsync -a "${WS}/hf_cache/hub/" "${HOME}/.cache/huggingface/hub/"
  echo "HF cache merged into ~/.cache/huggingface/hub"
fi

chmod +x "${WS}/Deploy/scripts/"*.sh
echo ""
echo "Configured workspace: ${WS}"
echo "students.env written with absolute paths."
echo "Next: follow REMOTE_H1_H4_BENCHMARK.md"
EOF
chmod +x "${STAGE}/configure_remote.sh"

echo ""
echo "Creating tarball..."
mkdir -p "${OUT_ROOT}"
tar -C "${OUT_ROOT}" -czf "${TAR}" "llava-deploy-h1-h4-${STAMP}"

echo ""
echo "=== Done ==="
du -sh "${STAGE}" "${TAR}"
echo ""
echo "Transfer this file to the remote server:"
echo "  ${TAR}"
echo ""
echo "On the remote server:"
echo "  tar -xzf llava-deploy-h1-h4-${STAMP}.tar.gz"
echo "  cd llava-deploy-h1-h4-${STAMP}"
echo "  bash configure_remote.sh"
echo "  # then follow REMOTE_H1_H4_BENCHMARK.md"
