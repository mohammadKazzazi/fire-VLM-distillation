#!/usr/bin/env bash
# Convert one KD_LM student → Q8_0 GGUF (+ CLIP mmproj if not MobileCLIP).
# Args: registry JSON path, slug
# Cleans F16 + lean HF export after success to save disk.
set -euo pipefail

REGISTRY="${1:?registry}"
SLUG="${2:?slug}"

DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${DEPLOY_ROOT}/../.venv/bin/activate"
LLAMA_CPP="${LLAMA_CPP:-/home/ubuntu/llavamod/llama.cpp}"
GGUF_DIR="${DEPLOY_ROOT}/bundles/gguf"
EXPORT_ROOT="${DEPLOY_ROOT}/bundles/llamacpp_export/3seeds"
CLIP_CACHE="${DEPLOY_ROOT}/cache_dir"
export HF_HOME="${HF_HOME:-/home/ubuntu/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
# llama-quantize / llama-server need build/bin on the library path
export LD_LIBRARY_PATH="${LLAMA_CPP}/build/bin:${LD_LIBRARY_PATH:-}"

mkdir -p "${GGUF_DIR}" "${EXPORT_ROOT}" /home/ubuntu/llavamod/pipeline_logs

python3 - <<PY
import json, sys
from pathlib import Path
reg=json.loads(Path("${REGISTRY}").read_text())
slug="${SLUG}"
m=next((x for x in reg["models"] if x["slug"]==slug), None)
if not m:
    sys.exit(f"slug not in registry: {slug}")
Path("/tmp/kd_lm_one.json").write_text(json.dumps(m))
print(json.dumps(m, indent=2))
PY

# shellcheck disable=SC1091
eval "$(python3 - <<'PY'
import json
m=json.loads(open("/tmp/kd_lm_one.json").read())
for k in ("lora","base","arch","model_type","jinja","gguf","mmproj","backend","family","slug"):
    v=str(m[k]).replace("'","'\\''")
    print(f"export M_{k.upper()}='{v}'")
PY
)"

EXPORT_DIR="${EXPORT_ROOT}/${M_SLUG}"
Q8="${M_GGUF}"
MMPROJ="${M_MMPROJ}"
F16="${GGUF_DIR}/fire-${M_SLUG}-f16.gguf"

# MobileCLIP: no mtmd mmproj; accuracy path is PyTorch. Skip GGUF convert.
if [[ "${M_BACKEND}" == "pytorch" ]]; then
  echo "[convert] SKIP GGUF for MobileCLIP ${M_SLUG} (eval=pytorch)"
  exit 0
fi

if [[ -f "${Q8}" && -f "${MMPROJ}" ]]; then
  sz=$(stat -c%s "${Q8}" 2>/dev/null || echo 0)
  if [[ "${sz}" -gt 1000000 ]]; then
    echo "[convert] SKIP already have Q8+mmproj for ${M_SLUG}"
    exit 0
  fi
  echo "[convert] removing stub/broken GGUF ${Q8}"
  rm -f "${Q8}" "${MMPROJ}"
fi

echo "=== convert ${M_SLUG} backend=${M_BACKEND} $(date -Is) ==="
free -h | head -2
df -h / | tail -1

echo "=== 1) lean HF export ==="
python3 "${DEPLOY_ROOT}/scripts/export_student_generic.py" \
  --lora "${M_LORA}" \
  --base "${M_BASE}" \
  --out "${EXPORT_DIR}" \
  --arch "${M_ARCH}" \
  --model-type "${M_MODEL_TYPE}" \
  --jinja "${M_JINJA}"

echo "=== 2) HF → GGUF F16 ==="
python3 "${LLAMA_CPP}/convert_hf_to_gguf.py" \
  "${EXPORT_DIR}/llm" \
  --outfile "${F16}" \
  --outtype f16

echo "=== 2b) bake chat template ==="
TMP="${F16}.tmp"
PYTHONPATH="${LLAMA_CPP}/gguf-py:${PYTHONPATH:-}" \
  python3 "${LLAMA_CPP}/gguf-py/gguf/scripts/gguf_new_metadata.py" \
    --force \
    --chat-template-file "${M_JINJA}" \
    "${F16}" "${TMP}"
mv -f "${TMP}" "${F16}"

echo "=== 3) quantize Q8_0 ==="
"${LLAMA_CPP}/build/bin/llama-quantize" "${F16}" "${Q8}" Q8_0
rm -f "${F16}"
echo "removed F16 to save disk"

echo "=== 4) mmproj F16 (select_layer=-2, pad) ==="
CLIP_SNAP="$(find "${CLIP_CACHE}/models--openai--clip-vit-large-patch14-336/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1 || true)"
if [[ -z "${CLIP_SNAP}" ]]; then
  CLIP_SNAP="$(find "${HF_HOME}/hub/models--openai--clip-vit-large-patch14-336/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1 || true)"
fi
[[ -n "${CLIP_SNAP}" ]] || { echo "CLIP snapshot missing"; exit 1; }

CFG="${CLIP_SNAP}/config.json"
CFG_BAK="${CLIP_SNAP}/config.json.kd_lm_bak"
cp -f "${CFG}" "${CFG_BAK}"
python3 - <<PY
import json
from pathlib import Path
cfg = Path("${CFG}")
d = json.loads(cfg.read_text())
d["mm_vision_select_layer"] = -2
vc = d.get("vision_config") if isinstance(d.get("vision_config"), dict) else None
if isinstance(vc, dict):
    vc["image_aspect_ratio"] = "pad"
else:
    d["image_aspect_ratio"] = "pad"
cfg.write_text(json.dumps(d, indent=2) + "\n")
print("patched CLIP", cfg)
PY

MMPROJ_TMP="$(mktemp -d)"
cleanup_clip() {
  if [[ -f "${CFG_BAK}" ]]; then
    mv -f "${CFG_BAK}" "${CFG}"
  fi
  rm -rf "${MMPROJ_TMP}"
}
trap cleanup_clip EXIT

python3 "${LLAMA_CPP}/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py" \
  -m "${CLIP_SNAP}" \
  --llava-projector "${EXPORT_DIR}/llava.projector" \
  --projector-type mlp \
  --output-dir "${MMPROJ_TMP}" \
  || python3 "${LLAMA_CPP}/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py" \
       -m "${CLIP_SNAP}" \
       --llava-projector "${EXPORT_DIR}/llava.projector" \
       --projector-type mlp \
       --clip-model-is-vision \
       --output-dir "${MMPROJ_TMP}"
cp -f "$(ls "${MMPROJ_TMP}"/mmproj-*.gguf | head -1)" "${MMPROJ}"
trap - EXIT
cleanup_clip

ls -lh "${Q8}" "${MMPROJ}"

# Drop merged HF weights (large); keep projector for debugging
rm -rf "${EXPORT_DIR}/llm"
echo "=== convert done ${M_SLUG} $(date -Is) ==="
