#!/usr/bin/env bash
# Sequential convert+eval for all KD_LM 3-seed students (skips Only_KD via registry).
# Resume-safe. Cursor-safe when launched via setsid/nohup (run_bench_durable.sh).
#
# Disk: deletes Q8+mmproj after successful score unless KEEP_GGUF=1.
set -euo pipefail

DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${DEPLOY_ROOT}/../.venv/bin/activate"

REGISTRY="${REGISTRY:-${DEPLOY_ROOT}/config/kd_lm_3seeds_registry.json}"
VAL_ROOT="${VAL_ROOT:-/home/ubuntu/llavamod/detectium-val-only}"
OUT_DIR="${OUT_DIR:-${DEPLOY_ROOT}/results/gguf_val}"
LOG_DIR="${LOG_DIR:-/home/ubuntu/llavamod/pipeline_logs}"
LLAMA_BIN="${LLAMA_BIN:-/home/ubuntu/llavamod/llama.cpp/build/bin/llama-server}"
KEEP_GGUF="${KEEP_GGUF:-0}"
PORT="${PORT:-8090}"
EXPECTED_N="${EXPECTED_N:-1452}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
STATUS="${LOG_DIR}/kd_lm_3seeds_status.jsonl"
MASTER_LOG="${LOG_DIR}/kd_lm_3seeds_master.log"

# Prefer secondary HF caches for Llama-3.2-3B / Qwen3-4B without duplicating.
HUB="${HF_HOME:-/home/ubuntu/.cache/huggingface}/hub"
mkdir -p "${HUB}"
for pair in \
  "models--meta-llama--Llama-3.2-3B-Instruct|/home/ubuntu/llavamod/llama3-3b-qwen3-4b-students/hf_cache/hub/models--meta-llama--Llama-3.2-3B-Instruct" \
  "models--Qwen--Qwen3-4B|/home/ubuntu/llavamod/llama3-3b-qwen3-4b-students/hf_cache/hub/models--Qwen--Qwen3-4B"
do
  name="${pair%%|*}"
  src="${pair##*|}"
  dst="${HUB}/${name}"
  if [[ -d "${src}" && ! -e "${dst}" ]]; then
    ln -sfn "${src}" "${dst}"
    echo "[setup] linked ${name}"
  fi
done
export HF_HOME="${HF_HOME:-/home/ubuntu/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export CPU_OPT="${CPU_OPT:-1}"
export CPU_USE_CACHE="${CPU_USE_CACHE:-1}"
export CPU_DTYPE="${CPU_DTYPE:-bfloat16}"

cd "${DEPLOY_ROOT}"

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

# Small → large so we validate the pipeline before heavy models.
ORDER=(
  qwen2.5-0.5b-seed42 qwen2.5-0.5b-seed123 qwen2.5-0.5b-seed456
  qwen3-0.6b-mobileclip-s2-seed42 qwen3-0.6b-mobileclip-s2-seed123 qwen3-0.6b-mobileclip-s2-seed456
  llama3-1b-seed42 llama3-1b-seed123 llama3-1b-seed456
  qwen2.5-1.5b-seed42 qwen2.5-1.5b-seed123 qwen2.5-1.5b-seed456
  gemma2-2b-seed42 gemma2-2b-seed123 gemma2-2b-seed456
  qwen2.5-3b-seed42 qwen2.5-3b-seed123 qwen2.5-3b-seed456
  llama3-3b-seed42 llama3-3b-seed123 llama3-3b-seed456
  qwen3-4b-seed42 qwen3-4b-seed123 qwen3-4b-seed456
)

if [[ "${#}" -gt 0 ]]; then
  ORDER=("$@")
fi

answers_complete() {
  local f="$1"
  [[ -f "${f}" ]] || return 1
  local n
  n=$(wc -l < "${f}" | tr -d ' ')
  [[ "${n}" -ge "${EXPECTED_N}" ]]
}

model_done() {
  local metrics="$1" table4="$2" answers="$3"
  [[ -f "${metrics}" && -f "${table4}" ]] && answers_complete "${answers}"
}

record() {
  local slug="$1" stage="$2" ok="$3" extra="${4:-}"
  python3 - <<PY >>"${STATUS}"
import json
print(json.dumps({
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "slug": "${slug}",
  "stage": "${stage}",
  "ok": ("${ok}" in ("1", "true", "True", "yes")),
  "extra": """${extra}""",
}))
PY
}

eval_gguf_one() {
  local slug="$1" gguf="$2" mmproj="$3" jinja="$4" answers="$5" metrics="$6" table4="$7"
  local state="/tmp/bench_llamacpp/kd_lm_${slug}"
  mkdir -p "${state}"
  pkill -TERM -f "/llama.cpp/build/bin/llama-server" 2>/dev/null || true
  sleep 1

  python3 fire_detect/eval_gguf_detectium_val.py \
    --question-file "${VAL_ROOT}/json/val_questions.jsonl" \
    --image-folder "${VAL_ROOT}/images" \
    --answers-file "${answers}" \
    --model "${gguf}" \
    --mmproj "${mmproj}" \
    --llama-bin "${LLAMA_BIN}" \
    --port "${PORT}" \
    --max-new-tokens 64 \
    --state-dir "${state}" \
    --chat-template-file "${jinja}"

  python3 "${VAL_ROOT}/eval/score_detectium_val.py" \
    --val-json "${VAL_ROOT}/json/detectium_llavamod_real_full_val.json" \
    --answers "${answers}" \
    --json-out "${metrics}"

  python3 "${VAL_ROOT}/eval/score_fire_reasoning_table4.py" \
    --val-json "${VAL_ROOT}/json/detectium_llavamod_real_full_val.json" \
    --model "${slug}" "${answers}" \
    --json-out "${table4}"
}

eval_pytorch_one() {
  local slug="$1" lora="$2" answers_name="$3" metrics="$4" table4="$5"
  local answers="${OUT_DIR}/${answers_name}"

  python3 -u scripts/eval_qwen3_0.6b_mobileclip_pytorch.py \
    --lora "${lora}" \
    --max-new-tokens 64 \
    --answers-name "${answers_name}" \
    --skip-table4

  python3 "${VAL_ROOT}/eval/score_detectium_val.py" \
    --val-json "${VAL_ROOT}/json/detectium_llavamod_real_full_val.json" \
    --answers "${answers}" \
    --json-out "${metrics}"

  python3 "${VAL_ROOT}/eval/score_fire_reasoning_table4.py" \
    --val-json "${VAL_ROOT}/json/detectium_llavamod_real_full_val.json" \
    --model "${slug}" "${answers}" \
    --json-out "${table4}"
}

log "=== KD_LM 3-seeds pipeline start n=${#ORDER[@]} ==="
log "registry=${REGISTRY} keep_gguf=${KEEP_GGUF}"

for slug in "${ORDER[@]}"; do
  mapfile -t META < <(python3 - <<PY
import json
from pathlib import Path
m=next(x for x in json.loads(Path("${REGISTRY}").read_text())["models"] if x["slug"]=="${slug}")
for k in ("backend","lora","jinja","gguf","mmproj","answers","metrics","table4"):
    print(m[k])
PY
)
  backend="${META[0]}"
  lora="${META[1]}"
  jinja="${META[2]}"
  gguf="${META[3]}"
  mmproj="${META[4]}"
  answers_name="${META[5]}"
  metrics_name="${META[6]}"
  table4_name="${META[7]}"
  answers="${OUT_DIR}/${answers_name}"
  metrics="${OUT_DIR}/${metrics_name}"
  table4="${OUT_DIR}/${table4_name}"

  log "----- ${slug} backend=${backend} -----"
  if model_done "${metrics}" "${table4}" "${answers}"; then
    log "SKIP done ${slug}"
    record "${slug}" "all" true "already_done"
    continue
  fi

  # Convert (GGUF backends only)
  if [[ "${backend}" == "gguf" ]]; then
    log "CONVERT ${slug}"
    if bash "${DEPLOY_ROOT}/scripts/convert_one_kd_lm_gguf.sh" "${REGISTRY}" "${slug}"; then
      record "${slug}" "convert" true
    else
      log "FAIL convert ${slug}"
      record "${slug}" "convert" false
      continue
    fi
  fi

  log "EVAL ${slug}"
  set +e
  if [[ "${backend}" == "gguf" ]]; then
    eval_gguf_one "${slug}" "${gguf}" "${mmproj}" "${jinja}" "${answers}" "${metrics}" "${table4}"
    rc=$?
  else
    eval_pytorch_one "${slug}" "${lora}" "${answers_name}" "${metrics}" "${table4}"
    rc=$?
  fi
  set -e

  if [[ "${rc}" -ne 0 ]] || ! model_done "${metrics}" "${table4}" "${answers}"; then
    log "FAIL eval ${slug} rc=${rc}"
    record "${slug}" "eval" false "rc=${rc}"
    continue
  fi

  log "OK ${slug}"
  record "${slug}" "eval" true
  if [[ -f "${metrics}" ]]; then
    python3 -c "import json; d=json.load(open('${metrics}')); print('  Yes/No acc=', d.get('accuracy', d.get('acc', d)))" 2>/dev/null || true
  fi

  if [[ "${KEEP_GGUF}" != "1" && "${backend}" == "gguf" ]]; then
    log "CLEAN gguf ${slug}"
    rm -f "${gguf}" "${mmproj}"
  fi
done

log "=== KD_LM 3-seeds pipeline finished $(date -Is) ==="
# Summary
python3 - <<PY
import json
from pathlib import Path
reg=json.loads(Path("${REGISTRY}").read_text())
out=Path("${OUT_DIR}")
rows=[]
for m in reg["models"]:
    met=out/m["metrics"]
    t4=out/m["table4"]
    ans=out/m["answers"]
    ok=met.is_file() and t4.is_file() and ans.is_file() and sum(1 for _ in open(ans))>=${EXPECTED_N}
    acc=None
    if met.is_file():
        try:
            d=json.loads(met.read_text())
            acc=d.get("accuracy", d.get("acc"))
        except Exception:
            pass
    rows.append((m["slug"], ok, acc))
print("SUMMARY:")
for slug, ok, acc in rows:
    print(f"  {'OK' if ok else 'MISS':4} {slug:40} acc={acc}")
done=sum(1 for _,ok,_ in rows if ok)
print(f"done {done}/{len(rows)}")
PY
