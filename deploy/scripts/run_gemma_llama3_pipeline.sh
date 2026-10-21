#!/usr/bin/env bash
# Durable pipeline: Gemma2 export→GGUF, then resume Llama3 Detectium val.
# Launch via: bash scripts/run_bench_durable.sh --name gemma-llama3-pipeline --log /tmp/logs/gemma_llama3_pipeline.log -- \
#   bash scripts/run_gemma_llama3_pipeline.sh
set -euo pipefail

DEPLOY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${DEPLOY_ROOT}/../.venv/bin/activate"
LLAMA_CPP="${LLAMA_CPP:-/home/ubuntu/llavamod/llama.cpp}"
STUDENTS_ROOT="/home/ubuntu/llavamod/llama3-gemma2-students"
EXPORT="${DEPLOY_ROOT}/bundles/llamacpp_export/gemma2-2b"
GGUF="${DEPLOY_ROOT}/bundles/gguf"
JINJA="${GGUF}/llava_gemma2.jinja"

source "${VENV}"
export HF_HOME="${STUDENTS_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GEMMA2_2B_BASE=google/gemma-2-2b-it
# Keep token out of this job; use local cache only.
unset HF_TOKEN || true

echo "=== pipeline start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
free -h | head -2

echo "=== 1) Gemma lean HF export $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
cd "${STUDENTS_ROOT}/workspace/Deploy"
python3 "${DEPLOY_ROOT}/scripts/export_student_lean.py" \
  --preset gemma2-2b \
  --config "${STUDENTS_ROOT}/workspace/Deploy/config/students.env" \
  --out "${EXPORT}"

echo "=== 2) Gemma GGUF convert $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 "${LLAMA_CPP}/convert_hf_to_gguf.py" \
  "${EXPORT}/llm" \
  --outfile "${GGUF}/fire-gemma2-2b-f16.gguf" \
  --outtype f16

TMP="${GGUF}/fire-gemma2-2b-f16.gguf.tmp"
PYTHONPATH="${LLAMA_CPP}/gguf-py:${PYTHONPATH:-}" \
  python3 "${LLAMA_CPP}/gguf-py/gguf/scripts/gguf_new_metadata.py" \
    --force \
    --chat-template-file "${JINJA}" \
    "${GGUF}/fire-gemma2-2b-f16.gguf" "${TMP}"
mv -f "${TMP}" "${GGUF}/fire-gemma2-2b-f16.gguf"

"${LLAMA_CPP}/build/bin/llama-quantize" \
  "${GGUF}/fire-gemma2-2b-f16.gguf" \
  "${GGUF}/fire-gemma2-2b-Q8_0.gguf" Q8_0
"${LLAMA_CPP}/build/bin/llama-quantize" \
  "${GGUF}/fire-gemma2-2b-f16.gguf" \
  "${GGUF}/fire-gemma2-2b-Q4_K_M.gguf" Q4_K_M

CLIP_SNAP="$(find "${DEPLOY_ROOT}/cache_dir/models--openai--clip-vit-large-patch14-336/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)"
MMPROJ_TMP="$(mktemp -d)"
python3 "${LLAMA_CPP}/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py" \
  -m "${CLIP_SNAP}" \
  --llava-projector "${EXPORT}/llava.projector" \
  --projector-type mlp \
  --output-dir "${MMPROJ_TMP}" \
  || python3 "${LLAMA_CPP}/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py" \
       -m "${CLIP_SNAP}" \
       --llava-projector "${EXPORT}/llava.projector" \
       --projector-type mlp \
       --clip-model-is-vision \
       --output-dir "${MMPROJ_TMP}"
cp -f "$(ls "${MMPROJ_TMP}"/mmproj-*.gguf | head -1)" "${GGUF}/mmproj-fire-gemma2-clip-vit-l-f16.gguf"
python3 "${DEPLOY_ROOT}/scripts/quantize_mmproj_q8.py" \
  "${GGUF}/mmproj-fire-gemma2-clip-vit-l-f16.gguf" \
  "${GGUF}/mmproj-fire-gemma2-clip-vit-l-Q8_0.gguf"
rm -rf "${MMPROJ_TMP}"
ls -lh "${GGUF}"/fire-gemma2* "${GGUF}"/mmproj-fire-gemma2* || true
echo "=== Gemma GGUF done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

echo "=== 3) Llama3 Detectium val resume $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
cd "${DEPLOY_ROOT}"
export BENCH_LLAMACPP_STATE_DIR="${BENCH_LLAMACPP_STATE_DIR:-/tmp/bench_llamacpp/gguf_val_llama3_q8}"
mkdir -p "${BENCH_LLAMACPP_STATE_DIR}" results/gguf_val
python3 fire_detect/eval_gguf_detectium_val.py \
  --question-file /home/ubuntu/llavamod/detectium-val-only/json/val_questions.jsonl \
  --image-folder /home/ubuntu/llavamod/detectium-val-only/images \
  --answers-file results/gguf_val/val_answers_fire_llama3_1b_gguf_q8.jsonl \
  --model bundles/gguf/fire-llama3.2-1b-Q8_0.gguf \
  --mmproj bundles/gguf/mmproj-fire-llama3-clip-vit-l-f16.gguf \
  --llama-bin "${LLAMA_CPP}/build/bin/llama-server" \
  --port 8091 \
  --max-new-tokens 64 \
  --state-dir "${BENCH_LLAMACPP_STATE_DIR}" \
  --chat-template-file bundles/gguf/llava_llama3.jinja

python3 /home/ubuntu/llavamod/detectium-val-only/eval/score_detectium_val.py \
  --val-json /home/ubuntu/llavamod/detectium-val-only/json/detectium_llavamod_real_full_val.json \
  --answers results/gguf_val/val_answers_fire_llama3_1b_gguf_q8.jsonl \
  --json-out results/gguf_val/metrics_fire_llama3_1b_gguf_q8.json

echo "=== 4) Gemma Detectium val $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
export BENCH_LLAMACPP_STATE_DIR=/tmp/bench_llamacpp/gguf_val_gemma2_q8
mkdir -p "${BENCH_LLAMACPP_STATE_DIR}"
python3 fire_detect/eval_gguf_detectium_val.py \
  --question-file /home/ubuntu/llavamod/detectium-val-only/json/val_questions.jsonl \
  --image-folder /home/ubuntu/llavamod/detectium-val-only/images \
  --answers-file results/gguf_val/val_answers_fire_gemma2_2b_gguf_q8.jsonl \
  --model bundles/gguf/fire-gemma2-2b-Q8_0.gguf \
  --mmproj bundles/gguf/mmproj-fire-gemma2-clip-vit-l-f16.gguf \
  --llama-bin "${LLAMA_CPP}/build/bin/llama-server" \
  --port 8092 \
  --max-new-tokens 64 \
  --state-dir "${BENCH_LLAMACPP_STATE_DIR}" \
  --chat-template-file bundles/gguf/llava_gemma2.jinja

python3 /home/ubuntu/llavamod/detectium-val-only/eval/score_detectium_val.py \
  --val-json /home/ubuntu/llavamod/detectium-val-only/json/detectium_llavamod_real_full_val.json \
  --answers results/gguf_val/val_answers_fire_gemma2_2b_gguf_q8.jsonl \
  --json-out results/gguf_val/metrics_fire_gemma2_2b_gguf_q8.json

echo "=== 5) H4 latency benches (reduced protocol) $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 fire_detect/bench_llamacpp.py \
  --image-list config/benchmark_images.txt \
  --warmup 5 --runs 20 --restarts 1 --mode both \
  --hardware-id H4 --port 8080 \
  --model bundles/gguf/fire-llama3.2-1b-Q8_0.gguf \
  --mmproj bundles/gguf/mmproj-fire-llama3-clip-vit-l-f16.gguf \
  --preset llama3-1b \
  --model-base meta-llama/Llama-3.2-1B-Instruct \
  --chat-template-file bundles/gguf/llava_llama3.jinja \
  --state-dir /tmp/bench_llamacpp/h4_llama3 \
  --output results/benchmark_llama3_1b_H4_llamacpp.json

python3 fire_detect/bench_llamacpp.py \
  --image-list config/benchmark_images.txt \
  --warmup 5 --runs 20 --restarts 1 --mode both \
  --hardware-id H4 --port 8081 \
  --model bundles/gguf/fire-gemma2-2b-Q8_0.gguf \
  --mmproj bundles/gguf/mmproj-fire-gemma2-clip-vit-l-f16.gguf \
  --preset gemma2-2b \
  --model-base google/gemma-2-2b-it \
  --chat-template-file bundles/gguf/llava_gemma2.jinja \
  --state-dir /tmp/bench_llamacpp/h4_gemma2 \
  --output results/benchmark_gemma2_2b_H4_llamacpp.json

echo "=== pipeline done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
