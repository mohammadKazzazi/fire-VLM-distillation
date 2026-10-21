# Detectium evaluation (`eval/`)

## Run a new student eval

```bash
cd MoD/LLava_MoD
bash shells/eval/plan4_d2d_multiquery_v4_val.sh          # Qwen2.5-3B default
bash shells/train/gemma2/plan4_d2d_multiquery_v4_val.sh  # Gemma2-2B
bash shells/train/llama3/plan4_d2d_multiquery_v4_val.sh  # Llama3-3B
bash "shells/train/llama3/student 1B/plan4_d2d_multiquery_v4_val.sh"  # Llama3-1B edge
bash shells/train/qwen3/plan4_d2d_multiquery_v4_val.sh  # Qwen3-4B
bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4_val.sh"  # Qwen3-0.6B edge
```

Answers land under `runs/students/<backbone>/<plan-slug>/val_answers.jsonl`.

## Train + eval — Qwen3-0.6B edge (full pipeline)

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
bash "shells/train/qwen3/student 0.6B/verify_tokenizer.sh"
bash "shells/train/qwen3/student 0.6B/pretrain_student_0.6b.sh"
tail -f logs/pretrain_qwen3_student_0.6b_latest.log
bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4.sh"
tail -f logs/plan4_d2d_multiquery_v4_kd_lm_qwen3_0.6b_latest.log
NUM_GPUS=8 bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4_val.sh"
cd detectioumfire/eval/launchers/qwen2.5 && ./run_table4_student_validation.sh
```

See [`shells/train/qwen3/student 0.6B/README.md`](../LLava_MoD/shells/train/qwen3/student%200.6B/README.md).

## Per-checkpoint layout

```
runs/students/<backbone>/<plan-slug>/
  val_answers.jsonl    # required
  metrics.json         # optional (score_detectium_val.py)
  table4.json          # optional (score_fire_reasoning_table4.py)
```

## Active runs (2026-07-05)

| Folder | Checkpoint |
|--------|------------|
| `students/qwen2.5-0.5b/plan4-v4-kd-lm-seed42/` | Qwen edge deploy |
| `students/qwen2.5-1.5b/plan4-v4-kd-lm-seed42/` | Qwen middle tier |
| `students/qwen2.5-3b/plan4-v4-kd-lm-seed42/` | Qwen best env/sev |
| `students/gemma2-2b/plan4-v4-kd-lm-seed42/` | Gemma2 deploy student |
| `students/qwen3-4b/plan4-v4-kd-lm-seed42/` | Qwen3 deploy student |
| `students/qwen3-0.6b/plan4-v4-kd-lm-seed42/` | Qwen3 0.6B edge (pending) |
| `students/llama3-3b/plan4-v4-kd-lm-seed42/` | Llama3 3B deploy student |
| `students/llama3-1b/plan4-v4-kd-lm-seed42/` | **Llama3 1B edge deploy** (99.79% Yes/No, FP=1) |
| `teacher/qwen2.5-7b-lora/` | Qwen2.5 reference teacher |
| `teacher/gemma2-9b-lora/` | Gemma2 reference teacher |
| `teacher/llama3-8b-lora/` | Llama3 reference teacher |
| `teacher/qwen3-8b-lora/` | Qwen3 reference teacher |
| `teacher/gemma2-9b-pretrain/` | Gemma2 projector-only (phase 1) |

## Table 4 batch score (all families)

```bash
cd eval/launchers/qwen2.5
./run_table4_student_validation.sh
# → comparisons/table4_all_plans.json
# → comparisons/table4_all_plans_with_yesno.json  (prefix Yes/No + yesno_content)

cd ../..
python3 lib/print_comparison_table.py
```

`yesno_content` scores fire detection from response body when models lack `Yes.`/`No.` prefixes (historical `only_kd` ablations — removed from repo).

## Inference pitfalls (Llama 3 / Qwen3 — 2026-07-04)

`llavamod/eval/model_vqa_loader.py` must use each backbone's **native end-of-turn stop token**. If the wrong stop is used (e.g. Llama 3 `<|begin_of_text|>` instead of `<|eot_id|>`), generation runs to `max_new_tokens` and answers repeat (~1,200 chars) while the leading `Yes.`/`No.` stays correct — metrics look fine but `val_answers.jsonl` is unusable.

**Fixes in repo:** correct stops per `conv_mode`, `MAX_NEW_TOKENS=64`, `trim_repetitive_answer()`, KV cache. See [`shells/train/llama3/README.md`](../LLava_MoD/shells/train/llama3/README.md) and [`shells/train/qwen3/README.md`](../LLava_MoD/shells/train/qwen3/README.md).

## Launchers by backbone

| Directory | `--conv-mode` | Training eval wrapper |
|-----------|---------------|------------------------|
| `launchers/qwen2.5/` | `qwen` | `shells/eval/plan4_d2d_multiquery_v4_val.sh` |
| `launchers/gemma2/` | `gemma_2` | `shells/train/gemma2/plan4_d2d_multiquery_v4_val.sh` |
| `launchers/llama3/` | `llama_3` | `shells/train/llama3/plan4_d2d_multiquery_v4_val.sh` or `student 1B/plan4_d2d_multiquery_v4_val.sh` |
| `launchers/qwen3/` | `qwen3` | `shells/train/qwen3/plan4_d2d_multiquery_v4_val.sh` or `student 0.6B/plan4_d2d_multiquery_v4_val.sh` |

All launchers share `eval_detectium_val_parallel.sh` (symlinked to `qwen2.5/`); defaults are chosen from the launcher directory name.
