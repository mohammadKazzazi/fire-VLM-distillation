# Llama 3 validation (Detectium val)

Eval scripts for Llama-3.1-8B teacher and Llama-3.2-3B / 1B Plan 4 v4 student checkpoints.

**Status:** **Done** (2026-07-05) — teacher + 3B + 1B edge students evaluated.

**Conv mode:** `llama_3` (default when run from this directory)

## Completed eval runs

| Answers file | Model |
|--------------|-------|
| `runs/teacher/llama3-8b-lora/val_answers.jsonl` | Teacher LoRA (99.86% Yes/No, 753/754) |
| `runs/students/llama3-3b/plan4-v4-kd-lm-seed42/val_answers.jsonl` | Student 3B kd_lm (99.72% Yes/No, 752/754, FN=2, FP=2) |
| `runs/students/llama3-1b/plan4-v4-kd-lm-seed42/val_answers.jsonl` | Student 1B kd_lm (99.79% Yes/No, 752/754, FN=2, FP=1) |

## Usage

```bash
cd detectioumfire/eval/launchers/llama3
MODEL_PATH=/path/to/checkpoint MODEL_BASE=meta-llama/Llama-3.2-3B-Instruct \
  ./eval_detectium_val_parallel.sh
```

Or from the training repo:

```bash
cd MoD/LLava_MoD
# 3B
NUM_GPUS=8 bash shells/train/llama3/plan4_d2d_multiquery_v4_val.sh
# 1B edge
NUM_GPUS=8 bash "shells/train/llama3/student 1B/plan4_d2d_multiquery_v4_val.sh"
```

Default answer paths:
- 3B: `eval/runs/students/llama3-3b/plan4-v4-kd-lm-seed42/val_answers.jsonl`
- 1B: `eval/runs/students/llama3-1b/plan4-v4-kd-lm-seed42/val_answers.jsonl`

`eval_detectium_val_parallel.sh` is a symlink to `../qwen2.5/eval_detectium_val_parallel.sh`; backbone defaults are selected from the launcher directory name.

After eval, refresh batch table: `../qwen2.5/run_table4_student_validation.sh` (includes Llama3 teacher + 3B + 1B students).

## Eval inference note (2026-07-04)

First Llama 3 eval hit **runaway generation** (~1,200-char repetitive answers) because `model_vqa_loader.py` used the wrong stop token for `llama_3`. Fixed in `llavamod/eval/model_vqa_loader.py` — see [`shells/train/llama3/README.md`](../../../../LLava_MoD/shells/train/llama3/README.md) § Eval inference note. Re-run eval after any template change; expect ~100–180 chars per answer.
