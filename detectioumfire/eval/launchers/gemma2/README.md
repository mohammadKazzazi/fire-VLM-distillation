# Gemma 2 validation (Detectium val)

Eval scripts for Gemma2-9B teacher and Gemma2-2B Plan 4 v4 student checkpoints.

**Conv mode:** `gemma_2` (default when run from this directory)

## Completed eval runs

| Answers file | Model |
|--------------|-------|
| `runs/teacher/gemma2-9b-lora/val_answers.jsonl` | Teacher LoRA (99.93% Yes/No) |
| `runs/teacher/gemma2-9b-pretrain/val_answers.jsonl` | Teacher projector-only |
| `runs/students/gemma2-2b/plan4-v4-kd-lm-seed42/val_answers.jsonl` | Student kd_lm (752/754) |

> **Historical:** Gemma2 `only_kd` ablation eval run was removed.

Teacher pretrain vs LoRA: `./compare_pretrain_vs_lora.sh`

## Usage

```bash
cd detectioumfire/eval/launchers/gemma2
MODEL_PATH=/path/to/checkpoint MODEL_BASE=google/gemma-2-2b-it \
  ./eval_detectium_val_parallel.sh
```

Or from the training repo:

```bash
cd MoD/LLava_MoD
bash shells/train/gemma2/plan4_d2d_multiquery_v4_val.sh
```

`eval_detectium_val_parallel.sh` is a symlink to `../qwen2.5/eval_detectium_val_parallel.sh`; backbone defaults are selected from the launcher directory name.

Batch comparison (all families): `../qwen2.5/run_table4_student_validation.sh`
