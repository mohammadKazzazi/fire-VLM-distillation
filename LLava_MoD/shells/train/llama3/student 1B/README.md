# Llama-3.2-1B edge student pipeline

**Status:** **Done** (2026-07-05) — pretrain → Plan 4 v4 D2D → eval complete.

Distill **meta-llama/Llama-3.2-1B-Instruct** from the Llama-3.1-8B LoRA teacher. Same Plan 4 v4 JSON as 3B; **separate CLIP projector** (`llavallama3.2-1b-student-pretrain/`).

## Val results (Detectium 1,452 rows)

| Checkpoint | Yes/No | Fire→Yes | FN | FP | Obj / Env / Sev |
|------------|-------:|---------:|---:|---:|-----------------|
| **Student 1B kd_lm** | **99.79%** | **752/754** | **2** | **1** | 68.97% / 77.97% / 87.74% |

Best Plan 4 Yes/No among all students (FP=1). Answers: `detectioumfire/eval/runs/students/llama3-1b/plan4-v4-kd-lm-seed42/val_answers.jsonl`

## Prerequisites

| Artifact | Path |
|----------|------|
| Teacher LoRA | `checkpoints/llavallama3.1-8b-finetune-lora/` |
| Meta HF license | [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |
| Vision encoder | `openai/clip-vit-large-patch14-336` (must match teacher) |

Run tokenizer gate before KD:
```bash
bash "shells/train/llama3/student 1B/verify_tokenizer.sh"
```

## Run order

```bash
cd MoD/LLava_MoD

# 1) Projector pretrain (detached by default; MASTER_PORT=29501 avoids port conflicts)
bash "shells/train/llama3/student 1B/pretrain_student_1b.sh"
tail -f logs/pretrain_llama3_student_1b_latest.log

# 2) Plan 4 v4 D2D (detached by default)
bash "shells/train/llama3/student 1B/plan4_d2d_multiquery_v4.sh"
tail -f logs/plan4_d2d_multiquery_v4_kd_lm_llama3_1b_latest.log

# 3) Eval
NUM_GPUS=8 bash "shells/train/llama3/student 1B/plan4_d2d_multiquery_v4_val.sh"

# 4) Table 4 batch
cd ../detectioumfire/eval/launchers/qwen2.5 && ./run_table4_student_validation.sh
```

## Checkpoints

| Stage | Directory |
|-------|-----------|
| Pretrain | `llavallama3.2-1b-student-pretrain/` |
| Plan 4 v4 | `plan4-d2d-multiquery-v4-kd-lm-llama3-1b-seed42/` |
| Eval answers | `detectioumfire/eval/runs/students/llama3-1b/plan4-v4-kd-lm-seed42/` |

## Notes

- Do **not** reuse 3B `mm_projector.bin` (hidden size differs: 1B vs 3B).
- Same `llama_3` conv / eval path as 3B (`use_cache=False` in `model_vqa_loader.py`).
- Shared teacher paths: `shells/train/llama3/common_llama3.env.sh`
- Pretrain uses `MASTER_PORT` default **29501** — override if another DeepSpeed job holds the port.

See parent [`../README.md`](../README.md) · [`Important Notes/CURRENT_STATUS.md`](../../../../Important%20Notes/CURRENT_STATUS.md).
