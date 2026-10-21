# Qwen2.5-1.5B student pipeline

Distill **Qwen/Qwen2.5-1.5B-Instruct** from the Qwen2.5-7B LoRA teacher. Same Plan 4 v4 JSON as 3B/0.5B; **separate CLIP projector** (`llavaqwen2.5-1.5b-student-pretrain/`).

**Status (2026-06-30):** Plan 4 v4 trained and evaluated.

## Val results — Plan 4 v4 1.5B

| Metric | 1.5B v4 | 0.5B v4 | 3B v4 | Teacher 7B |
|--------|--------:|--------:|------:|-----------:|
| Yes/No | 99.38% | **99.72%** | 98.62% | 99.79% |
| Fire→Yes | **752/754** | **752/754** | 747/754 | 753/754 |
| FN / FP | 2 / 7 | **2 / 2** | 7 / 13 | 1 / 2 |
| Table4 env | 76% | 73% | **82%** | 79% |
| Table4 sev | 87% | 87% | **89%** | 90% |

**Role:** Middle tier — ties 0.5B on fire→Yes and FN; more false alarms (FP=7). Better environment than 0.5B; worse than 3B.

## Prerequisites

| Artifact | Path |
|----------|------|
| Teacher LoRA | `checkpoints/llavaqwen2.5-7b-finetune-lora/` |
| 1.5B projector | `checkpoints/llavaqwen2.5-1.5b-student-pretrain/mm_projector.bin` |
| Vision encoder | `openai/clip-vit-large-patch14-336` (must match teacher) |

## Run order

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
source shells/conda_env.sh
export PYTHONPATH="${PWD}:${PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES

# 1) Projector pretrain
NUM_GPUS=8 bash "shells/train/qwen/student 1.5B/pretrain_student_1.5b.sh"

# 2) Plan 4 v4 D2D (detached by default)
bash "shells/train/qwen/student 1.5B/plan4_d2d_multiquery_v4.sh"

# 3) Eval
NUM_GPUS=8 bash "shells/train/qwen/student 1.5B/plan4_d2d_multiquery_v4_val.sh"
```

## Checkpoints

| Stage | Directory |
|-------|-----------|
| Pretrain | `llavaqwen2.5-1.5b-student-pretrain/` |
| **Plan 4 v4** | `plan4-d2d-multiquery-v4-kd-lm-1.5b-seed42/` |

Eval answers: `detectioumfire/eval/runs/students/qwen2.5-1.5b/plan4-v4-kd-lm-seed42/val_answers.jsonl`

## Notes

- Do **not** reuse 3B/0.5B `mm_projector.bin` (hidden 2048 / 896 vs **1536** for 1.5B).
- Same Qwen2.5 tokenizer as teacher — required for mimic KD.
- For lowest FP, prefer **0.5B**; for env/sev, prefer **3B**.
