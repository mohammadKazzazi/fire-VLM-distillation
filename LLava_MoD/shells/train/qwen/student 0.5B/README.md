# Qwen2.5-0.5B student pipeline

Distill **Qwen/Qwen2.5-0.5B-Instruct** from the Qwen2.5-7B LoRA teacher. Same Plan 4 v4 JSON as 3B; **separate CLIP projector** (`llavaqwen2.5-0.5b-student-pretrain/`).

**Status (2026-06-27):** Plan 4 v4 trained and evaluated — **primary edge deploy candidate**.

## Val results — Plan 4 v4 0.5B

| Metric | 0.5B v4 | 3B v4 | Teacher 7B |
|--------|--------:|------:|-----------:|
| Yes/No | **99.72%** | 98.62% | 99.79% |
| Fire→Yes | **752/754** | 747/754 | 753/754 |
| FN / FP | **2 / 2** | 7 / 13 | 1 / 2 |
| Table4 env | 73% | **82%** | 79% |
| Table4 sev | 87% | **89%** | 90% |

## Prerequisites

| Artifact | Path |
|----------|------|
| Teacher LoRA | `checkpoints/llavaqwen2.5-7b-finetune-lora/` |
| 0.5B projector | `checkpoints/llavaqwen2.5-0.5b-student-pretrain/mm_projector.bin` |
| Vision encoder | `openai/clip-vit-large-patch14-336` (must match teacher) |

## Run order

```bash
cd MoD/LLava_MoD
unset CUDA_VISIBLE_DEVICES

# 1) Projector pretrain (detached)
LOG_DIR=logs; STAMP=$(date +%Y%m%d_%H%M%S)
nohup env NUM_GPUS=8 bash "shells/train/qwen/student 0.5B/pretrain_student_0.5b.sh" \
  >> "${LOG_DIR}/pretrain_student_0.5b_${STAMP}.log" 2>&1 &

# 2) Plan 4 v4 D2D (detached by default)
bash "shells/train/qwen/student 0.5B/plan4_d2d_multiquery_v4.sh"

# 3) Eval (after training completes)
NUM_GPUS=8 bash "shells/train/qwen/student 0.5B/plan4_d2d_multiquery_v4_val.sh"
```

## Checkpoints

| Stage | Directory |
|-------|-----------|
| Pretrain | `llavaqwen2.5-0.5b-student-pretrain/` |
| **Plan 4 v4 (deploy)** | `plan4-d2d-multiquery-v4-kd-lm-0.5b-seed42/` |

## Notes

- Do **not** use SigLIP with the current CLIP teacher without retraining the teacher.
- Do **not** reuse 3B `mm_projector.bin` (hidden 2048 vs 896).
