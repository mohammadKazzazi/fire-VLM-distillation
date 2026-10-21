# D2D results — Plan 4 v4 kd_lm (Detectium val, 1,452 rows)

**Last updated:** 2026-07-09  
**Recipe:** Plan 4 v4 only (`dense2dense_fire_nonbbox_v4.json`, `kd_lm`)  
**Batch JSON:** `detectioumfire/eval/comparisons/table4_all_plans_with_yesno.json`

> **Historical:** Plans 1–6 and `only_kd` ablations were run during development; artifacts removed. See [`DISTILLATION_TRAINING_DATASET.md`](DISTILLATION_TRAINING_DATASET.md).

---

## Plan 4 v4 students

| Backbone | Params | Burning Obj | Environment | Severity | Yes/No | Fire→Yes | FN | FP |
|----------|-------:|------------:|------------:|---------:|-------:|---------:|---:|---:|
| **Qwen2.5** | **0.5B** | 69.76% | 73.30% | 87.13% | **99.72%** | **752/754** | **2** | **2** |
| Qwen2.5 | 1.5B | 68.97% | 76.05% | 87.30% | 99.38% | 752/754 | 2 | 7 |
| Qwen2.5 | 3B | 69.36% | **82.09%** | **88.96%** | 98.62% | 747/754 | 7 | 13 |
| Gemma2 | 2B | 68.57% | 76.35% | 88.18% | 99.31% | 752/754 | 2 | 8 |
| Llama3 | 3B | 70.16% | 80.49% | 89.22% | 99.72% | 752/754 | 2 | 2 |
| **Llama3** | **1B** | 68.97% | 77.97% | 87.74% | **99.79%** | **752/754** | **2** | **1** |
| Qwen3 | 4B | 70.29% | 79.74% | 89.54% | 99.31% | 752/754 | 2 | 8 |
| Qwen3 | 0.6B | — | — | — | — | — | — | — |

## Teachers (reference)

| Backbone | Params | Burning Obj | Environment | Severity | Yes/No | Fire→Yes | FN | FP |
|----------|-------:|------------:|------------:|---------:|-------:|---------:|---:|---:|
| Qwen2.5 | 7B | 70.82% | 79.35% | 89.76% | 99.79% | 753/754 | 1 | 2 |
| Gemma2 | 9B | 70.56% | 80.63% | 89.69% | **99.93%** | **754/754** | **0** | **1** |
| Llama3 | 8B | **72.02%** | **83.12%** | 89.23% | 99.86% | 753/754 | 1 | 1 |
| Qwen3 | 8B | 71.88% | 83.66% | **90.34%** | 99.86% | 753/754 | 1 | 1 |

---

## Deploy candidates

| Use case | Checkpoint |
|----------|------------|
| **Edge (best Yes/No)** | `plan4-d2d-multiquery-v4-kd-lm-llama3-1b-seed42/` (99.79%, FP=1) |
| **Edge (Qwen ecosystem)** | `plan4-d2d-multiquery-v4-kd-lm-0.5b-seed42/` |
| Middle tier | `plan4-d2d-multiquery-v4-kd-lm-1.5b-seed42/` |
| Best env + severity | `plan4-d2d-multiquery-v4-kd-lm-seed42/` (Qwen2.5-3B) |

---

## Reproduce

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/detectioumfire/eval/launchers/qwen2.5
./run_table4_student_validation.sh
python3 ../../lib/print_comparison_table.py
```
