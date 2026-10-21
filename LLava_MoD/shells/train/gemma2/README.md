# Gemma 2 fire KD pipeline

**Status:** **Complete** (2026-07-01) — teacher + student Plan 4 v4 kd_lm evaluated.

**Teacher:** `google/gemma-2-9b-it`  
**Student:** `google/gemma-2-2b-it`  
**Conv / version:** `--version gemma_2`  
**Attention:** `eager` (required — do not use `sdpa` for Gemma2 in this repo)

> **Historical:** During development we also ran a Gemma2 `only_kd` ablation on Plan 4 v4; it produced strong Table 4 scores but no deployable `Yes.`/`No.` prefixes (FN=0, FP=157 on content-based scoring). That checkpoint and its eval run were removed — use **kd_lm** only.

## Val results (Detectium 1,452 rows)

| Checkpoint | Yes/No | Fire→Yes | FN | FP | Obj / Env / Sev |
|------------|-------:|---------:|---:|---:|-----------------|
| Teacher LoRA 9B | 99.93% | 754/754 | 0 | 1 | 71% / 81% / 90% |
| **Student kd_lm 2B** | **99.31%** | **752/754** | **2** | **8** | 69% / 76% / 88% |

Batch comparison: `detectioumfire/eval/comparisons/table4_all_plans_with_yesno.json`

## Prerequisites

1. Accept Google Gemma license on Hugging Face and `huggingface-cli login`
2. Run tokenizer gate:
   ```bash
   bash shells/train/gemma2/verify_tokenizer.sh
   ```

Model wrapper `LlavaGemma2ForCausalLM` already exists in `llavamod/model/language_model/llava_gemma2.py` — no extra pip upgrade beyond base `requirements.txt`.

## Pipeline

| Phase | Script | Artifact |
|-------|--------|----------|
| 1a Teacher projector | `pretrain/pretrain_teacher.sh` | `checkpoints/llavagemma2-9b-pretrain/mm_projector.bin` |
| 1b Student projector | `pretrain/pretrain_student.sh` | `checkpoints/llavagemma2-2b-student-pretrain/mm_projector.bin` |
| 2 Teacher LoRA SFT | `finetune_lora.sh` | `checkpoints/llavagemma2-9b-finetune-lora/` |
| 3 Plan 4 v4 D2D | `plan4_d2d_multiquery_v4.sh` | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-gemma2-2b-seed42/` |
| Eval | `plan4_d2d_multiquery_v4_val.sh` | `detectioumfire/eval/runs/students/gemma2-2b/plan4-v4-kd-lm-seed42/` |
| Teacher compare | `detectioumfire/eval/launchers/gemma2/compare_pretrain_vs_lora.sh` | pretrain vs LoRA |

Shared paths: `common_gemma2.env.sh`

See [`Important Notes/LLAMA3_QWEN3_SCAFFOLD_README.md`](../../../Important%20Notes/LLAMA3_QWEN3_SCAFFOLD_README.md) (covers all non-Qwen2.5 backbones).
