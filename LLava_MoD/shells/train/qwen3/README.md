# Qwen3 fire KD pipeline

**Status:** **4B done** (2026-07-05). **0.6B edge** — vision encoder ablation in `vision_encoders/`.

**Teacher:** `Qwen/Qwen3-8B`  
**Students:** `Qwen/Qwen3-4B` · `Qwen/Qwen3-0.6B` (edge)  
**Conv / version:** `--version qwen3`

> HF repo IDs are `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B`, and `Qwen/Qwen3-0.6B` (no `-Instruct` suffix).

## Prerequisites

1. `hf auth login`
2. **Required:** Qwen3 needs **torch ≥ 2.4** and **transformers ≥ 4.51** (breaks torch 2.1):
   ```bash
   cd MoD/LLava_MoD
   pip install 'torch==2.4.1' --index-url https://download.pytorch.org/whl/cu121
   pip install 'torchvision==0.19.1' --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements-backbones-llama3-qwen3.txt
   ```
3. Run tokenizer gate (prove 8B ↔ student token IDs match on fire prompts):
   ```bash
   bash shells/train/qwen3/verify_tokenizer.sh              # 4B
   bash "shells/train/qwen3/student 0.6B/verify_tokenizer.sh"  # 0.6B edge
   ```
4. **transformers ≥ 4.57** renames `--evaluation_strategy` → `--eval_strategy` in all Qwen3 shell scripts.
5. **CLIP + transformers 4.57:** `clip_encoder.py` loads vision tower with `use_safetensors=True`.
6. **transformers 4.57 trainer API:** `LLaVATrainer` / `AlignTrainer` / `DPOTrainer` updated for `_get_train_sampler`, `_save_checkpoint`, and `log(logs, start_time)`.

## Important

- Do **not** reuse `LlavaQwen2*` or Qwen2.5 projectors/checkpoints
- Use native `conv_qwen3` + `preprocess_qwen` — **not** `conv_phi` / Vicuna `USER:`
- Disable thinking mode in training/eval for deployable `Yes.` / `No.` prefixes
- Qwen3 vocab (151,669) differs from Qwen2.5 (151,643) — cross-generation KD is invalid

## Eval inference note

New chat templates (`llama_3`, `qwen3`) must use the **native end-of-turn token** in `llavamod/eval/model_vqa_loader.py`. Wrong stops cause ~1,200-char repetitive answers while Yes/No prefix stays correct.

| Backbone | Stop token | Eval `conv_mode` |
|----------|------------|------------------|
| Qwen2.5 | `</s>` | `qwen` |
| Gemma2 | `<eos>` | `gemma_2` |
| Llama 3 | `<|eot_id|>` | `llama_3` |
| Qwen3 | `` (im_end) | `qwen3` |

Defaults: `MAX_NEW_TOKENS=64`, `trim_repetitive_answer()` — see [`shells/train/llama3/README.md`](../llama3/README.md) § Eval inference note.

Student eval uses `~/.cache/huggingface` for Qwen3 student base weights (4B, 0.6B — not `cache_dir/`) to avoid re-downloading when disk is tight.

## Pipeline

| Phase | Script | Artifact | Status |
|-------|--------|----------|--------|
| 1a Teacher projector | `pretrain/pretrain_teacher.sh` | `checkpoints/llavaqwen3-8b-pretrain/mm_projector.bin` | **Done** |
| 2 Teacher LoRA SFT | `finetune_lora.sh` | `checkpoints/llavaqwen3-8b-finetune-lora/` | **Done** |
| 1b Student projector (4B) | `pretrain/pretrain_student.sh` | `checkpoints/llavaqwen3-4b-student-pretrain/mm_projector.bin` | **Done** |
| 3 Plan 4 v4 D2D (4B) | `plan4_d2d_multiquery_v4.sh` | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen3-4b-seed42/` | **Done** |
| Eval (4B) | `plan4_d2d_multiquery_v4_val.sh` | `detectioumfire/eval/runs/students/qwen3-4b/plan4-v4-kd-lm-seed42/` | **Done** |
| 1b Student projector (0.6B) | `student 0.6B/pretrain_student_0.6b.sh` or `vision_encoders/` | `llavaqwen3-0.6b-student-pretrain[-VE]/` | Pending |
| 3 Plan 4 v4 D2D (0.6B) | `vision_encoders/scripts/plan4_d2d_multiquery_v4.sh` | `plan4-...-qwen3-0.6b-{ve}-seed42/` | Pending |
| Eval (0.6B) | `vision_encoders/scripts/plan4_d2d_multiquery_v4_val.sh` | `runs/students/qwen3-0.6b/{ve}/` | Pending |

**Vision encoder ablation:** [`vision_encoders/README.md`](vision_encoders/README.md) · [`VISION_ENCODER_ROADMAP.md`](../../../Important%20Notes/VISION_ENCODER_ROADMAP.md)

Shared paths: `common_qwen3.env.sh`

Model code: `llavamod/model/language_model/llava_qwen3.py`

## Val results (Table 4, 1,452 rows)

| Model | Yes/No | Fire→Yes | FN | FP | Obj / Env / Sev |
|-------|-------:|---------:|---:|---:|-----------------|
| Teacher 8B LoRA | 99.86% | 753/754 | 1 | 1 | 71.88% / 83.66% / 90.34% |
| **Student 4B kd_lm** | **99.31%** | **752/754** | **2** | **8** | **70.29% / 79.74% / 89.54%** |

## Quick start — 4B student

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
bash shells/train/qwen3/pretrain/pretrain_student.sh
bash shells/train/qwen3/plan4_d2d_multiquery_v4.sh
NUM_GPUS=8 bash shells/train/qwen3/plan4_d2d_multiquery_v4_val.sh
```

## Quick start — 0.6B edge student

See [`student 0.6B/README.md`](student%200.6B/README.md).

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
bash "shells/train/qwen3/student 0.6B/verify_tokenizer.sh"
bash "shells/train/qwen3/student 0.6B/pretrain_student_0.6b.sh"
tail -f logs/pretrain_qwen3_student_0.6b_latest.log
bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4.sh"
tail -f logs/plan4_d2d_multiquery_v4_kd_lm_qwen3_0.6b_latest.log
NUM_GPUS=8 bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4_val.sh"
cd ../detectioumfire/eval/launchers/qwen2.5 && ./run_table4_student_validation.sh
```

## Quick start (full pipeline — teacher + 4B)

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
bash shells/train/qwen3/pretrain/pretrain_teacher.sh
bash shells/train/qwen3/pretrain/pretrain_student.sh
bash shells/train/qwen3/finetune_lora.sh
bash shells/train/qwen3/plan4_d2d_multiquery_v4.sh
tail -f logs/plan4_d2d_multiquery_v4_kd_lm_qwen3_4b_latest.log
NUM_GPUS=8 bash shells/train/qwen3/plan4_d2d_multiquery_v4_val.sh
```

See [`Important Notes/BACKBONE_FAMILIES_AND_BASELINES.md`](../../../Important%20Notes/BACKBONE_FAMILIES_AND_BASELINES.md) · [`LLAMA3_QWEN3_SCAFFOLD_README.md`](../../../Important%20Notes/LLAMA3_QWEN3_SCAFFOLD_README.md).
