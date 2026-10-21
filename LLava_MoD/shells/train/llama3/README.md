# Llama 3 fire KD pipeline

**Status:** **3B + 1B edge done** (2026-07-05).

**Teacher:** `meta-llama/Llama-3.1-8B-Instruct`  
**Students:** `meta-llama/Llama-3.2-3B-Instruct` · `meta-llama/Llama-3.2-1B-Instruct` (edge)  
**Conv / version:** `--version llama_3`

Mirror the completed Gemma2 pipeline (`shells/train/gemma2/README.md`): phases 1a → 1b → 2 → Plan 4 v4 `kd_lm`.

## Val results (Detectium 1,452 rows)

| Checkpoint | Params | Yes/No | Fire→Yes | FN | FP | Obj / Env / Sev |
|------------|-------:|-------:|---------:|---:|---:|-----------------|
| Teacher LoRA 8B | 8B | 99.86% | 753/754 | 1 | 1 | 72.02% / 83.12% / 89.23% |
| **Student 3B kd_lm** | **3B** | **99.72%** | **752/754** | **2** | **2** | 70.16% / 80.49% / 89.22% |
| **Student 1B kd_lm** | **1B** | **99.79%** | **752/754** | **2** | **1** | 68.97% / 77.97% / 87.74% |

**1B edge** has the best Yes/No and lowest FP among Plan 4 kd_lm students. Batch JSON: `detectioumfire/eval/comparisons/table4_all_plans_with_yesno.json`

## Prerequisites (do these first)

1. **Accept Meta Llama license** on Hugging Face (login alone is not enough):
   - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
   - https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
   - https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct (edge student)
   Click **Agree and access** on each page while logged in.
2. `hf auth login` (or `export HF_TOKEN=...`)
3. Upgrade transformers (env may already be on 4.57 from Qwen3 work):
   ```bash
   cd MoD/LLava_MoD
   pip install -r requirements-backbones-llama3.txt
   ```
4. **transformers ≥ 4.57** renames `--evaluation_strategy` → `--eval_strategy` in all Llama3 shell scripts.
5. Run tokenizer gate:
   ```bash
   bash shells/train/llama3/verify_tokenizer.sh
   ```

**Verify download works** (should not return 403):
```bash
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')"
```

## Pipeline

| Phase | Script | Artifact | Status |
|-------|--------|----------|--------|
| 1a Teacher projector | `pretrain/pretrain_teacher.sh` | `checkpoints/llavallama3.1-8b-pretrain/mm_projector.bin` | **Done** |
| 2 Teacher LoRA SFT | `finetune_lora.sh` | `checkpoints/llavallama3.1-8b-finetune-lora/` | **Done** (99.86% val) |
| 1b Student projector (3B) | `pretrain/pretrain_student.sh` | `checkpoints/llavallama3.2-3b-student-pretrain/` | **Done** |
| 3 Plan 4 v4 D2D (3B) | `plan4_d2d_multiquery_v4.sh` | `plan4-d2d-multiquery-v4-kd-lm-llama3-3b-seed42/` | **Done** (99.72% Yes/No) |
| Eval (3B) | `plan4_d2d_multiquery_v4_val.sh` | `runs/students/llama3-3b/plan4-v4-kd-lm-seed42/` | **Done** |
| 1b Student projector (1B) | `student 1B/pretrain_student_1b.sh` | `checkpoints/llavallama3.2-1b-student-pretrain/` | **Done** |
| 3 Plan 4 v4 D2D (1B) | `student 1B/plan4_d2d_multiquery_v4.sh` | `plan4-d2d-multiquery-v4-kd-lm-llama3-1b-seed42/` | **Done** (99.79% Yes/No) |
| Eval (1B) | `student 1B/plan4_d2d_multiquery_v4_val.sh` | `runs/students/llama3-1b/plan4-v4-kd-lm-seed42/` | **Done** |

**Loss:** `kd_lm` only (same as Qwen/Gemma deploy students — not `only_kd`).

Shared paths: `common_llama3.env.sh`

## Quick start — 3B student

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
bash shells/train/llama3/pretrain/pretrain_student.sh
bash shells/train/llama3/plan4_d2d_multiquery_v4.sh
NUM_GPUS=8 bash shells/train/llama3/plan4_d2d_multiquery_v4_val.sh
```

**Artifacts (3B):**

| Phase | Output |
|-------|--------|
| 1b | `checkpoints/llavallama3.2-3b-student-pretrain/mm_projector.bin` |
| 3 | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-llama3-3b-seed42/` |
| Eval | `detectioumfire/eval/runs/students/llama3-3b/plan4-v4-kd-lm-seed42/val_answers.jsonl` |

## Quick start — 1B edge student

See [`student 1B/README.md`](student%201B/README.md).

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
bash "shells/train/llama3/student 1B/verify_tokenizer.sh"
bash "shells/train/llama3/student 1B/pretrain_student_1b.sh"
bash "shells/train/llama3/student 1B/plan4_d2d_multiquery_v4.sh"
NUM_GPUS=8 bash "shells/train/llama3/student 1B/plan4_d2d_multiquery_v4_val.sh"
```

**Artifacts (1B):**

| Phase | Output |
|-------|--------|
| 1b | `checkpoints/llavallama3.2-1b-student-pretrain/mm_projector.bin` |
| 3 | `checkpoints/plan4-d2d-multiquery-v4-kd-lm-llama3-1b-seed42/` |
| Eval | `detectioumfire/eval/runs/students/llama3-1b/plan4-v4-kd-lm-seed42/val_answers.jsonl` |

**Teacher eval:** `detectioumfire/eval/runs/teacher/llama3-8b-lora/val_answers.jsonl` — 99.86% Yes/No, 753/754 fire→Yes.

**Note:** D2D uses the same `align_trainer.py` fixes as Qwen3 (transformers 4.57). If D2D crashes on `outputs.labels`, apply the same `CausalLMOutputWithPast` wrap in `llava_llama3.py` as in `llava_qwen3.py`.

## Eval inference note (2026-07-04)

Llama 3 uses a **new** chat template (`SeparatorStyle.LLAMA_3`, stop token `<|eot_id|>`). The shared eval path in `llavamod/eval/model_vqa_loader.py` originally picked the wrong stop string (`<|begin_of_text|>`), so generation ran to `max_new_tokens=256` and answers looped (~1,200 chars) even though the leading `Yes.` / `No.` was correct.

**Fixes applied (eval only, no retrain):**

| Change | File | Effect |
|--------|------|--------|
| Stop on `<|eot_id|>` for `llama_3` | `model_vqa_loader.py` | Correct end-of-turn criteria |
| `eos_token_id` / `pad_token_id` in `generate()` | `model_vqa_loader.py` | Fewer pad/eos warnings |
| `MAX_NEW_TOKENS=64` default | `detectium_vqa_worker.py`, `eval_detectium_val_parallel.sh` | Caps runaway length |
| `trim_repetitive_answer()` | `model_vqa_loader.py` | Trims residual loops on fire captions |
| `use_cache=False` for `llama_3` | `model_vqa_loader.py` | Stable multimodal decode (transformers 4.57) |

Qwen2.5 (`conv_phi` → `</s>`) and Gemma2 (`gemma_2` → `<eos>`) were unaffected. **Qwen3** (`qwen3` → ``) must use the same eval path — do not reuse Vicuna/`conv_phi` stops.

Re-eval after any template change; monitor `avg` answer length in `val_answers.jsonl` (expect ~100–180 chars, not 1,000+).

Add results to batch table:
```bash
cd ../detectioumfire/eval/launchers/qwen2.5 && ./run_table4_student_validation.sh
```

See [`Important Notes/BACKBONE_FAMILIES_AND_BASELINES.md`](../../../Important%20Notes/BACKBONE_FAMILIES_AND_BASELINES.md) · [`CURRENT_STATUS.md`](../../../Important%20Notes/CURRENT_STATUS.md).
