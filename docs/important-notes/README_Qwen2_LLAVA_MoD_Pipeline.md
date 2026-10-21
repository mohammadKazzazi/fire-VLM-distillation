# Qwen2.5-7B LLaVA Teacher + LLaVA-MoD Distillation

Operational workflow: train a **multimodal LLaVA-style model** with a **Qwen2.5-7B** backbone, then **distill** to **Qwen2.5-3B** or **Qwen2.5-0.5B** students using **LLaVA-MoD** in `MoD/LLava_MoD`.

**Location:** `MoD/Important Notes/` (project handoff docs). **Resume:** [`CURRENT_STATUS.md`](CURRENT_STATUS.md).

Classic [`LLaVA`](../../LLaVA/) is Vicuna/Llama-centric; **Qwen** support lives in **LLaVA-MoD** (`--version qwen`).

---

## Teacher switch: Qwen2-7B → Qwen2.5-7B

The project originally trained a **Qwen2-7B** teacher (`llavaqwen-2-7b-*`). Phase 3 mimic KD failed because **teacher and student tokenizers must match**:

| Model | `vocab_size` | KD-compatible with Qwen2.5-3B? |
|-------|-------------:|----------------------------------|
| `Qwen/Qwen2-7B-Instruct` | 152064 | **No** — different tokenizer / token IDs |
| `Qwen/Qwen2.5-7B-Instruct` | 152064 | **Yes** — same Qwen2.5 tokenizer family |
| `Qwen/Qwen2.5-3B-Instruct` | 151643 | (student 3B) |
| `Qwen/Qwen2.5-0.5B-Instruct` | 151643 | (student 0.5B, edge) |
| `Qwen/Qwen2.5-1.5B-Instruct` | 151643 | (student 1.5B) |

**Rule:** Use **Qwen2.5-7B** teacher for distillation into **Qwen2.5-3B**, **1.5B**, or **0.5B**. Legacy Qwen2-7B checkpoints are ablation only.

**Other backbone families:** Gemma2, **Llama3**, and **Qwen3-4B** done; **Qwen3-0.6B edge** scaffold ready — see [`BACKBONE_FAMILIES_AND_BASELINES.md`](BACKBONE_FAMILIES_AND_BASELINES.md) and [`CURRENT_STATUS.md`](CURRENT_STATUS.md).

---

## Progress checkpoint (edit when you resume)

**Last updated:** 2026-07-01

**Where we are:**

- **Teacher Phase 1–2 — done:** Qwen2.5-7B LoRA teacher val **99.79%** → `REF_MLLM`.
- **Student projectors — done:** 3B, 0.5B, **1.5B** `mm_projector.bin`.
- **Plan 4 v4 0.5B — done:** **752/754 fire→Yes**, FN=2, FP=2 — **primary edge deploy**.
- **Plan 4 v4 1.5B — done:** 752/754, FN=2, FP=7 — middle tier.
- **Plan 4 v4 3B — done:** 747/754, best student env (82%) / sev (89%).
- **Plan 4 v2/v3 — archived.**
- **Optional next:** Plan 5 champion. See [`CURRENT_STATUS.md`](CURRENT_STATUS.md).

*Replace the paragraph above when you advance plans/phases so the next session knows which plan and checkpoint path you are on.*

---

## Why LLaVA-MoD (not only haotian-liu LLaVA)?

| Goal | Where |
|------|--------|
| Two-stage LLaVA idea (projector pretrain → multimodal tuning) | Same concept as [LLaVA README](../../LLaVA/README.md) |
| **Qwen2** backbone + multimodal code paths | [`MoD/LLava_MoD`](../../LLava_MoD) (`LlavaQwen2*`, `shells/train/qwen/`) |
| Paper-aligned **teacher → student** distillation (mimic + preference, optional MoE) | [LLaVA-MoD README](../../LLava_MoD/README.md), [TRAIN_EVAL.md](../../LLava_MoD/docs/TRAIN_EVAL.md) |

---

## Phase 1 — Adaptor / projector pretrain (frozen LLM)

**Purpose:** Align CLIP features to the language model with caption-style data (LLaVA 558K-style pretrain).

| Role | Script | `LLM` | `OUTPUT_DIR` |
|------|--------|-------|--------------|
| **Teacher** | [`pretrain_teacher.sh`](../../LLava_MoD/shells/train/qwen/pretrain_teacher.sh) | `Qwen/Qwen2.5-7B-Instruct` | `./checkpoints/llavaqwen2.5-7b-pretrain` |
| **Student** | [`pretrain_student.sh`](../../LLava_MoD/shells/train/qwen/pretrain_student.sh) | `Qwen/Qwen2.5-3B-Instruct` | `./checkpoints/llavaqwen2.5-3b-student-pretrain` |

Both use the same LCS-558K `JSON_FILE` / `IMAGE_FOLDER` under `llavamod/data/PreTrain/`. Teacher and student projectors are **separate** — do not reuse the 7B projector for the student.

**Multi-GPU:** DeepSpeed `--num_gpus` from `NUM_GPUS` env (default 4; use `NUM_GPUS=8` and `unset CUDA_VISIBLE_DEVICES` on full node).

**Artifact:** `mm_projector.bin` under `OUTPUT_DIR`.

**Detached run example:**

```bash
cd MoD/LLava_MoD
nohup bash shells/train/qwen/pretrain_student.sh > logs/pretrain_student.log 2>&1 &
```

---

## Phase 2 — Visual instruction tuning (fire-focused Qwen2.5-7B teacher)

**Purpose:** Multimodal SFT on a **fire-first** mixture. See [`TEACHER_TRAINING_DATASET.md`](TEACHER_TRAINING_DATASET.md).

### Data prep (once per machine)

```bash
# From workspace root (llava-test/)
bash LLaVA-Instruct-150K/download_coco_vg.sh
bash LLaVA-Instruct-150K/link_detectium_images.sh
python3 LLaVA_MoD/MoD/detectioumfire/build_detectium_bbox_synthetic2k_nonfire3k.py
python3 LLaVA-Instruct-150K/build_teacher_training_data.py
```

### Launch training

1. Edit [`finetune.sh`](../../LLava_MoD/shells/train/qwen/finetune.sh) or [`finetune_lora.sh`](../../LLava_MoD/shells/train/qwen/finetune_lora.sh):
   - `MLP_ADAPTOR` — Phase 1 teacher `mm_projector.bin` (`llavaqwen2.5-7b-pretrain`)
   - `JSON_FILE` / `IMAGE_FOLDER` — teacher training JSON + unified image root
2. `cd MoD/LLava_MoD` && `source shells/conda_env.sh` && `export PYTHONPATH="${PWD}:${PYTHONPATH}"`
3. `bash shells/train/qwen/finetune.sh` or `finetune_lora.sh`

**Do not train on** `detectium_llavamod_real_full_val.json`.

**Project default teacher for KD:** LoRA checkpoint + `REF_MLLM_BASE=Qwen/Qwen2.5-7B-Instruct`.

---

## Phase 3 — LLaVA-MoD distillation (7B teacher → 3B student)

Follow [TRAIN_EVAL.md](../../LLava_MoD/docs/TRAIN_EVAL.md) for upstream stages. **Active recipe: Plan 4 v4 only** (`dense2dense_fire_nonbbox_v4.json`, `kd_lm`).

1. **Student projector pretrain — done:** [`pretrain_student.sh`](../../LLava_MoD/shells/train/qwen/pretrain_student.sh).
2. **Dense → dense mimic KD — Plan 4 v4 kd_lm.** Ladder status:

| Backbone | Params | Output dir | Status |
|----------|-------:|------------|--------|
| Qwen2.5 | 0.5B | `plan4-d2d-multiquery-v4-kd-lm-0.5b-seed42/` | ✓ **edge deploy** |
| Qwen2.5 | 1.5B | `plan4-d2d-multiquery-v4-kd-lm-1.5b-seed42/` | ✓ middle tier |
| Qwen2.5 | 3B | `plan4-d2d-multiquery-v4-kd-lm-seed42/` | ✓ best env/sev |

> **Historical:** Plans 1–3, 4 v2/v3, 5, 6 and `only_kd` ablations were explored during development; artifacts removed. See [`DISTILLATION_TRAINING_DATASET.md`](DISTILLATION_TRAINING_DATASET.md).

0.5B scripts: `shells/train/qwen/student 0.5B/` — see [`student 0.5B/README.md`](../../LLava_MoD/shells/train/qwen/student%200.5B/README.md).  
1.5B scripts: `shells/train/qwen/student 1.5B/` — see [`student 1.5B/README.md`](../../LLava_MoD/shells/train/qwen/student%201.5B/README.md).

```bash
LOSS_TYPE=kd_lm  # default in dense2dense_distillation.sh
bash shells/train/qwen/plan4_d2d_multiquery_v4.sh
```

3. **Deploy** best D2D checkpoint: **`plan4-d2d-multiquery-v4-kd-lm-0.5b-seed42/`** (edge) or `plan4-d2d-multiquery-v4-kd-lm-seed42/` (3B env/sev). **Optional Phase 3b:** preference distillation — [`PREFERENCE_DISTILLATION_DATASET.md`](PREFERENCE_DISTILLATION_DATASET.md). MoE (D2S) optional later.

---

## Evaluation (Detectium / custom val)

Use **`MoD/detectioumfire/eval/`** — launchers under `launchers/qwen2.5/`:

| Path | Checkpoints | Key scripts |
|------|-------------|-------------|
| [`eval/launchers/qwen2.5/`](../../detectioumfire/eval/launchers/qwen2.5/) | `llavaqwen2.5-7b-finetune`, `llavaqwen2.5-7b-finetune-lora`, Plan 4 students | `compare_full_vs_lora.sh`, `eval_detectium_val_parallel.sh`, `run_table4_student_validation.sh` |

Legacy Qwen2-7B teacher metrics are in project notes only (old `Validation Code/Qwen2-7B/` removed).

**Qwen2.5-7B val results** (1,452 held-out rows, Yes/No prefix match):

| Checkpoint | Accuracy | TP | FP | TN | FN |
|------------|----------|----|----|----|-----|
| Full finetune | **99.79%** | 753 | 2 | 696 | 1 |
| LoRA finetune | **99.79%** | 753 | 2 | 696 | 1 |

Metrics JSON: `eval/comparisons/compare_full_vs_lora.json`.

Shared utilities: `eval/lib/` (`caption_student_pretrain.py`, `score_detectium_val.py`, `detectium_vqa_worker.py`).

For Qwen checkpoints: **`--conv-mode qwen`**. LoRA teacher loads need **`MODEL_BASE=Qwen/Qwen2.5-7B-Instruct`**. Student loads need **`model-base=Qwen/Qwen2.5-3B-Instruct`** + `projector-path` (projector-only).

---

## Pretrain batch sizes

Effective batch = `per_device_train_batch_size × gradient_accumulation_steps × NUM_GPUS`. Projector pretrain is usually forgiving; tune for memory on A100s. See stock values in `pretrain_teacher.sh` / `pretrain_student.sh`.

---

## References

| Doc | Purpose |
|-----|---------|
| [`CURRENT_STATUS.md`](CURRENT_STATUS.md) | Short resume for new chats |
| [`TEACHER_TRAINING_DATASET.md`](TEACHER_TRAINING_DATASET.md) | Teacher Phase 2 JSON |
| [`DISTILLATION_TRAINING_DATASET.md`](DISTILLATION_TRAINING_DATASET.md) | D2D Plans 1→5 |
| [`PREFERENCE_DISTILLATION_DATASET.md`](PREFERENCE_DISTILLATION_DATASET.md) | Phase 3b preference JSON |
| [`FIRE_MLLM_PROJECT_README.md`](FIRE_MLLM_PROJECT_README.md) | Strategy & mix rationale |
| [`BACKBONE_FAMILIES_AND_BASELINES.md`](BACKBONE_FAMILIES_AND_BASELINES.md) | Multi-family KD roadmap + native VLM baselines |
| [TRAIN_EVAL.md](../../LLava_MoD/docs/TRAIN_EVAL.md) | Upstream LLaVA-MoD stages |

---

## LoRA visual instruction tuning

[`finetune_lora.sh`](../../LLava_MoD/shells/train/qwen/finetune_lora.sh) wraps `train.py` with PEFT LoRA on LM linears (projector handled via `mm_projector_lr` / non-LoRA trainables). At distillation/inference, load adapter dir + base model (`REF_MLLM` + `REF_MLLM_BASE`). Trained on this project; used as default teacher.
