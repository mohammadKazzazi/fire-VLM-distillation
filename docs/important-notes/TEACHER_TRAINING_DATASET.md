# Teacher phase-2 training dataset (fire-focused)

This document describes the **merged multimodal** JSON for **Qwen2 / LLaVA-MoD phase-2 (visual instruction tuning)** of the fire-focused teacher. The mix prioritizes **DetectiumFire** hazard QA and bbox supervision, with a **capped** general-vision prior from filtered `llava_v1_5_mix665k.json` rows.

**Status:** Built and used for both teacher SFT runs (full + LoRA). **Next phase (student KD):** five-plan D2D ladder — [`DISTILLATION_TRAINING_DATASET.md`](DISTILLATION_TRAINING_DATASET.md).

**Design rationale:** [`FIRE_MLLM_PROJECT_README.md`](FIRE_MLLM_PROJECT_README.md)

---

## 1. Primary artifact

| Item | Path |
|------|------|
| **Training JSON** | [`LLaVA-Instruct-150K/teacher_training_data.json`](../../../LLaVA-Instruct-150K/teacher_training_data.json) |
| **Merge builder** | [`LLaVA-Instruct-150K/build_teacher_training_data.py`](../../../LLaVA-Instruct-150K/build_teacher_training_data.py) |
| **Detectium image symlinks** | [`LLaVA-Instruct-150K/link_detectium_images.sh`](../../../LLaVA-Instruct-150K/link_detectium_images.sh) |
| **COCO + VG download** | [`LLaVA-Instruct-150K/download_coco_vg.sh`](../../../LLaVA-Instruct-150K/download_coco_vg.sh) |
| **mix665k source** | [`LLaVA-Instruct-150K/llava_v1_5_mix665k.json`](../../../LLaVA-Instruct-150K/llava_v1_5_mix665k.json) |
| **Detectium VQA train** | [`detectium_llavamod_real_full_train.json`](../detectioumfire/detectium_llavamod_real_full_train.json) |
| **Detectium VQA val (hold-out)** | [`detectium_llavamod_real_full_val.json`](../detectioumfire/detectium_llavamod_real_full_val.json) |
| **Detectium bbox block** | [`detectium_bbox_synthetic2k_nonfire3k.json`](../detectioumfire/detectium_bbox_synthetic2k_nonfire3k.json) |
| **Bbox builder** | [`build_detectium_bbox_synthetic2k_nonfire3k.py`](../detectioumfire/build_detectium_bbox_synthetic2k_nonfire3k.py) |

| Metric | Value |
|--------|------:|
| **Record count** | **35,486** (all rows have an `image`) |
| **On-disk size (approx.)** | ~**24 MiB** |
| **Vision path check** | All `image` paths resolve under `IMAGE_FOLDER` after symlinks + COCO/VG download |

**Approximate mix:** ~**51%** Detectium (VQA + bbox), ~**43%** general vision prior, ~**7%** mix665k bbox-format prior.

---

## 2. Composition (concatenation order)

Built in this **fixed order** by `build_teacher_training_data.py` (`RNG_SEED = 42`):

| # | Block | Target | Actual (current build) | Source |
|---|--------|-------:|-----------------------:|--------|
| 1 | mix665k short description | 12,000 | **12,000** | Filtered mix665k (§3.1) |
| 2 | mix665k bbox format | 7,500 | **2,409** | Filtered mix665k (§3.2) |
| 3 | mix665k simple COCO/VG VQA | 3,000 | **3,000** | Filtered mix665k (§3.3) |
| 4 | Detectium real VQA train | full | **13,077** | Full train JSON |
| 5 | Detectium bbox + non-fire | full | **5,000** | Full bbox JSON |
| | **Total** | | **35,486** | |

---

## 3. mix665k filtering rules

All sampled mix665k rows must:

- Include an `image` path under **`coco/train2017/`** or **`vg/VG_100K/`**.
- Be **single-turn:** `len(conversations) == 2`.
- Pass **no-CJK** filter (§4).

### 3.1 Description block (Block 1)

Describe/detail-style prompts; not bbox prompts.

### 3.2 mix665k bbox block (Block 2)

BBox/localization prompts; **COCO `train2017/` only** — pool cap **~2,409** single-turn rows.

### 3.3 Simple VQA block (Block 3)

Single-turn COCO/VG rows that are not description or bbox; excludes heavy reasoning prompts.

---

## 4. CJK filter (mix665k general rows)

Drops rows with CJK / Hiragana / Katakana / Hangul in conversation text.

---

## 5. JSON schema (per record)

- **`id`:** string
- **`image`:** path **relative to `IMAGE_FOLDER`**
- **`conversations`:** `[{ "from": "human"|"gpt", "value": "..." }]` — `<image>` in human when applicable
- **`meta`:** on Detectium rows; loaders may ignore

**Val set:** `detectium_llavamod_real_full_val.json` (**1,452**) — **never** merge into training or distillation JSON.

---

## 6. Images and `IMAGE_FOLDER`

**Example root:** `/home/ubuntu/llava-test/LLaVA-Instruct-150K`

| Prefix in JSON | On disk |
|----------------|---------|
| `coco/train2017/…` | COCO train2017 |
| `vg/VG_100K/…` | Visual Genome |
| `real_fire/…`, `real_non_fire/…`, `stable_diff_v15/…` | Detectium symlinks |

### Trainer variables

```bash
JSON_FILE='/home/ubuntu/llava-test/LLaVA-Instruct-150K/teacher_training_data.json'
IMAGE_FOLDER='/home/ubuntu/llava-test/LLaVA-Instruct-150K'
```

### Phase 2 training status (complete)

**Current teacher (`Qwen/Qwen2.5-7B-Instruct`)** — eval: `eval/launchers/qwen2.5/`:

| Mode | Checkpoint | Val Yes/No |
|------|------------|------------|
| Full SFT | `checkpoints/llavaqwen2.5-7b-finetune/` | **99.79%** |
| LoRA SFT | `checkpoints/llavaqwen2.5-7b-finetune-lora/` | **99.79%** → **`REF_MLLM`** |

**Legacy teacher (`Qwen/Qwen2-7B-Instruct`)** — historical metrics only (superseded for KD):

| Mode | Checkpoint | Val Yes/No |
|------|------------|------------|
| Full SFT | `checkpoints/llavaqwen-2-7b-finetune/` | 99.72% |
| LoRA SFT | `checkpoints/llavaqwen-2-7b-finetune-lora/` | 99.86% |

---

## 7. Detectium bbox block (5,000 rows)

- **2,000** synthetic fire — YOLO → xyxy bbox answers
- **3,000** real non-fire — explicit “no boxes” negatives

---

## 8. Detectium VQA train block

**13,077** rows; GPT answers begin with **`Yes.`** / **`No.`**

---

## 9. Regeneration

```bash
bash LLaVA-Instruct-150K/download_coco_vg.sh
bash LLaVA-Instruct-150K/link_detectium_images.sh
python3 LLaVA_MoD/MoD/detectioumfire/build_detectium_bbox_synthetic2k_nonfire3k.py
python3 LLaVA-Instruct-150K/build_teacher_training_data.py
```

---

## 10. Distillation (Phase 3) — five-plan ladder

For **dense2dense mimic KD**, the same JSON schema applies. Do **not** default to reusing this file as-is for student D2D — it includes ~7.4k bbox rows the student plans exclude.

Run **Plans 1→5** in order — full spec in [`DISTILLATION_TRAINING_DATASET.md`](DISTILLATION_TRAINING_DATASET.md):

| Plan | Relationship to this file |
|------|---------------------------|
| 2 | Detectium block only (`detectium_llavamod_real_full_train.json`) |
| 3 | This file **minus bbox rows** (~28k) |
| 4 | Multi-query Detectium + subset of mix665k blocks 1 & 3 (no bbox) |

Plan 1 uses a 2k Detectium sample; Plan 5 tunes hyperparams on the best Plan 3/4 recipe.

---

## 11. Related docs

- Pipeline: [`README_Qwen2_LLAVA_MoD_Pipeline.md`](README_Qwen2_LLAVA_MoD_Pipeline.md)
- Resume: [`CURRENT_STATUS.md`](CURRENT_STATUS.md)
- KD data: [`DISTILLATION_TRAINING_DATASET.md`](DISTILLATION_TRAINING_DATASET.md)

---

## 12. Known limitations

1. **`id` not globally unique** on mix665k-derived rows.
2. **mix665k bbox cap** — ~2.4k single-turn COCO bbox rows only.
3. **General vision is capped** — not a full 665k LLaVA assistant.
4. **Rebuild RAM** — loads full mix665k (~1 GiB) once.

---

*Last updated: 2026-06-17 — Qwen2.5-7B teacher Phase 2 complete (val 99.79%); student projector complete; next = D2D Plans 1→5.*
