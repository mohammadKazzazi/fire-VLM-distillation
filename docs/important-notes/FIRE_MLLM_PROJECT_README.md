# Fire-Focused Multimodal LLM — Project Overview & Dataset Strategy

**Audience:** Collaborators, advisors, and future contributors reviewing our training plan.  
**Last updated:** 2026-07-01  
**Status:** Teacher **done** (99.79%). **Plan 4 v4 0.5B** = primary edge deploy (FN=2, FP=2). **1.5B** = middle tier (752/754, FP=7). **3B** = best env/sev. Metrics: [`D2D_RESULTS.md`](D2D_RESULTS.md).

---

## Table of contents

1. [Mission and constraints](#1-mission-and-constraints)  
2. [What we are building](#2-what-we-are-building)  
3. [End-to-end training pipeline](#3-end-to-end-training-pipeline)  
4. [Codebase map](#4-codebase-map)  
5. [Data inventory](#5-data-inventory)  
6. [LLaVA-Instruct-150K: file-by-file analysis](#6-llava-instruct-150k-file-by-file-analysis)  
7. [Recommended training mixture (fire-first)](#7-recommended-training-mixture-fire-first)  
8. [Why not bbox-only, and why captions still matter](#8-why-not-bbox-only-and-why-captions-still-matter)  
9. [JSON schema and path conventions](#9-json-schema-and-path-conventions)  
10. [Building and regenerating datasets](#10-building-and-regenerating-datasets)  
11. [Evaluation and success criteria](#11-evaluation-and-success-criteria)  
12. [Infrastructure notes (AWS EC2)](#12-infrastructure-notes-aws-ec2)  
13. [Open questions for consultation](#13-open-questions-for-consultation)  
14. [Backbone families & native VLM baselines](#14-backbone-families--native-vlm-baselines)  
15. [Related documents](#15-related-documents)

---

## 1. Mission and constraints

### Application

Train a **multimodal large language model (MLLM)** specialized for **fire understanding** in **industrial / factory** and similar hazard-aware settings — not only generic vision benchmarks.

### Deployment constraints

| Constraint | Implication |
|------------|-------------|
| **Low latency / high throughput** at inference | Final deployed model must be **small** (student), not a 7B+ teacher |
| **Limited edge compute** | Teacher is for **training/distillation**; student is the product |
| **Reliable hazard behavior** | Need strong **fire vs non-fire** discrimination, scene context, and optional **localization** — not open-ended chat |

### Research framing

- **Method:** [LLaVA-MoD](https://arxiv.org/abs/2408.15881) — projector pretrain → multimodal SFT (teacher) → mimic / preference distillation (student).  
- **Domain data:** [DetectiumFire](https://arxiv.org/abs/2511.02495) (Kaggle: `yimengfuyao/detectiumfire`) — real fire/non-fire images, synthetic fire, YOLO labels, preference pairs, video (optional).  
- **Publication target:** ACL-oriented contribution (domain adaptation, safety under hazard scenes, data efficiency, deployment metrics — exact claim TBD).

---

## 2. What we are building

### Teacher (large)

- **Backbone:** `Qwen/Qwen2.5-7B-Instruct` (same Qwen2.5 tokenizer family as the student — required for mimic KD).  
- **Vision:** CLIP ViT-L/336 + 2-layer MLP projector (`openai/clip-vit-large-patch14-336`).  
- **Role:** Strong **fire-aware** multimodal teacher producing logits / alignments for distillation.

**Note:** An initial **Qwen2-7B** teacher (`llavaqwen-2-7b-*`) was trained and validated (LoRA val **99.86%**) but **cannot** distill into Qwen2.5-3B because the tokenizers are incompatible (`Qwen2-7B` vs `Qwen2.5-3B` differ in `vocab_size` and token↔id mappings). Legacy checkpoints remain for ablation only.

### Student (small)

- **Backbone:** `Qwen2.5-0.5B` (edge deploy), **`Qwen2.5-1.5B`** (middle tier), or `Qwen2.5-3B` (env/severity) — Plan 4 v4.  
- **Role:** Deployable MLLM after LLaVA-MoD mimic (+ optional preference) stages.

**Cross-family comparison (planned):** Gemma2 (9B→2B), Llama 3 (8B→3B), Qwen3 (8B→4B) — full spec in [`BACKBONE_FAMILIES_AND_BASELINES.md`](BACKBONE_FAMILIES_AND_BASELINES.md). Each family requires a separate Phase 1–3 pipeline and tokenizer verification (same rule as Qwen2 vs Qwen2.5).

### Parallel track (out of scope for this doc’s main path)

- **Vicuna LLaVA v1.5 + LoRA** on Detectium — documented in [`README_Detectium_Finetune.md`](README_Detectium_Finetune.md). Same JSON layout ideas; different codebase and `--conv-mode`.

---

## 3. End-to-end training pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Projector pretrain (frozen LLM)                                │
│   Data: LLaVA LCS-558K (general captions) — NOT fire-specific           │
│   Scripts: pretrain_teacher.sh (7B) / pretrain_student.sh (3B) / pretrain_student_0.5b.sh │
│   Output: mm_projector.bin                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Visual instruction tuning — FIRE-FOCUSED TEACHER               │
│   Data: teacher_training_data.json + unified IMAGE_FOLDER                 │
│   Scripts: finetune.sh (full) and finetune_lora.sh (LoRA)               │
│   Output: llavaqwen2.5-7b-finetune/ and/or llavaqwen2.5-7b-finetune-lora/│
│           → REF_MLLM for Phase 3                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 3: LLaVA-MoD distillation (7B teacher → 3B / 0.5B students)     │
│   Student projectors: 3B + 0.5B — done                                 │
│   D2D Plans 1→5 — Plan 4 v4 done (3B + 0.5B); deploy 0.5B for edge     │
│   See MoD/LLaVA-MoD/docs/TRAIN_EVAL.md                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 4 (future): Native VLM baselines — compare on same Detectium val   │
│   Off-the-shelf VLMs (Qwen2.5-VL, InternVL, MiniCPM-V, Gemma 3, …)      │
│   Zero-shot and/or Detectium train fine-tune — NOT LLaVA-MoD KD          │
│   See BACKBONE_FAMILIES_AND_BASELINES.md §Track B                        │
└─────────────────────────────────────────────────────────────────────────┘
```

**Checkpoint status (2026-07-05):**

| Stage | Status | Path / note |
|-------|--------|-------------|
| Phase 1–3 Qwen2.5 | **Done** | Plan 4 v4 0.5B / 1.5B / 3B — [`DISTILLATION_TRAINING_DATASET.md`](DISTILLATION_TRAINING_DATASET.md) |
| **Gemma2 teacher + student** | **Done** | `llavagemma2-9b-finetune-lora/`, `plan4-d2d-multiquery-v4-kd-lm-gemma2-2b-seed42/` |
| **Llama3 teacher + students** | **Done** | 3B + 1B edge — `plan4-d2d-multiquery-v4-kd-lm-llama3-{3b,1b}-seed42/` |
| **Qwen3 teacher + student** | **Done** | `plan4-d2d-multiquery-v4-kd-lm-qwen3-4b-seed42/` |
| Phase 3 D2S / preference | Not started | Optional after best D2D deployed |

**Legacy (Qwen2-7B teacher — superseded):** `llavaqwen-2-7b-pretrain`, `llavaqwen-2-7b-finetune`, `llavaqwen-2-7b-finetune-lora` (LoRA val **99.86%**). Historical metrics only (old eval tree removed).

---

## 4. Codebase map

| Path | Purpose |
|------|---------|
| [`MoD/LLaVA-MoD/`](MoD/LLaVA-MoD/) | Main training code (`llavamod/`), Qwen shells, DeepSpeed configs |
| [`MoD/detectioumfire/`](MoD/detectioumfire/) | Detectium Kaggle data, LLaVA-MoD JSONs, bbox builder, eval JSONL |
| [`MoD/DetectiumFire/`](MoD/DetectiumFire/) | Extra metadata (`fire_prompts.json` aligned with real_fire images) |
| [`LLaVA-Instruct-150K/`](../../../LLaVA-Instruct-150K/) | Hugging Face instruct JSONs + unified image root for teacher SFT |
| [`LLaVA-Instruct-150K/build_teacher_training_data.py`](../../../LLaVA-Instruct-150K/build_teacher_training_data.py) | Merges mix665k + Detectium → `teacher_training_data.json` |
| [`README_Qwen2_LLAVA_MoD_Pipeline.md`](README_Qwen2_LLAVA_MoD_Pipeline.md) | Operational pipeline & hyperparameter notes |
| [`TEACHER_TRAINING_DATASET.md`](TEACHER_TRAINING_DATASET.md) | Fire-focused 35k mixture, schema & regeneration |
| [`PREFERENCE_DISTILLATION_DATASET.md`](PREFERENCE_DISTILLATION_DATASET.md) | Phase 3b fire QA preference JSON (after D2D) |
| [`CURRENT_STATUS.md`](CURRENT_STATUS.md) | Short resume for new chats |
| [`BACKBONE_FAMILIES_AND_BASELINES.md`](BACKBONE_FAMILIES_AND_BASELINES.md) | Multi-family KD + native VLM baselines |

**Workspace root (this server):** `/home/ubuntu/llava-test/`

---

## 5. Data inventory

### 5.1 DetectiumFire (domain — **primary**)

**Location:** `MoD/detectioumfire/`  
**Source:** [Kaggle — DetectiumFire](https://www.kaggle.com/datasets/yimengfuyao/detectiumfire)

| On-disk folder | Approx. size | Contents |
|----------------|--------------|----------|
| `real_images/real_images/real_fire/` | part of ~8.3G | ~7,549 fire images + YOLO labels |
| `real_images/real_images/real_non_fire/` | | ~6,980 non-fire images |
| `synthetic_images/synthetic_images/stable_diff_v15/train/` | ~369M | ~3,915 synthetic fire images + labels |
| `preference_dataset/` | ~4.8G | `preference.json` + img1/img2/img3 (needs conversion for DPO) |
| `real_video/` | (optional) | Fire/non-fire clips — not yet in core teacher JSON |

### 5.2 Generated LLaVA-MoD JSONs (Detectium)

| File | Records | Use |
|------|--------:|-----|
| `detectium_llavamod_real_full.json` | 14,529 | Full set (fire + non-fire) |
| `detectium_llavamod_real_full_train.json` | 13,077 | **Phase 2 train** — always include in full |
| `detectium_llavamod_real_full_val.json` | 1,452 | **Held-out eval** — do not train on this |
| `detectium_bbox_synthetic2k_nonfire3k.json` | 5,000 | 2k synthetic fire bbox + 3k non-fire “no box” negatives |
| `detectium_llavamod_real_small.json` | pilot | Quick experiments |

**Conversation pattern (fire):**

- Human: `<image>` + fire-focused question (e.g. “Carefully analyze the image for signs of fire or smoke.”).  
- GPT: starts with **`Yes.`** or **`No.`** for binary clarity, then short hazard/scene explanation.

### 5.3 LLaVA-Instruct-150K (general instruct — **selective use only**)

**Location:** `/home/ubuntu/llava-test/LLaVA-Instruct-150K/`  
**Source:** [Hugging Face — liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)

| File | Size | Records |
|------|------|--------:|
| `llava_v1_5_mix665k.json` | ~1.0 GB | 665,298 |
| `llava_instruct_150k.json` | 219 MB | 157,712 |
| `llava_instruct_80k.json` | 125 MB | 80,000 |
| `conversation_58k.json` | 121 MB | 56,681 |
| `complex_reasoning_77k.json` | 76 MB | 76,643 |
| `detail_23k.json` | 20 MB | 23,240 |

**Images for mix665k subsets** (if used): require separate download of COCO `train2017`, Visual Genome `VG_100K`, etc., under the same root as Detectium symlinks — see §9.

### 5.4 Phase 1 only (not for fire SFT)

**LLaVA LCS-558K** — `blip_laion_cc_sbu_558k.json` + `Images/` under `MoD/LLaVA-MoD/llavamod/data/PreTrain/`. Used only for **projector alignment**, not fire specialization.

---

## 6. LLaVA-Instruct-150K: file-by-file analysis

Analysis performed on the downloaded JSONs (heuristic tagging of human prompts + structure). Counts are approximate.

### 6.1 `llava_v1_5_mix665k.json` — **the only file worth mining**

| Slice in mix665k | ~Count | % | Fire-detection relevance | Recommendation |
|------------------|--------|---|------------------------|----------------|
| Multi-turn dialogue | ~356k | 53% | **Low** — generic chat | **Exclude** |
| COCO general VQA | ~147k | 22% | **Low** | **Exclude** (or tiny capped sample) |
| **BBox / localization** | **~67k** | **10%** | **Medium** — teaches xyxy format, not fire semantics | **Optional subset** (e.g. 5–20k) |
| Text-only (no image) | ~41k | 6% | **None** for vision fire model | **Exclude** |
| OCR-VQA + TextVQA | ~102k | 15% | **Low** | **Exclude** |
| GQA + misc reasoning | ~19k | 3% | **Low** | **Exclude** |
| Short description | ~13k | 2% | **Medium** — scene/object prior | **Include capped** (see §7) |

**BBox example (useful format, non-fire content):**

```text
Human: Please provide the bounding box coordinate of the region this sentence describes: white doughnut right.
GPT:   [0.52, 0.59, 0.82, 0.83]
```

**Image path prefixes inside mix665k:**

| Prefix | ~Rows |
|--------|------:|
| `coco/train2017/` | 364,100 |
| `ocr_vqa/images` | 80,000 |
| `gqa/images` | 72,140 |
| `vg/VG_100K/` | 53,343 |
| `vg/VG_100K_2/` | 33,074 |
| `textvqa/train_images` | 21,953 |
| *(no image)* | 40,688 |

### 6.2 Standalone 150K component files

These use **bare COCO filenames** (`000000442786.jpg`), not `coco/train2017/...` paths. They are **not** row-aligned with mix665k paths but represent the **same task families**.

| File | Dominant task type | Fire relevance | Recommendation |
|------|-------------------|----------------|----------------|
| **`conversation_58k.json`** | ~92% multi-turn conversation | **Very low** | **Do not use** for fire teacher |
| **`complex_reasoning_77k.json`** | ~80% general VQA + ~18% long reasoning | **Very low** | **Do not use** |
| **`detail_23k.json`** | ~74% general VQA + ~18% “describe image” | **Low–medium** (caption prior only) | **Optional**; prefer mix665k description rows + path consistency |
| **`llava_instruct_150k.json`** | Mix of conversation + general VQA | **Low** | **Do not use** as a whole |
| **`llava_instruct_80k.json`** | Same, smaller | **Low** | **Do not use** |

**Conclusion for collaborators:** For fire detection, **do not train on conversation_58k or complex_reasoning**. Use **`mix665k` as a filtered quarry**, not as a full 665k mixture. The standalone shards add little beyond what you can filter from `mix665k` with consistent image paths.

---

## 7. Recommended training mixture (fire-first)

This section captures **current team intent** after discussion. It **replaces** the older 215,577-row recipe in [`TEACHER_TRAINING_DATASET.md`](TEACHER_TRAINING_DATASET.md) (150k COCO + 30k VG + 20k text-only + Detectium), which overweights **general assistant** behavior.

### Design principles

1. **Detectium is the center of gravity** (~60–80% of training steps or samples).  
2. **General vision is supporting, not dominant** — captions + simple single-turn VQA, heavily capped.  
3. **BBox data is supplementary** — teaches localization format aligned with synthetic fire boxes; not the main objective.  
4. **No multi-turn conversation** from LLaVA-Instruct for this teacher.  
5. **Val set is sacred** — never merge `detectium_llavamod_real_full_val.json` into training.

### Proposed blocks (starting point for consultation)

| # | Block | Target count | Source | Role |
|---|--------|-------------:|--------|------|
| A | Detectium real VQA train | **13,077** | `detectium_llavamod_real_full_train.json` | Fire/non-fire QA, severity, scene language |
| B | Fire bbox + non-fire negatives | **5,000** | `detectium_bbox_synthetic2k_nonfire3k.json` | Spatial supervision + hard negatives |
| C | Scene caption / short describe | **10,000–25,000** | Filtered from `mix665k` (`describe`, single-turn, no CJK) | Object/scene understanding prior |
| D | Bbox-format prior (optional) | **5,000–20,000** | Filtered from `mix665k` (bbox prompts only) | xyxy output format consistency |
| E | Simple COCO/VG VQA (optional) | **0–15,000** | Filtered single-turn from `coco/` and `vg/VG_100K/` | Mild diversity cap only |

| Block | Exclude |
|-------|---------|
| — | `conversation_58k`, `complex_reasoning_77k`, full `llava_instruct_*` |
| — | 150k COCO + 30k VG + 20k text-only (legacy recipe) |
| — | OCR-VQA, TextVQA, GQA at scale |
| — | Multi-turn rows from mix665k (~356k rows) |

**Illustrative total:** ~30k–55k rows (vs 215k legacy) — **smaller, denser, fire-heavy**. Exact counts should be tuned after baseline eval.

### Legacy mixture (for reference only)

Documented in [`TEACHER_TRAINING_DATASET.md`](TEACHER_TRAINING_DATASET.md):

| Block | Count |
|-------|------:|
| Text-only (no CJK) | 20,000 |
| COCO train2017 | 150,000 |
| VG_100K | 30,000 |
| Detectium train | 13,077 |
| Bbox subset | 2,500 |
| **Total** | **215,577** |

We consider this **too general-purpose** for a deployable fire specialist, but it remains a reproducible baseline if needed.

---

## 8. Why not bbox-only, and why captions still matter

### Bbox-only is insufficient

Industrial fire deployment requires:

| Capability | Bbox-only training | Detectium QA + captions |
|------------|-------------------|-------------------------|
| “Is there fire?” | Weak | Strong (`Yes.` / `No.`) |
| False alarm control (steam, lights, welding) | Weak | Strong (non-fire set) |
| Severity / context language | Weak | Strong |
| “Where is the flame?” | Strong | Supported via bbox block |

BBox JSON teaches **coordinates**, not **hazard reasoning**.

### Captions / scene descriptions help

Fire is **contextual**: the model must interpret **objects, layout, and environment** before judging risk. A moderate amount of **short, factual captioning** (detail-style prompts) improves visual grounding without turning the model into a generic chatbot — **if capped** and kept **single-turn**.

### What to avoid

- **Long multi-turn conversations** → model learns dialogue, not deployment QA.  
- **Massive generic COCO VQA** → dilutes fire signal; hurts specialization.  
- **Replacing Detectium text with generic LLaVA captions** → loses domain language from `fire_prompts.json` and hazard templates.

---

## 9. JSON schema and path conventions

### LLaVA-style record (all training JSONs)

```json
{
  "id": "fire_00269",
  "image": "real_fire/images/example.jpg",
  "conversations": [
    { "from": "human", "value": "<image>\nWhat is burning, where is it happening, and how severe is the fire?" },
    { "from": "gpt", "value": "Yes. A small flame from a lighter..." }
  ],
  "meta": { }
}
```

- **`image`:** path **relative** to a single `IMAGE_FOLDER`.  
- **`<image>`** in human text: standard LLaVA token.  
- Text-only rows omit `"image"` (we plan to **exclude** these for fire teacher).

### Recommended unified `IMAGE_FOLDER`

```
LLaVA-Instruct-150K/          # or repo-relative path — one root for all vision samples
├── coco/train2017/           # only if using mix665k COCO rows
├── vg/VG_100K/               # only if using mix665k VG rows
├── real_fire/                # symlink or copy from detectioumfire
├── real_non_fire/
└── stable_diff_v15/train/images/
```

**Trainer variables** (`finetune.sh`):

```bash
JSON_FILE='.../teacher_training_data.json'
IMAGE_FOLDER='.../LLaVA-Instruct-150K'
MLP_ADAPTOR='.../checkpoints/llavaqwen-2-7b-pretrain/mm_projector.bin'
```

Update hardcoded paths in shell scripts from the original developer machine (`/home/mohammadkazzazi/...`) to your server paths before training.

---

## 10. Building and regenerating datasets

### Detectium bbox subset

```bash
cd MoD/detectioumfire
# Edit DETECTIUM_ROOT at top of script first
python3 build_detectium_bbox_synthetic2k_nonfire3k.py
# → detectium_bbox_synthetic2k_nonfire3k.json
```

### Teacher merge (legacy script — needs fire-first update)

```bash
# Current builder (legacy 215k recipe):
python3 utilities/LLaVA-Instruct-150K/build_teacher_training_data.py
# → teacher_training_data.json (next to mix665k if ROOT is configured there)
```

**TODO for engineering:** extend `build_teacher_training_data.py` with:

- Configurable block sizes (§7)  
- Filters: single-turn only, bbox pool, description pool, exclude multi-turn  
- Paths pointing at `MoD/detectioumfire` on this server  

### Download commands (reference)

```bash
# Detectium (Kaggle) — see MoD/detectioumfire/Download_Dataset.py
# LLaVA-Instruct JSONs (Hugging Face)
/opt/pytorch/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('liuhaotian/LLaVA-Instruct-150K', repo_type='dataset',
  local_dir='LLaVA-Instruct-150K')
"
```

---

## 11. Evaluation and success criteria

### Held-out data

| Asset | Path | Records |
|-------|------|--------:|
| Detectium val QA | `detectium_llavamod_real_full_val.json` | 1,452 |
| Val questions JSONL | `val_questions.jsonl` | for parallel eval scripts |

### Metrics (student D2D — completed)

On `detectium_llavamod_real_full_val.json` (1,452 rows), via `MoD/detectioumfire/eval/`:

| Student | Yes/No | Fire→Yes | FN | FP | Table 4 sev | Deploy |
|---------|-------:|---------:|---:|---:|------------:|--------|
| **Plan 4 v4 0.5B kd_lm** | **99.72%** | **752/754** | **2** | **2** | 87.13% | **✅ edge** |
| Plan 4 v4 1.5B kd_lm | 99.38% | 752/754 | 2 | 7 | 87.30% | ✅ middle |
| Plan 4 v4 3B kd_lm | 98.62% | 747/754 | 7 | 13 | **88.96%** | ✅ env/sev |
| Plan 3 3B kd_lm | 98.97% | 742/754 | 12 | 3 | 80.03% | legacy |
| Plan 4 v3 3B kd_lm | 98.18%* | 0/754 | 13† | 1* | 85.14% | ❌ |
| Teacher LoRA 7B | 99.79% | 753/754 | 1 | 2 | 89.76% | reference |

\*v3 metrics on parseable subset only; 794 preds lack Yes/No prefix. †semantic FN.

### Metrics (teacher val — completed)

On `detectium_llavamod_real_full_val.json` (1,452 examples), Yes/No prefix scoring via `MoD/detectioumfire/eval/`:

**Current teacher (Qwen2.5-7B)** — `eval/comparisons/compare_full_vs_lora.json`:

| Teacher | Accuracy | TP | FP | TN | FN | Notes |
|---------|----------|----|----|----|-----|--------|
| Full finetune | **99.79%** | 753 | 2 | 696 | 1 | `llavaqwen2.5-7b-finetune` |
| LoRA finetune | **99.79%** | 753 | 2 | 696 | 1 | **`REF_MLLM`** (`REF_MLLM_BASE=Qwen/Qwen2.5-7B-Instruct`) |

**Legacy teacher (Qwen2-7B)** — historical ablation (99.86% LoRA); no launcher folder retained.

| Teacher | Accuracy | Notes |
|---------|----------|--------|
| Full finetune | 99.72% | Superseded — vocab incompatible with Qwen2.5-3B student |
| LoRA finetune | 99.86% | Superseded |

BBox smoke tests: `eval/launchers/qwen2.5/compare_bbox_full_vs_lora.sh` (not yet run for Qwen2.5-7B; Qwen2-7B run showed full finetune slightly higher mean IoU than LoRA on 5 synthetic images).

### Metrics to agree on with collaborators (student / deployment)

1. **Binary fire detection accuracy** (Yes/No prefix match + semantic judge).  
2. **False positive rate** on `real_non_fire`.  
3. **BBox IoU or format validity** on synthetic + labeled subsets (if claiming localization).  
4. **Latency / throughput** on target hardware (deployment claim).  
5. **Optional:** comparison to Vicuna LoRA baseline ([`README_Detectium_Finetune.md`](README_Detectium_Finetune.md)).

**Qwen eval:** use `--conv-mode qwen` and MoD eval shells — not Vicuna `llava_v1` scripts.

---

## 12. Infrastructure notes (AWS EC2)

Example setup: **Deep Learning OSS Nvidia Driver AMI** (Ubuntu 24.04, PyTorch 2.11).

| Topic | Note |
|-------|------|
| **GPU training env** | `source /opt/pytorch/bin/activate` |
| **Small utilities / Kaggle** | Project venv e.g. `MoD/detectioumfire/llava-training` |
| **System `pip`** | Often missing on minimal AMIs; use venv |
| **Downloads** | Network/disk bound — GPU does not speed HTTP downloads |

---

## 13. Open questions for consultation

We welcome feedback on the following:

1. **Mix ratios:** Is ~60–80% Detectium / ~20–40% general vision the right band, or should general data be even smaller?  
2. **Caption source:** Filtered `mix665k` describe rows vs `detail_23k.json` (path remapping cost)?  
3. **BBox prior size:** Full ~67k mix665k bbox rows vs capped 5–20k — tradeoff between localization and fire QA?  
4. **Preference data:** How to convert Detectium `preference.json` (image-pair preferences) into LLaVA-MoD `chosen`/`rejected` for Phase 3?  
5. **Video:** Include `real_video` in a later teacher revision or keep image-only for ACL v1?  
6. **Teacher size:** ~~Qwen2-7B vs Qwen2.5-7B~~ — **resolved:** use **Qwen2.5-7B** teacher for Qwen2.5-3B student (tokenizer compatibility required for mimic KD).  
7. **Evaluation protocol:** Single numeric score vs human rubric for “controlled fire (no risk)” style answers.  
8. **Baseline:** Should we reproduce the **legacy 215k** mixture once as an ablation before switching to fire-first?

---

## 14. Backbone families & native VLM baselines

Full detail: [`BACKBONE_FAMILIES_AND_BASELINES.md`](BACKBONE_FAMILIES_AND_BASELINES.md)

### Track A — LLaVA-MoD KD (teacher → student)

Approved families for cross-backbone comparison. Same Plan 4 v4 JSON and Detectium val; **separate** projector + teacher + student per family.

| Priority | Family | Teacher → Student | Status |
|----------|--------|-------------------|--------|
| 1 | **Qwen2.5** | 7B → 0.5B / 1.5B / 3B | Primary — Plan 4 v4 done (all three) |
| 2 | **Gemma 2** | 9B → 2B | **Done** |
| 3 | **Llama 3** | 3.1-8B → 3.2-3B / 3.2-1B | **Done** (3B + 1B edge, 2026-07-05) |
| 4 | **Qwen3** | 8B → 4B | **Done** (2026-07-05) |

**Tokenizer rule:** Mimic KD requires identical tokenizer within a family (see Qwen2 vs Qwen2.5 failure). Qwen3 is **not** a drop-in replacement for Qwen2.5.

### Track B — Native VLM baselines (after KD students)

Compare distilled students against standard open VLMs on `detectium_llavamod_real_full_val.json`:

- **Qwen2.5-VL** (3B, 7B)
- **InternVL2.5** (4B, 8B)
- **MiniCPM-V 2.6**
- **Gemma 3** (4B-it)
- **Phi-4-multimodal**
- **LLaVA-OneVision**, **PaliGemma 2** (optional)

These use native vision encoders — not CLIP + `mm_projector`. Fine-tune on Detectium train optional; report zero-shot vs fine-tuned.

---

## 15. Related documents

| Document | Content |
|----------|---------|
| [`README_Qwen2_LLAVA_MoD_Pipeline.md`](README_Qwen2_LLAVA_MoD_Pipeline.md) | Phases 1–3, LoRA notes, batch size guidance |
| [`TEACHER_TRAINING_DATASET.md`](TEACHER_TRAINING_DATASET.md) | Fire-focused 35k teacher JSON spec |
| [`DISTILLATION_TRAINING_DATASET.md`](DISTILLATION_TRAINING_DATASET.md) | **5-plan D2D ladder** (Plans 1→5) |
| [`PREFERENCE_DISTILLATION_DATASET.md`](PREFERENCE_DISTILLATION_DATASET.md) | Phase 3b fire QA preference JSON (after D2D) |
| [`CURRENT_STATUS.md`](CURRENT_STATUS.md) | Short resume for new chats |
| [`BACKBONE_FAMILIES_AND_BASELINES.md`](BACKBONE_FAMILIES_AND_BASELINES.md) | Backbone families + native VLM comparison list |
| [`TRAIN_EVAL.md`](../../LLava_MoD/docs/TRAIN_EVAL.md) | Upstream distillation stages |
| [LLaVA-Instruct-150K on Hugging Face](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main) | Source dataset card |
| [DetectiumFire on Kaggle](https://www.kaggle.com/datasets/yimengfuyao/detectiumfire) | Domain images & labels |

---

## Summary (one paragraph)

We are building a **fire-specialized Qwen2.5-7B LLaVA teacher** and distilling to **Qwen2.5-0.5B / 1.5B / 3B** students via LLaVA-MoD. **Plan 4 v4 0.5B** and **Llama3-1B** are the top edge deploy checkpoints (99.72% / 99.79% Yes/No). Cross-family KD for **Gemma2**, **Llama3**, and **Qwen3** is **complete**. See [`BACKBONE_FAMILIES_AND_BASELINES.md`](BACKBONE_FAMILIES_AND_BASELINES.md).

---

*Please comment on §7 and §13 when reviewing. Update this file when mixture counts or checkpoints change.*
