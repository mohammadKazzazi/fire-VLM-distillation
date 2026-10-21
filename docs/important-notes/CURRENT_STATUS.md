# Current status (resume here)

**Last updated:** 2026-08-06  
**Workspace:** `/home/ubuntu/llava-test/`

Open a new chat with:

> Read `@MoD/Important Notes/CURRENT_STATUS.md` and continue from **Next step**.

Full metrics: [`D2D_RESULTS.md`](D2D_RESULTS.md) · Batch table: `detectioumfire/eval/comparisons/table4_all_plans_with_yesno.json`  
A100 ops guide: [`README_QWEN3_VE_A100.md`](../../../README_QWEN3_VE_A100.md)

---

## Done

| Stage | Checkpoint / artifact |
|-------|------------------------|
| Teacher LoRA SFT (Qwen2.5) | `llavaqwen2.5-7b-finetune-lora/` — val **99.79%** → **`REF_MLLM`** |
| Student projector 0.5B / 1.5B / 3B | `llavaqwen2.5-{0.5b,1.5b,3b}-student-pretrain/mm_projector.bin` |
| Plan 4 v4 kd_lm (Qwen2.5) | 0.5B / 1.5B / 3B — primary deploy ladder |
| **Gemma2 teacher** | pretrain + LoRA — `llavagemma2-9b-finetune-lora/` (99.93% Yes/No) |
| **Gemma2 student projector** | `llavagemma2-2b-student-pretrain/mm_projector.bin` |
| **Gemma2 Plan 4 v4 kd_lm** | `plan4-d2d-multiquery-v4-kd-lm-gemma2-2b-seed42/` — deployable student |
| **Llama3 teacher projector** | `llavallama3.1-8b-pretrain/mm_projector.bin` — Phase 1a (2026-07-01) |
| **Llama3 teacher LoRA** | `llavallama3.1-8b-finetune-lora/` — val **99.86%** (753/754, FN=1, FP=1) |
| **Qwen3 teacher projector** | `llavaqwen3-8b-pretrain/mm_projector.bin` — Phase 1a (2026-07-04) |
| **Qwen3 teacher LoRA** | `llavaqwen3-8b-finetune-lora/` — val **99.86%** (753/754, FN=1, FP=1) |
| **Qwen3 student projector** | `llavaqwen3-4b-student-pretrain/mm_projector.bin` |
| **Qwen3 Plan 4 v4 kd_lm** | `plan4-d2d-multiquery-v4-kd-lm-qwen3-4b-seed42/` — deployable student |
| **Llama3 student projector (3B)** | `llavallama3.2-3b-student-pretrain/mm_projector.bin` |
| **Llama3 Plan 4 v4 kd_lm (3B)** | `plan4-d2d-multiquery-v4-kd-lm-llama3-3b-seed42/` — deployable student |
| **Llama3 student projector (1B)** | `llavallama3.2-1b-student-pretrain/mm_projector.bin` |
| **Llama3 Plan 4 v4 kd_lm (1B)** | `plan4-d2d-multiquery-v4-kd-lm-llama3-1b-seed42/` — edge deploy student |

---

## Results summary (val 1,452 rows)

### Qwen2.5 (primary deploy)

| Model | Params | Yes/No | Fire→Yes | FN | FP | Table4 obj / env / sev |
|-------|-------:|-------:|---------:|---:|---:|------------------------|
| **4 v4 0.5B** | **0.5B** | **99.72%** | **752/754** | **2** | **2** | 70% / 73% / 87% |
| 4 v4 1.5B | 1.5B | 99.38% | 752/754 | 2 | 7 | 69% / 76% / 87% |
| 4 v4 3B | 3B | 98.62% | 747/754 | 7 | 13 | 69% / **82%** / **89%** |
| Teacher 7B | 7B | 99.79% | 753/754 | 1 | 2 | 71% / 79% / 90% |

### Gemma 2 (cross-family — done)

| Model | Yes/No | Fire→Yes | FN | FP | Table4 obj / env / sev | Notes |
|-------|-------:|---------:|---:|---:|------------------------|-------|
| Teacher 9B LoRA | 99.93% | 754/754 | 0 | 1 | 71% / 81% / 90% | reference |
| **Student 2B kd_lm** | **99.31%** | **752/754** | **2** | **8** | 69% / 76% / 88% | **deploy** |

> **Historical:** Gemma2 `only_kd` ablation (no Yes/No prefixes) was run during development and removed.

### Llama 3 (done — 2026-07-05)

| Model | Params | Yes/No | Fire→Yes | FN | FP | Table4 obj / env / sev | Notes |
|-------|-------:|-------:|---------:|---:|---:|------------------------|-------|
| Teacher 8B LoRA | 8B | 99.86% | 753/754 | 1 | 1 | **72%** / **83%** / 89% | reference |
| **Student 3B kd_lm** | **3B** | **99.72%** | **752/754** | **2** | **2** | 70% / 80% / 89% | **deploy** |
| **Student 1B kd_lm** | **1B** | **99.79%** | **752/754** | **2** | **1** | 69% / 78% / 88% | **edge deploy** |

### Qwen3 (4B done — 0.6B VE ablation next)

| Model | Params | Yes/No | Fire→Yes | FN | FP | Table4 obj / env / sev | Notes |
|-------|-------:|-------:|---------:|---:|---:|------------------------|-------|
| Teacher 8B LoRA (CLIP) | 8B | 99.86% | 753/754 | 1 | 1 | 72% / **84%** / **90%** | reference — **reuse for CLIP track** |
| **Student 4B kd_lm** | **4B** | **99.31%** | **752/754** | **2** | **8** | 70% / 80% / 90% | **deploy** |
| Student 0.6B (CLIP / MobileCLIP) | 0.6B | — | — | — | — | — | projectors + D2D pending |

**Vision encoder (Track A default):** `openai/clip-vit-large-patch14-336`.

---

## Deploy recommendation (Qwen2.5)

| Use case | Checkpoint |
|----------|------------|
| **Edge / best FN·FP** | `plan4-d2d-multiquery-v4-kd-lm-0.5b-seed42/` |
| **Middle tier** | `plan4-d2d-multiquery-v4-kd-lm-1.5b-seed42/` |
| **Best env + severity** | `plan4-d2d-multiquery-v4-kd-lm-seed42/` |

Gemma2-2B kd_lm and Qwen3-4B kd_lm are valid alternate backbones (752/754, FP=8) if you need those tokenizer ecosystems. **Llama3-1B** is the best edge option on Yes/No (99.79%, FP=1).

---

**VE ablation (Qwen3-0.6B):** **CLIP-L vs MobileCLIP-S2 only** (SigLIP2 dropped).  
See [`VISION_ENCODER_ROADMAP.md`](VISION_ENCODER_ROADMAP.md) · [`README_QWEN3_VE_A100.md`](../../../README_QWEN3_VE_A100.md).

---

## Next step — CLIP + MobileCLIP (Qwen3-0.6B)

**Hardware now:** 4× A100 80GB (`NUM_GPUS=4`). Same recipe as 8-GPU runs; global batch is half unless you retune `GRAD_ACCUM`.

**Do now (teacher + student Phase 1 only):**

1. CLIP: reuse existing teacher; train 0.6B student **projector** only  
2. MobileCLIP-S2: train teacher (projector + LoRA), then 0.6B student **projector**  
3. **Do not train SigLIP2**

Full plan: [`VISION_ENCODER_ROADMAP.md`](VISION_ENCODER_ROADMAP.md) · ops: [`README_QWEN3_VE_A100.md`](../../../README_QWEN3_VE_A100.md)  
**Remote / TeamViewer + Google Drive handoff:** [`HANDOFF_MOBILECLIP_GOOGLE_DRIVE.md`](HANDOFF_MOBILECLIP_GOOGLE_DRIVE.md)  
**Multi-seed addon (merge with 01–04):** [`HANDOFF_MULTISEED_ADDON.md`](HANDOFF_MULTISEED_ADDON.md) · pack: `handoff_multiseed_addon/05_d2d_multiseed_addon.zip`  
**All-models seed runner:** `shells/train/run_all_models_multiseed.sh`

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
source shells/conda_env.sh
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export NUM_GPUS=4
pip install -r requirements-vision-encoders.txt

# CLIP student Phase 1 (teacher reused)
VE_SLUG=clip_vitl336 bash shells/train/qwen3/vision_encoders/scripts/pretrain_student_0.6b.sh

# MobileCLIP teacher (both phases) + student Phase 1
VE_SLUG=mobileclip_s2 bash shells/train/qwen3/vision_encoders/scripts/run_teacher_pipeline.sh
VE_SLUG=mobileclip_s2 bash shells/train/qwen3/vision_encoders/scripts/pretrain_student_0.6b.sh
```

**Later (D2D — after projectors; 4×80GB should be enough):**

```bash
for VE in clip_vitl336 mobileclip_s2; do
  VE_SLUG=$VE SEED=42 bash shells/train/qwen3/vision_encoders/scripts/plan4_d2d_multiquery_v4.sh
  NUM_GPUS=4 VE_SLUG=$VE SEED=42 bash shells/train/qwen3/vision_encoders/scripts/plan4_d2d_multiquery_v4_val.sh
done

# Remaining seeds
for VE in clip_vitl336 mobileclip_s2; do
  VE_SLUG=$VE SEEDS="123 456" bash shells/train/qwen3/vision_encoders/scripts/run_multi_seed.sh
done
```

Shells: `shells/train/qwen3/vision_encoders/`

---

## Future roadmap

| Priority | Item | Status |
|----------|------|--------|
| 1 | Qwen2.5 | **Done** |
| 2 | Gemma 2 | **Done** |
| 3 | Qwen3 (4B + CLIP teacher) | **Done** |
| 4 | Llama 3 | **Done** |
| 5 | **CLIP + MobileCLIP × Qwen3-0.6B** | **Next** — teachers/projectors now; D2D after |
| 6 | Multi-seed (123 / 456) | After seed-42 D2D |
| — | SigLIP2 | **Skipped** |

Track B native VLMs: [`BACKBONE_FAMILIES_AND_BASELINES.md`](BACKBONE_FAMILIES_AND_BASELINES.md) §Track B.

---

## Handoff docs

| File | Purpose |
|------|---------|
| [`CURRENT_STATUS.md`](CURRENT_STATUS.md) | This file |
| [`README_QWEN3_VE_A100.md`](../../../README_QWEN3_VE_A100.md) | A100 / 4-GPU ops commands for VE ablation |
| [`VISION_ENCODER_ROADMAP.md`](VISION_ENCODER_ROADMAP.md) | CLIP-L vs MobileCLIP-S2 design + roadmap |
| [`BACKBONE_FAMILIES_AND_BASELINES.md`](BACKBONE_FAMILIES_AND_BASELINES.md) | Backbone families + native VLM baselines |
| [`D2D_RESULTS.md`](D2D_RESULTS.md) | Full metrics |
| [`shells/train/llama3/README.md`](../LLava_MoD/shells/train/llama3/README.md) | Llama 3 train/eval |
| [`shells/train/qwen3/README.md`](../LLava_MoD/shells/train/qwen3/README.md) | Qwen3 train/eval (completed) |
| [`shells/train/gemma2/README.md`](../LLava_MoD/shells/train/gemma2/README.md) | Gemma 2 (completed) |
