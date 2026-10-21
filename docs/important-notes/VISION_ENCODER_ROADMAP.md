# Vision encoder roadmap — deployable fire detection

**Last updated:** 2026-08-03  
**Scope:** Qwen3-8B teacher → **Qwen3-0.6B** edge student (Plan 4 v4, `kd_lm`)  
**Workspace:** `/home/ubuntu/llava-test/`  
**Active comparison:** **CLIP ViT-L/336** (baseline) vs **MobileCLIP-S2** only  

Related: [`CURRENT_STATUS.md`](CURRENT_STATUS.md) · [`BACKBONE_FAMILIES_AND_BASELINES.md`](BACKBONE_FAMILIES_AND_BASELINES.md) · [`Deploy/README.md`](../../Deploy/README.md) · [`README_QWEN3_VE_A100.md`](../../../README_QWEN3_VE_A100.md)

---

## LLaVA uses vision only (important)

CLIP-family checkpoints on Hugging Face may list **vision + text** parameters. LLaVA-MoD uses:

```text
Image → vision encoder (frozen) → mm_projector → Qwen3 LM
Text  → Qwen3 tokenizer + LM only
```

The text tower of dual-encoder models (if present) is **not used** at inference. For deploy sizing, count **vision + projector + 0.6B LM** only.

---

## Encoders in this study

| Short name | `VE_SLUG` | HuggingFace / tower ID | Role |
|------------|-----------|------------------------|------|
| **CLIP ViT-L/336** *(baseline)* | `clip_vitl336` | `openai/clip-vit-large-patch14-336` | All prior Track A results; teacher already trained |
| **MobileCLIP-S2** | `mobileclip_s2` | `mobileclip:MobileCLIP-S2:datacompdr` | **~36M vision**, 64 patches — edge candidate |

**Out of scope for this campaign:** SigLIP2-B/384 and SigLIP SO400M (code may remain in the repo; do **not** train).

Student LM: **`Qwen/Qwen3-0.6B`**. Shells: `shells/train/qwen3/vision_encoders/`.

---

## Comparison (architecture & deploy)

| Metric | CLIP ViT-L/336 | MobileCLIP-S2 |
|--------|---------------:|--------------:|
| **Vision params loaded** | ~304M | **~36M** |
| Hidden dim → projector | 1024 | 1280 |
| Resolution | 336px | 256px |
| Patches → LM | 576 | **64** |
| Vision depth | 24 layers | FastViT |
| Code | `clip_encoder.py` | `mobileclip_encoder.py` |
| Extra deps | none | `open-clip-torch` |

---

## Strengths & weaknesses (fire task)

### 1. CLIP ViT-L/336 *(baseline)*

**Strengths:** Proven val numbers; 336px; LLaVA-MoD paper match; all Track A students already on this encoder.  
**Weaknesses:** Largest vision tower (~304M); 576 patches; slowest deploy profile.  
**Best when:** Main paper baseline and comparison to all prior work.

### 2. MobileCLIP-S2 *(smallest / fastest)*

**Strengths:** ~36M vision; **64 patches** → fastest LM stage; designed for mobile / edge.  
**Weaknesses:** 256px — highest FN risk on distant smoke; `open-clip-torch` dependency.  
**Best when:** Maximum latency/power savings; accept re-validation on Detectium val.

---

## Expected outcome (hypothesis)

| Encoder | Yes/No (hypothesis) | Latency (hypothesis) | Paper story |
|---------|---------------------|----------------------|-------------|
| CLIP-L/336 | **Highest** (baseline) | Slowest | Reference |
| MobileCLIP-S2 | Risk −1–3% | **Fastest** | Extreme edge |

**Decision rule:** Prefer MobileCLIP for production if **FN ≤ CLIP-L + 2** and **p50 latency** improves ≥ **25%** vs CLIP-L (g6/L4 or Jetson proxy).

---

## Training rule (critical)

Within one `VE_SLUG` track, all stages must share the same `VISION_ENCODER`:

```
teacher projector → teacher LoRA → student projector → Plan 4 v4 D2D
```

Never mix encoders across stages or reuse another VE’s `mm_projector.bin`.

**CLIP teacher:** reuse legacy `llavaqwen3-8b-pretrain/` + `llavaqwen3-8b-finetune-lora/` — **do not retrain**.

---

## Checkpoint naming

| Stage | CLIP-L | MobileCLIP-S2 |
|-------|--------|---------------|
| Teacher projector | `llavaqwen3-8b-pretrain/` *(legacy, shared)* | `llavaqwen3-8b-pretrain-mobileclip-s2/` |
| Teacher LoRA | `llavaqwen3-8b-finetune-lora/` *(legacy, shared)* | `llavaqwen3-8b-finetune-lora-mobileclip-s2/` |
| Student projector | `llavaqwen3-0.6b-student-pretrain-clip-vitl336/` | `...-mobileclip-s2/` |
| Plan 4 student | `plan4-...-qwen3-0.6b-clip-vitl336-seed42/` | `...-mobileclip-s2-seed42/` |

---

## Roadmap (execution order)

### Phase 0 — Setup ✅

- [x] `mobileclip_encoder.py`, VE shells, CLIP reuse wiring
- [x] Active study narrowed to **CLIP + MobileCLIP-S2** (SigLIP dropped)
- [x] A100 (40GB) run guide: [`README_QWEN3_VE_A100.md`](../../../README_QWEN3_VE_A100.md)

### Phase 1 — Teachers (MobileCLIP only)

CLIP-L teacher already exists: `llavaqwen3-8b-finetune-lora/` (99.86% Yes/No).

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
pip install -r requirements-vision-encoders.txt

VE_SLUG=mobileclip_s2 bash shells/train/qwen3/vision_encoders/scripts/run_teacher_pipeline.sh
```

**Gate:** Teacher LoRA **≥ 99.5% Yes/No** on val.

### Phase 1b — Student projectors only (A100 40GB–safe)

D2D (Phase 2) needs 8B teacher + 0.6B student in memory — **skip on A100 40GB**. Train projectors only:

```bash
VE_SLUG=clip_vitl336 bash shells/train/qwen3/vision_encoders/scripts/pretrain_student_0.6b.sh
VE_SLUG=mobileclip_s2 bash shells/train/qwen3/vision_encoders/scripts/pretrain_student_0.6b.sh
```

### Phase 2 — Qwen3-0.6B Plan 4 v4 D2D @ seed 42 *(larger GPU)*

```bash
for VE in clip_vitl336 mobileclip_s2; do
  VE_SLUG=$VE SEED=42 bash shells/train/qwen3/vision_encoders/scripts/plan4_d2d_multiquery_v4.sh
  NUM_GPUS=8 VE_SLUG=$VE SEED=42 bash shells/train/qwen3/vision_encoders/scripts/plan4_d2d_multiquery_v4_val.sh
done
```

Run `Deploy/scripts/benchmark_edge_deploy.sh` per checkpoint.

### Phase 3 — Multi-seed (after Phase 2)

```bash
for VE in clip_vitl336 mobileclip_s2; do
  VE_SLUG=$VE SEEDS="123 456" \
    bash shells/train/qwen3/vision_encoders/scripts/run_multi_seed.sh
done
```

### Phase 4 — Paper table

| VE | Yes/No | FN | FP | p50 ms | Vision RAM |
|----|--------|---:|---:|-------:|-----------:|
| CLIP-L | | | | ~832 | TBD |
| MobileCLIP-S2 | | | | TBD | TBD |

---

## Quick commands (MobileCLIP)

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
export VE_SLUG=mobileclip_s2

# Teacher both phases (CLIP teacher is skipped / reused separately)
bash shells/train/qwen3/vision_encoders/scripts/run_teacher_pipeline.sh

# Student projector only (A100-safe)
bash shells/train/qwen3/vision_encoders/scripts/pretrain_student_0.6b.sh

# Full student + D2D + val — larger GPU only
# SEED=42 bash shells/train/qwen3/vision_encoders/scripts/run_student_pipeline.sh
```

---

## Code locations

| Component | Path |
|-----------|------|
| VE registry | `shells/train/qwen3/vision_encoders/common_ve.env.sh` |
| MobileCLIP tower | `llavamod/model/multimodal_encoder/mobileclip_encoder.py` |
| Router | `llavamod/model/multimodal_encoder/builder.py` |
| A100 ops guide | [`README_QWEN3_VE_A100.md`](../../../README_QWEN3_VE_A100.md) |

---

## Optional / deferred encoders (not in this campaign)

| Encoder | Notes |
|---------|--------|
| `google/siglip2-base-patch16-384` | Code exists (`siglip2_encoder.py`); **not trained** in this study |
| `google/siglip-so400m-patch14-384` | ~400M vision + 729 patches — poor edge choice |
| MobileCLIP2-B | Newer gen; needs new integration |
| CLIP ViT-B/336 | Smallest CLIP-family fallback |

---

## Handoff checklist

- [x] CLIP teacher reused (no retrain)  
- [ ] Phase 1: MobileCLIP teacher trained  
- [ ] Phase 1b: CLIP + MobileCLIP 0.6B student projectors  
- [ ] Phase 2: 2× Plan 4 v4 D2D @ seed 42 *(larger GPU)*  
- [ ] Deploy benchmarks (latency + RAM)  
- [ ] Phase 3: multi-seed on CLIP + MobileCLIP  
- [ ] Update `CURRENT_STATUS.md`
