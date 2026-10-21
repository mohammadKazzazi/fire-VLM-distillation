# Vision encoder ablation — Qwen3-0.6B edge student

Compare three vision encoders with **full teacher + student retraining** (projector and LoRA are VE-specific).

| `VE_SLUG` | HF / tower ID | Checkpoint suffix |
|-----------|---------------|-------------------|
| `clip_vitl336` | `openai/clip-vit-large-patch14-336` | `-clip-vitl336` |
| `siglip2_b384` | `google/siglip2-base-patch16-384` | `-siglip2-b384` |
| `mobileclip_s2` | `mobileclip:MobileCLIP-S2:datacompdr` | `-mobileclip-s2` |

**Docs:** [`Important Notes/VISION_ENCODER_ROADMAP.md`](../../../../Important%20Notes/VISION_ENCODER_ROADMAP.md)

## Prerequisites

```bash
cd MoD/LLava_MoD
pip install -r requirements-backbones-llama3-qwen3.txt
pip install -r requirements-vision-encoders.txt   # open-clip-torch for MobileCLIP
bash shells/train/qwen3/student\ 0.6B/verify_tokenizer.sh
```

## Run one track

```bash
cd MoD/LLava_MoD
export VE_SLUG=siglip2_b384   # or clip_vitl336 | mobileclip_s2

# Teacher (Phase 1a + 2)
bash shells/train/qwen3/vision_encoders/scripts/run_teacher_pipeline.sh

# Student 0.6B (Phase 1b + Plan 4 v4 + val)
SEED=42 bash shells/train/qwen3/vision_encoders/scripts/run_student_pipeline.sh
```

## Run all three encoders

```bash
bash shells/train/qwen3/vision_encoders/run_all_ve_ablation.sh
```

## Multi-seed (after VE ablation)

```bash
VE_SLUG=clip_vitl336 SEEDS="42 43 44" \
  bash shells/train/qwen3/vision_encoders/scripts/run_multi_seed.sh
```

## Layout

```
vision_encoders/
  common_ve.env.sh          # VE registry + checkpoint paths
  scripts/                  # Shared training scripts (parametric on VE_SLUG)
  clip_vitl336/README.md
  siglip2_b384/README.md
  mobileclip_s2/README.md
  run_all_ve_ablation.sh
```

## Note on CLIP-L Qwen3 teacher

The default Qwen3 pipeline (`llavaqwen3-8b-pretrain/` + `llavaqwen3-8b-finetune-lora/`) is CLIP-L.  
The `clip_vitl336` track **reuses these checkpoints** — no teacher retrain. Student checkpoints still use the `-clip-vitl336` suffix.
