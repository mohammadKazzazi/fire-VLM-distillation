# CLIP ViT-L/336 track (baseline)

**VE_SLUG:** `clip_vitl336`  
**Tower:** `openai/clip-vit-large-patch14-336`

**Teacher:** reuses existing legacy checkpoints (no retrain):

| Stage | Path |
|-------|------|
| Projector | `checkpoints/llavaqwen3-8b-pretrain/` |
| LoRA | `checkpoints/llavaqwen3-8b-finetune-lora/` |

Student artifacts still use the `-clip-vitl336` suffix for fair comparison naming.

```bash
cd MoD/LLava_MoD
export VE_SLUG=clip_vitl336

# Teacher — no-op (legacy reuse); only trains student:
SEED=42 bash shells/train/qwen3/vision_encoders/scripts/run_student_pipeline.sh
```

See [`../README.md`](../README.md) and [`VISION_ENCODER_ROADMAP.md`](../../../../Important%20Notes/VISION_ENCODER_ROADMAP.md).
