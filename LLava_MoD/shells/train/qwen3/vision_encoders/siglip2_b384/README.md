# SigLIP2-B /384 track (replaces SigLIP SO400M)

**VE_SLUG:** `siglip2_b384`  
**Tower:** `google/siglip2-base-patch16-384`  
**Vision params at runtime:** ~**93M** (text tower dropped; LLM handles text)

```bash
cd MoD/LLava_MoD
export VE_SLUG=siglip2_b384
bash shells/train/qwen3/vision_encoders/scripts/run_teacher_pipeline.sh
SEED=42 bash shells/train/qwen3/vision_encoders/scripts/run_student_pipeline.sh
```

See [`../README.md`](../README.md) and [`VISION_ENCODER_ROADMAP.md`](../../../../Important%20Notes/VISION_ENCODER_ROADMAP.md).
