# MobileCLIP-S2 track (edge)

**VE_SLUG:** `mobileclip_s2`  
**Tower:** `mobileclip:MobileCLIP-S2:datacompdr` (via `open-clip-torch`)

Requires: `pip install open-clip-torch`

```bash
cd MoD/LLava_MoD
export VE_SLUG=mobileclip_s2
bash shells/train/qwen3/vision_encoders/scripts/run_teacher_pipeline.sh
SEED=42 bash shells/train/qwen3/vision_encoders/scripts/run_student_pipeline.sh
```

See [`../README.md`](../README.md) and [`VISION_ENCODER_ROADMAP.md`](../../../../Important%20Notes/VISION_ENCODER_ROADMAP.md).
