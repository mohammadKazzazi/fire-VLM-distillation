# MobileCLIP-S2 — Qwen2.5-3B student

**VE_SLUG:** `mobileclip_s2`  
**Teacher:** Qwen2.5-7B-Instruct  
**Student:** Qwen2.5-3B-Instruct  
**Tower:** `mobileclip:MobileCLIP-S2:datacompdr` (`open-clip-torch`)

Requires: `pip install -r requirements-vision-encoders.txt`

```bash
cd MoD/LLava_MoD
export VE_SLUG=mobileclip_s2
export DEEPSPEED_CONFIG=./llavamod/config/dpconfig/zero2.json
bash shells/train/qwen/vision_encoders/scripts/run_full_pipeline.sh
```

Do **not** reuse `llavaqwen2.5-7b-finetune-lora` or `llavaqwen2.5-3b-student-pretrain` (those are CLIP ViT-L).

See [`../README.md`](../README.md).
