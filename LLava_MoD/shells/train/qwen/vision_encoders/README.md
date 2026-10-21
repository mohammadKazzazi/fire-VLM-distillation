# Qwen2.5-3B × MobileCLIP-S2

Train **from base** (do **not** reuse CLIP Qwen2.5-7B / 3B checkpoints).

| Role | Model | Vision |
|------|--------|--------|
| Teacher | `Qwen/Qwen2.5-7B-Instruct` | MobileCLIP-S2 |
| Student | `Qwen/Qwen2.5-3B-Instruct` | MobileCLIP-S2 |

**VE_SLUG:** `mobileclip_s2`  
**Tower:** `mobileclip:MobileCLIP-S2:datacompdr`  
**Conv:** `qwen`

## Checkpoints (created by this track)

```text
checkpoints/llavaqwen2.5-7b-pretrain-mobileclip-s2/mm_projector.bin
checkpoints/llavaqwen2.5-7b-finetune-lora-mobileclip-s2/
checkpoints/llavaqwen2.5-3b-student-pretrain-mobileclip-s2/mm_projector.bin
checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen2.5-3b-mobileclip-s2-seed42/
checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen2.5-3b-mobileclip-s2-seed123/
checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen2.5-3b-mobileclip-s2-seed456/
```

Eval answers:

```text
detectioumfire/eval/runs/students/qwen2.5-3b/mobileclip-s2/plan4-v4-kd-lm-seed{42,123,456}/val_answers.jsonl
```

## Run (from `MoD/LLava_MoD`)

```bash
export VE_SLUG=mobileclip_s2
export NUM_GPUS=4
export DEEPSPEED_CONFIG=./llavamod/config/dpconfig/zero2.json
export LLAVAMOD_SKIP_CONDA=1   # Triton: after activating your env
export PYTHONPATH="${PWD}:${PYTHONPATH}"

bash shells/train/qwen/vision_encoders/scripts/verify_layout.sh
bash shells/train/qwen/vision_encoders/scripts/run_full_pipeline.sh
```

Or stage by stage:

```bash
VE_SLUG=mobileclip_s2 bash shells/train/qwen/vision_encoders/scripts/run_teacher_pipeline.sh
VE_SLUG=mobileclip_s2 bash shells/train/qwen/vision_encoders/scripts/pretrain_student_3b.sh
VE_SLUG=mobileclip_s2 SEEDS="42 123 456" RUN_VAL=0 \
  bash shells/train/qwen/vision_encoders/scripts/run_multi_seed.sh
```

Val (after eval fix pack):

```bash
VE_SLUG=mobileclip_s2 SEED=42 NUM_GPUS=1 \
  bash shells/train/qwen/vision_encoders/scripts/plan4_d2d_multiquery_v4_val.sh
```

See [`HANDOFF_QWEN25_3B_MOBILECLIP.md`](../../../../../Important%20Notes/HANDOFF_QWEN25_3B_MOBILECLIP.md).
