# Qwen3-0.6B edge student pipeline

**Status:** Scaffold ready — **prefer [`vision_encoders/`](../vision_encoders/) ablation** for deploy study.

Distill **Qwen/Qwen3-0.6B** from the Qwen3-8B LoRA teacher. Same Plan 4 v4 JSON as 4B.

### Vision encoder tracks (recommended)

Compare three encoders with full teacher+student retrain per track:

| Track | `VE_SLUG` | Doc |
|-------|-----------|-----|
| CLIP ViT-L/336 | `clip_vitl336` | [`vision_encoders/clip_vitl336/`](../vision_encoders/clip_vitl336/README.md) |
| SigLIP2-B/384 | `siglip2_b384` | [`vision_encoders/siglip2_b384/`](../vision_encoders/siglip2_b384/README.md) |
| MobileCLIP-S2 | `mobileclip_s2` | [`vision_encoders/mobileclip_s2/`](../vision_encoders/mobileclip_s2/README.md) |

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
pip install -r requirements-vision-encoders.txt
export VE_SLUG=siglip2_b384
bash shells/train/qwen3/vision_encoders/scripts/run_teacher_pipeline.sh
SEED=42 bash shells/train/qwen3/vision_encoders/scripts/run_student_pipeline.sh
```

See [`Important Notes/VISION_ENCODER_ROADMAP.md`](../../../../Important%20Notes/VISION_ENCODER_ROADMAP.md).

### Legacy — single CLIP-L path (unsuffixed checkpoints)

All commands from `MoD/LLava_MoD` (or use absolute path `/home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD`).

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD

# 0) Tokenizer gate (8B ↔ 0.6B)
bash "shells/train/qwen3/student 0.6B/verify_tokenizer.sh"

# 1) Projector pretrain (detached by default; MASTER_PORT=29502)
bash "shells/train/qwen3/student 0.6B/pretrain_student_0.6b.sh"
tail -f logs/pretrain_qwen3_student_0.6b_latest.log

# 2) Plan 4 v4 D2D kd_lm (detached by default)
bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4.sh"
tail -f logs/plan4_d2d_multiquery_v4_kd_lm_qwen3_0.6b_latest.log

# 3) Detectium val eval
NUM_GPUS=8 bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4_val.sh"

# 4) Table 4 batch score
cd ../detectioumfire/eval/launchers/qwen2.5 && ./run_table4_student_validation.sh
```

### Monitor logs

| Phase | Command |
|-------|---------|
| Pretrain | `tail -f logs/pretrain_qwen3_student_0.6b_latest.log` |
| D2D | `tail -f logs/plan4_d2d_multiquery_v4_kd_lm_qwen3_0.6b_latest.log` |
| PID files | `logs/pretrain_qwen3_student_0.6b.pid`, `logs/plan4_d2d_multiquery_v4_kd_lm_qwen3_0.6b.pid` |

### Options

```bash
# Foreground (no nohup)
DETACH=0 bash "shells/train/qwen3/student 0.6B/pretrain_student_0.6b.sh"
DETACH=0 bash "shells/train/qwen3/student 0.6B/plan4_d2d_multiquery_v4.sh"

# Alternate DeepSpeed port if 29502 is in use
MASTER_PORT=29503 bash "shells/train/qwen3/student 0.6B/pretrain_student_0.6b.sh"
```

## Checkpoints

| Stage | Directory |
|-------|-----------|
| Pretrain | `llavaqwen3-0.6b-student-pretrain/` |
| Plan 4 v4 | `plan4-d2d-multiquery-v4-kd-lm-qwen3-0.6b-seed42/` |
| Eval answers | `detectioumfire/eval/runs/students/qwen3-0.6b/plan4-v4-kd-lm-seed42/` |

## Notes

- Do **not** reuse 4B `mm_projector.bin` (hidden size differs: 0.6B vs 4B).
- Same `qwen3` conv / eval path as 4B (`use_cache=False` in `model_vqa_loader.py`).
- Student eval uses `~/.cache/huggingface` for Qwen3-0.6B base weights.
- Shared teacher paths: `shells/train/qwen3/common_qwen3.env.sh`
- Pretrain uses `MASTER_PORT` default **29502** — override if another DeepSpeed job holds the port.

See parent [`../README.md`](../README.md) · [`Important Notes/CURRENT_STATUS.md`](../../../../Important%20Notes/CURRENT_STATUS.md).
