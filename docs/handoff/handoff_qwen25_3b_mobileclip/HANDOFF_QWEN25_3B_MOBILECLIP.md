# Handoff: Qwen2.5-3B × MobileCLIP-S2 (train from base → eval)

**Last updated:** 2026-08-14

This is a **new** track. It does **not** reuse CLIP Qwen2.5 teacher/student checkpoints.

| Role | Hugging Face id | Vision |
|------|-----------------|--------|
| Teacher | `Qwen/Qwen2.5-7B-Instruct` | MobileCLIP-S2 |
| Student | `Qwen/Qwen2.5-3B-Instruct` | MobileCLIP-S2 |

**Recipe:** Plan 4 v4 `kd_lm`, 1 epoch D2D, seeds **42 / 123 / 456**.  
**DeepSpeed on Triton:** `zero2.json` (not `zero2_offload.json`).

---

## 0. Copy-paste prompt for GPT on Triton

Send this block to GPT on the Triton session (fill `$WS` if different):

```text
You are on the Triton cluster. Unpack and run the Qwen2.5-3B MobileCLIP track.

Workspace root is $WS (folder that contains LLaVA_MoD/ and LLaVA-Instruct-150K/).
Repo is $WS/LLaVA_MoD/MoD/LLava_MoD.

1) Unpack the zip into $WS WITHOUT overwriting existing files except the new vision_encoders tree:
   unzip -n qwen25_3b_mobileclip_scripts.zip -d "$WS"
   If vision_encoders already exists, prefer:
   unzip -o qwen25_3b_mobileclip_scripts.zip -d "$WS"

2) Activate the existing llavamod training env (llavamod_aws or equivalent). Then:
   cd $WS/LLaVA_MoD/MoD/LLava_MoD
   export LLAVAMOD_SKIP_CONDA=1
   export PYTHONPATH="${PWD}:${PYTHONPATH}"
   export NUM_GPUS=4
   export DEEPSPEED_CONFIG=./llavamod/config/dpconfig/zero2.json
   export VE_SLUG=mobileclip_s2
   unset CUDA_VISIBLE_DEVICES
   pip install -r requirements-vision-encoders.txt

3) Recreate relative image symlinks if needed:
   cd $WS/LLaVA-Instruct-150K
   rm -f real_fire real_non_fire stable_diff_v15
   ln -s ../LLaVA_MoD/MoD/detectioumfire/real_images/real_images/real_fire real_fire
   ln -s ../LLaVA_MoD/MoD/detectioumfire/real_images/real_images/real_non_fire real_non_fire
   ln -s ../LLaVA_MoD/MoD/detectioumfire/synthetic_images/synthetic_images/stable_diff_v15 stable_diff_v15

4) Do NOT reuse CLIP checkpoints:
   llavaqwen2.5-7b-pretrain/
   llavaqwen2.5-7b-finetune-lora/
   llavaqwen2.5-3b-student-pretrain/
   plan4-d2d-multiquery-v4-kd-lm-seed*/
   Those are CLIP ViT-L. This track writes *-mobileclip-s2* names only.

5) Verify then train (Slurm, 4+ GPUs, c++ not required because we use zero2.json):
   cd $WS/LLaVA_MoD/MoD/LLava_MoD
   bash shells/train/qwen/vision_encoders/scripts/verify_layout.sh
   VE_SLUG=mobileclip_s2 SEEDS="42 123 456" RUN_VAL=0 SKIP_EXISTING=1 \
     bash shells/train/qwen/vision_encoders/scripts/run_full_pipeline.sh

   Or staged:
   VE_SLUG=mobileclip_s2 bash shells/train/qwen/vision_encoders/scripts/run_teacher_pipeline.sh
   VE_SLUG=mobileclip_s2 bash shells/train/qwen/vision_encoders/scripts/pretrain_student_3b.sh
   VE_SLUG=mobileclip_s2 SEEDS="42 123 456" RUN_VAL=0 \
     bash shells/train/qwen/vision_encoders/scripts/run_multi_seed.sh

6) After D2D, eval (needs the eval fix pack: GenerationMixin + use_cache=False + peft==0.4.0):
   for SEED in 42 123 456; do
     VE_SLUG=mobileclip_s2 SEED=$SEED NUM_GPUS=1 \
       bash shells/train/qwen/vision_encoders/scripts/plan4_d2d_multiquery_v4_val.sh
   done

7) Success checkpoints:
   checkpoints/llavaqwen2.5-7b-pretrain-mobileclip-s2/mm_projector.bin
   checkpoints/llavaqwen2.5-7b-finetune-lora-mobileclip-s2/adapter_config.json
   checkpoints/llavaqwen2.5-3b-student-pretrain-mobileclip-s2/mm_projector.bin
   checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen2.5-3b-mobileclip-s2-seed{42,123,456}/adapter_config.json

If verify_layout.sh fails, print the MISS lines and stop. Do not invent paths.
Keep DEEPSPEED_CONFIG=zero2.json. Do not use zero2_offload.json.
```

---

## 1. What to send to Triton

Small zip (scripts + this guide only; **no weights**):

```text
handoff_qwen25_3b_mobileclip/qwen25_3b_mobileclip_scripts.zip
```

Build on AWS:

```bash
cd /path/to/LLava_MoD
bash shells/train/qwen/vision_encoders/scripts/pack_qwen25_mobileclip_scripts.sh
```

Triton **already needs** (from earlier MobileCLIP / D2D packs):

| Need | Why |
|------|-----|
| LCS-558K JSON + Images (`llavamod/data/PreTrain/`) | teacher + student projector |
| `teacher_training_data.json` + COCO / Detectium images | teacher LoRA SFT |
| `dense2dense_fire_nonbbox_v4.json` | Plan 4 D2D |
| Detectium image symlinks | D2D + eval |
| `open-clip-torch` | MobileCLIP tower |
| HF cache: `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-3B-Instruct` | bases |

---

## 2. Unpack

```bash
export WS=/scratch/work/haghigs1/llava-test   # your workspace
unzip -o qwen25_3b_mobileclip_scripts.zip -d "$WS"
ls "$WS/LLaVA_MoD/MoD/LLava_MoD/shells/train/qwen/vision_encoders/scripts/run_full_pipeline.sh"
```

---

## 3. Env (Triton)

```bash
cd "$WS/LLaVA_MoD/MoD/LLava_MoD"
conda activate llavamod_aws          # or your env
export LLAVAMOD_SKIP_CONDA=1
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export NUM_GPUS=4
export DEEPSPEED_CONFIG=./llavamod/config/dpconfig/zero2.json
export VE_SLUG=mobileclip_s2
unset CUDA_VISIBLE_DEVICES
pip install -r requirements-vision-encoders.txt
```

---

## 4. Train order

| Step | Script | Output |
|------|--------|--------|
| 0 | `verify_layout.sh` | must PASS |
| 1a | `pretrain_teacher.sh` | `llavaqwen2.5-7b-pretrain-mobileclip-s2/mm_projector.bin` |
| 1b | `finetune_lora.sh` | `llavaqwen2.5-7b-finetune-lora-mobileclip-s2/` |
| 2 | `pretrain_student_3b.sh` | `llavaqwen2.5-3b-student-pretrain-mobileclip-s2/mm_projector.bin` |
| 3 | `run_multi_seed.sh` | `plan4-d2d-multiquery-v4-kd-lm-qwen2.5-3b-mobileclip-s2-seed{42,123,456}/` |
| 4 | `plan4_d2d_multiquery_v4_val.sh` | `eval/runs/students/qwen2.5-3b/mobileclip-s2/.../val_answers.jsonl` |

One command:

```bash
VE_SLUG=mobileclip_s2 SEEDS="42 123 456" RUN_VAL=0 SKIP_EXISTING=1 \
  bash shells/train/qwen/vision_encoders/scripts/run_full_pipeline.sh
```

### Example Slurm

```bash
#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --job-name=q25_mc_3b

export WS=/scratch/work/haghigs1/llava-test
cd "$WS/LLaVA_MoD/MoD/LLava_MoD"
source activate llavamod_aws
export LLAVAMOD_SKIP_CONDA=1 PYTHONPATH="${PWD}:${PYTHONPATH}"
export NUM_GPUS=4
export DEEPSPEED_CONFIG=./llavamod/config/dpconfig/zero2.json
export VE_SLUG=mobileclip_s2
unset CUDA_VISIBLE_DEVICES
mkdir -p logs

VE_SLUG=mobileclip_s2 SEEDS="42 123 456" RUN_VAL=0 SKIP_EXISTING=1 \
  bash shells/train/qwen/vision_encoders/scripts/run_full_pipeline.sh \
  >> logs/qwen25_3b_mobileclip_full.log 2>&1
```

Teacher + student projector can be one job; D2D seeds another if walltime is tight.

---

## 5. Eval

Apply the **eval fix pack** first (`GenerationMixin` on vendored Qwen2, `use_cache=False`, `peft==0.4.0`). Then:

```bash
cd "$WS/LLaVA_MoD/MoD/LLava_MoD"
export LLAVAMOD_SKIP_CONDA=1 PYTHONPATH="${PWD}:${PYTHONPATH}" VE_SLUG=mobileclip_s2
for SEED in 42 123 456; do
  VE_SLUG=mobileclip_s2 SEED="${SEED}" NUM_GPUS=1 \
    bash shells/train/qwen/vision_encoders/scripts/plan4_d2d_multiquery_v4_val.sh
done
```

`CONV_MODE=qwen`. Vision tower is read from the checkpoint (`mm_image_tower`).

---

## 6. Do not mix with CLIP Qwen2.5

| CLIP (existing) | MobileCLIP (this track) |
|-----------------|-------------------------|
| `llavaqwen2.5-7b-finetune-lora` | `llavaqwen2.5-7b-finetune-lora-mobileclip-s2` |
| `llavaqwen2.5-3b-student-pretrain` | `llavaqwen2.5-3b-student-pretrain-mobileclip-s2` |
| `plan4-d2d-multiquery-v4-kd-lm-seed42` | `plan4-d2d-multiquery-v4-kd-lm-qwen2.5-3b-mobileclip-s2-seed42` |

---

## 7. Failures

| Symptom | Fix |
|---------|-----|
| `Unknown image tower: mobileclip:...` | `pip install -r requirements-vision-encoders.txt` |
| `Conda env not found` | `export LLAVAMOD_SKIP_CONDA=1` after activating env |
| CPU Adam / missing `c++` | `DEEPSPEED_CONFIG=./llavamod/config/dpconfig/zero2.json` |
| Missing `real_fire` | relative Detectium symlinks |
| `no attribute generate` on val | eval fix pack (vendored Qwen2 `GenerationMixin`) |
| OOM on D2D | 4+ GPUs (7B teacher + 3B student) |

Lean save is enough: `adapter_*` + `non_lora_trainables.bin` + `config.json`. Delete extra `checkpoint-*` to save disk.
