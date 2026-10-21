# Multi-seed addon — CLIP-family only (merge with packs 01–04)

**Last updated:** 2026-08-07  
**Goal right now:** Run Plan 4 v4 **multi-seed (`123`, `456`)** for **CLIP-family students only** on the remote GPU box.

**CLIP-family models (this pack):**

| Student | Needs teacher LoRA | Needs projector |
|---------|--------------------|-----------------|
| Qwen2.5-0.5B / 1.5B / 3B | `llavaqwen2.5-7b-finetune-lora` | `llavaqwen2.5-{0.5b,1.5b,3b}-student-pretrain` |
| Gemma2-2B | `llavagemma2-9b-finetune-lora` | `llavagemma2-2b-student-pretrain` |
| Llama3-1B / 3B | `llavallama3.1-8b-finetune-lora` | `llavallama3.2-{1b,3b}-student-pretrain` |
| Qwen3-4B | `llavaqwen3-8b-finetune-lora` | `llavaqwen3-4b-student-pretrain` |

Seed **42** is already done for these; this sweep trains **123** and **456** only.  
**Not included:** MobileCLIP / Qwen3-0.6B VE tracks (`INCLUDE_VE=0`).

---

## What you already have (01–04)

| Pack | Role for multi-seed |
|------|---------------------|
| 01 code | Training code / shells |
| 02 PreTrain | Not required for D2D multi-seed |
| 03 slim COCO + teacher JSON | Not required for D2D (Detectium-only JSON) |
| 04 Detectium images | **Required** — D2D `IMAGE_FOLDER` via Instruct symlinks |

---

## Packs to send now (CLIP-family)

Built under `llava-test/handoff_multiseed_addon/`:

| Pack | Approx size | Contents |
|------|-------------|----------|
| **`05_d2d_multiseed_addon.zip`** | ~2 MB | `dense2dense_fire_nonbbox_v4.json`, val JSON, `detectioumfire/eval`, `run_all_models_multiseed.sh` |
| **`06_teacher_loras_slim.zip`** (+ `.z01` if split) | ~3 GB | Slim teacher LoRAs (**no** huge `checkpoint-*` folders) |
| **`07_student_projectors.zip`** | ~100 MB | All CLIP-family student `mm_projector.bin` |
| **`README_MERGE.md`** | — | Copy of this guide |

Send **05 + 06 + 07** (+ all split parts for 06).

---

## Merge on remote (same `$WS` as 01–04)

```bash
export WS=~/llava-test   # MUST be the same workspace root as packs 01–04

cd ~/downloads/handoff_multiseed_addon   # or wherever you put the zips

# 05 — small
unzip -n 05_d2d_multiseed_addon.zip -d "$WS"

# Prefer new multiseed script (overwrite that file only)
unzip -o 05_d2d_multiseed_addon.zip \
  "LLaVA_MoD/MoD/LLava_MoD/shells/train/run_all_models_multiseed.sh" \
  "LLaVA_MoD/MoD/Important Notes/HANDOFF_MULTISEED_ADDON.md" \
  -d "$WS"

# 06 — may be split (~2GB parts)
if ls 06_teacher_loras_slim.z[0-9][0-9] >/dev/null 2>&1; then
  zip -s 0 06_teacher_loras_slim.zip --out 06_teacher_loras_slim_full.zip
  unzip -n 06_teacher_loras_slim_full.zip -d "$WS"
else
  unzip -n 06_teacher_loras_slim.zip -d "$WS"
fi

# 07
unzip -n 07_student_projectors.zip -d "$WS"
```

**`unzip -n`** = never overwrite existing 01–04 files (safe).

### Symlinks (if not already set)

```bash
cd "$WS/LLaVA-Instruct-150K"
rm -f real_fire real_non_fire stable_diff_v15
ln -s ../LLaVA_MoD/MoD/detectioumfire/real_images/real_images/real_fire real_fire
ln -s ../LLaVA_MoD/MoD/detectioumfire/real_images/real_images/real_non_fire real_non_fire
ln -s ../LLaVA_MoD/MoD/detectioumfire/synthetic_images/synthetic_images/stable_diff_v15 stable_diff_v15
```

### Quick verify

```bash
ls -lh "$WS/LLaVA-Instruct-150K/dense2dense_fire_nonbbox_v4.json"
ls "$WS/LLaVA_MoD/MoD/LLava_MoD/checkpoints/llavaqwen2.5-7b-finetune-lora/adapter_config.json"
ls "$WS/LLaVA_MoD/MoD/LLava_MoD/checkpoints/llavaqwen2.5-0.5b-student-pretrain/mm_projector.bin"
# …repeat for other teachers/projectors if you want
test -e "$WS/LLaVA-Instruct-150K/real_fire" && echo "symlink OK"
```

---

## Run CLIP-family multi-seed only

```bash
cd "$WS/LLaVA_MoD/MoD/LLava_MoD"
export LLAVAMOD_SKIP_CONDA=1    # if no bundled miniconda on that box
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export NUM_GPUS=4               # match remote GPU count
mkdir -p logs

# Preview (no training)
INCLUDE_CLIP_FAMILY=1 INCLUDE_VE=0 SEEDS="123 456" DRY_RUN=1 \
  bash shells/train/run_all_models_multiseed.sh

# Real run: seeds 123 & 456 for all CLIP-family students + val
INCLUDE_CLIP_FAMILY=1 INCLUDE_VE=0 SEEDS="123 456" RUN_VAL=1 NUM_GPUS=4 \
  nohup bash shells/train/run_all_models_multiseed.sh \
  >> logs/clip_family_multiseed_123_456.log 2>&1 &

tail -f logs/clip_family_multiseed_123_456.log
```

Expected new checkpoints (examples):

```text
checkpoints/plan4-d2d-multiquery-v4-kd-lm-0.5b-seed123/
checkpoints/plan4-d2d-multiquery-v4-kd-lm-seed456/          # Qwen2.5-3B
checkpoints/plan4-d2d-multiquery-v4-kd-lm-gemma2-2b-seed123/
checkpoints/plan4-d2d-multiquery-v4-kd-lm-llama3-1b-seed456/
checkpoints/plan4-d2d-multiquery-v4-kd-lm-qwen3-4b-seed123/
…
```

`SKIP_EXISTING=1` (default) skips any seed already finished.

### One model only (optional)

```bash
ONLY=Llama3-1B INCLUDE_VE=0 SEEDS="123 456" RUN_VAL=1 \
  bash shells/train/run_all_models_multiseed.sh
```

---

## HF models still needed on remote

Teachers/students download from Hugging Face on first use (or pre-cache):

- `Qwen/Qwen2.5-7B-Instruct`, `Qwen2.5-{0.5B,1.5B,3B}-Instruct`
- `google/gemma-2-9b-it`, `google/gemma-2-2b-it`
- `meta-llama/Llama-3.1-8B-Instruct`, `Llama-3.2-{1B,3B}-Instruct` (gated — `hf auth login`)
- `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B`
- Vision: `openai/clip-vit-large-patch14-336`

---

## Build these packs (this machine)

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
INCLUDE_CHECKPOINTS=1 bash shells/train/qwen3/vision_encoders/scripts/pack_multiseed_addon.sh
# → handoff_multiseed_addon/05_*.zip  06_*.zip  07_*.zip  README_MERGE.md
```

---

## Later (not this pack)

MobileCLIP / CLIP ViT-L **Qwen3-0.6B VE** multi-seed: train those teachers/projectors first, then:

```bash
INCLUDE_CLIP_FAMILY=0 INCLUDE_VE=1 SEEDS="42 123 456" RUN_VAL=1 \
  bash shells/train/run_all_models_multiseed.sh
```
