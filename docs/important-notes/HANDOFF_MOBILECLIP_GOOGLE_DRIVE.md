# Handoff: MobileCLIP teacher + student Phase 1 (Google Drive → remote SSH)

**Yes — Drive zip → download on the TeamViewer SSH server is fine.**  
Scripts use **paths relative to the unpacked tree** (no hardcoded `/home/ubuntu/...`).

**What you train on the remote box:**

1. MobileCLIP teacher (projector + LoRA)  
2. Qwen3-0.6B student projector (Phase 1 only)  

Do **not** run Plan 4 D2D unless that GPU can hold 8B teacher + 0.6B student.

---

## Required directory layout (must preserve)

Unpack so this relative tree exists (the top folder name can be anything):

```text
<WORKSPACE_ROOT>/                          # e.g. ~/llava-test or /data/exp/fire
  LLaVA-Instruct-150K/
    teacher_training_data.json
    coco/train2017/                        # slim set OK (referenced images only)
    real_fire          -> ../LLaVA_MoD/MoD/detectioumfire/.../real_fire
    real_non_fire      -> ../LLaVA_MoD/MoD/detectioumfire/.../real_non_fire
    stable_diff_v15    -> ../LLaVA_MoD/MoD/detectioumfire/.../stable_diff_v15
  LLaVA_MoD/MoD/
    LLava_MoD/                             # code + PreTrain JSON/Images
    detectioumfire/real_images/...
    detectioumfire/synthetic_images/...
```

`REPO_ROOT` and `WORKSPACE_ROOT` are auto-detected from script location.  
Optional override: `export WORKSPACE_ROOT=/your/unpacked/root`

---

## Part A — On THIS machine (Cursor / current server)

### 1. Build Drive packs

```bash
cd /home/ubuntu/llava-test/LLaVA_MoD/MoD/LLava_MoD
bash shells/train/qwen3/vision_encoders/scripts/pack_mobileclip_handoff.sh
```

Outputs under `llava-test/handoff_mobileclip/`:

| Archive | Contents | Approx size |
|---------|----------|-------------|
| `01_code.zip` | code + shells + Important Notes | small |
| `02_pretrain_json.zip` | LCS-558K JSON | ~300MB |
| `02_pretrain_images.zip` (+ `.z01`…) | PreTrain Images | ~27GB (split ~2GB) |
| `03_teacher_json_slim_coco.zip` (+ splits) | teacher JSON + only referenced COCO | ~3GB |
| `04_detectium_images.zip` (+ splits) | real fire/non-fire + synthetic | ~9GB |

Upload **all** parts (`.zip`, `.z01`, `.z02`, …) to Google Drive.

### 2. Do **not** rely on absolute symlinks

Instruct image links are **relative** (already fixed). The pack script re-asserts that.

---

## Part B — On TeamViewer → SSH server

### 1. Download from Drive

Use browser, `gdown`, or rclone into e.g. `~/downloads/handoff_mobileclip/`.

### 2. Unpack into one workspace (order matters for split zips)

```bash
export WS=~/llava-test          # pick any path
mkdir -p "$WS"
cd ~/downloads/handoff_mobileclip

# Unsplit Drive parts then extract into $WS
unsplit_and_unzip() {
  local base="$1"   # e.g. 02_pretrain_images.zip
  if ls "${base%.zip}".z[0-9][0-9] >/dev/null 2>&1; then
    zip -s 0 "$base" --out "${base%.zip}_full.zip"
    unzip -n "${base%.zip}_full.zip" -d "$WS"
  else
    unzip -n "$base" -d "$WS"
  fi
}

unzip -n 01_code.zip -d "$WS"
unzip -n 02_pretrain_json.zip -d "$WS"
unsplit_and_unzip 02_pretrain_images.zip
unsplit_and_unzip 03_teacher_json_slim_coco.zip
unsplit_and_unzip 04_detectium_images.zip
```

After unpack, recreate relative symlinks (safe even if zip stored them wrong):

```bash
cd "$WS/LLaVA-Instruct-150K"
rm -f real_fire real_non_fire stable_diff_v15
ln -s ../LLaVA_MoD/MoD/detectioumfire/real_images/real_images/real_fire real_fire
ln -s ../LLaVA_MoD/MoD/detectioumfire/real_images/real_images/real_non_fire real_non_fire
ln -s ../LLaVA_MoD/MoD/detectioumfire/synthetic_images/synthetic_images/stable_diff_v15 stable_diff_v15
```

### 3. Python env (remote usually has no bundled miniconda)

```bash
cd "$WS/LLaVA_MoD/MoD/LLava_MoD"
# activate YOUR pytorch env first, then:
export LLAVAMOD_SKIP_CONDA=1
# or: export LLAVAMOD_CONDA_ENV=/path/to/conda/envs/training

pip install -r requirements.txt
pip install -r requirements-backbones-llama3-qwen3.txt
pip install -r requirements-vision-encoders.txt   # open-clip-torch for MobileCLIP
```

Hugging Face models download on first run (needs network + disk):

- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-0.6B`
- MobileCLIP-S2 weights via `open-clip-torch`

### 4. Verify layout (must PASS)

```bash
cd "$WS/LLaVA_MoD/MoD/LLava_MoD"
export LLAVAMOD_SKIP_CONDA=1
bash shells/train/qwen3/vision_encoders/scripts/verify_portable_layout.sh
```

### 5. Train MobileCLIP teacher + student Phase 1

```bash
cd "$WS/LLaVA_MoD/MoD/LLava_MoD"
export LLAVAMOD_SKIP_CONDA=1
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export NUM_GPUS=4          # set to that machine's GPU count
unset CUDA_VISIBLE_DEVICES
mkdir -p logs

nohup bash -lc '
  export LLAVAMOD_SKIP_CONDA=1
  export PYTHONPATH="'"${PWD}"':${PYTHONPATH}"
  export NUM_GPUS='"${NUM_GPUS}"'
  VE_SLUG=mobileclip_s2 bash shells/train/qwen3/vision_encoders/scripts/run_teacher_pipeline.sh
  VE_SLUG=mobileclip_s2 bash shells/train/qwen3/vision_encoders/scripts/pretrain_student_0.6b.sh
' >> logs/ve_mobileclip_teacher_student_p1.log 2>&1 &
echo $! > logs/ve_mobileclip.pid
tail -f logs/ve_mobileclip_teacher_student_p1.log
```

### 6. Expected checkpoints (copy these back via Drive)

```text
checkpoints/llavaqwen3-8b-pretrain-mobileclip-s2/mm_projector.bin
checkpoints/llavaqwen3-8b-finetune-lora-mobileclip-s2/
checkpoints/llavaqwen3-0.6b-student-pretrain-mobileclip-s2/mm_projector.bin
```

---

## Path design (why it works on any server)

| Variable | How set |
|----------|---------|
| `REPO_ROOT` | Directory of `LLava_MoD` (from script path) |
| `WORKSPACE_ROOT` | Three levels above `REPO_ROOT` (= folder that contains `LLaVA_MoD/` and `LLaVA-Instruct-150K/`) |
| Pretrain data | Relative: `./llavamod/data/PreTrain/...` after `cd $REPO_ROOT` |
| Teacher SFT | `${WORKSPACE_ROOT}/LLaVA-Instruct-150K/...` |

No `/home/ubuntu/llava-test` in the MobileCLIP VE scripts.

---

## Common failures

| Symptom | Fix |
|---------|-----|
| `Conda env not found` | `export LLAVAMOD_SKIP_CONDA=1` after activating your env |
| Missing `real_fire` images | Recreate **relative** symlinks (Part B §2) |
| `Unknown image tower: mobileclip:...` | `pip install -r requirements-vision-encoders.txt` |
| OOM on teacher | Lower `PER_DEVICE_BATCH` / raise `GRAD_ACCUM`; set `NUM_GPUS` correctly |
| unzip split fails | Upload **all** `.z01`, `.z02`, … parts; use `zip -s 0 … --out full.zip` |

---

## Related

- [`VISION_ENCODER_ROADMAP.md`](VISION_ENCODER_ROADMAP.md)  
- [`CURRENT_STATUS.md`](CURRENT_STATUS.md)  
- [`README_QWEN3_VE_A100.md`](../../../README_QWEN3_VE_A100.md)
