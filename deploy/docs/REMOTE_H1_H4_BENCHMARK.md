# Remote H1 / H4 Deployment Benchmark Guide

**Purpose:** Run the remaining hardware profiles (**H1**, **H4**) for Qwen2.5-0.5B fire-detection on a **different AWS instance**. This document is self-contained — paste it into a new chat on the remote server and follow the steps.

**Already done on g6.xlarge (do not re-run there):**
- **H3** — full NVIDIA L4 → `benchmark_qwen0.5b_H3.json` (class. p50 ≈ 297 ms)
- **H2** — same L4, 50% GPU memory cap → `benchmark_qwen0.5b_H2.json` (≈ identical to H3)

**Last updated:** 2026-07-16

---

## 1. Which instance to launch

| Paper ID | AWS instance | Arch | GPU | Role |
|----------|--------------|------|-----|------|
| **H4** | **c7g.2xlarge** | ARM64 (Graviton3) | none | CPU-only feasibility / limitation study |
| **H1** | **g5g.xlarge** | ARM64 (Graviton2) | NVIDIA T4G 16 GiB | ARM host + discrete GPU |

### Recommendation

1. Start with **H4** (`c7g.2xlarge`) — cheaper, documents CPU limits.
2. Then **H1** (`g5g.xlarge`) — ARM+GPU edge-oriented row.

### AMI / OS

- Ubuntu **22.04 LTS ARM64**
- For **H1 only:** Deep Learning AMI or Ubuntu + NVIDIA driver that supports T4G (CUDA 12.x preferred)

### Disk

- At least **40 GB** root volume (models + caches + env)

---

## 2. What to transfer (files)

### Option A — packaged tarball (recommended)

On the **current g6.xlarge** (this workspace), run:

```bash
cd /home/ubuntu/llava-test/Deploy
bash scripts/pack_for_remote_benchmark.sh
# Optional fully-offline (larger ~4–5 GB+):
# bash scripts/pack_for_remote_benchmark.sh --with-hf-cache
```

Transfer the resulting file:

```text
Deploy/transfer/llava-deploy-h1-h4-YYYYMMDD.tar.gz
```

Copy with `scp` / S3 / `rsync`, e.g.:

```bash
# Example via S3
aws s3 cp Deploy/transfer/llava-deploy-h1-h4-*.tar.gz s3://YOUR_BUCKET/deploy/
# On remote:
aws s3 cp s3://YOUR_BUCKET/deploy/llava-deploy-h1-h4-YYYYMMDD.tar.gz .
tar -xzf llava-deploy-h1-h4-YYYYMMDD.tar.gz
cd llava-deploy-h1-h4-YYYYMMDD
bash configure_remote.sh
```

### Option B — manual copy (if you prefer)

Copy **only** these paths (preserve relative layout):

| Path | Why | Approx. size |
|------|-----|--------------|
| `Deploy/` (whole folder) | Benchmark scripts, daemon, docs | ~tens of MB (exclude `transfer/`, optional exclude `bundles/hf_cache`) |
| `LLaVA_MoD/MoD/LLava_MoD/llavamod/` | Model code | ~15 MB (**exclude `llavamod/data/`** — 28 GB, not needed) |
| `LLaVA_MoD/MoD/LLava_MoD/checkpoints/plan4-d2d-multiquery-v4-kd-lm-0.5b-seed42/` | Student LoRA | ~138 MB |
| `LLaVA_MoD/MoD/LLava_MoD/requirements.txt` | Dependency list | tiny |
| `LLaVA_MoD/MoD/LLava_MoD/shells/conda_env.sh` | Optional helper | tiny |
| Sample images under `Deploy/samples/bench_images/` + `Deploy/config/benchmark_images.txt` | Fixed latency images | ~few MB |

**Do NOT bring:**
- Full `checkpoints/` tree (other students/teachers)
- `llavamod/data/`
- Training conda env (`miniconda3/`) — rebuild on ARM
- Video / full Detectium dataset
- A100 training artifacts

**HF models (choose one):**
1. Download on the remote while online (default), **or**
2. Pack with `--with-hf-cache`:
   - `Qwen/Qwen2.5-0.5B-Instruct` (~1 GB)
   - `openai/clip-vit-large-patch14-336` (~1–3 GB)

---

## 3. Expected workspace layout after unpack

```text
llava-deploy-h1-h4-YYYYMMDD/
  configure_remote.sh
  REMOTE_H1_H4_BENCHMARK.md          ← this guide
  workspace/
    Deploy/
      config/students.env            ← absolute paths after configure_remote.sh
      config/benchmark_images.txt
      fire_detect/
      scripts/aws_benchmark_hardware.sh
      samples/bench_images/
      results/                       ← write H1/H4 JSON here
    LLaVA_MoD/MoD/LLava_MoD/
      llavamod/
      checkpoints/plan4-d2d-multiquery-v4-kd-lm-0.5b-seed42/
      requirements.txt
    hf_cache/                        ← only if packed with --with-hf-cache
```

After `bash configure_remote.sh`, `students.env` points at this workspace.

---

## 4. Python environment (ARM — rebuild, do not copy x86 conda)

### Shared packages (H1 and H4)

```bash
# Use python 3.10
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip git

cd /path/to/llava-deploy-h1-h4-*/workspace
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
```

### H4 — CPU only (c7g.2xlarge)

```bash
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Inference deps (lighter than full training requirements.txt)
pip install \
  "transformers>=4.45,<5" \
  "peft>=0.10" \
  "accelerate" \
  "sentencepiece" \
  "protobuf" \
  "einops" \
  "pillow" \
  "timm" \
  "shortuuid" \
  "huggingface_hub"
```

### H1 — CUDA on T4G (g5g.xlarge)

Install NVIDIA driver + CUDA first (`nvidia-smi` must work). Then:

```bash
source .venv/bin/activate
# Match CUDA on the instance; cu121 is typical
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install \
  "transformers>=4.45,<5" \
  "peft>=0.10" \
  "accelerate" \
  "sentencepiece" \
  "protobuf" \
  "einops" \
  "pillow" \
  "timm" \
  "shortuuid" \
  "huggingface_hub"
```

### Important compatibility notes

- Do **not** upgrade Torch just to satisfy Transformers’ `.bin` CVE check if it breaks other things. Prefer **safetensors** downloads (the cache script already does).
- Runtime on g6 used: **Transformers 4.57.x**, **PyTorch 2.4.x**, **bf16 on CUDA**. On H4 use **fp32 CPU**.
- If `generate()` fails on LLaVA wrappers, `Deploy/fire_detect/model_utils.py` already patches `GenerationMixin` — keep that file.

Verify:

```bash
python - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("arch", __import__("platform").machine())
PY
```

Expected:
- H4: `cuda False`, arch `aarch64`
- H1: `cuda True`, GPU name contains `T4G`, arch `aarch64`

---

## 5. Download models once (while online)

```bash
cd /path/to/.../workspace/Deploy
source ../.venv/bin/activate   # or your venv path

# Cache CLIP (safetensors-aware script)
bash scripts/cache_vision_tower.sh

# Cache Qwen base LM
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
mid = "Qwen/Qwen2.5-0.5B-Instruct"
print("Downloading", mid)
AutoTokenizer.from_pretrained(mid)
AutoModelForCausalLM.from_pretrained(mid, torch_dtype="auto")
print("Done")
PY
```

Then for the real benchmark runs:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

If you packed with `--with-hf-cache` and ran `configure_remote.sh`, caches are already under `~/.cache/huggingface/hub`.

---

## 6. Configure paths

If you used the packer:

```bash
cd /path/to/llava-deploy-h1-h4-YYYYMMDD
bash configure_remote.sh
source workspace/.venv/bin/activate
cd workspace/Deploy
```

Check `config/students.env`:

- `STUDENT_PRESET=qwen0.5b`
- `QWEN0_5B_LORA=.../checkpoints/plan4-d2d-multiquery-v4-kd-lm-0.5b-seed42`
- Paths exist on disk

Set `PYTHONPATH`:

```bash
export LLAVA_MOD_ROOT="$(pwd)/../LLaVA_MoD/MoD/LLava_MoD"
export PYTHONPATH="${LLAVA_MOD_ROOT}:$(pwd):${PYTHONPATH:-}"
```

(`scripts/common.sh` also sets this when you run the bash entrypoints.)

---

## 7. Smoke test (required before full protocol)

### H4 smoke (CPU)

```bash
cd workspace/Deploy
source ../.venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

bash scripts/aws_benchmark_hardware.sh H4 \
  --image-list config/benchmark_images.txt \
  --warmup 1 --runs 2 --restarts 1 --mode classification
```

Expect slow latency (seconds, not milliseconds). Confirm JSON is written:

```text
results/benchmark_qwen0.5b_H4.json
```

### H1 smoke (GPU)

```bash
bash scripts/aws_benchmark_hardware.sh H1 \
  --image-list config/benchmark_images.txt \
  --warmup 2 --runs 3 --restarts 1 --mode classification
```

Expect hundreds of ms for classification (similar ballpark to H3 if T4G is healthy).

---

## 8. Full benchmark commands

### H4 — reduced protocol (CPU is slow)

Full GPT protocol (100×3) can take many hours on CPU. Use a reduced but documented protocol:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

bash scripts/aws_benchmark_hardware.sh H4 \
  --image-list config/benchmark_images.txt \
  --warmup 5 --runs 20 --restarts 1 --mode both
```

In the paper, note: *H4 used a reduced measurement budget due to CPU latency.*

Optional overnight full protocol (only if you have time):

```bash
bash scripts/aws_benchmark_hardware.sh H4 \
  --image-list config/benchmark_images.txt
```

### H1 — full protocol (same as H3)

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

bash scripts/aws_benchmark_hardware.sh H1 \
  --image-list config/benchmark_images.txt
```

Defaults: warmup 20, runs 100, restarts 3, modes classification + reasoning.

Optional concurrency:

```bash
BENCHMARK_CONCURRENCY=4 bash scripts/aws_benchmark_hardware.sh H1 \
  --image-list config/benchmark_images.txt \
  --output results/benchmark_qwen0.5b_H1_c4.json
```

---

## 9. What “good” looks like

| Profile | Class. p50 expectation | Notes |
|---------|------------------------|-------|
| H3 (done) | ~297 ms | Full L4 |
| H2 (done) | ~298 ms | Same L4, 50% mem cap |
| **H1** | likely ~300–800 ms | T4G; may be slower than L4 |
| **H4** | likely **several seconds** | CPU-only; documents why GPU is required |

H4 being slow is **the intended result**, not a failure.

---

## 10. Copy results back

After runs, collect:

```text
workspace/Deploy/results/benchmark_qwen0.5b_H4.json
workspace/Deploy/results/benchmark_qwen0.5b_H1.json
```

Copy them back to the main workspace:

```text
/home/ubuntu/llava-test/Deploy/results/
```

Then update `Deploy/docs/EDGE_DEPLOY_AWS_BENCHMARK.md` paper table (H1/H4 rows).

---

## 11. Paper wording (copy-ready)

> We evaluate deployment feasibility on four virtualized hardware configurations (H1–H4). H3 and H2 use an L4-class accelerator (unconstrained vs 50% GPU-memory budget). H1 uses an ARM64 host with a discrete T4G-class GPU. H4 uses ARM64 CPU only. Cloud instances are used solely for reproducible benchmarking; production inference is offline.

Do **not** write “Raspberry Pi performance” or “Jetson-equivalent” unless you measured those physical devices.

---

## 12. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Unknown STUDENT_PRESET` / missing LoRA | Run `configure_remote.sh`; check checkpoint path exists |
| Torch CVE / `.bin` load error on CLIP | Re-run `bash scripts/cache_vision_tower.sh` (safetensors) |
| `generate()` missing on LlavaQwen2 | Ensure `fire_detect/model_utils.py` is present and used |
| Wrong arch (`x86_64` on H1/H4) | You launched the wrong instance family — need ARM64 |
| H1: `nvidia-smi` missing | Install NVIDIA driver for T4G / use DLAMI |
| OOM on H4 | Close other processes; ensure `DEVICE=cpu`; do not set 4bit unless tested |
| Images not found | Use `--image-list config/benchmark_images.txt` after configure |
| `conda_env.sh` fails | Ignore — use the venv from §4; Deploy scripts tolerate missing conda if `python3` is correct |

---

## 13. Chat prompt for the remote agent

Paste this into a new Cursor chat on the remote server:

```text
Follow Deploy/docs/REMOTE_H1_H4_BENCHMARK.md (or REMOTE_H1_H4_BENCHMARK.md
at the tarball root).

Goal:
1) Confirm we are on the correct instance (H4 = c7g.2xlarge ARM CPU,
   or H1 = g5g.xlarge ARM + T4G).
2) Unpack/configure the transfer package if needed.
3) Create the Python venv for this arch (CPU or CUDA).
4) Cache CLIP + Qwen2.5-0.5B if not already local.
5) Smoke-test, then run the benchmark for this profile.
6) Report the classification/reasoning p50/p95, peak RAM/GPU, and the
   path to the results JSON.

Do not re-run H2/H3. Do not start training. Deployment benchmarking only.
```

---

## 14. Related files on the main server

| File | Role |
|------|------|
| `Deploy/scripts/pack_for_remote_benchmark.sh` | Builds the transfer tarball |
| `Deploy/scripts/aws_benchmark_hardware.sh` | Entry point (`H1` / `H4`) |
| `Deploy/docs/EDGE_DEPLOY_AWS_BENCHMARK.md` | Full methodology + H2/H3 results |
| `Deploy/results/benchmark_qwen0.5b_H{2,3}.json` | Completed GPU baselines |
