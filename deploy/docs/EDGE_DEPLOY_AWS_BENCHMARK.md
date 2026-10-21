# Edge Deployment — Hardware Benchmarking (Deployment Only)

**Scope:** Offline inference packaging and **hardware-proximate benchmarking** for the Qwen2.5-0.5B fire-detection student (and later other presets). Training, seeds, and vision-encoder ablations are **out of scope** here.

**Last updated:** 2026-07-16  
**Status:** H3 + H2 complete · H4 pending · H1 optional · **Pi ONNX classifier path added**

---

## Two-tier deployment (important)

| Tier | Hardware | Runtime | Code |
|------|----------|---------|------|
| **A — Multimodal VLM** | GPU edge (H1–H3), Jetson/IPC class | PyTorch | `Deploy/fire_detect/` |
| **B — Raspberry Pi class** | Pi / ARM CPU, no CUDA | **ONNX Runtime (no Torch)** | **`Deploy/pi/`** |

The VLM is **not** ported to Raspberry Pi. Pi gets a **MobileNetV3-Small binary fire classifier** exported to ONNX. See `Deploy/pi/README.md`.

VirtualBox / Docker-emulated Pi OS is **not** required to “have Pi code”; the ONNX package *is* the Pi code. Emulators do not produce valid Pi timing.

---

## Paper claim (safe wording)

> We evaluate deployment feasibility of the distilled student as an offline, containerized inference service. Experiments use reproducible virtualized hardware configurations (H1–H4) to measure load time, classification latency, memory, and throughput. These runs study resource requirements and inference behaviour; they do not claim physical edge deployment, video-stream evaluation, or production operation.

---

## Finalized hardware matrix

Use **H1–H4** in the paper. Do **not** name cloud providers or instance types in the main text.

| ID | AWS instance to launch | CPU / RAM | GPU | Role | Status |
|----|------------------------|-----------|-----|------|--------|
| **H3** | **g6.xlarge** | x86, 4 vCPU, 16 GiB | NVIDIA L4 (24 GiB) | Full GPU upper bound | **Done** |
| **H2** | **g6.xlarge** | same as H3 | L4 capped at **50%** memory | Strict accelerator budget | **Done** |
| **H1** | **g5g.xlarge** | ARM64, 4 vCPU, 16 GiB | NVIDIA T4G (16 GiB) | ARM host + discrete GPU | Optional |
| **H4** | **c7g.2xlarge** | ARM64, 8 vCPU, 16 GiB | none | CPU-only feasibility | Pending |

### Minimum set for deployment paper table

1. **H3** — primary numbers for Qwen2.5-0.5B ✅  
2. **H2** — shows the model still fits under a tight GPU budget ✅  
3. **H4** — documents CPU-only limits (expect slow; that's the point)

Add **H1** when you want the ARM+GPU row (optional but strengthens the edge story).

---

## Measured results (Qwen2.5-0.5B + CLIP ViT-L/336)

**Date:** 2026-07-16  
**Software:** Python 3.10.20 · PyTorch 2.4.1+cu121 · Transformers 4.57.6 · bf16 on CUDA

### H3 vs H2 comparison (same g6.xlarge, L4)

H2 applies `gpu_memory_fraction=0.5` (~11 GiB cap). H3 uses the full L4 (~22 GiB).

| Metric | H3 (full GPU) | H2 (50% GPU cap) | Δ |
|--------|---------------|------------------|---|
| Class. p50 | 297 ms | 298 ms | ~same |
| Class. p95 | 312 ms | 313 ms | ~same |
| Reason. p50 | 928 ms | 930 ms | ~same |
| Reason. p95 | 948 ms | 950 ms | ~same |
| Peak GPU | 3,449 MB | 3,449 MB | same |
| Peak RAM | 2,813 MB | 2,813 MB | same |

**Interpretation:** The model uses only **~3.5 GB GPU**, well below even the H2 cap (~11 GB). Latency and memory are effectively identical — this confirms the model is **not memory-bound** on L4-class hardware and would fit on smaller GPUs (T4G 16 GiB, Jetson Orin class).

Sources: `results/benchmark_qwen0.5b_H3.json`, `results/benchmark_qwen0.5b_H2.json`

### H3 detail (full GPU)

| Metric | Classification (alarm) | Reasoning (explanation) |
|--------|------------------------|-------------------------|
| Warm p50 latency | **297 ms** | **928 ms** |
| Warm p95 latency | **312 ms** | **948 ms** |
| Warm p99 latency | 320 ms | 954 ms |
| Cold first request (mean) | 461 ms | 929 ms |
| Load time (mean) | 4.5 s | 3.3 s |
| Peak GPU memory | **3,449 MB** | **3,460 MB** |
| Peak system RAM | **2,813 MB** | **2,813 MB** |
| Throughput | 3.34 req/s | 1.08 req/s |
| Time to first token | — | 42 ms (p50) |

### H2 detail (50% GPU memory cap)

| Metric | Classification (alarm) | Reasoning (explanation) |
|--------|------------------------|-------------------------|
| Warm p50 latency | **298 ms** | **930 ms** |
| Warm p95 latency | **313 ms** | **950 ms** |
| Peak GPU memory | **3,449 MB** | **3,460 MB** |
| Peak system RAM | **2,813 MB** | **2,813 MB** |

### Deployment interpretation

- **Classification mode** (~297 ms p50) supports roughly **3 frames/sec** on L4-class hardware — suitable for periodic camera polling (e.g. 1 fps with headroom).
- **Reasoning mode** (~928 ms p50) is ~3× slower; use on-demand, not for continuous alarm polling.
- **Memory footprint** (~3.5 GB GPU, ~2.8 GB RAM) fits comfortably on L4 and should fit on T4G (16 GiB) for H1.
- The earlier single-number benchmark (~832 ms) mixed in 64-token generation; the split protocol shows classification is much faster.

---

## Benchmark protocol (implemented)

Matches the GPT deployment spec, **without** video-stream tests.

| Setting | Value |
|---------|-------|
| Modes | **classification** (`Yes.` / `No.`, 8 tokens) and **reasoning** (64 tokens) — reported separately |
| Warm-up | 20 requests per restart |
| Measured | 100 requests per restart |
| Restarts | 3 independent model loads |
| Concurrency | 1 (primary), then optional 4 |
| Batch size | 1 |
| Total timed requests | 300 per mode (3 × 100) |

### What data the benchmark uses

This is a **speed / memory benchmark**, not an accuracy evaluation. Labels are not used.

Image selection priority (`fire_detect/bench.py`):

1. `--image-list file.txt` — fixed list, one path per line (recommended for paper reproducibility)
2. `--image /path/to/frame.jpg` — single image repeated
3. **`auto` (default)** — up to 20 `.jpg` files from `LLaVA-Instruct-150K/real_fire/` and `real_non_fire/` (top level only), cycled

**Current auto behaviour on this workspace:** fire images live in a subdirectory (`images/`), so auto mode currently picks **20 non-fire Detectium images** only. For a balanced fixed set, create `config/benchmark_images.txt`:

```bash
find LLaVA_MoD/MoD/detectioumfire/real_images/real_images/real_fire/images -name '*.jpg' | head -10 \
  > Deploy/config/benchmark_images.txt
find LLaVA-Instruct-150K/real_non_fire -maxdepth 1 -name '*.jpg' | head -10 \
  >> Deploy/config/benchmark_images.txt

bash scripts/aws_benchmark_hardware.sh H2 \
  --image-list config/benchmark_images.txt
```

### Metrics written to JSON

| JSON field | Meaning |
|------------|---------|
| `classification.warm_latency_ms.p50/p95/p99` | Alarm latency after warmup |
| `reasoning.warm_latency_ms.p50/p95/p99` | Explanation latency after warmup |
| `classification.load_time_s.mean` | Model load time |
| `classification.cold_first_request_ms` | First inference after each restart |
| `reasoning.time_to_first_token_ms` | TTFT for explanation mode |
| `classification.peak_gpu_memory_mb` | Peak GPU allocation |
| `classification.peak_rss_mb` | Peak system RAM |
| `bundle_size_gb` | LoRA checkpoint size on disk |

Entry point: `scripts/aws_benchmark_hardware.sh <H1|H2|H3|H4>`

---

## Step-by-step runbook

### 0. One-time setup

```bash
cd /home/ubuntu/llava-test/Deploy
cp config/students.env.example config/students.env   # if missing
# STUDENT_PRESET=qwen0.5b

bash scripts/cache_vision_tower.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Note: `cache_vision_tower.sh` uses safetensors to avoid a Torch < 2.6 loading block in Transformers 4.57.

### 1. Smoke test (~5 min)

```bash
bash scripts/aws_benchmark_hardware.sh H3 \
  --warmup 3 --runs 10 --restarts 1 --mode classification
```

### 2. Full H3 benchmark ✅ DONE

```bash
bash scripts/aws_benchmark_hardware.sh H3
```

Output: `results/benchmark_qwen0.5b_H3.json`

### 3. H2 on the same g6.xlarge ✅ DONE

```bash
bash scripts/aws_benchmark_hardware.sh H2
```

Output: `results/benchmark_qwen0.5b_H2.json`

### 4. H4 / H1 on a **separate ARM server** — **NEXT**

Full transfer + setup guide: **`docs/REMOTE_H1_H4_BENCHMARK.md`**

```bash
# On this g6.xlarge — build the transfer package
bash scripts/pack_for_remote_benchmark.sh
# Copy Deploy/transfer/llava-deploy-h1-h4-*.tar.gz to the remote host
```

On the remote host (after unpack + `configure_remote.sh`):

```bash
# H4 — c7g.2xlarge (CPU, reduced protocol)
bash scripts/aws_benchmark_hardware.sh H4 \
  --image-list config/benchmark_images.txt \
  --warmup 5 --runs 20 --restarts 1

# H1 — g5g.xlarge (ARM + T4G, full protocol)
bash scripts/aws_benchmark_hardware.sh H1 \
  --image-list config/benchmark_images.txt
```

### 5. Concurrency 4 on H3 (optional)

```bash
BENCHMARK_CONCURRENCY=4 bash scripts/aws_benchmark_hardware.sh H3 \
  --output results/benchmark_qwen0.5b_H3_c4.json
```

### 6. Offline detection sanity check

```bash
bash scripts/run_offline_detect.sh --image /path/to/test.jpg
```

### 7. H1 (optional: g5g.xlarge, ARM64 + T4G)

```bash
bash scripts/aws_benchmark_hardware.sh H1
```

---

## Paper deployment table (copy-ready)

| Hardware | Class. p50 | Class. p95 | Reason. p50 | Load (s) | GPU (MB) | RAM (MB) |
|----------|------------|------------|-------------|----------|----------|----------|
| H3 (L4, full) | 297 | 312 | 928 | 4.5 | 3,449 | 2,813 |
| H2 (L4, 50% cap) | 298 | 313 | 930 | 4.4 | 3,449 | 2,813 |
| H4 | — | — | — | — | — | — |
| H1 | — | — | — | — | — | — |

Fill H4 (and optionally H1) to complete the hardware matrix.

---

## Not in scope (by design)

- Video-stream / temporal alarm evaluation
- Multi-seed training comparisons
- Vision-encoder ablation (wait for A100 training)
- Claiming Jetson, Raspberry Pi, or production deployment without physical hardware
- Yes/No accuracy (use val worker separately)

---

## Legacy scripts

| Script | Maps to |
|--------|---------|
| `scripts/aws_benchmark_gpu.sh` | H3 |
| `scripts/aws_benchmark_arm.sh` | H4 |
| `scripts/benchmark_edge_deploy.sh` | Direct call to `fire_detect/bench.py` |

---

## Related files

- `Deploy/scripts/aws_benchmark_hardware.sh` — main entry point
- `Deploy/scripts/pack_for_remote_benchmark.sh` — builds H1/H4 transfer tarball
- `Deploy/docs/REMOTE_H1_H4_BENCHMARK.md` — **use this on the remote ARM server**
- `Deploy/config/benchmark.env.example` — protocol defaults (warmup, runs, restarts)
- `Deploy/config/students.env` — active student preset and paths
- `Deploy/config/fire_prompt.txt` — fixed detection prompt
- `Deploy/results/benchmark_qwen0.5b_H3.json` — completed H3 results
- `Deploy/results/benchmark_qwen0.5b_H2.json` — completed H2 results
- `Deploy/results/benchmark_qwen0.5b_H*.json` — all hardware profile results
- `Deploy/transfer/llava-deploy-h1-h4-*.tar.gz` — package to copy to H1/H4 instances
