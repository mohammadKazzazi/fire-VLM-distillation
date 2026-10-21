# Edge Deployment — Hardware Benchmarking (Deployment Only)

**Scope:** Offline inference packaging and **hardware-proximate benchmarking** for the Qwen2.5-0.5B fire-detection student (and later other presets). Training, seeds, and vision-encoder ablations are **out of scope** here.

**Last updated:** 2026-07-16  
**Status:** H3 + H2 complete · H4 pending · H1 optional

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

**Date:** 2026-07-16 (class / H2); **reasoning H3 re-run 2026-07-23** under `request_ready` timing  
**Software:** Python 3.10.20 · PyTorch 2.4.1+cu121 · Transformers 4.57.6 · bf16 on CUDA

### H3 vs H2 comparison (same g6.xlarge, L4)

H2 applies `gpu_memory_fraction=0.5` (~11 GiB cap). H3 uses the full L4 (~22 GiB).  
Classification / H2 rows are 2026-07-16. H3 reasoning row is **2026-07-23** (`request_ready`).

| Metric | H3 (full GPU) | H2 (50% GPU cap) | Δ |
|--------|---------------|------------------|---|
| Class. p50 | 297 ms | 298 ms | ~same |
| Class. p95 | 312 ms | 313 ms | ~same |
| Reason. p50 | **940 ms** | 930 ms† | ~same |
| Reason. p95 | **1104 ms** | 950 ms† | — |
| Reason. TTFT p50 | **52 ms** | 42 ms† | — |
| Peak GPU | 3,457 MB | 3,449 MB | ~same |
| Peak RAM | 3,496 MB | 2,813 MB | — |

† H2 not re-run under `request_ready` yet (old `generate`-only clock).

**Interpretation:** The model uses only **~3.5 GB GPU**, well below even the H2 cap (~11 GB). Latency and memory are effectively identical on L4 — **not memory-bound**; would fit on smaller GPUs (T4G 16 GiB, Jetson Orin class).

Sources: `results/benchmark_qwen0.5b_H3.json`, `results/benchmark_qwen0.5b_H2.json`, `results/benchmark_qwen0.5b_H3_request_ready_reasoning.json`

### H3 detail (full GPU)

| Metric | Classification (alarm)† | Reasoning (explanation)‡ |
|--------|------------------------:|-------------------------:|
| Warm p50 latency | **297 ms** | **940 ms** |
| Warm p95 latency | **312 ms** | **1104 ms** |
| Warm p99 latency | 320 ms | 1121 ms |
| Cold first request (mean) | 461 ms | 1123 ms |
| Load time (mean) | 4.5 s | 4.6 s |
| Peak GPU memory | **3,449 MB** | **3,457 MB** |
| Peak system RAM | **2,813 MB** | **3,496 MB** |
| Throughput | 3.34 req/s | 1.05 req/s |
| Time to first token (p50) | — | **52 ms** |

† 2026-07-16 (timer at `generate`). ‡ 2026-07-23 (`request_ready`).

### H2 detail (50% GPU memory cap)

| Metric | Classification (alarm) | Reasoning (explanation) |
|--------|------------------------|-------------------------|
| Warm p50 latency | **298 ms** | **930 ms** |
| Warm p95 latency | **313 ms** | **950 ms** |
| Peak GPU memory | **3,449 MB** | **3,460 MB** |
| Peak system RAM | **2,813 MB** | **2,813 MB** |

### Deployment interpretation

- **Classification mode** (~297 ms p50) supports roughly **3 frames/sec** on L4-class hardware — suitable for periodic camera polling (e.g. 1 fps with headroom).
- **Reasoning mode** (~940 ms p50 under aligned clock) is ~3× slower; use on-demand, not for continuous alarm polling.
- **Memory footprint** (~3.5 GB GPU, ~3.5 GB RSS on reasoning re-run) fits comfortably on L4 and should fit on T4G (16 GiB) for H1.
- Aligned H3 vs H4 Qwen Q8 reasoning E2E: **940 ms vs 7368 ms** (~8×); TTFT **52 vs 7215 ms** (CPU is prefill-bound).

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
| `reasoning.time_to_first_token_ms` | TTFT for explanation mode (`request_ready` boundary — see `notes/TIMING_PROTOCOL.md`) |
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

### 4. H4 (separate ARM instance: c7g.2xlarge) — **NEXT**

```bash
bash scripts/cache_vision_tower.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash scripts/aws_benchmark_hardware.sh H4 --runs 20 --warmup 5 --restarts 1
```

Use fewer runs on H4 — CPU inference is slow.

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

Timing: classification / H2 = 2026-07-16; H3 reasoning = 2026-07-23 `request_ready`; H4 = llama.cpp Q8.

| Hardware | Class. p50 | Class. p95 | Reason. TTFT p50 | Reason. E2E p50 | Load (s) | GPU (MB) | RAM (MB) |
|----------|------------|------------|------------------|-----------------|----------|----------|----------|
| H3 (L4, full) | 297 | 312 | **52** | **940** | 4.6 | 3,457 | 3,496 |
| H2 (L4, 50% cap) | 298 | 313 | 42† | 930† | 4.4 | 3,449 | 2,813 |
| H4 (Arm CPU, Qwen Q8) | — | — | **7215** | **7368** | — | — | 1,736 |
| H1 | — | — | — | — | — | — | — |

† H2 not re-measured under `request_ready`.

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

| Path | Purpose |
|------|---------|
| `scripts/aws_benchmark_hardware.sh` | Main benchmark entry point |
| `config/benchmark.env.example` | Protocol defaults (warmup, runs, restarts) |
| `config/students.env` | Active student preset and paths |
| `config/fire_prompt.txt` | Fixed detection prompt |
| `results/benchmark_qwen0.5b_H3.json` | Completed H3 results |
| `results/benchmark_qwen0.5b_H*.json` | All hardware profile results |
