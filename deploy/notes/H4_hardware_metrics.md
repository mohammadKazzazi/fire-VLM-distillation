# H4 hardware / systems metrics (GGUF students)

**No accuracy numbers in the latency table** — see `gguf_val/metrics_fire_*.json` separately.

**Hardware:** AWS `c7g.2xlarge` (paper **H4**) · 8× Arm Graviton3 (Neoverse-V1) · 16 GB RAM · no GPU · `eu-central-1`

**Timing:** shared `request_ready` boundary (see `TIMING_PROTOCOL.md`). H3 GPU reasoning re-measured 2026-07-23; H4 GGUF rows below are prior llama.cpp HTTP-client clocks (same conceptual boundary; not yet re-run after the code sync).

**Protocol (Llama3 / Gemma2):** llama.cpp `llama-server` · OpenBLAS+OpenMP · **8 threads** · `ctx=4096` · `temp=0` · fresh encode per image · reasoning `max_new_tokens=64` · **warmup 5 + runs 20 × 1 restart** (reduced vs full Qwen protocol)

**Protocol (Qwen 0.5B H4):** same stack · **20 warmup + 100 measured × 3 restarts** (full)  
**Protocol (Qwen 0.5B H3):** PyTorch CUDA · **20 warmup + 100 measured × 3 restarts** · reasoning only re-run

### Latency — reasoning focus (aligned clock)

| Hardware | Config | Engine | Reason. TTFT p50 (ms) | Reason. E2E p50 (ms) | Peak mem |
|----------|--------|--------|----------------------:|---------------------:|---------:|
| **H3** (L4 GPU) | Qwen2.5-0.5B fire | PyTorch bf16 | **52** | **940** | 3457 MB GPU / 3496 MB RSS |
| H4 (Arm CPU) | Qwen2.5-0.5B fire | llama.cpp Q8_0 + mmproj F16 | **7215** | **7368** | 1736 MB RSS |
| H4 | Qwen2.5-0.5B fire | llama.cpp Q4_K_M + mmproj F16 | 8666 | 8894 | 1341 MB RSS |
| H4 | **Llama-3.2-1B fire** | llama.cpp Q8_0 + mmproj F16 | **7855** | **8660** | 3645 MB RSS |
| H4 | **Gemma-2-2B fire** | llama.cpp Q8_0 + mmproj F16 | **11500** | **12194** | 7464 MB RSS |

H3 source: `results/benchmark_qwen0.5b_H3_request_ready_reasoning.json` (2026-07-23).  
H4 sources: unchanged GGUF benches below.

### H4 GGUF latency detail (class + reason)

| Config | Engine | LLM | mmproj | Threads | Class. E2E p50 (ms) | Reason. TTFT p50 (ms) | Reason. E2E p50 (ms) | Peak RSS (MB) |
|--------|--------|-----|--------|--------:|--------------------:|----------------------:|---------------------:|--------------:|
| Qwen2.5-0.5B fire | llama.cpp | Q8_0 | F16 | 8 | — | **7215** | **7368** | **1736** |
| Qwen2.5-0.5B fire | llama.cpp | Q4_K_M | F16 | 8 | — | 8666 | 8894 | 1341 |
| **Llama-3.2-1B fire** | llama.cpp | Q8_0 | F16 | 8 | **7943** | **7855** | **8660** | **3645** |
| **Gemma-2-2B fire** | llama.cpp | Q8_0 | F16 | 8 | **11714** | **11500** | **12194** | **7464** |

### Detectium Yes/No — full val (1452)

GPU from `detectium-val-only/eval/comparisons/table4_all_plans_with_yesno.json`.  
**GGUF Q8 fixed** — fixed mmproj + native template + stock llama.cpp preprocess (`*_gguf_q8_fixed`).  
**GGUF Q8 Pillow** — same weights/templates + `RESIZE_ALGO_BICUBIC_PILLOW` in `clip.cpp` (`*_gguf_q8_pillow`).

| Model | GPU acc | GPU FP | GPU FN | GGUF fixed acc | FP | FN | GGUF Pillow acc | FP | FN |
|-------|--------:|-------:|-------:|---------------:|---:|---:|----------------:|---:|---:|
| Llama-3.2-1B | 99.79% | 1 | 2 | 99.45% | 4 | 4 | 99.38% | 4 | 5 |
| Gemma-2-2B | 99.31% | 8 | 2 | 99.04% | 13 | 1 | 98.83% | 15 | 2 |
| Qwen2.5-0.5B | 99.72% | 2 | 2 | 99.38% | 7 | 2 | 99.45% | 6 | 2 |

### Table 4 — fire reasoning (754 fire rows, same val)

Heuristic scorer: `detectium-val-only/eval/score_fire_reasoning_table4.py`.

| Model | Runtime | Burning object | Environment | Fire severity |
|-------|---------|---------------:|------------:|--------------:|
| Llama-3.2-1B | GPU PyTorch | 68.97% | 77.97% | 87.74% |
| Llama-3.2-1B | GGUF Q8 fixed | 67.51% | 75.49% | 86.42% |
| Llama-3.2-1B | GGUF Q8 Pillow | 65.92% | 77.05% | 86.42% |
| Gemma-2-2B | GPU PyTorch | 68.57% | 76.35% | 88.18% |
| Gemma-2-2B | GGUF Q8 fixed | 67.37% | 77.78% | 88.24% |
| Gemma-2-2B | GGUF Q8 Pillow | 67.51% | 77.94% | 87.97% |
| Qwen2.5-0.5B | GPU PyTorch | 69.76% | 72.96% | 87.13% |
| Qwen2.5-0.5B | GGUF Q8 fixed | 67.90% | 73.29% | 87.35% |
| Qwen2.5-0.5B | GGUF Q8 Pillow | 69.23% | 75.89% | 87.47% |

### Definitions
| Metric | Definition |
|--------|------------|
| TTFT | **`request_ready`**: RGB image in memory → first output token (includes vision preprocess/encode + prefill). See `TIMING_PROTOCOL.md`. |
| Reasoning E2E | Same t0 → full request done (`max_new_tokens=64`) |
| Classification E2E | Same t0 → short Yes/No-style decode budget |
| Peak RSS | Peak process RSS during measured runs |

> **Note (2026-07-23):** H3 Qwen reasoning re-measured under `request_ready` → TTFT p50 **52 ms**, E2E p50 **940 ms** (was 42 / 928 with timer at `generate` only). H4 GGUF not re-run yet; prefer **Reason. E2E** for H3 vs H4 claims. Gap remains ~**8×** E2E (940 ms vs 7368 ms), not a timer artifact.

### Quantization note (Qwen F16 recheck)
Unquantized LLM **F16** on Q4∪Q8 FP/FN (`gguf_val/fp_fn_q4_q8_f16_recheck.jsonl`): F16 fixed **4/9** Q4 errors and **0/9** Q8 errors. Prefer **Q8_0 + mmproj F16**.

### Root cause (Llama3 / Gemma2 GGUF vs GPU gap)
mmproj conversion was **missing `mm_vision_select_layer=-2`**, so llama.cpp used CLIP layer output **off-by-one vs training**. Also fixed Llama3 **double-BOS** (jinja emitted `<|begin_of_text|>` while mtmd `add_special=true` already prepends BOS).

| Model | Before (Q8) | Error-set after fix | Still wrong on error-set |
|-------|------------:|--------------------:|-------------------------:|
| Llama3-1B | 98.14% (28 errs) | **24/28 fixed** | 4 |
| Gemma2-2B | 97.18% (41 errs) | **28/41 fixed** | 13 |

Full Detectium re-val (fixed mmproj): `pipeline_logs/full_val_mmproj_fix.log` → `metrics_fire_*_gguf_q8_fixed.json`.  
Pillow preprocess experiment (`pipeline_logs/full_val_pillow_patch.log`): net negative on Llama/Gemma; **Qwen improves** vs fixed (Yes/No +0.07 pp; Table 4 all three axes up).

### Quantization note (Llama3 / Gemma2 F16 recheck on Q8 FP/FN)
On each model’s **Q8 error set** *before* the mmproj fix (`fp_fn_*_q8_f16_recheck.jsonl`):

| Model | Q8 errors | F16 fixed | Still wrong on F16 |
|-------|----------:|----------:|-------------------:|
| Llama3-1B | 28 | **4** | 24 (23 FN + 1 FP) |
| Gemma2-2B | 41 | **1** | 40 (39 FP + 1 FN) |

F16 alone barely helped — the gap was **not** mainly Q8 quant.

### Sources
- H3 Qwen reasoning (`request_ready`): `results/benchmark_qwen0.5b_H3_request_ready_reasoning.json`
- Qwen Q8 reasoning H4: `benchmark_qwen0.5b_H4_llamacpp_q8_reasoning.json`
- Llama3: `benchmark_llama3_1b_H4_llamacpp.json` + `llava_llama3.jinja`
- Gemma2: `benchmark_gemma2_2b_H4_llamacpp.json` + `llava_gemma2.jinja`
- Accuracy: `metrics_fire_*_gguf_q8_fixed.json`, `metrics_fire_*_gguf_q8_pillow.json`, `table4_all_students_gguf_fixed.json`, `table4_*_pillow*.json`
