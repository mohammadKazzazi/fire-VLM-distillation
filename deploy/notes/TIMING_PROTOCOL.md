# Aligned timing protocol (GPU PyTorch ↔ GGUF llama.cpp)

**Boundary name:** `request_ready`  
**Code:** `Deploy/fire_detect/bench.py` (GPU/CPU Torch) · `Deploy/fire_detect/bench_llamacpp.py` (llama-server)

## Shared clock

| Event | Definition |
|-------|------------|
| **Before t0** | Model already loaded. RGB image (or base64 bytes) already in memory. |
| **t0** | GPU: immediately before `process_images` / tokenize / H2D / `generate`. GGUF: immediately before HTTP `POST /v1/chat/completions`. |
| **TTFT** | First **output text token** available to the client. |
| **E2E** | Generation finished (GPU: after `torch.cuda.synchronize()`). |

**Excluded from latency:** model load, disk I/O, llama-server process startup.

## Why older H3 vs H4 TTFT numbers were not line-upable

| Stack (old) | Timer started at |
|-------------|------------------|
| GPU PyTorch | `model.generate(...)` — **after** PIL resize/normalize + tensor copy |
| GGUF | HTTP request send — **includes** server receive + CLIP encode + prefill |

That mismatch is tens of ms at most. The ~170× H3↔H4 gap is still mostly **L4 GPU vs 8-core Graviton CPU**.

## How to re-run fairly

```bash
# GPU (H2/H3) — same machine as before
python -m fire_detect.bench --hardware-id H3 --mode both

# GGUF (H4) — llama-server already up with the student GGUF + mmproj
python -m fire_detect.bench_llamacpp \
  --base-url http://127.0.0.1:8080 \
  --hardware-id H4 \
  --preset qwen0.5b \
  --warmup 5 --runs 20
```

JSON reports include a top-level `"timing"` block with `boundary: "request_ready"`.

## Meeting guidance

- Prefer **Reason. E2E** and **Class. E2E** for H3 vs H4 hardware claims.
- Report TTFT only with the shared boundary noted; do not treat old 42 ms vs 7215 ms as the same clock.
- Re-measured rows after this change may show slightly higher GPU TTFT/E2E (preprocess + CUDA sync now inside the window).
