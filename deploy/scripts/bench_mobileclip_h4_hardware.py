#!/usr/bin/env python3
"""H4 hardware bench for Qwen3-0.6B + MobileCLIP (PyTorch CPU bf16).

Matches reduced GGUF protocol: warmup 5 + measured 20, classification (8 tok)
and reasoning (64 tok), with TTFT on reasoning.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(
    HF_HUB_OFFLINE="0",
    TRANSFORMERS_OFFLINE="0",
    CPU_OPT="1",
    CPU_USE_CACHE="1",
    CPU_DTYPE="bfloat16",
)

import sys

sys.path.insert(0, "/home/ubuntu/llavamod/llava-deploy-h1-h4-20260716/workspace/Deploy")
from fire_detect.inference import FireDetector  # noqa: E402

DEPLOY = Path("/home/ubuntu/llavamod/llava-deploy-h1-h4-20260716/workspace/Deploy")
OUT = DEPLOY / "results" / "benchmark_qwen3_0.6b_mobileclip_H4_pytorch.json"
LORA = (
    "/home/ubuntu/llavamod/qwen3-0.6b-mobileclip-students/checkpoints/"
    "plan4-d2d-multiquery-v4-kd-lm-qwen3-0.6b-mobileclip-s2-seed42"
)
LLAVA = "/home/ubuntu/llavamod/llava-deploy-h1-h4-20260716/workspace/LLaVA_MoD/MoD/LLava_MoD"
PROMPT = "Carefully analyze the image for signs of fire or smoke. Answer Yes or No only."
WARMUP = 5
RUNS = 20


def rss_mb() -> float:
    # Linux ru_maxrss is KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = max(0, min(len(s) - 1, int(len(s) * p) - 1))
    return s[idx]


def stats(vals: list[float]) -> dict:
    if not vals:
        return {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}
    return {
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "mean": round(statistics.mean(vals), 2),
        "p50": round(pct(vals, 0.50), 2),
        "p95": round(pct(vals, 0.95), 2),
        "p99": round(pct(vals, 0.99), 2),
    }


def pick_images(n: int) -> list[str]:
    lst = DEPLOY / "config" / "benchmark_images.txt"
    if lst.is_file():
        paths = [ln.strip() for ln in lst.read_text().splitlines() if ln.strip()]
        if paths:
            return [paths[i % len(paths)] for i in range(n)]
    # fallback: detectium val fire/nonfire mix
    val = Path("/home/ubuntu/llavamod/detectium-val-only/images")
    imgs = sorted(val.glob("*.jpg"))[: max(n, 1)]
    return [str(imgs[i % len(imgs)]) for i in range(n)]


def run_mode(det: FireDetector, images: list[str], max_new: int, measure_ttft: bool) -> dict:
    # warmup
    for img in images[:WARMUP]:
        det.detect(img, PROMPT, max_new_tokens=max_new, measure_ttft=False)
    latencies = []
    ttfts = []
    for img in images[WARMUP : WARMUP + RUNS]:
        r = det.detect(img, PROMPT, max_new_tokens=max_new, measure_ttft=measure_ttft)
        latencies.append(r.latency_ms)
        if r.ttft_ms is not None:
            ttfts.append(r.ttft_ms)
        print(
            f"  tok={max_new} e2e={r.latency_ms:.0f}ms ttft={r.ttft_ms} :: {r.raw_text[:60]!r}",
            flush=True,
        )
    return {
        "mode": "reasoning" if max_new >= 32 else "classification",
        "max_new_tokens": max_new,
        "warmup_per_restart": WARMUP,
        "measured_per_restart": RUNS,
        "restarts": 1,
        "concurrency": 1,
        "total_measured_requests": len(latencies),
        "warm_latency_ms": stats(latencies),
        "time_to_first_token_ms": stats(ttfts) if ttfts else None,
        "peak_rss_mb": round(rss_mb(), 2),
        "throughput_rps": round(1000.0 / statistics.mean(latencies), 3) if latencies else 0,
    }


def main() -> None:
    images = pick_images(WARMUP + RUNS)
    print(f"images={len(images)} first={images[0]}", flush=True)
    t0 = time.time()
    det = FireDetector(
        model_path=LORA,
        model_base="Qwen/Qwen3-0.6B",
        conv_mode="qwen3",
        device="cpu",
        max_new_tokens=64,
        temperature=0.0,
        llava_mod_root=LLAVA,
    )
    load_s = time.time() - t0
    print(f"loaded {load_s:.1f}s cache={det._use_cache} dtype={getattr(det,'_cpu_dtype',None)}", flush=True)

    print("=== classification (8) ===", flush=True)
    classification = run_mode(det, images, 8, measure_ttft=False)
    classification["load_time_s"] = {"mean": round(load_s, 3), "min": round(load_s, 3), "max": round(load_s, 3)}

    print("=== reasoning (64) ===", flush=True)
    reasoning = run_mode(det, images, 64, measure_ttft=True)

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "gpt-deployment-v1",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_arch": platform.machine(),
        "hardware_id": "H4",
        "preset": "qwen3-0.6b-mobileclip",
        "model_path": LORA,
        "mmproj_path": None,
        "model_base": "Qwen/Qwen3-0.6B",
        "conv_mode": "qwen3",
        "device": "cpu",
        "runtime": "pytorch+mobileclip",
        "prompt": PROMPT,
        "inference_settings": {
            "precision": "bf16",
            "cpu_use_cache": True,
            "cpu_dtype": "bfloat16",
            "classification_max_new_tokens": 8,
            "reasoning_max_new_tokens": 64,
            "threads": int(os.environ.get("CPU_THREADS") or (os.cpu_count() or 1)),
            "notes": "No GGUF mmproj (MobileCLIP/FastViT unsupported in llama.cpp mtmd).",
        },
        "benchmark_procedure": {
            "warmup_per_restart": WARMUP,
            "measured_per_restart": RUNS,
            "restarts": 1,
            "concurrency": 1,
            "modes": ["classification", "reasoning"],
        },
        "classification": classification,
        "reasoning": reasoning,
        "hardware_profile": {
            "id": "H4",
            "expected_instance_type": "c7g.2xlarge",
            "maps_to": "ARM64 CPU-only PyTorch (MobileCLIP student)",
        },
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"classification_p50": classification["warm_latency_ms"]["p50"],
                      "reasoning_p50": reasoning["warm_latency_ms"]["p50"],
                      "reasoning_ttft_p50": (reasoning["time_to_first_token_ms"] or {}).get("p50"),
                      "peak_rss_mb": reasoning["peak_rss_mb"]}, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
