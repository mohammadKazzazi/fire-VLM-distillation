#!/usr/bin/env python3
"""Deployment benchmark — ChatGPT protocol (classification + reasoning modes)."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

_DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(_DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_ROOT))

from fire_detect.daemon import _load_env_file, _resolve_preset  # noqa: E402
from fire_detect.inference import FireDetectResult, FireDetector  # noqa: E402
from fire_detect.cpu_opt import cpu_inference_settings  # noqa: E402

CLASSIFICATION_MAX_TOKENS = 8
REASONING_MAX_TOKENS = 64


def _ru_max_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _disk_usage_gb(path: Path) -> float:
    total = 0
    if not path.exists():
        return 0.0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024**3)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(len(ordered) * pct) - 1))
    return ordered[idx]


def _latency_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "mean": round(statistics.mean(values), 2),
        "p50": round(_percentile(values, 0.50), 2),
        "p95": round(_percentile(values, 0.95), 2),
        "p99": round(_percentile(values, 0.99), 2),
    }


def _collect_images(image_arg: str, image_list: Optional[str], count: int) -> list[str]:
    if image_list:
        paths = [ln.strip() for ln in Path(image_list).read_text().splitlines() if ln.strip()]
        if not paths:
            raise ValueError(f"No images in {image_list}")
        return [paths[i % len(paths)] for i in range(count)]

    if image_arg:
        return [image_arg] * count

    candidate_root = _DEPLOY_ROOT.parent / "LLaVA-Instruct-150K"
    found: list[str] = []
    for sub in ("real_fire", "real_non_fire"):
        d = candidate_root / sub
        if d.is_dir():
            found.extend(str(p) for p in sorted(d.glob("*.jpg"))[:20])
            found.extend(str(p) for p in sorted(d.glob("*.png"))[:20])
    if not found:
        demo = _DEPLOY_ROOT / "samples" / "demo.jpg"
        if demo.is_file():
            return [str(demo)] * count
        raise ValueError("Provide --image or --image-list")
    return [found[i % len(found)] for i in range(count)]


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return info
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    info.update(
        {
            "device_index": idx,
            "name": props.name,
            "total_memory_gib": round(props.total_memory / (1024**3), 2),
            "driver": None,
            "cuda_version": torch.version.cuda,
        }
    )
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0]
        info["driver"] = smi
    except Exception:
        pass
    return info


def _apply_gpu_memory_fraction(fraction: Optional[float]) -> None:
    if fraction is None or not torch.cuda.is_available():
        return
    torch.cuda.set_per_process_memory_fraction(float(fraction), device=0)


def _load_detector(cfg: dict, device: str, max_new_tokens: int) -> tuple[FireDetector, float]:
    llava_root = os.environ.get("LLAVA_MOD_ROOT")
    t0 = time.perf_counter()
    detector = FireDetector(
        model_path=cfg["model_path"],
        model_base=cfg["model_base"],
        conv_mode=cfg["conv_mode"],
        device=device,
        load_4bit=os.environ.get("LOAD_4BIT") == "1",
        load_8bit=os.environ.get("LOAD_8BIT") == "1",
        max_new_tokens=max_new_tokens,
        llava_mod_root=llava_root,
    )
    return detector, time.perf_counter() - t0


def _unload_detector(detector: FireDetector) -> None:
    del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _run_sequential(
    detector: FireDetector,
    images: list[str],
    prompt: str,
    *,
    max_new_tokens: int,
    measure_ttft: bool,
) -> tuple[list[float], list[float], list[str]]:
    latencies: list[float] = []
    ttfts: list[float] = []
    answers: list[str] = []
    for img in images:
        r = detector.detect(
            img,
            prompt,
            max_new_tokens=max_new_tokens,
            measure_ttft=measure_ttft,
        )
        latencies.append(r.latency_ms)
        if r.ttft_ms is not None:
            ttfts.append(r.ttft_ms)
        answers.append(r.raw_text)
    return latencies, ttfts, answers


def _run_concurrent(
    detector: FireDetector,
    images: list[str],
    prompt: str,
    *,
    max_new_tokens: int,
    concurrency: int,
    measure_ttft: bool,
) -> tuple[list[float], list[float]]:
    lock = threading.Lock()
    latencies: list[float] = []
    ttfts: list[float] = []

    def _one(img: str) -> FireDetectResult:
        with lock:
            return detector.detect(
                img,
                prompt,
                max_new_tokens=max_new_tokens,
                measure_ttft=measure_ttft,
            )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one, img) for img in images]
        for fut in as_completed(futures):
            r = fut.result()
            latencies.append(r.latency_ms)
            if r.ttft_ms is not None:
                ttfts.append(r.ttft_ms)
    return latencies, ttfts


def _benchmark_mode(
    cfg: dict,
    device: str,
    prompt: str,
    images: list[str],
    *,
    mode: str,
    warmup: int,
    runs: int,
    restarts: int,
    concurrency: int,
    gpu_memory_fraction: Optional[float],
) -> dict[str, Any]:
    max_new_tokens = CLASSIFICATION_MAX_TOKENS if mode == "classification" else REASONING_MAX_TOKENS
    measure_ttft = mode == "reasoning"

    all_warm_latencies: list[float] = []
    all_ttfts: list[float] = []
    load_times: list[float] = []
    cold_latencies: list[float] = []
    peak_gpu_mb: list[float] = []
    peak_rss_mb: list[float] = []

    for restart_idx in range(restarts):
        _apply_gpu_memory_fraction(gpu_memory_fraction)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        detector, load_s = _load_detector(cfg, device, max_new_tokens)
        load_times.append(load_s)

        cold_img = images[0]
        cold = detector.detect(
            cold_img,
            prompt,
            max_new_tokens=max_new_tokens,
            measure_ttft=measure_ttft,
        )
        cold_latencies.append(cold.latency_ms)

        warmup_images = images[:warmup]
        _run_sequential(
            detector,
            warmup_images,
            prompt,
            max_new_tokens=max_new_tokens,
            measure_ttft=False,
        )

        measured_images = images[warmup : warmup + runs]
        if concurrency <= 1:
            latencies, ttfts, _ = _run_sequential(
                detector,
                measured_images,
                prompt,
                max_new_tokens=max_new_tokens,
                measure_ttft=measure_ttft,
            )
        else:
            latencies, ttfts = _run_concurrent(
                detector,
                measured_images,
                prompt,
                max_new_tokens=max_new_tokens,
                concurrency=concurrency,
                measure_ttft=measure_ttft,
            )

        all_warm_latencies.extend(latencies)
        all_ttfts.extend(ttfts)

        if torch.cuda.is_available():
            peak_gpu_mb.append(torch.cuda.max_memory_allocated() / (1024**2))
        peak_rss_mb.append(_ru_max_rss_mb())

        _unload_detector(detector)

    wall_s = sum(all_warm_latencies) / 1000.0
    throughput = (len(all_warm_latencies) / wall_s) if wall_s > 0 else 0.0

    return {
        "mode": mode,
        "max_new_tokens": max_new_tokens,
        "warmup_per_restart": warmup,
        "measured_per_restart": runs,
        "restarts": restarts,
        "concurrency": concurrency,
        "total_measured_requests": len(all_warm_latencies),
        "load_time_s": {
            "mean": round(statistics.mean(load_times), 3),
            "min": round(min(load_times), 3),
            "max": round(max(load_times), 3),
        },
        "cold_first_request_ms": _latency_stats(cold_latencies),
        "warm_latency_ms": _latency_stats(all_warm_latencies),
        "time_to_first_token_ms": _latency_stats(all_ttfts) if all_ttfts else None,
        "throughput_rps": round(throughput, 3),
        "peak_gpu_memory_mb": round(max(peak_gpu_mb), 2) if peak_gpu_mb else None,
        "peak_rss_mb": round(max(peak_rss_mb), 2) if peak_rss_mb else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deployment benchmark (GPT protocol)")
    parser.add_argument("--config", type=str, default=str(_DEPLOY_ROOT / "config" / "students.env"))
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--image-list", type=str, default=None, help="Text file with one image path per line")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1, choices=[1, 4])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mode", type=str, default="both", choices=["classification", "reasoning", "both"])
    parser.add_argument("--hardware-id", type=str, default=None, help="Paper label: H1, H2, H3, H4")
    parser.add_argument("--gpu-memory-fraction", type=float, default=None, help="H2: e.g. 0.5 for half GPU")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    bench_env = _DEPLOY_ROOT / "config" / "benchmark.env"
    if bench_env.is_file():
        _load_env_file(bench_env)
    _load_env_file(Path(args.config))

    cfg = _resolve_preset(dict(os.environ))
    prompt_path = _DEPLOY_ROOT / "config" / "fire_prompt.txt"
    prompt = prompt_path.read_text().strip()
    device = args.device or os.environ.get("DEVICE", "cuda")

    gpu_fraction = args.gpu_memory_fraction
    if gpu_fraction is None and os.environ.get("GPU_MEMORY_FRACTION"):
        gpu_fraction = float(os.environ["GPU_MEMORY_FRACTION"])

    total_images_needed = args.restarts * (args.warmup + args.runs)
    if args.concurrency > 1:
        total_images_needed = max(total_images_needed, args.runs * args.restarts)

    images = _collect_images(args.image, args.image_list, total_images_needed)

    modes: list[str]
    if args.mode == "both":
        modes = ["classification", "reasoning"]
    else:
        modes = [args.mode]

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "gpt-deployment-v1",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_arch": platform.machine(),
        "cpu_model": platform.processor() or platform.machine(),
        "hardware_id": args.hardware_id or os.environ.get("HARDWARE_ID"),
        "preset": cfg["preset"],
        "model_path": cfg["model_path"],
        "model_base": cfg.get("model_base"),
        "conv_mode": cfg["conv_mode"],
        "device": device,
        "prompt": prompt,
        "image_source": args.image or args.image_list or "auto",
        "gpu": _gpu_info(),
        "bundle_size_gb": round(_disk_usage_gb(Path(cfg["model_path"])), 4),
        "inference_settings": {
            "precision": "bf16-on-cuda" if device == "cuda" else "fp32-on-cpu",
            "quantization": "4bit" if os.environ.get("LOAD_4BIT") == "1" else (
                "8bit" if os.environ.get("LOAD_8BIT") == "1" else (
                    "dynamic_int8" if device == "cpu" and os.environ.get("CPU_INT8") == "1" else "none"
                )
            ),
            "batch_size": 1,
            "temperature": float(os.environ.get("TEMPERATURE", "0.0")),
            "classification_max_new_tokens": CLASSIFICATION_MAX_TOKENS,
            "reasoning_max_new_tokens": REASONING_MAX_TOKENS,
            "gpu_memory_fraction": gpu_fraction,
            "cpu_adaptations": cpu_inference_settings() if device == "cpu" else None,
        },
        "benchmark_procedure": {
            "warmup_per_restart": args.warmup,
            "measured_per_restart": args.runs,
            "restarts": args.restarts,
            "concurrency": args.concurrency,
            "modes": modes,
        },
    }

    for mode in modes:
        report[mode] = _benchmark_mode(
            cfg,
            device,
            prompt,
            images,
            mode=mode,
            warmup=args.warmup,
            runs=args.runs,
            restarts=args.restarts,
            concurrency=args.concurrency,
            gpu_memory_fraction=gpu_fraction,
        )

    out_path = Path(args.output) if args.output else (
        _DEPLOY_ROOT
        / "results"
        / f"benchmark_{cfg['preset']}_{report.get('hardware_id') or device}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
