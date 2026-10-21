#!/usr/bin/env python3
"""llama.cpp / GGUF deployment benchmark — same timing boundary as fire_detect/bench.py.

Timing contract (request_ready):
  - Preload image bytes before t0 (disk I/O excluded).
  - t0 = immediately before HTTP POST to llama-server.
  - TTFT = first streamed text token content.
  - E2E = stream / response complete.

Requires a running llama-server with multimodal (mtmd) support, e.g.:
  ./llama-server -m model.gguf --mmproj mmproj.gguf -c 4096 -t 8 --port 8080
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import platform
import resource
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(_DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_ROOT))

from fire_detect.inference import TIMING_BOUNDARY, TIMING_BOUNDARY_DEF  # noqa: E402

CLASSIFICATION_MAX_TOKENS = 8
REASONING_MAX_TOKENS = 64


def _ru_max_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024


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
        p = Path(image_arg)
        if not p.is_file():
            raise FileNotFoundError(image_arg)
        return [str(p)] * count

    candidates: list[str] = []
    for root in (
        _DEPLOY_ROOT / "samples" / "bench_images",
        _DEPLOY_ROOT.parent / "LLaVA-Instruct-150K" / "real_non_fire",
    ):
        if root.is_dir():
            candidates.extend(sorted(str(p) for p in root.glob("*.jpg"))[:20])
    if not candidates:
        raise FileNotFoundError("No benchmark images found; pass --image or --image-list")
    return [candidates[i % len(candidates)] for i in range(count)]


def _preload_b64(paths: list[str]) -> list[tuple[str, str]]:
    """Return (mime, base64) pairs; disk read happens here (untimed)."""
    out: list[tuple[str, str]] = []
    for p in paths:
        data = Path(p).read_bytes()
        mime = mimetypes.guess_type(p)[0] or "image/jpeg"
        out.append((mime, base64.b64encode(data).decode("ascii")))
    return out


def _extract_delta_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        # Some llama-server builds stream {content: "..."} at top level.
        return str(payload.get("content") or "")
    ch0 = choices[0]
    delta = ch0.get("delta") or {}
    if isinstance(delta, dict) and delta.get("content"):
        return str(delta["content"])
    msg = ch0.get("message") or {}
    if isinstance(msg, dict) and msg.get("content"):
        return str(msg["content"])
    if ch0.get("text"):
        return str(ch0["text"])
    return ""


def _chat_once(
    base_url: str,
    prompt: str,
    mime: str,
    b64: str,
    *,
    max_tokens: int,
    temperature: float,
    measure_ttft: bool,
    timeout_s: float,
) -> tuple[float, Optional[float], str]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": bool(measure_ttft),
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream, application/json"},
        method="POST",
    )

    ttft_ms: Optional[float] = None
    chunks: list[str] = []
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if measure_ttft:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text or text.startswith(":"):
                        continue
                    if text.startswith("data:"):
                        text = text[5:].strip()
                    if text == "[DONE]":
                        break
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    piece = _extract_delta_text(payload)
                    if piece:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000.0
                        chunks.append(piece)
            else:
                raw_body = resp.read().decode("utf-8", errors="replace")
                payload = json.loads(raw_body)
                piece = _extract_delta_text(payload)
                if not piece:
                    choices = payload.get("choices") or []
                    if choices:
                        msg = choices[0].get("message") or {}
                        piece = str(msg.get("content") or "")
                chunks.append(piece)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server HTTP {e.code}: {detail[:500]}") from e

    e2e_ms = (time.perf_counter() - t0) * 1000.0
    return e2e_ms, ttft_ms, "".join(chunks).strip()


def _wait_ready(base_url: str, timeout_s: float = 120.0) -> float:
    """Poll /health (or /v1/models). Returns seconds until ready."""
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    health = base_url.rstrip("/") + "/health"
    models = base_url.rstrip("/") + "/v1/models"
    last_err = ""
    while time.perf_counter() < deadline:
        for url in (health, models):
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        return time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
        time.sleep(0.5)
    raise TimeoutError(f"llama-server not ready at {base_url}: {last_err}")


def _benchmark_mode(
    base_url: str,
    prompt: str,
    media: list[tuple[str, str]],
    *,
    mode: str,
    warmup: int,
    runs: int,
    temperature: float,
    timeout_s: float,
) -> dict[str, Any]:
    max_tokens = CLASSIFICATION_MAX_TOKENS if mode == "classification" else REASONING_MAX_TOKENS
    measure_ttft = mode == "reasoning"

    warm_latencies: list[float] = []
    ttfts: list[float] = []

    # Cold first request
    mime, b64 = media[0]
    cold_ms, cold_ttft, _ = _chat_once(
        base_url,
        prompt,
        mime,
        b64,
        max_tokens=max_tokens,
        temperature=temperature,
        measure_ttft=measure_ttft,
        timeout_s=timeout_s,
    )

    for i in range(warmup):
        mime, b64 = media[i % len(media)]
        _chat_once(
            base_url,
            prompt,
            mime,
            b64,
            max_tokens=max_tokens,
            temperature=temperature,
            measure_ttft=False,
            timeout_s=timeout_s,
        )

    for i in range(runs):
        mime, b64 = media[(warmup + i) % len(media)]
        e2e_ms, ttft_ms, _ = _chat_once(
            base_url,
            prompt,
            mime,
            b64,
            max_tokens=max_tokens,
            temperature=temperature,
            measure_ttft=measure_ttft,
            timeout_s=timeout_s,
        )
        warm_latencies.append(e2e_ms)
        if ttft_ms is not None:
            ttfts.append(ttft_ms)

    wall_s = sum(warm_latencies) / 1000.0
    throughput = (len(warm_latencies) / wall_s) if wall_s > 0 else 0.0
    return {
        "mode": mode,
        "max_new_tokens": max_tokens,
        "warmup_per_restart": warmup,
        "measured_per_restart": runs,
        "restarts": 1,
        "concurrency": 1,
        "total_measured_requests": len(warm_latencies),
        "timing_boundary": TIMING_BOUNDARY,
        "cold_first_request_ms": _latency_stats([cold_ms]),
        "cold_ttft_ms": cold_ttft,
        "warm_latency_ms": _latency_stats(warm_latencies),
        "time_to_first_token_ms": _latency_stats(ttfts) if ttfts else None,
        "throughput_rps": round(throughput, 3),
        "peak_rss_mb": round(_ru_max_rss_mb(), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="llama.cpp GGUF bench (aligned timing)")
    parser.add_argument("--base-url", type=str, default=os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--image-list", type=str, default=None)
    parser.add_argument("--prompt-file", type=str, default=str(_DEPLOY_ROOT / "config" / "fire_prompt.txt"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--mode", type=str, default="both", choices=["classification", "reasoning", "both"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--hardware-id", type=str, default="H4")
    parser.add_argument("--preset", type=str, default=os.environ.get("STUDENT_PRESET", "gguf"))
    parser.add_argument("--model-label", type=str, default=os.environ.get("GGUF_MODEL_LABEL", ""))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text().strip()
    need = max(1, args.warmup + args.runs)
    paths = _collect_images(args.image, args.image_list, need)
    media = _preload_b64(paths)

    ready_s = _wait_ready(args.base_url)
    modes = ["classification", "reasoning"] if args.mode == "both" else [args.mode]

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "gpt-deployment-v1-llamacpp",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_arch": platform.machine(),
        "hardware_id": args.hardware_id,
        "preset": args.preset,
        "model_label": args.model_label or None,
        "engine": "llama.cpp",
        "base_url": args.base_url,
        "prompt": prompt,
        "image_source": args.image or args.image_list or "auto",
        "server_ready_s": round(ready_s, 3),
        "inference_settings": {
            "classification_max_new_tokens": CLASSIFICATION_MAX_TOKENS,
            "reasoning_max_new_tokens": REASONING_MAX_TOKENS,
            "temperature": args.temperature,
            "batch_size": 1,
        },
        "benchmark_procedure": {
            "warmup_per_restart": args.warmup,
            "measured_per_restart": args.runs,
            "restarts": 1,
            "concurrency": 1,
            "modes": modes,
            "note": "Server restarts are external; pass --restarts via wrapper if needed.",
        },
        "timing": {
            "boundary": TIMING_BOUNDARY,
            "definition": TIMING_BOUNDARY_DEF,
            "ttft": "t0 (HTTP POST) → first streamed text token",
            "e2e": "t0 (HTTP POST) → response complete",
            "excluded": ["model_load", "disk_io", "server_startup"],
            "includes_localhost_http": True,
        },
    }

    for mode in modes:
        report[mode] = _benchmark_mode(
            args.base_url,
            prompt,
            media,
            mode=mode,
            warmup=args.warmup,
            runs=args.runs,
            temperature=args.temperature,
            timeout_s=args.timeout,
        )

    out_path = Path(args.output) if args.output else (
        _DEPLOY_ROOT / "results" / f"benchmark_{args.preset}_{args.hardware_id}_llamacpp.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
