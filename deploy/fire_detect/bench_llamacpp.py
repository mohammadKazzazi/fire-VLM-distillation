#!/usr/bin/env python3
"""
H2/H3-compatible deployment benchmark for llama.cpp + GGUF (mtmd).

Same protocol as fire_detect/bench.py:
  warmup 20, runs 100, restarts 3, modes classification (8) + reasoning (64).

Uses llama-server (model stays loaded) so warm latency is comparable to H2/H3.
"""
from __future__ import annotations

import argparse
import atexit
import base64
import json
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib import error, request

_DEPLOY = Path(__file__).resolve().parents[1]

CLASSIFICATION_MAX_TOKENS = 8
REASONING_MAX_TOKENS = 64

# Track live llama-server children so SIGTERM/SIGHUP/atexit never leave orphans.
_ACTIVE_SERVERS: list["LlamaServer"] = []
_CLEANUP_DONE = False
_DEFAULT_STATE_DIR = Path(os.environ.get("BENCH_LLAMACPP_STATE_DIR", "/tmp/bench_llamacpp"))


def _state_paths(state_dir: Path) -> dict[str, Path]:
    state_dir.mkdir(parents=True, exist_ok=True)
    return {
        "client_pid": state_dir / "client.pid",
        "server_pid": state_dir / "server.pid",
        "heartbeat": state_dir / "heartbeat",
    }


def _touch_heartbeat(state_dir: Path) -> None:
    path = _state_paths(state_dir)["heartbeat"]
    path.write_text(f"{time.time():.3f} pid={os.getpid()}\n")


def _cleanup_active_servers() -> None:
    global _CLEANUP_DONE
    if _CLEANUP_DONE:
        return
    _CLEANUP_DONE = True
    for srv in list(_ACTIVE_SERVERS):
        try:
            srv.stop()
        except Exception as exc:
            print(f"[cleanup] failed to stop server: {exc}", flush=True)
    _ACTIVE_SERVERS.clear()


def _install_lifecycle_hooks(state_dir: Path) -> None:
    paths = _state_paths(state_dir)
    paths["client_pid"].write_text(f"{os.getpid()}\n")
    _touch_heartbeat(state_dir)

    def _on_signal(signum: int, _frame: Any) -> None:
        print(f"[bench] caught signal {signum}; stopping llama-server...", flush=True)
        _cleanup_active_servers()
        try:
            paths["client_pid"].unlink(missing_ok=True)
            paths["server_pid"].unlink(missing_ok=True)
        except Exception:
            pass
        # 128+signum is conventional for shell signal exits.
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _on_signal)
        except Exception:
            pass
    atexit.register(_cleanup_active_servers)
    atexit.register(lambda: paths["client_pid"].unlink(missing_ok=True))


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


def _ru_max_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _proc_rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _collect_images(image_list: str, count: int) -> list[str]:
    paths = [ln.strip() for ln in Path(image_list).read_text().splitlines() if ln.strip()]
    if not paths:
        raise ValueError(f"No images in {image_list}")
    return [paths[i % len(paths)] for i in range(count)]


def _image_data_url(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    suffix = Path(path).suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{b64}"


class LlamaServer:
    def __init__(
        self,
        bin_path: Path,
        model: Path,
        mmproj: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        threads: int = 8,
        ctx: int = 4096,
        chat_template: Optional[str] = None,
        chat_template_file: Optional[Path] = None,
        extra_args: Optional[list[str]] = None,
        state_dir: Optional[Path] = None,
    ) -> None:
        self.bin_path = bin_path
        self.model = model
        self.mmproj = mmproj
        self.host = host
        self.port = port
        self.threads = threads
        self.ctx = ctx
        self.chat_template = chat_template
        self.chat_template_file = chat_template_file
        self.extra_args = list(extra_args or [])
        self.state_dir = state_dir or _DEFAULT_STATE_DIR
        self.proc: Optional[subprocess.Popen] = None
        self.base = f"http://{host}:{port}"

    def start(self) -> float:
        env = os.environ.copy()
        libdir = str(self.bin_path.parent)
        env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"
        cmd = [
            str(self.bin_path),
            "-m",
            str(self.model),
            "--mmproj",
            str(self.mmproj),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-t",
            str(self.threads),
            "-c",
            str(self.ctx),
            "--no-mmproj-offload",
            "-np",
            "1",
            # Do not reuse prompt/KV across different images (deployment = one frame → one request).
            "--slot-prompt-similarity",
            "0",
        ]
        if self.chat_template_file is not None:
            cmd.extend(["--jinja", "--chat-template-file", str(self.chat_template_file)])
        elif self.chat_template:
            cmd.extend(["--chat-template", self.chat_template])
        cmd.extend(self.extra_args)
        t0 = time.perf_counter()
        # Own process group so we can SIGTERM the whole tree, but register for cleanup
        # so agent/shell teardown does not leave an orphan spinning forever.
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            preexec_fn=os.setsid,
        )
        _ACTIVE_SERVERS.append(self)
        paths = _state_paths(self.state_dir)
        paths["server_pid"].write_text(f"{self.proc.pid}\n")
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self._forget()
                raise RuntimeError(f"llama-server exited early with code {self.proc.returncode}")
            try:
                with request.urlopen(f"{self.base}/health", timeout=2) as resp:
                    if resp.status == 200:
                        _touch_heartbeat(self.state_dir)
                        return time.perf_counter() - t0
            except Exception:
                time.sleep(0.5)
        self.stop()
        raise TimeoutError("llama-server did not become healthy in time")

    def _forget(self) -> None:
        if self in _ACTIVE_SERVERS:
            _ACTIVE_SERVERS.remove(self)
        try:
            _state_paths(self.state_dir)["server_pid"].unlink(missing_ok=True)
        except Exception:
            pass

    def stop(self) -> None:
        if self.proc is None:
            self._forget()
            return
        proc = self.proc
        self.proc = None
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        self._forget()

    def erase_slot_cache(self) -> None:
        """Clear server prompt/KV cache so the next image is fully re-encoded."""
        try:
            req = request.Request(
                f"{self.base}/slots/0?action=erase",
                method="POST",
                data=b"",
            )
            with request.urlopen(req, timeout=10) as resp:
                resp.read()
        except Exception:
            # Older servers may not expose erase; cache_prompt=false still applies.
            pass

    def complete(
        self,
        image: str,
        prompt: str,
        *,
        max_tokens: int,
        measure_ttft: bool,
    ) -> tuple[float, Optional[float], str]:
        # Realistic edge path: each camera frame is an independent request.
        self.erase_slot_cache()
        # Match PyTorch <image>\n{question}
        text_prompt = prompt if prompt.startswith("\n") else f"\n{prompt}"
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": measure_ttft,
            # Disable cross-request prompt/KV reuse (was making warm latency unrealistically low).
            "cache_prompt": False,
        }
        data = json.dumps(body).encode("utf-8")
        req = request.Request(
            f"{self.base}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        ttft: Optional[float] = None
        text = ""
        with request.urlopen(req, timeout=600) as resp:
            if measure_ttft:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if delta:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000.0
                        text += delta
            else:
                payload = json.loads(resp.read().decode("utf-8"))
                text = payload["choices"][0]["message"]["content"]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        _touch_heartbeat(self.state_dir)
        return latency_ms, ttft, text


def _benchmark_mode(
    server_factory,
    prompt: str,
    images: list[str],
    *,
    mode: str,
    warmup: int,
    runs: int,
    restarts: int,
) -> dict[str, Any]:
    max_tokens = CLASSIFICATION_MAX_TOKENS if mode == "classification" else REASONING_MAX_TOKENS
    measure_ttft = mode == "reasoning"
    all_warm: list[float] = []
    all_ttft: list[float] = []
    load_times: list[float] = []
    cold: list[float] = []
    peak_rss: list[float] = []

    total_needed = warmup + runs
    assert len(images) >= total_needed

    for r in range(restarts):
        print(f"[{mode}] restart {r+1}/{restarts}: starting server...", flush=True)
        srv = server_factory()
        load_s = srv.start()
        load_times.append(load_s)
        print(f"[{mode}] restart {r+1}: loaded in {load_s:.2f}s", flush=True)

        try:
            lat, ttft, _ = srv.complete(
                images[0], prompt, max_tokens=max_tokens, measure_ttft=measure_ttft
            )
            cold.append(lat)
            print(f"[{mode}] cold={lat:.0f}ms", flush=True)

            for i, img in enumerate(images[:warmup]):
                srv.complete(img, prompt, max_tokens=max_tokens, measure_ttft=False)
                print(f"[{mode}] warmup {i+1}/{warmup}", flush=True)

            for i, img in enumerate(images[warmup : warmup + runs]):
                lat, ttft, _ = srv.complete(
                    img, prompt, max_tokens=max_tokens, measure_ttft=measure_ttft
                )
                all_warm.append(lat)
                if ttft is not None:
                    all_ttft.append(ttft)
                print(
                    f"[{mode}] measured {i+1}/{runs} last={lat:.0f}ms "
                    f"p50_so_far={_percentile(all_warm, 0.5):.0f}ms",
                    flush=True,
                )

            if srv.proc is not None:
                peak_rss.append(_proc_rss_mb(srv.proc.pid))
        finally:
            srv.stop()
            time.sleep(1)

    wall_s = sum(all_warm) / 1000.0
    return {
        "mode": mode,
        "max_new_tokens": max_tokens,
        "warmup_per_restart": warmup,
        "measured_per_restart": runs,
        "restarts": restarts,
        "concurrency": 1,
        "total_measured_requests": len(all_warm),
        "load_time_s": {
            "mean": round(statistics.mean(load_times), 3),
            "min": round(min(load_times), 3),
            "max": round(max(load_times), 3),
        },
        "cold_first_request_ms": _latency_stats(cold),
        "warm_latency_ms": _latency_stats(all_warm),
        "time_to_first_token_ms": _latency_stats(all_ttft) if all_ttft else None,
        "throughput_rps": round((len(all_warm) / wall_s) if wall_s > 0 else 0.0, 3),
        "peak_gpu_memory_mb": None,
        "peak_rss_mb": round(max(peak_rss), 2) if peak_rss else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-list", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--mode", choices=["classification", "reasoning", "both"], default="both")
    parser.add_argument("--hardware-id", default="H4")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=int(os.cpu_count() or 8))
    parser.add_argument(
        "--gguf-dir",
        default=str(_DEPLOY / "bundles" / "gguf"),
    )
    parser.add_argument("--model", default=None, help="Path to LLM GGUF (overrides --gguf-dir default)")
    parser.add_argument("--mmproj", default=None, help="Path to mmproj GGUF")
    parser.add_argument("--preset", default="qwen0.5b")
    parser.add_argument("--model-base", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--chat-template",
        default="",
        help="Named chat template (e.g. chatml). Empty + default jinja file = LLaVA conv_phi.",
    )
    parser.add_argument(
        "--chat-template-file",
        default=str(_DEPLOY / "bundles" / "gguf" / "llava_qwen_phi.jinja"),
        help="Jinja file matching LLaVA-MoD conv_mode=qwen (conv_phi). Empty to disable.",
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra llama-server arg (repeatable), e.g. --server-arg=--reasoning --server-arg=off",
    )
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument(
        "--llama-bin",
        default=str(Path("/home/ubuntu/llavamod/llama.cpp/build/bin/llama-server")),
    )
    parser.add_argument("--prompt-file", default=str(_DEPLOY / "config" / "fire_prompt.txt"))
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--state-dir",
        default=str(_DEFAULT_STATE_DIR),
        help="Directory for client/server pid + heartbeat files",
    )
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    _install_lifecycle_hooks(state_dir)

    gguf = Path(args.gguf_dir)
    model = Path(args.model) if args.model else gguf / "fire-qwen2.5-0.5b-Q8_0.gguf"
    mmproj = Path(args.mmproj) if args.mmproj else gguf / "mmproj-fire-clip-vit-l-f16.gguf"
    bin_path = Path(args.llama_bin)
    if not bin_path.is_file():
        alt = gguf / "bin-arm64" / "llama-server"
        if alt.is_file():
            bin_path = alt
        else:
            raise FileNotFoundError(f"llama-server not found: {args.llama_bin}")
    if not model.is_file():
        raise FileNotFoundError(f"model not found: {model}")
    if not mmproj.is_file():
        raise FileNotFoundError(f"mmproj not found: {mmproj}")

    prompt = Path(args.prompt_file).read_text().strip() + " Answer Yes or No only."
    # Images are reused across restarts the same way as bench.py: one contiguous list
    # of warmup+runs, cycled from the file; restarts reload the same sequence.
    images = _collect_images(args.image_list, args.warmup + args.runs)
    chat_template = None if args.chat_template in {"", "auto", "none"} else args.chat_template
    chat_template_file = (
        Path(args.chat_template_file)
        if args.chat_template_file and args.chat_template_file not in {"", "none"}
        else None
    )
    if chat_template_file is not None and not chat_template_file.is_file():
        raise FileNotFoundError(f"chat template file not found: {chat_template_file}")
    # Named template wins only when jinja file is disabled.
    if chat_template_file is not None:
        chat_template = None
    extra_args = list(args.server_arg)

    def factory() -> LlamaServer:
        return LlamaServer(
            bin_path,
            model,
            mmproj,
            port=args.port,
            threads=args.threads,
            ctx=args.ctx,
            chat_template=chat_template,
            chat_template_file=chat_template_file,
            extra_args=extra_args,
            state_dir=state_dir,
        )

    modes = ["classification", "reasoning"] if args.mode == "both" else [args.mode]
    instance = "unknown"
    try:
        instance = (
            request.urlopen("http://169.254.169.254/latest/meta-data/instance-type", timeout=2)
            .read()
            .decode()
        )
    except Exception:
        try:
            token = (
                request.Request(
                    "http://169.254.169.254/latest/api/token",
                    method="PUT",
                    headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                )
            )
            # IMDSv2 optional; ignore failures
        except Exception:
            pass

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "gpt-deployment-v1",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_arch": platform.machine(),
        "cpu_model": platform.processor() or platform.machine(),
        "hardware_id": args.hardware_id,
        "preset": args.preset,
        "model_path": str(model),
        "mmproj_path": str(mmproj),
        "model_base": args.model_base,
        "conv_mode": (
            f"jinja:{chat_template_file.name}"
            if chat_template_file is not None
            else (chat_template or "model-default")
        ),
        "device": "cpu",
        "runtime": "llama.cpp+gguf",
        "prompt": prompt,
        "image_source": args.image_list,
        "gpu": {"available": False},
        "bundle_size_gb": round(
            (model.stat().st_size + mmproj.stat().st_size) / (1024**3), 4
        ),
        "inference_settings": {
            "precision": "gguf-q4_k_m+mmproj-f16",
            "quantization": model.name,
            "batch_size": 1,
            "temperature": 0.0,
            "classification_max_new_tokens": CLASSIFICATION_MAX_TOKENS,
            "reasoning_max_new_tokens": REASONING_MAX_TOKENS,
            "gpu_memory_fraction": None,
            "threads": args.threads,
            "ctx": args.ctx,
            "server": "llama-server",
            "server_extra_args": extra_args,
            "cache_prompt": False,
            "fresh_request_per_image": True,
            "notes": "Each request clears slot cache; matches one-frame-one-request deployment.",
        },
        "benchmark_procedure": {
            "warmup_per_restart": args.warmup,
            "measured_per_restart": args.runs,
            "restarts": args.restarts,
            "concurrency": 1,
            "modes": modes,
        },
    }

    for mode in modes:
        report[mode] = _benchmark_mode(
            factory,
            prompt,
            images,
            mode=mode,
            warmup=args.warmup,
            runs=args.runs,
            restarts=args.restarts,
        )

    report["hardware_profile"] = {
        "id": args.hardware_id,
        "expected_instance_type": "c7g.2xlarge",
        "actual_instance_type": instance,
        "maps_to": "ARM64 CPU-only + llama.cpp GGUF (edge/Raspberry-Pi-oriented runtime)",
        "virtualized": True,
    }
    report["software"] = {
        "python": platform.python_version(),
        "pytorch": None,
        "transformers": None,
        "llama_cpp_bin": str(bin_path),
    }

    out = Path(args.output) if args.output else (
        _DEPLOY / "results" / f"benchmark_{args.preset}_{args.hardware_id}_llamacpp.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
