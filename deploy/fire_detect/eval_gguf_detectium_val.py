#!/usr/bin/env python3
"""
Detectium held-out val accuracy for llama.cpp GGUF (fire student).

- Persistent llama-server (model stays loaded)
- Fresh encode per image (cache_prompt=false + slot erase)
- Resume-safe: skips question_ids already in answers JSONL
- Signal/atexit cleanup so server is not left orphaned
"""
from __future__ import annotations

import argparse
import atexit
import base64
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib import request

_DEPLOY = Path(__file__).resolve().parents[1]
_ACTIVE: list["LlamaServer"] = []
_CLEANUP_DONE = False


def _image_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{b64}"


def _load_done(answers_path: Path) -> set[str]:
    done: set[str] = set()
    if not answers_path.is_file():
        return done
    with answers_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get("question_id")
            if qid is not None:
                done.add(str(qid))
    return done


def _cleanup() -> None:
    global _CLEANUP_DONE
    if _CLEANUP_DONE:
        return
    _CLEANUP_DONE = True
    for srv in list(_ACTIVE):
        try:
            srv.stop()
        except Exception as exc:
            print(f"[cleanup] {exc}", flush=True)
    _ACTIVE.clear()


def _install_signals(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "client.pid").write_text(f"{os.getpid()}\n")

    def _on_signal(signum: int, _frame: Any) -> None:
        print(f"[eval] signal {signum}; stopping server...", flush=True)
        _cleanup()
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _on_signal)
        except Exception:
            pass
    atexit.register(_cleanup)


class LlamaServer:
    def __init__(
        self,
        bin_path: Path,
        model: Path,
        mmproj: Path,
        *,
        host: str,
        port: int,
        threads: int,
        ctx: int,
        state_dir: Path,
        chat_template_file: Optional[Path] = None,
    ) -> None:
        self.bin_path = bin_path
        self.model = model
        self.mmproj = mmproj
        self.host = host
        self.port = port
        self.threads = threads
        self.ctx = ctx
        self.state_dir = state_dir
        self.chat_template_file = chat_template_file
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
            "--slot-prompt-similarity",
            "0",
        ]
        # Match LLaVA-MoD train/val: conv_mode=qwen → conv_phi (USER:/ASSISTANT:), not ChatML.
        tmpl = self.chat_template_file
        if tmpl is None:
            tmpl = _DEPLOY / "bundles" / "gguf" / "llava_qwen_phi.jinja"
        if tmpl.is_file():
            cmd.extend(["--jinja", "--chat-template-file", str(tmpl)])
        else:
            cmd.extend(["--chat-template", "chatml"])
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            preexec_fn=os.setsid,
        )
        _ACTIVE.append(self)
        (self.state_dir / "server.pid").write_text(f"{self.proc.pid}\n")
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early: {self.proc.returncode}")
            try:
                with request.urlopen(f"{self.base}/health", timeout=2) as resp:
                    if resp.status == 200:
                        return time.perf_counter() - t0
            except Exception:
                time.sleep(0.5)
        self.stop()
        raise TimeoutError("llama-server did not become healthy")

    def stop(self) -> None:
        if self.proc is None:
            if self in _ACTIVE:
                _ACTIVE.remove(self)
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
        if self in _ACTIVE:
            _ACTIVE.remove(self)
        try:
            (self.state_dir / "server.pid").unlink(missing_ok=True)
        except Exception:
            pass

    def erase_slot_cache(self) -> None:
        try:
            req = request.Request(f"{self.base}/slots/0?action=erase", method="POST", data=b"")
            with request.urlopen(req, timeout=10) as resp:
                resp.read()
        except Exception:
            pass

    def complete(self, image: Path, prompt: str, max_tokens: int) -> str:
        self.erase_slot_cache()
        # PyTorch conv_phi places a newline after the image token: <image>\n{question}
        text = prompt if prompt.startswith("\n") else f"\n{prompt}"
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                        {"type": "text", "text": text},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
            "cache_prompt": False,
        }
        data = json.dumps(body).encode("utf-8")
        req = request.Request(
            f"{self.base}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return (payload["choices"][0]["message"].get("content") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mmproj", required=True)
    parser.add_argument(
        "--llama-bin",
        default="/home/ubuntu/llavamod/llama.cpp/build/bin/llama-server",
    )
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--threads", type=int, default=int(os.cpu_count() or 8))
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--append-yes-no",
        action="store_true",
        help="Append ' Answer Yes or No only.' to each question (deploy-style).",
    )
    parser.add_argument("--state-dir", default="/tmp/bench_llamacpp/gguf_val")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for smoke tests")
    parser.add_argument(
        "--chat-template-file",
        default=str(_DEPLOY / "bundles" / "gguf" / "llava_qwen_phi.jinja"),
        help="Jinja chat template matching LLaVA-MoD conv_mode=qwen (conv_phi).",
    )
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    _install_signals(state_dir)

    questions_path = Path(args.question_file)
    image_folder = Path(args.image_folder)
    answers_path = Path(args.answers_file)
    answers_path.parent.mkdir(parents=True, exist_ok=True)

    questions = []
    with questions_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if args.limit > 0:
        questions = questions[: args.limit]

    done = _load_done(answers_path)
    todo = [q for q in questions if str(q["question_id"]) not in done]
    print(
        f"[eval] total={len(questions)} done={len(done)} remaining={len(todo)} "
        f"answers={answers_path}",
        flush=True,
    )
    if not todo:
        print("[eval] nothing to do", flush=True)
        return

    model = Path(args.model)
    mmproj = Path(args.mmproj)
    bin_path = Path(args.llama_bin)
    if not bin_path.is_file():
        alt = _DEPLOY / "bundles" / "gguf" / "bin-arm64" / "llama-server"
        if alt.is_file():
            bin_path = alt
        else:
            raise FileNotFoundError(bin_path)

    tmpl = Path(args.chat_template_file) if args.chat_template_file else None
    srv = LlamaServer(
        bin_path,
        model,
        mmproj,
        host="127.0.0.1",
        port=args.port,
        threads=args.threads,
        ctx=args.ctx,
        state_dir=state_dir,
        chat_template_file=tmpl,
    )
    load_s = srv.start()
    print(f"[eval] server ready in {load_s:.2f}s on :{args.port}", flush=True)

    t_start = time.time()
    try:
        with answers_path.open("a") as out:
            for i, q in enumerate(todo, 1):
                qid = str(q["question_id"])
                img_rel = q["image"]
                img_path = image_folder / img_rel
                if not img_path.is_file():
                    print(f"[eval] MISSING image {img_path}", flush=True)
                    row = {"question_id": qid, "prompt": q.get("text", ""), "text": "", "error": "missing_image"}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    continue

                prompt = (q.get("text") or "").strip()
                if args.append_yes_no:
                    prompt = f"{prompt} Answer Yes or No only."

                t0 = time.perf_counter()
                try:
                    text = srv.complete(img_path, prompt, args.max_new_tokens)
                    err = None
                except Exception as exc:
                    text = ""
                    err = str(exc)
                dt_ms = (time.perf_counter() - t0) * 1000.0

                row: dict[str, Any] = {
                    "question_id": qid,
                    "prompt": prompt,
                    "text": text,
                    "latency_ms": round(dt_ms, 1),
                }
                if err:
                    row["error"] = err
                out.write(json.dumps(row) + "\n")
                out.flush()
                os.fsync(out.fileno())

                (state_dir / "heartbeat").write_text(
                    f"{time.time():.3f} done={len(done)+i}/{len(questions)} last={qid}\n"
                )
                preview = (text or err or "")[:80].replace("\n", " ")
                elapsed = time.time() - t_start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(todo) - i) / rate if rate > 0 else 0
                print(
                    f"[eval] {i}/{len(todo)} qid={qid} {dt_ms:.0f}ms "
                    f"eta={eta/3600:.1f}h :: {preview}",
                    flush=True,
                )
    finally:
        srv.stop()
        print("[eval] server stopped", flush=True)


if __name__ == "__main__":
    main()
