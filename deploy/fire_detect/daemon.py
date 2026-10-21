#!/usr/bin/env python3
"""
CLI daemon for offline fire detection.

Examples:
  python3 fire_detect/daemon.py --image frame.jpg
  python3 fire_detect/daemon.py --image frame.jpg --watch-dir /data/camera --interval 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(_DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_ROOT))

from fire_detect.inference import FireDetector  # noqa: E402


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if "${" in val:
            continue  # bash-only expansion; resolved by shell scripts
        os.environ.setdefault(key, val)


def _resolve_preset(env: dict) -> dict:
    preset = env.get("STUDENT_PRESET", "qwen0.5b")
    mapping = {
        "qwen0.5b": ("QWEN0_5B_BUNDLE", "QWEN0_5B_BASE", "QWEN0_5B_CONV", "QWEN0_5B_LORA"),
        "llama3-1b": ("LLAMA3_1B_BUNDLE", "LLAMA3_1B_BASE", "LLAMA3_1B_CONV", "LLAMA3_1B_LORA"),
        "qwen3-0.6b": ("QWEN3_0_6B_BUNDLE", "QWEN3_0_6B_BASE", "QWEN3_0_6B_CONV", "QWEN3_0_6B_LORA"),
    }
    if preset not in mapping:
        raise ValueError(f"Unknown STUDENT_PRESET={preset}")
    bundle_k, base_k, conv_k, lora_k = mapping[preset]
    bundle = env.get("BUNDLE_PATH") or env.get(bundle_k, "")
    base = env.get("MODEL_BASE") or env.get(base_k, "")
    conv = env.get("CONV_MODE") or env.get(conv_k, "qwen")
    lora = env.get("LORA_PATH") or env.get(lora_k, "")

    use_merged = env.get("USE_MERGED_BUNDLE", "0") == "1"
    if use_merged and bundle and Path(bundle).is_dir():
        model_path = bundle
        model_base = None
    elif lora and Path(lora).is_dir():
        model_path = lora
        model_base = base or None
    else:
        model_path = bundle
        model_base = None

    return {
        "preset": preset,
        "model_path": model_path,
        "model_base": model_base,
        "conv_mode": conv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline fire-detection daemon")
    parser.add_argument("--config", type=str, default=str(_DEPLOY_ROOT / "config" / "students.env"))
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--prompt-file", type=str, default=str(_DEPLOY_ROOT / "config" / "fire_prompt.txt"))
    parser.add_argument("--bundle", type=str, default=None, help="Override merged bundle directory name under bundles/")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--watch-dir", type=str, default=None, help="Poll newest image in directory")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--output", type=str, default=None, help="Write JSON result to file")
    args = parser.parse_args()

    _load_env_file(Path(args.config))
    env = dict(os.environ)
    if env.get("HF_HUB_OFFLINE") == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"
    if env.get("TRANSFORMERS_OFFLINE") == "1":
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    cfg = _resolve_preset(env)
    if args.bundle:
        cfg["model_path"] = str(_DEPLOY_ROOT / "bundles" / args.bundle)
        cfg["model_base"] = None

    prompt = Path(args.prompt_file).read_text().strip()
    llava_root = env.get("LLAVA_MOD_ROOT")

    detector = FireDetector(
        model_path=cfg["model_path"],
        model_base=cfg["model_base"],
        conv_mode=cfg["conv_mode"],
        device=args.device or env.get("DEVICE", "cuda"),
        load_4bit=args.load_4bit or env.get("LOAD_4BIT") == "1",
        load_8bit=args.load_8bit or env.get("LOAD_8BIT") == "1",
        max_new_tokens=args.max_new_tokens or int(env.get("MAX_NEW_TOKENS", "64")),
        temperature=float(env.get("TEMPERATURE", "0.0")),
        llava_mod_root=llava_root,
    )

    def run_once(image_path: str) -> dict:
        result = detector.detect(image_path, prompt)
        out = result.to_dict()
        out["image"] = image_path
        out["preset"] = cfg["preset"]
        return out

    if args.watch_dir:
        watch = Path(args.watch_dir)
        seen: set[str] = set()
        print(json.dumps({"status": "watching", "dir": str(watch), "interval_s": args.interval}))
        while True:
            images = sorted(
                [p for p in watch.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
                key=lambda p: p.stat().st_mtime,
            )
            if images:
                latest = str(images[-1])
                if latest not in seen:
                    seen.add(latest)
                    payload = run_once(latest)
                    line = json.dumps(payload)
                    print(line, flush=True)
                    if args.output:
                        Path(args.output).write_text(line + "\n")
            time.sleep(args.interval)
        return

    if not args.image:
        parser.error("Provide --image or --watch-dir")

    payload = run_once(args.image)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
