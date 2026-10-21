#!/usr/bin/env python3
"""
Build a JSONL file for llava.eval.model_vqa from LLaVA-format instruction data.

Each output line: {"question_id": ..., "image": ..., "text": ...}
`text` is the human question only (no <image> prefix); model_vqa adds the image token.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def human_text(row: dict) -> str:
    for turn in row.get("conversations", []):
        if turn.get("from") == "human":
            return (turn.get("value") or "").strip()
    return ""


def strip_image_tokens(s: str) -> str:
    """Remove all <image> placeholders; model_vqa prepends one image token."""
    out = s
    for _ in range(8):
        prev = out
        for token in ("<image>\n", "\n<image>", "<image>"):
            out = out.replace(token, "")
        if out == prev:
            break
    return out.strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "detectium_llavamod_real_full_val.json",
        help="LLaVA-format JSON list (e.g. val split).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "val_questions.jsonl",
    )
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with args.output.open("w") as f:
        for row in data:
            qid = row.get("id")
            image = row.get("image")
            if qid is None or not image:
                continue
            text = strip_image_tokens(human_text(row))
            if not text:
                continue
            rec = {"question_id": qid, "image": image, "text": text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"wrote {n} lines to {args.output}")


if __name__ == "__main__":
    main()
