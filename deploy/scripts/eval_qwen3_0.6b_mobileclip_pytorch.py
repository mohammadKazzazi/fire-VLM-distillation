#!/usr/bin/env python3
"""PyTorch CPU Detectium Yes/No + Table4 for Qwen3-0.6B + MobileCLIP-S2 student.

llama.cpp cannot run this student multimodal (no MobileCLIP mmproj). This is the
deployable accuracy path on H4 until FastViT lands in mtmd.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY))

from fire_detect.inference import FireDetector  # noqa: E402


def _patch_dynamic_cache_compat() -> None:
    """LLaVA vendored Qwen2 modeling still calls Cache.get_usable_length (removed in TF≥4.56)."""
    try:
        from transformers import DynamicCache
    except Exception:
        return
    if hasattr(DynamicCache, "get_usable_length"):
        return

    def get_usable_length(self, new_seq_length: int, layer_idx: int | None = 0) -> int:
        return int(self.get_seq_length(layer_idx or 0))

    DynamicCache.get_usable_length = get_usable_length  # type: ignore[method-assign]


_patch_dynamic_cache_compat()

VAL_ROOT = Path("/home/ubuntu/llavamod/detectium-val-only")
DEFAULT_LORA = Path(
    "/home/ubuntu/llavamod/qwen3-0.6b-mobileclip-students/checkpoints/"
    "plan4-d2d-multiquery-v4-kd-lm-qwen3-0.6b-mobileclip-s2-seed42"
)
LLAVA_MOD = Path(
    "/home/ubuntu/llavamod/llava-deploy-h1-h4-20260716/workspace/LLaVA_MoD/MoD/LLava_MoD"
)
OUT_DIR = DEPLOY / "results" / "gguf_val"


def lead_yes_no(text: str) -> str | None:
    for tok in re.split(r"[^a-zA-Z]+", (text or "").strip().lower()):
        if tok in ("yes", "no"):
            return tok
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--skip-table4", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--lora",
        type=Path,
        default=DEFAULT_LORA,
        help="Student LoRA checkpoint directory (MobileCLIP KD_LM).",
    )
    ap.add_argument(
        "--model-base",
        default="Qwen/Qwen3-0.6B",
        help="Base LLM id for FireDetector.",
    )
    ap.add_argument(
        "--conv-mode",
        default="qwen3",
        help="LLaVA conv template (qwen3 for Qwen3; qwen for Qwen2.5).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Match GPU Detectium val (default 64). Do not crop for Yes/No.",
    )
    ap.add_argument(
        "--answers-name",
        default="val_answers_fire_qwen3_0.6b_mobileclip_pytorch.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for answers JSONL (default: Deploy/results/gguf_val).",
    )
    ap.add_argument(
        "--append-yes-no-suffix",
        action="store_true",
        help="Append 'Answer Yes or No only.' (NOT used by GPU Detectium val; "
        "inflates FPs on this student).",
    )
    ap.add_argument("--num-chunks", type=int, default=1, help="Shard count for parallel workers.")
    ap.add_argument("--chunk-idx", type=int, default=0, help="0-based shard index.")
    ap.add_argument(
        "--score",
        action="store_true",
        help="Score Yes/No (+ Table4 unless --skip-table4). Off by default for shards.",
    )
    args = ap.parse_args()

    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / args.answers_name
    stem = answers_path.stem.replace("val_answers_", "score_", 1)
    if stem == answers_path.stem:
        stem = f"score_{answers_path.stem}"
    score_path = out_dir / f"{stem}.json"
    table4_path = out_dir / answers_path.name.replace("val_answers_", "table4_", 1).replace(
        ".jsonl", ".json"
    )

    t0 = time.time()
    lora = args.lora.resolve()
    print(f"Loading FireDetector (MobileCLIP) lora={lora} ...", flush=True)
    det = FireDetector(
        model_path=str(lora),
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        llava_mod_root=str(LLAVA_MOD),
    )
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    # Smoke one image
    questions = [
        json.loads(ln)
        for ln in (VAL_ROOT / "json" / "val_questions.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    if args.limit > 0:
        questions = questions[: args.limit]
    if args.num_chunks < 1:
        raise SystemExit("--num-chunks must be >= 1")
    if not (0 <= args.chunk_idx < args.num_chunks):
        raise SystemExit(f"--chunk-idx must be in [0, {args.num_chunks})")
    if args.num_chunks > 1:
        questions = [
            q for i, q in enumerate(questions) if i % args.num_chunks == args.chunk_idx
        ]

    done_ids: set[str] = set()
    if answers_path.is_file():
        for ln in answers_path.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                done_ids.add(str(json.loads(ln)["question_id"]))
            except Exception:
                pass
    mode = "a" if done_ids else "w"
    print(
        f"chunk {args.chunk_idx}/{args.num_chunks}: {len(questions)} qs, "
        f"resume {len(done_ids)} done → {answers_path.name}",
        flush=True,
    )

    with answers_path.open(mode) as fout:
        for i, q in enumerate(questions):
            qid = str(q["question_id"])
            if qid in done_ids:
                continue
            img = VAL_ROOT / "images" / q["image"]
            # Match GPU model_vqa / Detectium worker: use the human question as-is.
            # Appending "Answer Yes or No only." inflates FPs (~31→~2 on error recheck)
            # and is not used in GPU val.
            prompt = (q.get("text") or q.get("prompt") or "").rstrip()
            if args.append_yes_no_suffix and "Answer Yes or No" not in prompt:
                prompt = prompt + " Answer Yes or No only."
            t1 = time.time()
            r = det.detect(str(img), prompt=prompt)
            text = r.raw_text
            row = {
                "question_id": qid,
                "image": q["image"],
                "prompt": prompt,
                "text": text,
                "pred": lead_yes_no(text),
                "latency_ms": round((time.time() - t1) * 1000, 1),
                "backend": "pytorch_mobileclip",
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0 or i == 0:
                print(
                    f"[{i+1}/{len(questions)}] {qid} pred={row['pred']} "
                    f"{row['latency_ms']:.0f}ms :: {text[:80]!r}",
                    flush=True,
                )

    if args.score:
        import subprocess

        print("Scoring Yes/No ...", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                str(VAL_ROOT / "eval" / "score_detectium_val.py"),
                "--val-json",
                str(VAL_ROOT / "json" / "detectium_llavamod_real_full_val.json"),
                "--answers",
                str(answers_path),
                "--json-out",
                str(score_path),
            ]
        )
        print(score_path.read_text()[:800], flush=True)

        if not args.skip_table4:
            print("Table 4 scoring from Yes/No answers ...", flush=True)
            subprocess.check_call(
                [
                    sys.executable,
                    str(VAL_ROOT / "eval" / "score_fire_reasoning_table4.py"),
                    "--val-json",
                    str(VAL_ROOT / "json" / "detectium_llavamod_real_full_val.json"),
                    "--model",
                    args.answers_name.replace("val_answers_", "").replace(".jsonl", ""),
                    str(answers_path),
                    "--json-out",
                    str(table4_path),
                ]
            )
            print(f"wrote {table4_path}", flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
