#!/usr/bin/env python3
"""
Zero-shot Detectium val inference with Qwen2.5-VL (native vision-language model).

Reads val_questions.jsonl (same format as LLaVA-MoD eval) and writes JSONL answers
compatible with score_detectium_val.py and score_fire_reasoning_table4.py.

Example:
  python scripts/eval_qwen25_vl_zeroshot.py \\
    --question-file ../LLaVA_MoD/MoD/detectioumfire/val_questions.jsonl \\
    --image-folder ../LLaVA-Instruct-150K \\
    --answers-file models/qwen25_vl_3b/zeroshot/val_answers.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen2.5-VL zero-shot Detectium val VQA")
    p.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    p.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional PEFT LoRA adapter (from finetune_qwen25_vl_detectium.py).",
    )
    p.add_argument(
        "--question-file",
        type=Path,
        required=True,
        help="JSONL with question_id, image, text (human prompt).",
    )
    p.add_argument(
        "--image-folder",
        type=Path,
        required=True,
        help="Root for relative image paths (e.g. LLaVA-Instruct-150K).",
    )
    p.add_argument("--answers-file", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=0, help="Debug: cap number of rows (0 = all).")
    p.add_argument("--resume", action="store_true", help="Skip question_ids already in answers-file.")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def load_questions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("question_id")
            if qid is not None:
                done.add(str(qid))
    return done


def generate_one(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
    }
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return out.strip()


def main() -> None:
    args = parse_args()
    if not args.question_file.is_file():
        print(f"question file not found: {args.question_file}", file=sys.stderr)
        sys.exit(1)
    if not args.image_folder.is_dir():
        print(f"image folder not found: {args.image_folder}", file=sys.stderr)
        sys.exit(1)

    questions = load_questions(args.question_file)
    if args.limit > 0:
        questions = questions[: args.limit]

    args.answers_file.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.answers_file) if args.resume else set()
    todo = [q for q in questions if str(q["question_id"]) not in done]
    print(f"model:      {args.model_id}")
    print(f"questions:  {len(questions)} total, {len(todo)} to run ({len(done)} skipped)")
    print(f"image_root: {args.image_folder}")
    print(f"output:     {args.answers_file}")
    if not todo:
        print("Nothing to do.")
        return

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"loading model ({dtype})...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    if args.adapter_path is not None:
        if not args.adapter_path.is_dir():
            print(f"adapter not found: {args.adapter_path}", file=sys.stderr)
            sys.exit(1)
        print(f"loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, str(args.adapter_path))
    model.eval()
    processor = AutoProcessor.from_pretrained(
        str(args.adapter_path) if args.adapter_path else args.model_id
    )

    mode = "a" if args.resume and args.answers_file.exists() else "w"
    t0 = time.time()
    errors = 0

    with args.answers_file.open(mode, encoding="utf-8") as out_f:
        for i, row in enumerate(todo, start=1):
            qid = str(row["question_id"])
            prompt = row["text"]
            image_rel = row["image"]
            image_path = args.image_folder / image_rel

            if not image_path.is_file():
                print(f"[{i}/{len(todo)}] MISSING image {image_path}", file=sys.stderr)
                errors += 1
                continue

            try:
                text = generate_one(
                    model,
                    processor,
                    image_path,
                    prompt,
                    args.max_new_tokens,
                    args.temperature,
                )
            except Exception as exc:
                print(f"[{i}/{len(todo)}] ERROR {qid}: {exc}", file=sys.stderr)
                errors += 1
                continue

            rec = {
                "question_id": qid,
                "prompt": prompt,
                "text": text,
                "answer_id": uuid.uuid4().hex[:22],
                "model_id": args.model_id,
                "metadata": {"zeroshot": True},
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

            if i % args.log_every == 0 or i == len(todo):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                print(f"[{i}/{len(todo)}] {rate:.2f} img/s  last={qid}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. errors={errors}. wrote {args.answers_file}")


if __name__ == "__main__":
    main()
