#!/usr/bin/env python3
"""Smoke-test student projector pretrain with a simple image caption."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2] / "LLava_MoD"
sys.path.insert(0, str(REPO_ROOT))

from llavamod.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llavamod.conversation import SeparatorStyle, conv_templates
from llavamod.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llavamod.model.builder import load_pretrained_model
from llavamod.utils import disable_torch_init

DEFAULT_PROMPT = "Describe this image in detail."
DEFAULT_IMAGE_REL = "coco/train2017/000000000009.jpg"


def run_caption(
    model,
    tokenizer,
    image_processor,
    image_path: Path,
    conv_mode: str,
    max_new_tokens: int,
) -> str:
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
    image_tensor = image_tensor.to(device="cuda", dtype=torch.bfloat16)

    qs = DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_PROMPT
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = [KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=False,
            repetition_penalty=1.0,
            stopping_criteria=stopping_criteria,
        )

    text = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=False).strip()
    for suffix in ("<|endoftext|>", stop_str):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    detectium_dir = script_dir.parent
    workspace = detectium_dir.parents[2]

    parser = argparse.ArgumentParser(description="Caption test for student projector pretrain.")
    parser.add_argument(
        "--projector-path",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "llavaqwen2.5-3b-student-pretrain",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=workspace / "LLaVA-Instruct-150K" / DEFAULT_IMAGE_REL,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "caption_student_pretrain",
    )
    parser.add_argument("--conv-mode", type=str, default="qwen")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if not args.projector_path.is_dir():
        parser.error(f"Projector checkpoint not found: {args.projector_path}")
    if not (args.projector_path / "mm_projector.bin").is_file():
        parser.error(f"Missing mm_projector.bin in {args.projector_path}")
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_image = args.output_dir / args.image.name
    out_image.write_bytes(args.image.read_bytes())

    disable_torch_init()
    model_name = get_model_name_from_path(str(args.projector_path))
    tokenizer, model, processor, _ = load_pretrained_model(
        str(args.projector_path),
        args.model_base,
        model_name,
    )
    model.generation_config.repetition_penalty = 1.0

    caption = run_caption(
        model,
        tokenizer,
        processor["image"],
        args.image,
        args.conv_mode,
        args.max_new_tokens,
    )

    result = {
        "stage": "student_projector_pretrain",
        "projector_path": str(args.projector_path),
        "model_base": args.model_base,
        "image": str(args.image),
        "prompt": DEFAULT_PROMPT,
        "caption": caption,
        "note": (
            "Projector-only stage: LLM is frozen base weights + aligned mm_projector. "
            "No fire SFT or distillation yet."
        ),
    }
    result_path = args.output_dir / "caption_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))
    print(f"\nWrote {result_path}")
    print(f"Image copy: {out_image}")


if __name__ == "__main__":
    main()
