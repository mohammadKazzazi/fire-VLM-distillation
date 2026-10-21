#!/usr/bin/env python3
"""
Ask the finetuned LLaVA-MoD Qwen model for fire bounding boxes on a few images,
parse normalized [x1, y1, x2, y2] boxes (0–1), and save visualizations.

Usage:
  cd detectioumfire/eval/launchers/qwen2.5
  ./run_bbox_test.sh

  # custom images (paths relative to IMAGE_FOLDER or absolute):
  python test_fire_bbox.py --images stable_diff_v15/train/images/foo.jpg
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2] / "LLava_MoD"
sys.path.insert(0, str(REPO_ROOT))

from llavamod.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llavamod.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llavamod.mm_utils import (  # noqa: E402
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llavamod.model.builder import load_pretrained_model  # noqa: E402
from llavamod.utils import disable_torch_init  # noqa: E402

BBOX_PROMPT = (
    "Please provide all bounding box coordinates for regions with visible fire or flame."
)
BBOX_RE = re.compile(
    r"\[\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\]"
)
BOX_COLORS = ["#FF3333", "#3399FF", "#FFCC00", "#FF33FF", "#00FFFF"]
GT_COLOR = "#00FF00"
PRED_COLOR = "#FF3333"


def gpt_answer(row: dict) -> str:
    for turn in row.get("conversations", []):
        if turn.get("from") == "gpt":
            return (turn.get("value") or "").strip()
    return ""


def parse_bboxes(text: str) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for m in BBOX_RE.finditer(text or ""):
        x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        if x2 - x1 > 1e-4 and y2 - y1 > 1e-4:
            boxes.append((x1, y1, x2, y2))
    return boxes


def default_bbox_samples(
    bbox_json: Path,
    image_folder: Path,
    n: int = 5,
) -> list[dict]:
    """Pick n synthetic fire images with GT bboxes from detectium_bbox JSON."""
    rows = json.loads(bbox_json.read_text())
    with_boxes = [r for r in rows if parse_bboxes(gpt_answer(r))]
    if not with_boxes:
        raise RuntimeError(f"No bbox rows in {bbox_json}")

    # First n rows with bbox labels in the JSON file.
    pick_indices = list(range(min(n, len(with_boxes))))

    samples: list[dict] = []
    for i in pick_indices:
        row = with_boxes[i]
        rel = row["image"]
        path = image_folder / rel
        if not path.is_file():
            continue
        gt_text = gpt_answer(row)
        samples.append(
            {
                "id": row.get("id"),
                "image_path": path,
                "image_rel": rel,
                "gt_text": gt_text,
                "gt_boxes": parse_bboxes(gt_text),
            }
        )
        if len(samples) >= n:
            break

    if len(samples) < n:
        for row in with_boxes:
            rel = row["image"]
            path = image_folder / rel
            if not path.is_file():
                continue
            if any(s["image_path"] == path for s in samples):
                continue
            gt_text = gpt_answer(row)
            samples.append(
                {
                    "id": row.get("id"),
                    "image_path": path,
                    "image_rel": rel,
                    "gt_text": gt_text,
                    "gt_boxes": parse_bboxes(gt_text),
                }
            )
            if len(samples) >= n:
                break

    if len(samples) < n:
        raise RuntimeError(
            f"Only found {len(samples)} bbox images on disk under {image_folder}"
        )
    return samples[:n]


def default_image_paths(detectium_dir: Path, image_folder: Path, n: int = 5) -> list[Path]:
    bbox_json = detectium_dir / "detectium_bbox_synthetic2k_nonfire500.json"
    return [s["image_path"] for s in default_bbox_samples(bbox_json, image_folder, n)]


def draw_boxes_on_image(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    color: str,
    prefix: str,
) -> None:
    draw = ImageDraw.Draw(image)
    w, h = image.size
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        px1, py1, px2, py2 = x1 * w, y1 * h, x2 * w, y2 * h
        draw.rectangle([px1, py1, px2, py2], outline=color, width=max(2, w // 200))
        draw.text((px1 + 2, max(0, py1 - 14)), f"{prefix}{i + 1}", fill=color, font=font)


def draw_bboxes(
    image: Image.Image,
    pred_boxes: list[tuple[float, float, float, float]],
    gt_boxes: list[tuple[float, float, float, float]] | None = None,
    caption: str = "",
) -> Image.Image:
    out = image.copy().convert("RGB")
    if gt_boxes:
        draw_boxes_on_image(out, gt_boxes, GT_COLOR, "G")
    draw_boxes_on_image(out, pred_boxes, PRED_COLOR, "P")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    label = caption or f"GT:{len(gt_boxes or [])} pred:{len(pred_boxes)}"
    draw.text((8, 8), label[:120], fill="white", font=font)
    return out


def run_inference(
    model,
    tokenizer,
    image_processor,
    image_path: Path,
    conv_mode: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
    image_tensor = image_tensor.to(device="cuda", dtype=torch.bfloat16)

    qs = DEFAULT_IMAGE_TOKEN + "\n" + BBOX_PROMPT
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
    stopping_criteria = [
        KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    ]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=max_new_tokens,
            use_cache=False,
            repetition_penalty=1.0,
            stopping_criteria=stopping_criteria,
        )

    new_tokens = output_ids[0, input_ids.shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    for suffix in ("<|endoftext|>", stop_str):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text


def resolve_samples(
    args: argparse.Namespace, detectium_dir: Path
) -> list[dict]:
    image_folder = Path(args.image_folder)
    bbox_json = Path(args.bbox_json)

    if args.images:
        samples = []
        for p in args.images:
            path = Path(p)
            if not path.is_absolute():
                path = image_folder / path
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
            samples.append(
                {
                    "id": None,
                    "image_path": path,
                    "image_rel": str(path.relative_to(image_folder))
                    if path.is_relative_to(image_folder)
                    else str(path),
                    "gt_text": "",
                    "gt_boxes": [],
                }
            )
        return samples

    return default_bbox_samples(bbox_json, image_folder, n=args.num_images)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    detectium_dir = script_dir.parent
    workspace = detectium_dir.parents[1]

    parser = argparse.ArgumentParser(description="Fire bbox smoke test on a few images.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "llavaqwen-2-7b-finetune",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--image-folder",
        type=Path,
        default=workspace / "LLaVA-Instruct-150K",
    )
    parser.add_argument(
        "--bbox-json",
        type=Path,
        default=None,
        help="Default image list + GT boxes when --images is not set.",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Image paths (relative to --image-folder or absolute). "
        "Default: first 5 bbox rows from --bbox-json.",
    )
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--conv-mode", type=str, default="qwen")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs" / "bbox_vis",
    )
    args = parser.parse_args()
    if args.bbox_json is None:
        args.bbox_json = detectium_dir / "detectium_bbox_synthetic2k_nonfire500.json"

    if not args.model_path.is_dir():
        parser.error(f"Model not found: {args.model_path}")

    image_paths = resolve_samples(args, detectium_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "bbox_results.jsonl"

    disable_torch_init()
    model_name = get_model_name_from_path(str(args.model_path))
    tokenizer, model, processor, _ = load_pretrained_model(
        str(args.model_path), args.model_base, model_name
    )
    model.generation_config.repetition_penalty = 1.0
    image_processor = processor["image"]

    print(f"Model:     {args.model_path}")
    print(f"Bbox JSON: {args.bbox_json}")
    print(f"Output:    {args.output_dir}")
    print(f"Images:    {len(image_paths)}")
    print()

    with results_path.open("w") as results_f:
        for idx, sample in enumerate(image_paths, start=1):
            image_path = sample["image_path"]
            gt_boxes = sample.get("gt_boxes") or []
            print(f"[{idx}/{len(image_paths)}] {sample.get('id') or image_path.name}")
            if gt_boxes:
                print(f"  gt: {sample.get('gt_text', '')[:120]}")
            response = run_inference(
                model,
                tokenizer,
                image_processor,
                image_path,
                args.conv_mode,
                args.max_new_tokens,
                args.temperature,
            )
            pred_boxes = parse_bboxes(response)
            print(f"  response: {response[:160]}{'...' if len(response) > 160 else ''}")
            print(f"  parsed GT={len(gt_boxes)} pred={len(pred_boxes)}")

            image = Image.open(image_path).convert("RGB")
            annotated = draw_bboxes(
                image,
                pred_boxes,
                gt_boxes=gt_boxes or None,
                caption=f"G={len(gt_boxes)} P={len(pred_boxes)}",
            )
            out_name = f"{idx:02d}_{image_path.stem}_bbox.jpg"
            out_path = args.output_dir / out_name
            annotated.save(out_path, quality=95)
            print(f"  saved:    {out_path}")

            rec = {
                "index": idx,
                "id": sample.get("id"),
                "image": str(image_path),
                "image_rel": sample.get("image_rel")
                or (
                    str(image_path.relative_to(args.image_folder))
                    if image_path.is_relative_to(args.image_folder)
                    else str(image_path)
                ),
                "prompt": BBOX_PROMPT,
                "gt_text": sample.get("gt_text", ""),
                "gt_boxes_norm_xyxy": [list(b) for b in gt_boxes],
                "response": response,
                "pred_boxes_norm_xyxy": [list(b) for b in pred_boxes],
                "annotated_image": str(out_path),
            }
            results_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print()

    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
