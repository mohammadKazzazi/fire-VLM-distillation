import argparse
import json
import os
import random
import re
from typing import Dict, List, Tuple

import torch
from PIL import Image

from llavamod.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llavamod.conversation import SeparatorStyle, conv_templates
from llavamod.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llavamod.model.builder import load_pretrained_model
from llavamod.utils import disable_torch_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick fire/non-fire teacher benchmark.")
    parser.add_argument(
        "--dataset-json",
        type=str,
        default="/home/mohammadkazzazi/mohammadkazzazi/MoD/detectiumfire/detectium_llavamod_real_full.json",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="/home/mohammadkazzazi/mohammadkazzazi/MoD/detectiumfire/real_images/real_images",
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--query",
        type=str,
        default="Carefully analyze the image for signs of fire or smoke. Start your answer with Yes. or No.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--output-jsonl", type=str, default="./teacher_quick_predictions.jsonl")
    parser.add_argument("--summary-json", type=str, default="./teacher_quick_summary.json")
    return parser.parse_args()


def is_fire_sample(item: Dict) -> bool:
    if item.get("id", "").startswith("non_fire_"):
        return False
    meta_fire_type = item.get("meta", {}).get("fire_type")
    if meta_fire_type == "non_fire":
        return False
    answer = item.get("conversations", [{}, {"value": ""}])[1].get("value", "")
    if answer.strip().startswith("No."):
        return False
    return True


def sample_balanced(data: List[Dict], n_per_class: int, seed: int) -> List[Dict]:
    fire = [x for x in data if is_fire_sample(x)]
    non_fire = [x for x in data if not is_fire_sample(x)]
    random.seed(seed)
    random.shuffle(fire)
    random.shuffle(non_fire)
    n_fire = min(n_per_class, len(fire))
    n_non_fire = min(n_per_class, len(non_fire))
    subset = fire[:n_fire] + non_fire[:n_non_fire]
    random.shuffle(subset)
    return subset


def infer_conv_mode(model_name: str) -> str:
    m = model_name.lower()
    if "llama-2" in m:
        return "llava_llama_2"
    if "v1" in m:
        return "llava_v1"
    if "mpt" in m:
        return "mpt"
    if "qwen" in m:
        return "qwen"
    return "llava_v0"


def build_prompt(query: str, conv_mode: str, mm_use_im_start_end: bool) -> Tuple[str, str]:
    conv = conv_templates[conv_mode].copy()
    if mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + query
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + query
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return prompt, stop_str


def classify_prediction(text: str) -> str:
    t = text.strip().lower()
    if re.match(r"^yes[\.,\s]", t):
        return "fire"
    if re.match(r"^no[\.,\s]", t):
        return "non_fire"
    return "unknown"


def run_inference(args: argparse.Namespace) -> None:
    with open(args.dataset_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    subset = sample_balanced(data, args.samples_per_class, args.seed)

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    conv_mode = infer_conv_mode(model_name)
    prompt, stop_str = build_prompt(args.query, conv_mode, bool(model.config.mm_use_im_start_end))

    device = model.device
    use_fp16 = str(device).startswith("cuda")

    rows = []
    cm = {
        "fire_tp": 0,
        "fire_fn": 0,
        "non_fire_tn": 0,
        "non_fire_fp": 0,
        "unknown": 0,
    }

    for idx, item in enumerate(subset, start=1):
        image_rel = item["image"]
        image_abs = os.path.join(args.image_root, image_rel)
        image = Image.open(image_abs).convert("RGB")
        images_tensor = process_images([image], image_processor, model.config)
        if use_fp16:
            images_tensor = images_tensor.to(device, dtype=torch.float16)
        else:
            images_tensor = images_tensor.to(device)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        input_ids = input_ids.to(device)
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        in_len = input_ids.shape[1]
        out = tokenizer.batch_decode(output_ids[:, in_len:], skip_special_tokens=True)[0].strip()
        if out.endswith(stop_str):
            out = out[: -len(stop_str)].strip()

        pred = classify_prediction(out)
        gt = "fire" if is_fire_sample(item) else "non_fire"

        if pred == "unknown":
            cm["unknown"] += 1
        elif gt == "fire" and pred == "fire":
            cm["fire_tp"] += 1
        elif gt == "fire" and pred != "fire":
            cm["fire_fn"] += 1
        elif gt == "non_fire" and pred == "non_fire":
            cm["non_fire_tn"] += 1
        elif gt == "non_fire" and pred != "non_fire":
            cm["non_fire_fp"] += 1

        rows.append(
            {
                "idx": idx,
                "id": item.get("id"),
                "image": image_rel,
                "ground_truth": gt,
                "prediction": pred,
                "output": out,
            }
        )
        if idx % 20 == 0:
            print(f"Processed {idx}/{len(subset)}")

    total = len(rows)
    known = total - cm["unknown"]
    acc = (cm["fire_tp"] + cm["non_fire_tn"]) / known if known > 0 else 0.0
    fire_recall = cm["fire_tp"] / (cm["fire_tp"] + cm["fire_fn"]) if (cm["fire_tp"] + cm["fire_fn"]) > 0 else 0.0
    non_fire_fpr = cm["non_fire_fp"] / (cm["non_fire_fp"] + cm["non_fire_tn"]) if (cm["non_fire_fp"] + cm["non_fire_tn"]) > 0 else 0.0

    summary = {
        "model_path": args.model_path,
        "dataset_json": args.dataset_json,
        "image_root": args.image_root,
        "samples_per_class": args.samples_per_class,
        "evaluated_total": total,
        "confusion": cm,
        "metrics": {
            "accuracy_known_only": acc,
            "fire_recall": fire_recall,
            "non_fire_false_positive_rate": non_fire_fpr,
            "format_compliance_yes_no": known / total if total > 0 else 0.0,
        },
    }

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Predictions written to: {args.output_jsonl}")
    print(f"Summary written to: {args.summary_json}")


if __name__ == "__main__":
    run_inference(parse_args())
