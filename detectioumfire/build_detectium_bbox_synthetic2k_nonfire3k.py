#!/usr/bin/env python3
"""Build detectium_bbox_synthetic2k_nonfire3k.json (2k synthetic fire + 3k non-fire negatives)."""
from __future__ import annotations

import json
import random
from pathlib import Path

DETECTIUM_ROOT = Path(__file__).resolve().parent
SYN_ROOT = DETECTIUM_ROOT / "synthetic_images" / "synthetic_images" / "stable_diff_v15" / "train"
SYN_IMG_DIR = SYN_ROOT / "images"
SYN_LBL_DIR = SYN_ROOT / "labels"
NON_FIRE_DIR = DETECTIUM_ROOT / "real_images" / "real_images" / "real_non_fire"
OUT_JSON = DETECTIUM_ROOT / "detectium_bbox_synthetic2k_nonfire3k.json"

N_SYNTHETIC = 2000
N_NON_FIRE = 3000
RNG_SEED = 42

# Human side: same template as llava_v1_5_mix665k.json (no instructions about output shape).
BBOX_HUMAN_PROMPT = (
    "<image>\n"
    "Please provide all bounding box coordinates for regions with visible fire or flame."
)

NO_FIRE_GPT = (
    "No fire or flame is visible in this image. There are no bounding boxes to report."
)


def yolo_line_to_norm_xyxy(parts: list[str]) -> tuple[float, float, float, float]:
    _cls, xc, yc, bw, bh = parts
    xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
    x1 = max(0.0, min(1.0, xc - bw / 2.0))
    y1 = max(0.0, min(1.0, yc - bh / 2.0))
    x2 = max(0.0, min(1.0, xc + bw / 2.0))
    y2 = max(0.0, min(1.0, yc + bh / 2.0))
    return x1, y1, x2, y2


def _mix665k_style_num(x: float) -> str:
    """Match llava_v1_5_mix665k bbox strings like [0.52, 0.59, 0.82, 0.83] and [0.2, 0.59, 0.52, 0.84]."""
    x = round(float(x), 2)
    if abs(x) < 1e-9:
        return "0.0"
    s = f"{x:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0.0"


def format_xyxy_mix665k(box: tuple[float, float, float, float]) -> str:
    """One bracket: comma + space after each comma, same as mix665k GPT-only bbox replies."""
    return "[" + ", ".join(_mix665k_style_num(c) for c in box) + "]"


def label_file_to_gpt_value(label_path: Path) -> str:
    lines_out: list[str] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            box = yolo_line_to_norm_xyxy(parts)
            lines_out.append(format_xyxy_mix665k(box))
    return "\n".join(lines_out)


def main() -> None:
    rng = random.Random(RNG_SEED)

    syn_images = sorted(
        p
        for p in SYN_IMG_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if len(syn_images) < N_SYNTHETIC:
        raise SystemExit(f"Need {N_SYNTHETIC} synthetic images, found {len(syn_images)}")

    syn_pick = sorted(rng.sample(syn_images, N_SYNTHETIC), key=lambda p: p.name)

    records: list[dict] = []

    for i, img_path in enumerate(syn_pick):
        stem = img_path.stem
        lbl = SYN_LBL_DIR / f"{stem}.txt"
        if not lbl.is_file():
            raise FileNotFoundError(lbl)
        rel_image = f"stable_diff_v15/train/images/{img_path.name}"
        gpt_val = label_file_to_gpt_value(lbl)
        records.append(
            {
                "id": f"syn_bbox_{i:05d}_{stem}",
                "image": rel_image,
                "conversations": [
                    {"from": "human", "value": BBOX_HUMAN_PROMPT},
                    {"from": "gpt", "value": gpt_val},
                ],
                "meta": {
                    "source": "detectium_synthetic_stable_diff_v15",
                    "task": "fire_bbox_xyxy",
                    "label_file": f"stable_diff_v15/train/labels/{lbl.name}",
                },
            }
        )

    non_fire = sorted(
        p
        for p in NON_FIRE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if len(non_fire) < N_NON_FIRE:
        raise SystemExit(f"Need {N_NON_FIRE} non-fire images, found {len(non_fire)}")

    nf_pick = sorted(rng.sample(non_fire, N_NON_FIRE), key=lambda p: p.name)

    for i, img_path in enumerate(nf_pick):
        rel_image = f"real_non_fire/{img_path.name}"
        records.append(
            {
                "id": f"nonfire_bbox_{i:05d}_{img_path.stem}",
                "image": rel_image,
                "conversations": [
                    {"from": "human", "value": BBOX_HUMAN_PROMPT},
                    {"from": "gpt", "value": NO_FIRE_GPT},
                ],
                "meta": {
                    "source": "detectium_real_non_fire",
                    "task": "fire_bbox_xyxy_negative",
                    "fire_type": "non_fire",
                },
            }
        )

    OUT_JSON.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", OUT_JSON, "records", len(records))


if __name__ == "__main__":
    main()
