#!/usr/bin/env bash
# Fire bbox smoke test: Qwen2.5-7B full finetune vs LoRA on 5 synthetic images.
#
# Usage:
#   cd "detectioumfire/eval/launchers/qwen2.5"
#   ./compare_bbox_full_vs_lora.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../common.env.sh"
REPO_ROOT="$(cd "${DETECTIUM_DIR}/../LLava_MoD" && pwd)"

FULL_PATH="${FULL_PATH:-${REPO_ROOT}/checkpoints/llavaqwen2.5-7b-finetune}"
LORA_PATH="${LORA_PATH:-${REPO_ROOT}/checkpoints/llavaqwen2.5-7b-finetune-lora}"
MODEL_BASE="${MODEL_BASE:-Qwen/Qwen2.5-7B-Instruct}"
BBOX_OUT="${EVAL_RUNS}/_scratch/bbox_vis"
FULL_OUT="${BBOX_OUT}/full"
LORA_OUT="${BBOX_OUT}/lora"
SUMMARY_JSON="${EVAL_COMPARISONS}/bbox_compare_full_vs_lora.json"

if [[ ! -f "${FULL_PATH}/config.json" ]]; then
  echo "Missing full finetune checkpoint: ${FULL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${LORA_PATH}/adapter_config.json" ]]; then
  echo "Missing LoRA checkpoint: ${LORA_PATH}" >&2
  exit 1
fi

echo "=== Full finetune bbox (Qwen2.5-7B) ==="
echo "  ${FULL_PATH}"
MODEL_PATH="${FULL_PATH}" OUTPUT_DIR="${FULL_OUT}" "${SCRIPT_DIR}/run_bbox_test.sh"

echo
echo "=== LoRA finetune bbox (Qwen2.5-7B) ==="
echo "  adapter: ${LORA_PATH}"
echo "  base:    ${MODEL_BASE}"
MODEL_PATH="${LORA_PATH}" MODEL_BASE="${MODEL_BASE}" OUTPUT_DIR="${LORA_OUT}" "${SCRIPT_DIR}/run_bbox_test.sh"

echo
echo "=== IoU summary (best pred box vs each GT box) ==="
python - <<'PY' "${FULL_OUT}/bbox_results.jsonl" "${LORA_OUT}/bbox_results.jsonl" "${SUMMARY_JSON}"
import json
import sys
from pathlib import Path


def box_area(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 1e-8 else 0.0


def dedupe_boxes(boxes):
    seen = set()
    out = []
    for b in boxes:
        key = tuple(round(v, 4) for v in b)
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_row(row):
    gt = [tuple(b) for b in row.get("gt_boxes_norm_xyxy", [])]
    pred = dedupe_boxes([tuple(b) for b in row.get("pred_boxes_norm_xyxy", [])])
    if not gt:
        return {"gt_count": 0, "pred_count": len(pred), "mean_best_iou": None, "matched_ious": []}
    matched = []
    for g in gt:
        best = max((iou(g, p) for p in pred), default=0.0) if pred else 0.0
        matched.append(best)
    return {
        "gt_count": len(gt),
        "pred_count": len(pred),
        "mean_best_iou": sum(matched) / len(matched),
        "matched_ious": matched,
    }


full_path, lora_path, out_path = map(Path, sys.argv[1:4])
full_rows = {r.get("id") or r["index"]: r for r in load_jsonl(full_path)}
lora_rows = {r.get("id") or r["index"]: r for r in load_jsonl(lora_path)}
keys = sorted(set(full_rows) & set(lora_rows), key=lambda k: full_rows[k]["index"])

summary = {"images": [], "full_mean_iou": None, "lora_mean_iou": None}
full_ious, lora_ious = [], []

for key in keys:
    fr, lr = full_rows[key], lora_rows[key]
    fs, ls = score_row(fr), score_row(lr)
    if fs["mean_best_iou"] is not None:
        full_ious.append(fs["mean_best_iou"])
    if ls["mean_best_iou"] is not None:
        lora_ious.append(ls["mean_best_iou"])
    summary["images"].append(
        {
            "id": fr.get("id"),
            "index": fr["index"],
            "image_rel": fr.get("image_rel"),
            "full_annotated": fr.get("annotated_image"),
            "lora_annotated": lr.get("annotated_image"),
            "gt_boxes": fr.get("gt_boxes_norm_xyxy"),
            "full_pred_boxes": fr.get("pred_boxes_norm_xyxy"),
            "lora_pred_boxes": lr.get("pred_boxes_norm_xyxy"),
            "full": fs,
            "lora": ls,
        }
    )

if full_ious:
    summary["full_mean_iou"] = sum(full_ious) / len(full_ious)
if lora_ious:
    summary["lora_mean_iou"] = sum(lora_ious) / len(lora_ious)

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(summary, indent=2) + "\n")

print(f"Wrote {out_path}")
for item in summary["images"]:
    fi = item["full"]["mean_best_iou"]
    li = item["lora"]["mean_best_iou"]
    fi_s = f"{fi:.3f}" if fi is not None else "n/a"
    li_s = f"{li:.3f}" if li is not None else "n/a"
    print(f"  [{item['index']}] {item.get('id', '')[:48]}  full IoU={fi_s}  lora IoU={li_s}")
if summary["full_mean_iou"] is not None or summary["lora_mean_iou"] is not None:
    fm = summary["full_mean_iou"]
    lm = summary["lora_mean_iou"]
    fm_s = f"{fm:.3f}" if fm is not None else "n/a"
    lm_s = f"{lm:.3f}" if lm is not None else "n/a"
    print(f"Mean best IoU over GT boxes: full={fm_s}  lora={lm_s}")
PY

echo
echo "Annotated images:"
echo "  full: ${FULL_OUT}"
echo "  lora: ${LORA_OUT}"
echo "Done."
