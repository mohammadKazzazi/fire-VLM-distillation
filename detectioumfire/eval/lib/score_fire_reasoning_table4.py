#!/usr/bin/env python3
"""
Paper-aligned Table 4 scoring (DetectiumFire arXiv:2511.02495).

Evaluates fire-only val rows on three attributes:
  - Burning Object
  - Environment (indoor / outdoor)
  - Fire Severity (no_risk / low / moderate / high)

Ground truth is derived from val JSON reference captions + fire_type metadata.
Predictions are scored with the same heuristics (paper uses GPT-4o judges).

Usage:
  python3 score_fire_reasoning_table4.py \\
    --val-json ../detectium_llavamod_real_full_val.json \\
    --answers-full Qwen2.5-7B/outputs/val_answers_full.jsonl \\
    --answers-lora Qwen2.5-7B/outputs/val_answers_lora.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --- Ground truth from fire_type (Detectium taxonomy) ---

def environment_gt(fire_type: str) -> Optional[str]:
    if fire_type.startswith("Indoor_") or fire_type == "cooking_fire_no_risk":
        return "indoor"
    if fire_type.startswith("Outdoor_") or fire_type.startswith("Outddor_"):
        return "outdoor"
    if fire_type in ("gas_tank_fires",):
        return "outdoor"
    return None


OBJECT_KEYWORDS: Dict[str, List[str]] = {
    "Indoor_candle_fire": ["candle"],
    "Indoor_lighter_flame": ["lighter", "cigarette", "lantern", "cigar"],
    "Indoor_matches_flame": ["match"],
    "Indoor_stove_fire": ["stove", "fireplace", "heater", "wood-fueled", "wood fueled"],
    "Indoor_kitchen_fire": ["kitchen", "stove", "cooker", "frying", "cooking", "oil"],
    "Indoor_electrical_fire": ["electrical", "wire", "phone", "fan", "device", "appliance"],
    "Indoor_fire": ["room", "indoor", "inside", "interior"],
    "Outdoor_car_fire": ["car", "vehicle", "truck", "motorcycle", "automobile"],
    "Outdoor_house_fire": ["house", "residential", "home", "building", "apartment"],
    "Outdoor_forest_fire": ["forest", "wildfire", "tree", "wildland", "brush", "underbrush"],
    "Outdoor_fire": ["gas station", "structure", "building", "outdoor"],
    "Outddor_camp_fire": ["campfire", "camp", "bonfire"],
    "Outdoor_burning_debris": ["debris", "leaves", "woodpile", "wood pile", "paper"],
    "Outdoor_ship_fire": ["ship", "vessel", "marine", "boat"],
    "Outdoor_plane_fire": ["plane", "aircraft", "airplane"],
    "Outdoor_trash_bin_fire": ["trash", "dumpster", "bin"],
    "gas_tank_fires": ["gas", "tank", "lpg", "canister", "propane"],
    "cooking_fire_no_risk": ["cooking", "stove", "kitchen"],
    "pure_fire_only": ["flame", "fire"],
}

INDOOR_HINTS = [
    "indoor", "inside", "interior", "room", "kitchen", "living room",
    "bedroom", "office", "hallway", "stove", "fireplace",
]
OUTDOOR_HINTS = [
    "outdoor", "outside", "forest", "wildland", "road", "street",
    "field", "yard", "wildfire", "highway", "gas station", "open air",
]


def severity_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    if re.search(r"\bno\s+risk\b", t) or re.search(r"controlled fire\s*\(no risk\)", t):
        return "no_risk"
    if re.search(r"\blow\s+risk\b", t):
        return "low"
    if re.search(r"\bmoderate\b|\bmedium\s+risk\b", t):
        return "moderate"
    if re.search(r"\bsevere\b|\bhigh\s+risk\b|\buncontrolled\b", t):
        return "high"
    if "controlled fire" in t:
        return "no_risk"
    return None


def object_match(pred: str, fire_type: str) -> Optional[bool]:
    keywords = OBJECT_KEYWORDS.get(fire_type)
    if not keywords:
        return None
    t = (pred or "").lower()
    return any(k in t for k in keywords)


def environment_match(pred: str, gt_env: str) -> Optional[bool]:
    t = (pred or "").lower()
    indoor = any(h in t for h in INDOOR_HINTS)
    outdoor = any(h in t for h in OUTDOOR_HINTS)
    if indoor and not outdoor:
        pred_env = "indoor"
    elif outdoor and not indoor:
        pred_env = "outdoor"
    elif indoor and outdoor:
        return None
    else:
        return None
    return pred_env == gt_env


def load_val(path: Path) -> Dict[str, dict]:
    rows = json.loads(path.read_text())
    return {r["id"]: r for r in rows}


def load_preds(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("question_id")
            if qid is not None:
                text = (obj.get("text") or "").strip()
                text = re.sub(r"<\|endoftext\|>$", "", text).strip()
                out[str(qid)] = text
    return out


@dataclass
class AxisScore:
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class Table4Result:
    burning_object: AxisScore
    environment: AxisScore
    fire_severity: AxisScore
    fire_rows: int


def score_predictions(val_by_id: Dict[str, dict], preds: Dict[str, str]) -> Table4Result:
    obj_s, env_s, sev_s = AxisScore(), AxisScore(), AxisScore()
    fire_rows = 0

    for rid, row in val_by_id.items():
        meta = row.get("meta") or {}
        if meta.get("fire_type") == "non_fire":
            continue
        fire_rows += 1
        pred = preds.get(rid, "")
        if not pred:
            continue

        ref = row["conversations"][1]["value"]
        ft = meta.get("fire_type", "")

        om = object_match(pred, ft)
        if om is not None:
            obj_s.total += 1
            if om:
                obj_s.correct += 1

        gt_env = environment_gt(ft)
        if gt_env:
            em = environment_match(pred, gt_env)
            if em is not None:
                env_s.total += 1
                if em:
                    env_s.correct += 1

        gt_sev = severity_from_text(ref)
        pr_sev = severity_from_text(pred)
        if gt_sev and pr_sev:
            sev_s.total += 1
            if gt_sev == pr_sev:
                sev_s.correct += 1

    return Table4Result(obj_s, env_s, sev_s, fire_rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Table 4 fire reasoning scores")
    p.add_argument(
        "--val-json",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "detectium_llavamod_real_full_val.json",
    )
    p.add_argument("--answers-full", type=Path, default=None)
    p.add_argument("--answers-lora", type=Path, default=None)
    p.add_argument("--answers-base", type=Path, default=None)
    p.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("LABEL", "ANSWERS_JSONL"),
        default=[],
        help="Arbitrary model label + answers JSONL (repeatable; overrides --answers-full/lora/base)",
    )
    p.add_argument("--label-full", type=str, default="Full SFT")
    p.add_argument("--label-lora", type=str, default="LoRA SFT")
    p.add_argument("--label-base", type=str, default="Base (pretrain projector only)")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    val_by_id = load_val(args.val_json)

    models: List[Tuple[str, Path]] = []
    if args.model:
        models = [(label, Path(path)) for label, path in args.model]
    else:
        if not args.answers_full:
            p.error("Provide --answers-full or one or more --model LABEL PATH")
        if args.answers_base:
            models.append((args.label_base, args.answers_base))
        models.append((args.label_full, args.answers_full))
        if args.answers_lora:
            models.append((args.label_lora, args.answers_lora))

    results = {}
    for label, path in models:
        preds = load_preds(path)
        r = score_predictions(val_by_id, preds)
        results[label] = {
            "burning_object": {"accuracy": r.burning_object.accuracy, "n": r.burning_object.total},
            "environment": {"accuracy": r.environment.accuracy, "n": r.environment.total},
            "fire_severity": {"accuracy": r.fire_severity.accuracy, "n": r.fire_severity.total},
            "fire_rows_in_val": r.fire_rows,
        }

    print("Table 4-style: Accuracy on fire reasoning (fire-only val subset)")
    print(f"Val JSON: {args.val_json}")
    print(f"Fire images: {results[models[0][0]]['fire_rows_in_val']}")
    print()
    print(f"| {'Model':<45} | {'Burning Object':>14} | {'Environment':>12} | {'Fire Severity':>14} |")
    print(f"|{'-'*47}|{'-'*16}|{'-'*14}|{'-'*16}|")
    for label, _ in models:
        d = results[label]
        print(
            f"| {label:<45} "
            f"| {d['burning_object']['accuracy']*100:>12.2f}% "
            f"| {d['environment']['accuracy']*100:>10.2f}% "
            f"| {d['fire_severity']['accuracy']*100:>12.2f}% |"
        )
    print()
    print("Note: heuristic scoring on existing VQA answers; paper uses GPT-4o judges.")
    print("Severity/object skipped when reference or prediction is ambiguous.")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
