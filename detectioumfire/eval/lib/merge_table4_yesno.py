#!/usr/bin/env python3
"""Attach Yes/No metrics to each entry in a Table 4 comparison JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from score_detectium_val import (
    _gpt_answer,
    content_fire_label,
    load_answers_jsonl,
    score_against_val,
    score_against_val_content,
)


def _metrics_block(r, preds: dict[str, str], ref_by_id: dict[str, str]) -> dict:
    fire_total = sum(1 for ref in ref_by_id.values() if ref.strip().lower().startswith("yes"))
    fire_detected = sum(
        1
        for rid, ref in ref_by_id.items()
        if ref.strip().lower().startswith("yes")
        and content_fire_label(preds.get(rid, "")) is True
    )
    return {
        "yesno_accuracy": round(r.accuracy, 4),
        "fire_yes_lead": fire_detected,
        "fire_total": fire_total,
        "tp": r.tp,
        "fp": r.fp,
        "tn": r.tn,
        "fn": r.fn,
        "fp_fn": r.fp + r.fn,
        "ambig_pred": r.ambiguous_pred,
        "matched_eval": r.matched_eval,
    }


def yesno_block(ref_by_id: dict[str, str], answers_path: Path) -> dict:
    preds = load_answers_jsonl(answers_path)
    r = score_against_val(ref_by_id, preds)
    block = _metrics_block(r, preds, ref_by_id)
    # Prefix-only fire→Yes count (deploy gate).
    block["fire_yes_lead"] = sum(
        1
        for rid, ref in ref_by_id.items()
        if ref.strip().lower().startswith("yes")
        and preds.get(rid, "").strip().lower().startswith("yes")
    )
    return block


def yesno_content_block(ref_by_id: dict[str, str], answers_path: Path) -> dict:
    preds = load_answers_jsonl(answers_path)
    r = score_against_val_content(ref_by_id, preds)
    return _metrics_block(r, preds, ref_by_id)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table4-json", type=Path, required=True)
    p.add_argument("--val-json", type=Path, required=True)
    p.add_argument(
        "--models-json",
        type=Path,
        required=True,
        help="JSON object mapping model label -> val_answers.jsonl path",
    )
    p.add_argument("--json-out", type=Path, required=True)
    args = p.parse_args()

    table4 = json.loads(args.table4_json.read_text())
    models = json.loads(args.models_json.read_text())
    val_rows = json.loads(args.val_json.read_text())
    ref_by_id = {row["id"]: _gpt_answer(row) for row in val_rows}

    out: dict = {}
    for label, metrics in table4.items():
        entry = dict(metrics)
        answers = models.get(label)
        if answers and Path(answers).is_file():
            entry["yesno"] = yesno_block(ref_by_id, Path(answers))
            entry["yesno_content"] = yesno_content_block(ref_by_id, Path(answers))
        out[label] = entry

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {args.json_out} ({len(out)} models)")


if __name__ == "__main__":
    main()
