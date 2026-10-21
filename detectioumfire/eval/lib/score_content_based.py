#!/usr/bin/env python3
"""
Content-based scoring: classify predictions as fire/non-fire using full text
heuristics (not just leading Yes/No). Ground truth from question_id prefix.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


NON_FIRE_PATTERNS = [
    r"no\s+(visible\s+)?signs?\s+of\s+(fire|smoke|flame)",
    r"no\s+(fire|smoke|flame)",
    r"no\s+indication\s+of",
    r"no\s+evidence\s+of\s+(active\s+)?fire",
    r"there\s+are\s+no\s+(visible\s+)?(signs?|indication)",
    r"non[\-\s]fire\s+scene",
    r"this\s+is\s+a\s+non[\-\s]fire",
    r"not\s+(currently\s+)?experiencing\s+any\s+(wild)?fire",
    r"absence\s+of\s+any\s+indication",
    r"does\s+not\s+show\s+any\s+signs?\s+of\s+fire",
    r"without\s+any\s+(indication|sign)\s+of.*(fire|smoke)",
    r"no\s+active\s+fire",
    r"scene\s+is\s+(not|safe|serene|peaceful|picturesque)",
]

FIRE_PATTERNS = [
    r"(fire|flame|blaze)\s+(is\s+)?(burning|spreading|engulfing|visible|raging)",
    r"(on|caught)\s+fire",
    r"(flames?|fire)\s+(are\s+)?visible",
    r"(flames?|fire)\s+(shooting|reaching|rising|billowing)",
    r"smoke\s+(is\s+)?(billowing|rising|coming|visible)",
    r"(large|small|intense|significant|severe|moderate)\s+fire",
    r"fire\s+(is|appears|seems|could)\s+be",
    r"fire\s+(scene|incident|hazard|burning|engulf)",
    r"burning\s+(in|on|inside|through|across)",
    r"(lit|burning)\s+(cigarette|lighter|candle|match|flame)",
    r"fireplace\s+with\s+a?\s*fire",
    r"fire\s+burning\s+in",
    r"evidence\s+of\s+(active\s+|an\s+active\s+)?fire",
    r"yes[\.,\s]",
    r"controlled\s+fire",
    r"a\s+(lighter|candle|match)\s+is\s+(lit|ignit)",
]


def classify_text(text: str, honor_prefix: bool = True) -> Optional[bool]:
    """
    Classify full response text as fire (True), non-fire (False), or ambiguous (None).

    honor_prefix=True: leading Yes./No. wins (kd_lm deploy gate).
    honor_prefix=False: ignore leading token; score the body (only_kd).
    """
    t = (text or "").strip().lower()
    if not t:
        return None

    if honor_prefix:
        leading = re.match(r"^\s*(yes|no)[\.,\s]", t)
        if leading:
            return leading.group(1) == "yes"

    non_fire_score = sum(1 for p in NON_FIRE_PATTERNS if re.search(p, t))
    fire_score = sum(1 for p in FIRE_PATTERNS if re.search(p, t))

    if non_fire_score > 0 and fire_score == 0:
        return False
    if fire_score > 0 and non_fire_score == 0:
        return True
    if fire_score > non_fire_score:
        return True
    if non_fire_score > fire_score:
        return False

    return None


def ground_truth_from_qid(qid: str) -> Optional[bool]:
    """True = fire, False = non-fire, based on question_id prefix."""
    if qid.startswith("fire_"):
        return True
    if qid.startswith("non_fire_"):
        return False
    return None


def load_answers_jsonl(path: Path) -> Dict[str, str]:
    preds: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("question_id")
            if qid is not None:
                preds[str(qid)] = (obj.get("text") or "").strip()
    return preds


@dataclass
class ScoreResult:
    total: int = 0
    classified: int = 0
    ambiguous: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    ambiguous_examples: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        n = self.classified
        return (self.tp + self.tn) / n if n else 0.0


def score_model(preds: Dict[str, str]) -> ScoreResult:
    r = ScoreResult()
    for qid, text in preds.items():
        gt = ground_truth_from_qid(qid)
        if gt is None:
            continue
        r.total += 1
        pred = classify_text(text)
        if pred is None:
            r.ambiguous += 1
            r.ambiguous_examples.append({"qid": qid, "text": text[:120]})
            continue
        r.classified += 1
        if gt and pred:
            r.tp += 1
        elif gt and not pred:
            r.fn += 1
        elif not gt and pred:
            r.fp += 1
        else:
            r.tn += 1
    return r


def print_result(label: str, path: Path, r: ScoreResult) -> None:
    print(f"=== {label} ===")
    print(f"answers_file:  {path}")
    print(f"total samples: {r.total}")
    print(f"classified:    {r.classified} ({100*r.classified/r.total:.1f}%)")
    print(f"ambiguous:     {r.ambiguous} ({100*r.ambiguous/r.total:.1f}%)")
    print()
    print(f"accuracy:        {r.accuracy:.4f} ({r.tp + r.tn}/{r.classified})")
    print(f"confusion:       TP={r.tp}  FP={r.fp}  TN={r.tn}  FN={r.fn}")
    if r.tp + r.fn:
        recall = r.tp / (r.tp + r.fn)
        print(f"fire recall:     {recall:.4f} ({r.tp}/{r.tp + r.fn})")
    if r.tp + r.fp:
        prec = r.tp / (r.tp + r.fp)
        print(f"fire precision:  {prec:.4f} ({r.tp}/{r.tp + r.fp})")
    if r.tn + r.fp:
        spec = r.tn / (r.tn + r.fp)
        print(f"non-fire recall: {spec:.4f} ({r.tn}/{r.tn + r.fp})")
    if r.fn + r.tn:
        npv = r.tn / (r.tn + r.fn)
        print(f"non-fire prec:   {npv:.4f} ({r.tn}/{r.tn + r.fn})")
    print()
    if r.ambiguous_examples:
        print(f"  first 5 ambiguous examples:")
        for ex in r.ambiguous_examples[:5]:
            print(f"    [{ex['qid']}] {ex['text']}...")
        print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Content-based fire/non-fire scoring using full-text heuristics."
    )
    p.add_argument(
        "--answers-base", type=Path,
        default=Path(__file__).resolve().parent / "val_answers_base.jsonl",
    )
    p.add_argument(
        "--answers-finetuned", type=Path,
        default=Path(__file__).resolve().parent / "val_answers_lora.jsonl",
    )
    args = p.parse_args()

    pb = load_answers_jsonl(args.answers_base)
    pf = load_answers_jsonl(args.answers_finetuned)

    rb = score_model(pb)
    rf = score_model(pf)

    print_result("Base model (content-based classification)", args.answers_base, rb)
    print_result("Fine-tuned LoRA (content-based classification)", args.answers_finetuned, rf)

    # Side-by-side on all samples where both are classifiable
    both_right = both_wrong = base_wrong_ft_right = base_right_ft_wrong = 0
    n = 0
    for qid in pb:
        gt = ground_truth_from_qid(qid)
        if gt is None:
            continue
        pred_b = classify_text(pb[qid])
        pred_f = classify_text(pf.get(qid, ""))
        if pred_b is None or pred_f is None:
            continue
        n += 1
        cb = pred_b == gt
        cf = pred_f == gt
        if cb and cf:
            both_right += 1
        elif not cb and cf:
            base_wrong_ft_right += 1
        elif cb and not cf:
            base_right_ft_wrong += 1
        else:
            both_wrong += 1

    acc_b = (both_right + base_right_ft_wrong) / n if n else 0
    acc_f = (both_right + base_wrong_ft_right) / n if n else 0

    print(f"=== Side-by-side (both classifiable: {n} samples) ===")
    print(f"base accuracy:      {acc_b:.4f}")
    print(f"finetuned accuracy: {acc_f:.4f}")
    print(f"delta:              {acc_f - acc_b:+.4f}")
    print()
    print(f"  both correct:           {both_right}")
    print(f"  base wrong → FT fixed:  {base_wrong_ft_right}")
    print(f"  base right → FT broke:  {base_right_ft_wrong}")
    print(f"  both wrong:             {both_wrong}")
    print()


if __name__ == "__main__":
    main()
