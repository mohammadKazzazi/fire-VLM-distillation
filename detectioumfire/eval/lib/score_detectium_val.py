#!/usr/bin/env python3
"""Score model_vqa JSONL answers against Detectium val JSON (Yes/No fire labels)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


def _gpt_answer(row: dict) -> str:
    for turn in row.get("conversations", []):
        if turn.get("from") == "gpt":
            return (turn.get("value") or "").strip()
    return ""


def yes_no_label(text: str) -> Optional[bool]:
    """
    Map free text to a coarse label from a leading Yes/No token.
    True  = fire / positive (reference starts with "Yes.")
    False = non-fire (reference starts with "No.")
    None  = could not infer from leading wording.
    """
    t = (text or "").strip()
    if not t:
        return None
    m = re.match(r"^\s*([A-Za-z]+)", t)
    if not m:
        return None
    word = m.group(1).lower()
    if word.startswith("yes"):
        return True
    if word.startswith("no"):
        return False
    return None


# Content-based fire detection when preds lack Yes./No. prefixes (e.g. only_kd students).
_CONTENT_NEG = tuple(
    re.compile(p, re.I)
    for p in (
        r"\bno (?:active )?fire\b",
        r"\bnon[- ]fire\b",
        r"\bno (?:visible )?(?:flames?|smoke)\b",
        r"\bno evidence of (?:active )?fire\b",
        r"\bno signs? of (?:fire|flame|smoke)\b",
        r"\bno fire (?:is )?(?:present|visible|detected|seen)\b",
        r"\bnot (?:a )?(?:fire|burning)\b",
        r"\bwithout (?:any )?(?:fire|flames?|smoke)\b",
        r"\bindicating no fire\b",
        r"\bno fire risk\b",
        r"\bposes? no (?:fire )?risk\b",
        r"\bdoes not (?:show|contain|depict).{0,40}(?:fire|flames?|smoke)\b",
        r"\bthere is no\b.{0,30}(?:fire|flames?|smoke)\b",
    )
)
_CONTENT_POS = tuple(
    re.compile(p, re.I)
    for p in (
        r"\bcontrolled fire\b",
        r"\bfire is burning\b",
        r"\b(?:severe|moderate|active|intense|uncontrolled|wild|structure|building|forest|grass|field|bush|dumpster|trash|vehicle|room|interior|wildland)\w*\s+fire\b",
        r"\bfire (?:risk|hazard|incident|scene|spread)\b",
        r"\b(?:is|are) burning\b",
        r"\bflames?\b",
        r"\bburning\b",
        r"\bblaze\b",
        r"\bwildfire\b",
        r"\bsmoke (?:is )?(?:rising|billowing|visible|plume)\b",
        r"\bengulfed\b",
        r"\bignit",
    )
)


def content_fire_label(text: str) -> Optional[bool]:
    """
    Infer fire vs non-fire from the full response body.

    Leading Yes./No. is ignored so only_kd captions (usually starting with "A …")
    are scored from scene language, not from a missing prefix.
    """
    from score_content_based import classify_text

    return classify_text(text, honor_prefix=False)


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
    pred_lines: int
    matched_eval: int
    missing_ref: int
    ambiguous_ref: int
    ambiguous_pred: int
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def accuracy(self) -> float:
        n = self.matched_eval
        return (self.tp + self.tn) / n if n else 0.0


def _score_pairs(
    ref_by_id: Dict[str, str],
    preds: Dict[str, str],
    pred_label_fn,
) -> ScoreResult:
    tp = fp = tn = fn = 0
    missing_ref = 0
    ambiguous_ref = ambiguous_pred = 0
    both_labeled = 0

    for qid, pred_text in preds.items():
        ref_text = ref_by_id.get(qid)
        if ref_text is None:
            missing_ref += 1
            continue
        rl = yes_no_label(ref_text)
        pl = pred_label_fn(pred_text)
        if rl is None:
            ambiguous_ref += 1
        if pl is None:
            ambiguous_pred += 1
        if rl is None or pl is None:
            continue
        both_labeled += 1
        if rl and pl:
            tp += 1
        elif rl and not pl:
            fn += 1
        elif not rl and pl:
            fp += 1
        else:
            tn += 1

    return ScoreResult(
        pred_lines=len(preds),
        matched_eval=both_labeled,
        missing_ref=missing_ref,
        ambiguous_ref=ambiguous_ref,
        ambiguous_pred=ambiguous_pred,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def score_against_val(
    ref_by_id: Dict[str, str], preds: Dict[str, str]
) -> ScoreResult:
    return _score_pairs(ref_by_id, preds, yes_no_label)


def score_against_val_content(
    ref_by_id: Dict[str, str], preds: Dict[str, str]
) -> ScoreResult:
    """Score preds using content-based fire inference (prefix + body heuristics)."""
    return _score_pairs(ref_by_id, preds, content_fire_label)


def print_single(label: str, path: Path, r: ScoreResult) -> None:
    print(f"=== {label} ===")
    print(f"answers_file: {path}")
    print(f"pred_lines:   {r.pred_lines}")
    print(f"matched_eval: {r.matched_eval} (ref and pred both have Yes/No lead)")
    print(f"missing_ref:  {r.missing_ref}")
    print(f"ambig_ref:    {r.ambiguous_ref}  ambig_pred: {r.ambiguous_pred}")
    print()
    print(f"accuracy:     {r.accuracy:.4f}")
    print(f"confusion    TP={r.tp}  FP={r.fp}  TN={r.tn}  FN={r.fn}")
    if r.tp + r.fn:
        print(f"fire recall:    {r.tp / (r.tp + r.fn):.4f}")
    if r.tp + r.fp:
        print(f"fire precision: {r.tp / (r.tp + r.fp):.4f}")
    if r.tn + r.fp:
        print(f"non-fire recall: {r.tn / (r.tn + r.fp):.4f}")
    print()


def compare_models(
    ref_by_id: Dict[str, str],
    preds_base: Dict[str, str],
    preds_ft: Dict[str, str],
) -> None:
    """Same-example comparison: only qids where ref and both preds have Yes/No labels."""
    base_right_ft_right = 0
    base_wrong_ft_right = 0
    base_right_ft_wrong = 0
    base_wrong_ft_wrong = 0
    n = 0

    for qid, ref_text in ref_by_id.items():
        rl = yes_no_label(ref_text)
        if rl is None:
            continue
        tb = preds_base.get(qid)
        tf = preds_ft.get(qid)
        if tb is None or tf is None:
            continue
        lb = yes_no_label(tb)
        lf = yes_no_label(tf)
        if lb is None or lf is None:
            continue
        correct_b = lb == rl
        correct_f = lf == rl
        n += 1
        if correct_b and correct_f:
            base_right_ft_right += 1
        elif not correct_b and correct_f:
            base_wrong_ft_right += 1
        elif correct_b and not correct_f:
            base_right_ft_wrong += 1
        else:
            base_wrong_ft_wrong += 1

    acc_b = (base_right_ft_right + base_right_ft_wrong) / n if n else 0.0
    acc_f = (base_wrong_ft_right + base_right_ft_right) / n if n else 0.0

    print("=== Side-by-side (same examples) ===")
    print(
        f"examples_with_labeled_ref_and_both_preds: {n} "
        "(Yes/No parseable on reference, base, and fine-tuned)"
    )
    print()
    print(f"base accuracy on this set:      {acc_b:.4f}")
    print(f"finetuned accuracy on this set: {acc_f:.4f}")
    print(f"delta (finetuned - base):       {acc_f - acc_b:+.4f}")
    print()
    print("transition counts (reference is ground truth):")
    print(f"  both correct:           {base_right_ft_right}")
    print(f"  base wrong → FT fixed:  {base_wrong_ft_right}  (enhancements)")
    print(f"  base right → FT broke:  {base_right_ft_wrong}  (regressions)")
    print(f"  both wrong:             {base_wrong_ft_wrong}")
    print()


def default_val_json() -> Path:
    return Path(__file__).resolve().parent.parent / "detectium_llavamod_real_full_val.json"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Score model_vqa JSONL vs Detectium val JSON; optional base vs fine-tuned compare."
    )
    p.add_argument("--val-json", type=Path, default=default_val_json())
    p.add_argument(
        "--answers",
        type=Path,
        default=None,
        help="Single model: JSONL from model_vqa (question_id, text).",
    )
    p.add_argument(
        "--answers-base",
        type=Path,
        default=None,
        help="With --answers-finetuned: base model answers JSONL.",
    )
    p.add_argument(
        "--answers-finetuned",
        type=Path,
        default=None,
        help="With --answers-base: LoRA / fine-tuned answers JSONL.",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write metrics as JSON.",
    )
    args = p.parse_args()

    if not args.val_json.is_file():
        print(f"val JSON not found: {args.val_json}", file=sys.stderr)
        sys.exit(1)

    val_rows = json.loads(args.val_json.read_text())
    ref_by_id = {row["id"]: _gpt_answer(row) for row in val_rows}

    if args.answers_base is not None and args.answers_finetuned is not None:
        pb = load_answers_jsonl(args.answers_base)
        pf = load_answers_jsonl(args.answers_finetuned)
        print(f"val_json: {args.val_json}\n")
        rb = score_against_val(ref_by_id, pb)
        rf = score_against_val(ref_by_id, pf)
        print_single("Base model", args.answers_base, rb)
        print_single("Fine-tuned", args.answers_finetuned, rf)
        compare_models(ref_by_id, pb, pf)
        if args.json_out:
            _write_json_out(args.json_out, base=rb, finetuned=rf)
        return

    if args.answers is None:
        p.error("Pass --answers FILE, or both --answers-base and --answers-finetuned.")

    preds = load_answers_jsonl(args.answers)
    r = score_against_val(ref_by_id, preds)
    print(f"val_json:     {args.val_json}")
    print_single("Model", args.answers, r)
    if args.json_out:
        _write_json_out(args.json_out, model=r)


def _write_json_out(path: Path, model: ScoreResult | None = None, **kwargs) -> None:
    def as_dict(r: ScoreResult) -> dict:
        return {
            "pred_lines": r.pred_lines,
            "matched_eval": r.matched_eval,
            "missing_ref": r.missing_ref,
            "ambiguous_ref": r.ambiguous_ref,
            "ambiguous_pred": r.ambiguous_pred,
            "accuracy": r.accuracy,
            "tp": r.tp,
            "fp": r.fp,
            "tn": r.tn,
            "fn": r.fn,
        }

    payload: dict = {}
    if model is not None:
        payload["model"] = as_dict(model)
    for name, result in kwargs.items():
        payload[name] = as_dict(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote metrics JSON: {path}")


if __name__ == "__main__":
    main()
