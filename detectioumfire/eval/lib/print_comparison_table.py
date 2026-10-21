#!/usr/bin/env python3
"""Print a side-by-side comparison table from table4_all_plans_with_yesno.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.2f}%"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "comparisons" / "table4_all_plans_with_yesno.json",
    )
    args = p.parse_args()
    data = json.loads(args.json.read_text())

    hdr = (
        f"{'Model':<46} {'Obj':>6} {'Env':>6} {'Sev':>6} "
        f"{'Y/N':>6} {'F→Y':>8} {'FN':>4} {'FP':>4} "
        f"{'CntAcc':>7} {'CntFN':>6} {'CntFP':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, m in data.items():
        obj = m.get("burning_object", {}).get("accuracy")
        env = m.get("environment", {}).get("accuracy")
        sev = m.get("fire_severity", {}).get("accuracy")
        yn = m.get("yesno", {})
        yc = m.get("yesno_content", {})
        fy = yn.get("fire_yes_lead", 0)
        ft = yn.get("fire_total", 754)
        print(
            f"{label[:46]:<46} "
            f"{pct(obj):>6} {pct(env):>6} {pct(sev):>6} "
            f"{pct(yn.get('yesno_accuracy')):>6} "
            f"{fy:>3}/{ft:<4} "
            f"{yn.get('fn', '—'):>4} {yn.get('fp', '—'):>4} "
            f"{pct(yc.get('yesno_accuracy')):>7} "
            f"{yc.get('fn', '—'):>6} {yc.get('fp', '—'):>6}"
        )
    print()
    print("kd_lm Y/N = leading Yes/No (deploy gate).")
    print("only_kd: ignore Y/N prefix; CntAcc/CntFN/CntFP = full-text fire vs non-fire.")
    print("CntAcc is over classified rows only; ambig_pred is in the JSON.")


if __name__ == "__main__":
    main()
