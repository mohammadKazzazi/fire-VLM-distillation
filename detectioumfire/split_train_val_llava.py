#!/usr/bin/env python3
"""Stratified ~90/10 train/val split of LLaVA-format JSON by meta.fire_type."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "detectium_llavamod_real_full.json",
    )
    p.add_argument(
        "--train-out",
        type=Path,
        default=Path(__file__).resolve().parent / "detectium_llavamod_real_full_train.json",
    )
    p.add_argument(
        "--val-out",
        type=Path,
        default=Path(__file__).resolve().parent / "detectium_llavamod_real_full_val.json",
    )
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    rng = random.Random(args.seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(data):
        k = (row.get("meta") or {}).get("fire_type", "unknown")
        groups[str(k)].append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for idxs in groups.values():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            train_idx.extend(idxs)
            continue
        n_val = max(1, int(round(n * args.val_fraction)))
        if n_val >= n:
            n_val = n - 1
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    args.train_out.write_text(json.dumps([data[i] for i in train_idx], indent=2))
    args.val_out.write_text(json.dumps([data[i] for i in val_idx], indent=2))
    print(f"total={len(data)} train={len(train_idx)} val={len(val_idx)}")


if __name__ == "__main__":
    main()
