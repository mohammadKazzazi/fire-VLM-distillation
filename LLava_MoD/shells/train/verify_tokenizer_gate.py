#!/usr/bin/env python3
"""Compare teacher/student tokenizer encode() for KD safety."""

import argparse
import sys

from transformers import AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    args = parser.parse_args()

    t_tok = AutoTokenizer.from_pretrained(args.teacher, use_fast=False)
    s_tok = AutoTokenizer.from_pretrained(args.student, use_fast=False)

    print(f"Teacher: {args.teacher} vocab_size={t_tok.vocab_size}")
    print(f"Student: {args.student} vocab_size={s_tok.vocab_size}")

    ok = True
    for prompt in args.prompts:
        t_ids = t_tok.encode(prompt, add_special_tokens=False)
        s_ids = s_tok.encode(prompt, add_special_tokens=False)
        match = t_ids == s_ids
        ok = ok and match
        status = "OK" if match else "MISMATCH"
        print(f"[{status}] {prompt!r}: teacher={t_ids} student={s_ids}")

    if t_tok.vocab_size != s_tok.vocab_size:
        print(f"WARN: vocab_size differs ({t_tok.vocab_size} vs {s_tok.vocab_size})")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
