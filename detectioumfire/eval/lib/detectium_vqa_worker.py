#!/usr/bin/env python3
"""
Detectium VQA worker for LLaVA-MoD Qwen checkpoints.

Uses model_vqa_loader (pad aspect ratio, correct stop tokens) and disables
repetition_penalty from generation_config. The default 1.05 penalty crashes
during generate() because input_ids contain IMAGE_TOKEN_INDEX (-200).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3] / "LLava_MoD"
sys.path.insert(0, str(REPO_ROOT))

import llavamod.eval.model_vqa_loader as model_vqa_loader
from llavamod.model.builder import load_pretrained_model as _load_pretrained_model


def load_pretrained_model(*args, **kwargs):
    tokenizer, model, processor, context_len = _load_pretrained_model(*args, **kwargs)
    # peft>=0.1x can leave generation_config as None after adapter wrap; peft 0.4.0 is fine.
    from transformers import GenerationConfig
    gc = getattr(model, "generation_config", None)
    if gc is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.repetition_penalty = 1.0
    return tokenizer, model, processor, context_len


model_vqa_loader.load_pretrained_model = load_pretrained_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="qwen")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--return_gating_logit", type=str, default=None)
    args = parser.parse_args()
    model_vqa_loader.args = args
    model_vqa_loader.eval_model(args)


if __name__ == "__main__":
    main()
