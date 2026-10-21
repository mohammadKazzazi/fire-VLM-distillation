#!/usr/bin/env python3
"""Generic lean export: LoRA fire student → HF LLM + llava.projector (no vision load)."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_DEPLOY = Path(__file__).resolve().parents[1]


def _load_projector(lora: Path) -> dict:
    nlt = lora / "non_lora_trainables.bin"
    if not nlt.is_file():
        raise FileNotFoundError(nlt)
    raw = torch.load(nlt, map_location="cpu", weights_only=True)
    out = {}
    for k, v in raw.items():
        if k.endswith("mm_projector.image_spatial_proj.0.weight"):
            out["model.mm_projector.0.weight"] = v.detach().to(torch.float16).cpu().contiguous()
        elif k.endswith("mm_projector.image_spatial_proj.0.bias"):
            out["model.mm_projector.0.bias"] = v.detach().to(torch.float16).cpu().contiguous()
        elif k.endswith("mm_projector.image_spatial_proj.2.weight"):
            out["model.mm_projector.2.weight"] = v.detach().to(torch.float16).cpu().contiguous()
        elif k.endswith("mm_projector.image_spatial_proj.2.bias"):
            out["model.mm_projector.2.bias"] = v.detach().to(torch.float16).cpu().contiguous()
    need = [
        "model.mm_projector.0.weight",
        "model.mm_projector.0.bias",
        "model.mm_projector.2.weight",
        "model.mm_projector.2.bias",
    ]
    missing = [k for k in need if k not in out]
    if missing:
        raise KeyError(f"Projector keys missing {missing}; available={list(raw.keys())[:20]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True, type=Path)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arch", required=True, help="e.g. Qwen2ForCausalLM")
    ap.add_argument("--model-type", required=True, help="e.g. qwen2")
    ap.add_argument("--jinja", required=True, type=Path)
    ap.add_argument("--local-files-only", action="store_true", default=False)
    args = ap.parse_args()

    lora = args.lora.resolve()
    out = args.out.resolve()
    llm_dir = out / "llm"
    out.mkdir(parents=True, exist_ok=True)
    llm_dir.mkdir(parents=True, exist_ok=True)

    local = bool(args.local_files_only) or os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    print(
        f"Lean export\n  lora={lora}\n  base={args.base}\n  out={out}\n  local_only={local}",
        flush=True,
    )

    proj = _load_projector(lora)
    proj_path = out / "llava.projector"
    torch.save(proj, proj_path)
    print(f"Wrote {proj_path}", flush=True)
    del proj
    gc.collect()

    print("Loading base LM (fp16, no vision) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=False, local_files_only=local)
    lm = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=local,
    )
    print("Loading LoRA + merge ...", flush=True)
    lm = PeftModel.from_pretrained(lm, str(lora))
    lm = lm.merge_and_unload()
    lm = lm.to(torch.float16)
    gc.collect()

    print("Saving merged LM ...", flush=True)
    lm.save_pretrained(llm_dir, safe_serialization=True)
    tok.save_pretrained(llm_dir)
    del lm
    gc.collect()

    cfg_path = llm_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["architectures"] = [args.arch]
    cfg["model_type"] = args.model_type
    cfg["torch_dtype"] = "float16"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    if not args.jinja.is_file():
        raise FileNotFoundError(args.jinja)
    jinja_text = args.jinja.read_text()
    (llm_dir / "chat_template.jinja").write_text(jinja_text)
    tok_cfg_path = llm_dir / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_path.read_text()) if tok_cfg_path.is_file() else {}
    tok_cfg["chat_template"] = jinja_text
    tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + "\n")

    meta = {
        "lora": str(lora),
        "base": args.base,
        "arch": args.arch,
        "model_type": args.model_type,
        "jinja": str(args.jinja),
    }
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Export done → {llm_dir}", flush=True)


if __name__ == "__main__":
    main()
