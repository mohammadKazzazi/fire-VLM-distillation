#!/usr/bin/env python3
"""
Low-RAM export for Gemma2/Llama3 fire students → HF LLM + llava.projector.

Skips loading CLIP vision tower (not needed for GGUF LM/mmproj export).
Uses float16 + merge_and_unload to stay within ~16GB + swap.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_DEPLOY = Path(__file__).resolve().parents[1]

PRESETS = {
    "llama3-1b": {
        "lora_env": "LLAMA3_1B_LORA",
        "base_env": "LLAMA3_1B_BASE",
        "default_base": "meta-llama/Llama-3.2-1B-Instruct",
        "arch": "LlamaForCausalLM",
        "model_type": "llama",
        "conv_jinja": "llava_llama3.jinja",
        "conv_name": "llava_llama3 (llama_3)",
    },
    "gemma2-2b": {
        "lora_env": "GEMMA2_2B_LORA",
        "base_env": "GEMMA2_2B_BASE",
        "default_base": "google/gemma-2-2b-it",
        "arch": "Gemma2ForCausalLM",
        "model_type": "gemma2",
        "conv_jinja": "llava_gemma2.jinja",
        "conv_name": "llava_gemma2 (gemma_2)",
    },
}


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


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
        raise KeyError(f"Projector keys missing {missing}; available={list(raw.keys())}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_DEPLOY / "config" / "students.env"))
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    _load_env(Path(args.config))
    preset = PRESETS[args.preset]
    lora = Path(os.environ[preset["lora_env"]]).resolve()
    base = os.environ.get(preset["base_env"], preset["default_base"])
    clip_id = os.environ.get("CLIP_MODEL", "openai/clip-vit-large-patch14-336")
    out = Path(args.out).resolve() if args.out else (_DEPLOY / "bundles" / "llamacpp_export" / args.preset)
    llm_dir = out / "llm"
    out.mkdir(parents=True, exist_ok=True)
    llm_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lean export preset={args.preset}\n  lora={lora}\n  base={base}\n  out={out}", flush=True)

    # Projector from non_lora_trainables (no CLIP load)
    proj = _load_projector(lora)
    proj_path = out / "llava.projector"
    torch.save(proj, proj_path)
    print(f"Wrote {proj_path} keys={list(proj)}", flush=True)
    del proj
    gc.collect()

    print("Loading base LM (fp16, no vision) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(base, use_fast=False, local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    print("Loading LoRA + merge ...", flush=True)
    lm = PeftModel.from_pretrained(lm, str(lora))
    lm = lm.merge_and_unload()
    lm = lm.to(torch.float16)
    gc.collect()

    print("Saving merged LM ...", flush=True)
    lm.save_pretrained(llm_dir, safe_serialization=True)
    tok.save_pretrained(llm_dir)
    hidden_size = int(lm.config.hidden_size)
    del lm
    gc.collect()

    cfg_path = llm_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["architectures"] = [preset["arch"]]
    cfg["model_type"] = preset["model_type"]
    cfg["torch_dtype"] = "float16"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    jinja_src = _DEPLOY / "bundles" / "gguf" / preset["conv_jinja"]
    jinja_text = jinja_src.read_text()
    (llm_dir / "chat_template.jinja").write_text(jinja_text)
    tok_cfg_path = llm_dir / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_path.read_text()) if tok_cfg_path.is_file() else {}
    tok_cfg["chat_template"] = jinja_text
    tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + "\n")

    # mm_hidden from CLIP-L = 1024 for these students
    meta = {
        "preset": args.preset,
        "base": base,
        "lora": str(lora),
        "clip": clip_id,
        "llm_dir": str(llm_dir),
        "projector": str(proj_path),
        "image_projector_type": "mlp2x_gelu",
        "mm_hidden_size": 1024,
        "hidden_size": hidden_size,
        "mm_vision_select_layer": -2,
        "chat_template": preset["conv_name"],
        "image_aspect_ratio": "pad",
        "export_mode": "lean_no_clip",
    }
    (out / "clip_ref.txt").write_text(clip_id + "\n")
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Done. Export root: {out}", flush=True)


if __name__ == "__main__":
    main()
