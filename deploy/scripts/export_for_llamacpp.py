#!/usr/bin/env python3
"""
Merge LoRA fire student and export artifacts for llama.cpp / GGUF.

Supports presets: qwen0.5b, llama3-1b, gemma2-2b

Produces:
  <out>/llm/          — plain HF LM folder (for convert_hf_to_gguf.py)
  <out>/llava.projector — mlp2x_gelu weights in LLaVA-compatible key layout
  <out>/clip_ref.txt  — CLIP model id used for mmproj conversion
  <out>/export_meta.json

Usage (from Deploy/ with venv active):
  python3 scripts/export_for_llamacpp.py --preset llama3-1b
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_DEPLOY = Path(__file__).resolve().parents[1]
if str(_DEPLOY) not in sys.path:
    sys.path.insert(0, str(_DEPLOY))

PRESETS = {
    "qwen0.5b": {
        "lora_env": "QWEN0_5B_LORA",
        "base_env": "QWEN0_5B_BASE",
        "default_base": "Qwen/Qwen2.5-0.5B-Instruct",
        "arch": "Qwen2ForCausalLM",
        "model_type": "qwen2",
        "conv_jinja": "llava_qwen_phi.jinja",
        "conv_name": "llava_qwen_phi (conv_phi)",
    },
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_DEPLOY / "config" / "students.env"))
    parser.add_argument("--out", default=None, help="Export root (default: bundles/llamacpp_export/<preset>)")
    parser.add_argument("--preset", default=None, help="qwen0.5b | llama3-1b | gemma2-2b")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--llava-root", default=None, help="Override LLAVA_MOD_ROOT")
    args = parser.parse_args()

    _load_env(Path(args.config))
    preset_name = args.preset or os.environ.get("STUDENT_PRESET", "qwen0.5b")
    if preset_name not in PRESETS:
        raise SystemExit(f"Unknown preset {preset_name!r}; choose from {list(PRESETS)}")
    preset = PRESETS[preset_name]

    llava_root = Path(args.llava_root or os.environ["LLAVA_MOD_ROOT"]).resolve()
    lora = Path(os.environ[preset["lora_env"]]).resolve()
    base = os.environ.get(preset["base_env"], preset["default_base"])
    clip_id = os.environ.get("CLIP_MODEL", "openai/clip-vit-large-patch14-336")

    sys.path.insert(0, str(llava_root))
    from llavamod.mm_utils import get_model_name_from_path
    from llavamod.model.builder import load_pretrained_model
    from llavamod.utils import disable_torch_init

    out = Path(args.out).resolve() if args.out else (_DEPLOY / "bundles" / "llamacpp_export" / preset_name)
    llm_dir = out / "llm"
    out.mkdir(parents=True, exist_ok=True)
    llm_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preset={preset_name}\n  lora={lora}\n  base={base}\n  llava_root={llava_root}")
    disable_torch_init()
    model_name = get_model_name_from_path(str(lora))
    tokenizer, model, _processor, _ = load_pretrained_model(
        str(lora),
        base,
        model_name,
        device=args.device,
        load_4bit=False,
        load_8bit=False,
    )

    # Projector → LLaVA surgery key layout
    proj = {}
    state = model.state_dict()
    mapping = {
        "model.mm_projector.image_spatial_proj.0.weight": "model.mm_projector.0.weight",
        "model.mm_projector.image_spatial_proj.0.bias": "model.mm_projector.0.bias",
        "model.mm_projector.image_spatial_proj.2.weight": "model.mm_projector.2.weight",
        "model.mm_projector.image_spatial_proj.2.bias": "model.mm_projector.2.bias",
    }
    for src, dst in mapping.items():
        if src not in state:
            raise KeyError(f"Missing projector tensor {src}")
        proj[dst] = state[src].detach().to(torch.float16).cpu()
    proj_path = out / "llava.projector"
    torch.save(proj, proj_path)
    print(f"Wrote {proj_path} keys={list(proj)}")
    del proj

    # Snapshot config fields before freeing the multimodal model (RAM-critical on H4 16GB).
    import gc

    mm_hidden_size = int(getattr(model.config, "mm_hidden_size", 1024))
    hidden_size = int(model.config.hidden_size)
    mm_vision_select_layer = int(getattr(model.config, "mm_vision_select_layer", -2))
    embed_rows = model.get_input_embeddings().weight.shape[0]

    print(f"Saving language model ({preset['arch']}) ...")
    lm_cfg = AutoConfig.from_pretrained(base, trust_remote_code=True, local_files_only=True)
    if getattr(lm_cfg, "vocab_size", None) != embed_rows:
        print(f"Adjusting vocab_size {lm_cfg.vocab_size} -> {embed_rows}")
        lm_cfg.vocab_size = embed_rows

    # Drop vision tower early if present (CLIP ~1GB) before duplicating LM weights.
    try:
        if hasattr(model, "get_model") and hasattr(model.get_model(), "image_tower"):
            model.get_model().image_tower = None
        if hasattr(model, "model") and hasattr(model.model, "image_tower"):
            model.model.image_tower = None
    except Exception:
        pass
    gc.collect()

    # Build CPU fp16 weight dict using a meta-device probe (no RAM), then free LLaVA.
    with torch.device("meta"):
        shape_probe = AutoModelForCausalLM.from_config(lm_cfg)
        shape_ref = {k: tuple(v.shape) for k, v in shape_probe.state_dict().items()}
        del shape_probe

    llava_sd = model.state_dict()
    filtered = {}
    copied = 0
    skipped = []
    for k, shape in shape_ref.items():
        if k in llava_sd and tuple(llava_sd[k].shape) == shape:
            filtered[k] = llava_sd[k].detach().to(torch.float16).cpu().contiguous().clone()
            copied += 1
        else:
            skipped.append(k)
    del llava_sd, model, shape_ref
    gc.collect()

    lm = AutoModelForCausalLM.from_config(lm_cfg, torch_dtype=torch.float16)
    try:
        missing, unexpected = lm.load_state_dict(filtered, strict=False, assign=True)
    except TypeError:
        missing, unexpected = lm.load_state_dict(filtered, strict=False)
    print(f"Copied {copied} tensors; skipped={len(skipped)} missing={missing} unexpected={unexpected}")
    if skipped:
        print(f"  skipped sample: {skipped[:5]}")
    filtered.clear()
    del filtered
    gc.collect()

    lm.save_pretrained(llm_dir, safe_serialization=True)
    del lm
    gc.collect()
    tokenizer.save_pretrained(llm_dir)

    cfg_path = llm_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["architectures"] = [preset["arch"]]
    cfg["model_type"] = preset["model_type"]
    cfg["torch_dtype"] = "float16"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    # Bake training chat template
    jinja_src = _DEPLOY / "bundles" / "gguf" / preset["conv_jinja"]
    if not jinja_src.is_file():
        raise FileNotFoundError(f"Missing chat template: {jinja_src}")
    jinja_text = jinja_src.read_text()
    (llm_dir / "chat_template.jinja").write_text(jinja_text)
    tok_cfg_path = llm_dir / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_path.read_text()) if tok_cfg_path.is_file() else {}
    tok_cfg["chat_template"] = jinja_text
    tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + "\n")
    print(f"Baked {preset['conv_name']} into {llm_dir / 'chat_template.jinja'}")

    (out / "clip_ref.txt").write_text(clip_id + "\n")
    meta_out = {
        "preset": preset_name,
        "base": base,
        "lora": str(lora),
        "clip": clip_id,
        "llm_dir": str(llm_dir),
        "projector": str(proj_path),
        "image_projector_type": "mlp2x_gelu",
        "mm_hidden_size": mm_hidden_size,
        "hidden_size": hidden_size,
        "mm_vision_select_layer": mm_vision_select_layer,
        "chat_template": preset["conv_name"],
        "image_aspect_ratio": "pad",
    }
    (out / "export_meta.json").write_text(json.dumps(meta_out, indent=2) + "\n")
    print(f"Done. Export root: {out}")


if __name__ == "__main__":
    main()
