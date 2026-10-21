"""Compatibility shims for edge inference (transformers >= 4.50)."""
from __future__ import annotations

from transformers import GenerationConfig, GenerationMixin


def patch_model_for_inference(model, conv_mode: str) -> None:
    """Restore generate() and safe generation_config on LLaVA wrapper classes."""
    if model.generation_config is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)

    model.generation_config.repetition_penalty = 1.0
    model.generation_config.do_sample = False

    if not isinstance(model, GenerationMixin):
        model.__class__ = type(
            f"Patched{model.__class__.__name__}",
            (model.__class__, GenerationMixin),
            {},
        )


def _generation_base_class(conv_mode: str):
    from transformers import Gemma2ForCausalLM, LlamaForCausalLM, Qwen2ForCausalLM

    mapping = {
        "qwen": Qwen2ForCausalLM,
        "gemma_2": Gemma2ForCausalLM,
        "llama_3": LlamaForCausalLM,
    }
    if conv_mode in mapping:
        return mapping[conv_mode]
    if conv_mode == "qwen3":
        try:
            from transformers import Qwen3ForCausalLM

            return Qwen3ForCausalLM
        except ImportError:
            return Qwen2ForCausalLM
    return Qwen2ForCausalLM
