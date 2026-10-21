#!/usr/bin/env python3
"""
Single-image fire detection using merged LLaVA-MoD student checkpoints.

Designed for offline edge deployment (no cloud inference). Supports CPU and CUDA.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# Resolve LLaVA-MoD repo from env or default relative path.
_DEPLOY_ROOT = Path(__file__).resolve().parents[1]
_LLAVA_MOD_DEFAULT = _DEPLOY_ROOT.parent / "LLaVA_MoD" / "MoD" / "LLava_MoD"


def _ensure_llavamod_on_path(llava_mod_root: Optional[str] = None) -> Path:
    root = Path(llava_mod_root or _LLAVA_MOD_DEFAULT).resolve()
    if not (root / "llavamod").is_dir():
        raise FileNotFoundError(
            f"LLaVA-MoD not found at {root}. Set LLAVA_MOD_ROOT or pass llava_mod_root."
        )
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


# Shared with GGUF / llama.cpp bench (see Deploy/notes/TIMING_PROTOCOL.md).
TIMING_BOUNDARY = "request_ready"
TIMING_BOUNDARY_DEF = (
    "t0 after RGB image is in memory; includes vision preprocess, device H2D, "
    "encode, prefill, and decode. Excludes model load and disk I/O."
)


@dataclass
class FireDetectResult:
    prompt: str
    raw_text: str
    fire_detected: Optional[bool]
    latency_ms: float
    device: str
    model_path: str
    ttft_ms: Optional[float] = None
    total_generation_ms: Optional[float] = None
    timing_boundary: str = TIMING_BOUNDARY

    def to_dict(self) -> dict:
        return asdict(self)


class FireDetector:
    """Load once, run many images offline."""

    def __init__(
        self,
        model_path: str,
        model_base: Optional[str] = None,
        conv_mode: str = "qwen",
        device: str = "cuda",
        load_4bit: bool = False,
        load_8bit: bool = False,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        llava_mod_root: Optional[str] = None,
    ) -> None:
        _ensure_llavamod_on_path(llava_mod_root)

        from llavamod.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llavamod.conversation import SeparatorStyle, conv_templates
        from llavamod.eval.model_vqa_loader import (
            get_conv_stop_str,
            get_generate_token_kwargs,
            trim_repetitive_answer,
        )
        from llavamod.mm_utils import (
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
        )
        from llavamod.model.builder import load_pretrained_model
        from llavamod.utils import disable_torch_init

        from fire_detect.model_utils import patch_model_for_inference

        self._conv_mode = conv_mode
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._device = device
        self._model_path = model_path

        self._SeparatorStyle = SeparatorStyle
        self._conv_templates = conv_templates
        self._get_conv_stop_str = get_conv_stop_str
        self._get_generate_token_kwargs = get_generate_token_kwargs
        self._trim_repetitive_answer = trim_repetitive_answer
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self._DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self._KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token

        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, processor, _ = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            load_4bit=load_4bit,
            load_8bit=load_8bit,
            device=device,
        )
        patch_model_for_inference(model, conv_mode)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = processor["image"]

        if device == "cuda" and torch.cuda.is_available():
            self.model.eval()
        else:
            self._device = "cpu"
            self.model.eval()

    def _resolve_device(self) -> torch.device:
        if self._device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def parse_fire_yes_no(text: str) -> Optional[bool]:
        t = (text or "").strip()
        if re.match(r"^Yes\b", t, re.IGNORECASE):
            return True
        if re.match(r"^No\b", t, re.IGNORECASE):
            return False
        return None

    def detect(
        self,
        image: str | Path | Image.Image,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        measure_ttft: bool = False,
    ) -> FireDetectResult:
        # Disk I/O is outside the timed region (same contract as GGUF bench).
        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
        else:
            pil = image.convert("RGB")

        conv = self._conv_templates[self._conv_mode].copy()
        qs = prompt
        if getattr(self.model.config, "mm_use_im_start_end", False):
            qs = (
                self._DEFAULT_IM_START_TOKEN
                + self._DEFAULT_IMAGE_TOKEN
                + self._DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = self._DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_str = conv.get_prompt()

        stop_str = self._get_conv_stop_str(conv)
        gen_kwargs = self._get_generate_token_kwargs(conv, self.tokenizer)
        use_cache = False  # stable multimodal decode on transformers >= 4.50
        token_budget = self._max_new_tokens if max_new_tokens is None else max_new_tokens
        generate_kwargs = {
            "max_new_tokens": token_budget,
            "use_cache": use_cache,
            **gen_kwargs,
        }
        if self._temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self._temperature
        else:
            generate_kwargs["do_sample"] = False

        # request_ready: t0 → first token / completion (matches llama.cpp client clock).
        ttft_ms: Optional[float] = None
        t0 = time.perf_counter()
        with torch.inference_mode():
            image_tensor = self._process_images([pil], self.image_processor, self.model.config)
            if isinstance(image_tensor, list):
                image_tensor = image_tensor[0]

            input_ids = self._tokenizer_image_token(
                prompt_str, self.tokenizer, self._IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0)

            dev = self._resolve_device()
            input_ids = input_ids.to(device=dev)
            if isinstance(image_tensor, list):
                images = [
                    t.to(dev, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32)
                    for t in image_tensor
                ]
            else:
                dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
                images = image_tensor.to(dtype=dtype, device=dev)

            stopping = [self._KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)]
            generate_kwargs["stopping_criteria"] = stopping

            if measure_ttft:
                from threading import Thread

                from transformers import TextIteratorStreamer

                streamer = TextIteratorStreamer(
                    self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120.0
                )
                generate_kwargs["streamer"] = streamer
                thread = Thread(
                    target=self.model.generate,
                    kwargs={"input_ids": input_ids, "images": images, **generate_kwargs},
                )
                thread.start()
                chunks: list[str] = []
                for chunk in streamer:
                    if ttft_ms is None and chunk:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                    chunks.append(chunk)
                thread.join()
                raw = "".join(chunks).strip()
                output_ids = None
            else:
                output_ids = self.model.generate(
                    input_ids,
                    images=images,
                    **generate_kwargs,
                )

            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

        total_ms = (time.perf_counter() - t0) * 1000.0
        latency_ms = total_ms

        if output_ids is not None:
            input_len = input_ids.shape[1]
            raw = self.tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]
        raw = self._trim_repetitive_answer(raw.strip())
        if stop_str and raw.endswith(stop_str):
            raw = raw[: -len(stop_str)].strip()

        return FireDetectResult(
            prompt=prompt,
            raw_text=raw,
            fire_detected=self.parse_fire_yes_no(raw),
            latency_ms=latency_ms,
            device=str(dev),
            model_path=self._model_path,
            ttft_ms=ttft_ms,
            total_generation_ms=total_ms,
            timing_boundary=TIMING_BOUNDARY,
        )
