"""CPU inference adaptations for H4 / DEVICE=cpu (same protocol as GPU profiles).

Enabled via env (H4 script sets defaults):
  CPU_OPT=1           thread + matmul + KV-cache runtime tuning (default on CPU)
  CPU_INT8=1          dynamic INT8 on nn.Linear (language + projector; vision stays fp32)
  CPU_COMPILE=1       torch.compile after load (optional; slow first compile on ARM)
  CPU_THREADS=N       override worker threads (default: os.cpu_count())
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def cpu_opt_enabled(device: str) -> bool:
    if device != "cpu":
        return False
    return _env_flag("CPU_OPT", default=True)


def configure_cpu_runtime() -> dict[str, Any]:
    """Pin BLAS/Torch threads and enable oneDNN-friendly settings."""
    n = int(os.environ.get("CPU_THREADS") or (os.cpu_count() or 1))
    n = max(1, n)

    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, str(n))

    torch.set_num_threads(n)
    if hasattr(torch, "set_num_interop_threads"):
        # Keep interop modest to avoid oversubscription on small VLM graphs.
        interop = max(1, min(2, n // 4 or 1))
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError:
            # May already have been set; ignore.
            interop = torch.get_num_interop_threads()
    else:
        interop = 1

    # Prefer oneDNN / MKLDNN kernels when available (x86 and many aarch64 builds).
    if hasattr(torch.backends, "mkldnn"):
        try:
            torch.backends.mkldnn.enabled = True
        except Exception:
            pass

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    return {
        "cpu_threads": n,
        "interop_threads": interop,
        "mkldnn": bool(getattr(getattr(torch.backends, "mkldnn", None), "enabled", False)),
        "use_cache": os.environ.get("CPU_USE_CACHE", "0").strip().lower()
        in {"1", "true", "yes", "on"},
    }


def apply_dynamic_int8(model: nn.Module) -> tuple[nn.Module, dict[str, Any]]:
    """Dynamic INT8 on Linear layers, skipping the CLIP vision tower."""
    info: dict[str, Any] = {
        "applied": False,
        "method": "torch.ao.quantize_dynamic",
        "dtype": "qint8",
        "scope": "model.layers_only",
    }
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError:
        from torch.quantization import quantize_dynamic  # type: ignore

    quantized_parts: list[str] = []
    try:
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            # Only decoder Linear layers — quantizing mm_projector breaks multimodal shapes.
            inner.layers = quantize_dynamic(inner.layers, {nn.Linear}, dtype=torch.qint8)
            quantized_parts.append("model.layers")

        info["applied"] = bool(quantized_parts)
        info["parts"] = quantized_parts
        if not quantized_parts:
            info["error"] = "no Linear modules quantized"
    except Exception as exc:
        info["applied"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"
    return model, info


def maybe_torch_compile(model: nn.Module) -> tuple[nn.Module, dict[str, Any]]:
    info: dict[str, Any] = {"applied": False, "backend": "inductor"}
    if not _env_flag("CPU_COMPILE", default=False):
        return model, info
    if not hasattr(torch, "compile"):
        info["error"] = "torch.compile unavailable"
        return model, info
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)  # type: ignore[assignment]
        info["applied"] = True
        info["mode"] = "reduce-overhead"
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return model, info


def optimize_cpu_model(model: nn.Module) -> tuple[nn.Module, dict[str, Any]]:
    """Apply optional INT8 + compile. Call after .to(cpu, float32) and eval()."""
    report: dict[str, Any] = {
        "int8": {"applied": False},
        "compile": {"applied": False},
    }
    if _env_flag("CPU_INT8", default=False):
        model, report["int8"] = apply_dynamic_int8(model)
    model, report["compile"] = maybe_torch_compile(model)
    return model, report


def cpu_inference_settings() -> dict[str, Any]:
    """Snapshot of CPU adaptation flags for benchmark JSON."""
    return {
        "cpu_opt": _env_flag("CPU_OPT", default=True),
        "cpu_int8": _env_flag("CPU_INT8", default=False),
        "cpu_compile": _env_flag("CPU_COMPILE", default=False),
        "cpu_threads": int(os.environ.get("CPU_THREADS") or (os.cpu_count() or 1)),
        "cpu_use_cache": os.environ.get("CPU_USE_CACHE", "0").strip().lower()
        in {"1", "true", "yes", "on"},
    }
