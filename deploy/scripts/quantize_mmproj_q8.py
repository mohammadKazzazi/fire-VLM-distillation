#!/usr/bin/env python3
"""Quantize a legacy CLIP/LLaVA mmproj GGUF F16 → Q8_0 (llama-quantize rejects clip arch)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_LLAMA = Path("/home/ubuntu/llavamod/llama.cpp/gguf-py")
sys.path.insert(0, str(_LLAMA))

import gguf  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()

    reader = gguf.GGUFReader(str(args.input))
    arch = reader.fields["general.architecture"].contents()
    if arch != "clip":
        raise SystemExit(f"expected clip architecture, got {arch!r}")

    writer = gguf.GGUFWriter(str(args.output), arch=arch)
    skip_prefix = ("GGUF.",)
    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith(skip_prefix):
            continue
        if field.name == "general.file_type":
            writer.add_uint32("general.file_type", int(gguf.LlamaFileType.MOSTLY_Q8_0))
            continue
        val_type = field.types[0]
        sub = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub)

    if "general.file_type" not in reader.fields:
        writer.add_uint32("general.file_type", int(gguf.LlamaFileType.MOSTLY_Q8_0))

    qtype = gguf.GGMLQuantizationType.Q8_0
    n_q = n_keep = 0
    for tensor in reader.tensors:
        data = np.asarray(tensor.data)
        # Q8_0 needs last dim divisible by 32; keep conv / odd shapes in f16.
        can_q = (
            data.dtype == np.float16
            and data.ndim == 2
            and data.shape[-1] % 32 == 0
            and data.size >= 32
        )
        if can_q:
            qdata = gguf.quants.quantize(data.astype(np.float32), qtype)
            writer.add_tensor(tensor.name, qdata, raw_dtype=qtype)
            n_q += 1
        else:
            writer.add_tensor(tensor.name, data)
            n_keep += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"Wrote {args.output} (quantized={n_q} kept={n_keep})")


if __name__ == "__main__":
    main()
